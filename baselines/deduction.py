import argparse
import json
import logging as log
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm


from read_data import *
from models import *


logger = log.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='out', type=str)
parser.add_argument("--log_dir", default='log.txt', type=str)
parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
parser.add_argument("--dataset", nargs='+', default=('Ego4D', 'Youtube'), type=str, help="Name of dataset, Ego4D or Youtube")

parser.add_argument("--gpu", default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--no_train", action="store_true", help="Whether to run training.")
parser.add_argument("--no_eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument("--no_test", action="store_true", help="Whether to run predictions on the test set.")
parser.add_argument("--no_evaluate_during_training", action="store_true",
                    help="Whether to run evaluation every epoch.")
parser.add_argument("--evaluate_period", default=10, type=int, help="evaluate every * epochs.")
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument('--eval_batch_size', default=128, type=int)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")

parser.add_argument("--num_train_epochs", default=1000, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--early_stopping_patience', default=10000, type=int, help="Patience for early stopping.")
parser.add_argument('--logging_steps', default=40, type=int, help="Log every X updates steps.")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument("--avalon", action="store_true", help="Testing on avalon data as well")
parser.add_argument("--video", action="store_true", help="Using video features")
parser.add_argument("--video_path", type=str, default='data/Ego4D/video_feature', help="Path to video features")

args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

if (os.path.exists(args.output_dir) and os.listdir(
		args.output_dir) and not args.no_train and not args.overwrite_output_dir):
	raise ValueError(
		"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
			args.output_dir))
logger.setLevel(log.INFO)
formatter = log.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

if not os.path.exists(args.output_dir):
	os.makedirs(args.output_dir)

fh = log.FileHandler(os.path.join(args.output_dir, args.log_dir))
fh.setLevel(log.INFO)
fh.setFormatter(formatter)

ch = log.StreamHandler()
ch.setLevel(log.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

Strategies = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]


def cal_score(labels, preds):
	all_total = 0
	all_correct = 0
	joint_a = 0
	for label, pred in zip(labels, preds):
		total = 0
		correct = 0
		for l, p in zip(label, pred):
			if l == -100:
				break
			total += 1
			correct += (1 if l == p else 0)
		all_total += total
		all_correct += correct
		joint_a += (1 if total == correct else 0)
	return all_correct / all_total, joint_a / len(labels)


def train(model, train_dataset, dev_dataset):
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	tb_writer = SummaryWriter(args.output_dir)

	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

	t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info(
		"  Total train batch size (w. parallel, accumulation) = %d",
		args.batch_size
		* args.gradient_accumulation_steps),
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	wait_step = 0
	epochs_trained = 0
	steps_trained_in_current_epoch = 0
	best_f1 = 0
	tr_loss, logging_loss = 0.0, 0.0

	model.zero_grad()

	train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc='Epoch')

	for epoch in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration")
		for step, batch in enumerate(epoch_iterator):
			if steps_trained_in_current_epoch > 0:
				steps_trained_in_current_epoch -= 1
				continue
			model.train()

			batch = tuple(t.to(args.device) for t in batch)
			# print(epoch, batch)
			loss, logits = model(inputs=batch[0], labels=batch[1])
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			loss.backward()
			tr_loss += loss.item()

			if (step + 1) % args.gradient_accumulation_steps == 0:
				optimizer.step()
				model.zero_grad()
				global_step += 1

				if args.logging_steps > 0 and global_step % args.logging_steps == 0:
					tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
					logging_loss = tr_loss

			# logger.info("logging train info!!!")
			# logger.info("*")

		# eval and save the best model based on dev set after each epoch
		if not args.no_evaluate_during_training and epoch % args.evaluate_period == 0:

			results = evaluate(model, dev_dataset, mode="dev", prefix=str(global_step))
			for i, (key, value) in enumerate(results.items()):
				tb_writer.add_scalar("eval_{}".format(key), value, epoch)
			# tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
			# tb_writer.add_scalar("loss", tr_loss - logging_loss, epoch)
			logging_loss = tr_loss
			logger.info(f"{results}")
			if results['f1'] >= best_f1:
				best_f1 = results['f1']
				wait_step = 0
				output_dir = os.path.join(args.output_dir, "best")
				if not os.path.exists(output_dir):
					os.makedirs(output_dir)
				logger.info("Saving best model to %s", output_dir)
				model_to_save = (
					model.module if hasattr(model, "module") else model)
				torch.save(model_to_save, os.path.join(output_dir, "model.bin"))
			else:
				wait_step += 1
				if wait_step >= args.early_stopping_patience:
					train_iterator.close()
					break

	tb_writer.close()
	return global_step, tr_loss / global_step, best_f1


def evaluate(model, eval_dataset=None, mode='dev', prefix=''):
	eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

	logger.info("***** Running evaluation %s *****", mode + '-' + prefix)
	logger.info("  Num examples = %d", len(eval_dataset))
	logger.info("  Batch size = %d", args.eval_batch_size)

	eval_loss = 0.0
	nb_eval_steps = 0
	preds = None
	labels = None
	model.eval()

	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		batch = tuple(t.to(args.device) for t in batch)

		with torch.no_grad():
			loss, logits = model(inputs=batch[0], labels=batch[1])
			eval_loss += loss.item()
		nb_eval_steps += 1

		if preds is None:
			preds = logits.detach().cpu().numpy()
			labels = batch[1].detach().cpu().numpy()
		else:
			preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
			labels = np.append(labels, batch[1].detach().cpu().numpy(), axis=0)

	eval_loss /= nb_eval_steps
	# print(preds.shape)
	preds = np.argmax(preds, axis=1)
	# print(preds.shape)
	labels = labels.tolist()
	preds = preds.tolist()
	# utterance_encodings = utterance_encodings.tolist()

	assert len(labels) == len(preds), f"{len(labels)}, {len(preds)}"

	# scores = cal_score(preds, labels)
	# if prefix == 'final':
	# 	print(preds)
	# 	print(labels)
	# 	results = {
	# 		'accuracy': scores[0],
	# 		'Joint-A': scores[1],
	# 		'preds': preds,
	# 	}
	# else:
	# 	results = {
	# 		'accuracy': scores[0],
	# 		'Joint-A': scores[1],
	# 		'loss': eval_loss
	# 	}
	if prefix == 'final':
		results = {
			'f1': f1_score(y_true=labels, y_pred=preds),
			'precision': precision_score(y_true=labels, y_pred=preds),
			'recall': recall_score(y_true=labels, y_pred=preds),
			'accuracy': accuracy_score(y_true=labels, y_pred=preds),
			'report': classification_report(y_true=labels, y_pred=preds),
			'preds': preds,
		}
	else:
		results={
			'f1': f1_score(y_true=labels, y_pred=preds),
			'loss': eval_loss
		}
	logger.info(results)
	return results


def main():
	logger.info("------NEW RUN-----")

	logger.info("device: %s, gpu: %s", args.device, args.gpu)
	logger.info("random seed %s", args.seed)
	logger.info("Training/evaluation parameters %s", args)

	splits = []

	if not args.no_eval:
		splits.append('dev')
	if not args.no_test:
		splits.append('test')
	if args.avalon:
		splits.append('avalon')

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if len(args.gpu) > 0:
		torch.cuda.manual_seed_all(args.seed)

	# model = Deduction_simple(120, 7)
	model = Deduction_simple_paired(14, 2)
	model.to(args.device)

	dataset = {}
	for split in ("train", "dev", "test"):
		dataset[split] = read_data_for_deduction_simple_paired(args, logger, split)

	_, _, best_score = train(model, dataset["train"], dataset["dev"])
	logger.info(f"best_score: {best_score}")

	model = torch.load(os.path.join(args.output_dir, "best", "model.bin"))
	model.to(args.device)

	evaluate(model, dataset["dev"], mode="dev", prefix="final")
	evaluate(model, dataset["test"], mode="test", prefix="final")


if __name__ == "__main__":
	main()