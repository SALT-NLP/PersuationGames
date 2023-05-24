# These codes are used to calculate and save the probabilities of prediction.

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

from transformers import *

from read_data import *
from models import *


MODEL_CLASSES = {"bert": BertForSequenceClassification, "roberta": RobertaForSequenceClassification}
MODEL_WITH_VIDEO_CLASSES = {"bert": BertForSequenceClassificationWithVideo,
                            "roberta": RobertaForSequenceClassificationWithVideo}
TOKENIZER_CLASSES = {"bert": BertTokenizer, "roberta": RobertaTokenizer}

logger = log.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default='bert', type=str)
parser.add_argument("--model_name", default='bert-base-uncased', type=str)
parser.add_argument("--output_dir", default='out', type=str)
parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
parser.add_argument("--dataset", default='Ego4D', type=str, help="Name of dataset, Ego4D or Youtube")

parser.add_argument("--gpu", default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--no_train", action="store_true", help="Whether to run training.")
parser.add_argument("--no_eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument("--no_test", action="store_true", help="Whether to run predictions on the test set.")
parser.add_argument("--no_evaluate_during_training", action="store_true",
                    help="Whether to run evaluation every epoch.")
parser.add_argument("--evaluate_period", default=1, type=int, help="evaluate every * epochs.")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument('--eval_batch_size', default=128, type=int)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--warmup_steps', default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument('--early_stopping_patience', default=10000, type=int, help="Patience for early stopping.")
parser.add_argument('--logging_steps', default=40, type=int, help="Log every X updates steps.")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument("--context_size", type=int, default=3, help="size of the context")
parser.add_argument("--avalon", action="store_true", help="Testing on avalon data as well")
parser.add_argument("--video", action="store_true", help="Using video features")
parser.add_argument("--video_path", type=str, default='data/Ego4D/video_feature', help="Path to video features")
parser.add_argument("--deduction", action="store_true", help="Deduction prediction")
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

fh = log.FileHandler(args.output_dir + '/log.txt')
fh.setLevel(log.INFO)
fh.setFormatter(formatter)

ch = log.StreamHandler()
ch.setLevel(log.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

Strategies = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]


def train(model, train_dataset, dev_dataset):
    tb_writer = SummaryWriter(args.output_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, accumulation) = %d",
                args.batch_size * args.gradient_accumulation_steps),
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

            # frames = read_video(args.dataset, 'train', batch[-1].numpy().tolist())
            batch = tuple(t.to(args.device) for t in batch)
            # print(epoch, batch)
            target = batch[2]
            if args.video:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "video_features": batch[3]}
                outputs = model(inputs['input_ids'], labels=target, attention_mask=inputs["attention_mask"],
                                video_features=inputs["video_features"])
            else:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                outputs = model(inputs['input_ids'], labels=target, attention_mask=inputs["attention_mask"], )

            loss = outputs['loss']

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    logger.info("logging train info!!!")
                    logger.info("*")

            # if args.save_steps > 0 and global_step % args.save_steps == 0:
            # 	output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            # 	if not os.path.exists(output_dir):
            # 		os.makedirs(output_dir)
            # 	model_to_save = (model.module if hasattr(model, "module") else model)
            # 	model_to_save.save_pretrained(output_dir)
            # 	torch.save(args, os.path.join(output_dir, "training_args.bin"))
            # 	logger.info("Saving model checkpoint to %s", output_dir)
            #
            # 	torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            # 	torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            # 	logger.info("Saving optimizer and scheduler states to %s", output_dir)

        # eval and save the best model based on dev set after each epoch
        if not args.no_evaluate_during_training and epoch % args.evaluate_period == 0:

            results, _ = evaluate(model, dev_dataset, mode="dev", prefix=str(global_step))
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
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
            else:
                wait_step += 1
                if wait_step >= args.early_stopping_patience:
                    train_iterator.close()
                    break

    tb_writer.close()
    return global_step, tr_loss / global_step, best_f1


def evaluate(model, eval_dataset=None, mode='dev', prefix='', output_dir=None):
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

    logger.info("***** Running evaluation %s *****", mode + '-' + prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    labels = None
    utterance_encodings = None
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            target = batch[2]
            if args.video:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "video_features": batch[3]}
                outputs = model(inputs['input_ids'], labels=target, attention_mask=inputs["attention_mask"],
                                video_features=inputs["video_features"], output_hidden_states=args.deduction)
            else:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                outputs = model(inputs['input_ids'], labels=target, attention_mask=inputs["attention_mask"],
                                output_hidden_states=args.deduction)

            # logger.info(outputs)
            logits, tmp_eval_loss = outputs['logits'], outputs['loss']
            if args.deduction:
                hidden_states = outputs['hidden_states'][-1][:][0]
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = target.detach().cpu().numpy()
            if args.deduction:
                utterance_encodings = hidden_states.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, target.detach().cpu().numpy(), axis=0)
            if args.deduction:
                utterance_encodings = np.append(utterance_encodings, hidden_states.detach().cpu().numpy(), axis=0)

    np.save(f'{output_dir}/avalon_logits_test.npy', preds)
    np.save(f'{output_dir}/avalon_labels_test.npy', labels)
    return None, None

    eval_loss /= nb_eval_steps

    preds = np.argmax(preds, axis=1)

    labels = labels.tolist()
    preds = preds.tolist()
    # utterance_encodings = utterance_encodings.tolist()

    assert len(labels) == len(preds), f"{len(labels)}, {len(preds)}"
    correct = [1 if pred == label else 0 for pred, label in zip(preds, labels)]
    if prefix == 'final':
        results = {
            'f1': f1_score(y_true=labels, y_pred=preds),
            'precision': precision_score(y_true=labels, y_pred=preds),
            'recall': recall_score(y_true=labels, y_pred=preds),
            'accuracy': accuracy_score(y_true=labels, y_pred=preds),
            'report': classification_report(y_true=labels, y_pred=preds),
            'correct': correct,
            'preds': preds,
        }
    else:
        results = {
            'f1': f1_score(y_true=labels, y_pred=preds),
            'loss': eval_loss
        }
    logger.info(results['f1'])
    return results, utterance_encodings


def log_predictions(dataset, splits, preds):
    for split in splits:
        with open(os.path.join('data', dataset, f'{split}.json'), 'r') as f:
            games = json.load(f)
        id = -1
        data = defaultdict(list)
        for game in games:
            # print(f"{game['EG_ID']}_{game['Game_ID']}")
            for record in game['Dialogue']:
                id += 1
                pred = []
                for strategy in Strategies:
                    if preds[split][strategy][id] == 1:
                        pred.append(strategy)
                if len(pred) == 0:
                    pred.append("No Strategy")
                for key, val in record.items():
                    data[key].append(val)
                data['prediction'].append(pred)
        # print(f"{record['Rec_Id']},{record['speaker']},{record['timestamp']},{record['utterance']},{record['annotation']},{pred}")
        df = pd.DataFrame.from_dict(data)
        df.to_csv(os.path.join(args.output_dir, f'predictions_{split}.csv'))


def train_deduction(model, train_dataset, dev_dataset):
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)
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

                    logger.info("logging train info!!!")
                    logger.info("*")

        # eval and save the best model based on dev set after each epoch
        if not args.no_evaluate_during_training and epoch % args.evaluate_period == 0:

            results = evaluate_deduction(model, dev_dataset, mode="dev", prefix=str(global_step))
            for i, (key, value) in enumerate(results.items()):
                tb_writer.add_scalar("eval_{}".format(key), value, epoch)
            # tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
            # tb_writer.add_scalar("loss", tr_loss - logging_loss, epoch)
            logging_loss = tr_loss
            logger.info(f"{results}")
            if results['f1'] >= best_f1:
                best_f1 = results['f1']
                wait_step = 0
            # output_dir = os.path.join(args.output_dir, "best")
            # if not os.path.exists(output_dir):
            # 	os.makedirs(output_dir)
            # logger.info("Saving best model to %s", output_dir)
            # model_to_save = (
            # 	model.module if hasattr(model, "module") else model)
            # model_to_save.save_pretrained(output_dir)
            # torch.save(args, os.path.join(output_dir, "training_args.bin"))
            else:
                wait_step += 1
                if wait_step >= args.early_stopping_patience:
                    train_iterator.close()
                    break

    tb_writer.close()
    return global_step, tr_loss / global_step, best_f1


def evaluate_deduction(model, eval_dataset=None, mode='dev', prefix=''):
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
    print(preds.shape)
    preds = np.argmax(preds, axis=1)

    labels = labels.tolist()
    preds = preds.tolist()
    # utterance_encodings = utterance_encodings.tolist()

    assert len(labels) == len(preds), f"{len(labels)}, {len(preds)}"

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
        results = {
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

    all_result = {'dev': {}, 'test': {}, 'avalon': {}}
    all_correct = {'dev': None, 'test': None, 'avalon': None}
    output_dir = args.output_dir
    if args.video:
        model_class = MODEL_WITH_VIDEO_CLASSES[args.model_type]
    else:
        model_class = MODEL_CLASSES[args.model_type]

    tokenizer_class = TOKENIZER_CLASSES[args.model_type]
    preds = {'dev': {}, 'test': {}, 'avalon': {}}
    averaged_f1 = {'dev': 0.0, 'test': 0.0, 'avalon': 0.0}
    splits = []

    utterance_encodings = {'dev': None, 'test': None, 'train': None}

    if not args.no_eval:
        splits.append('dev')
    if not args.no_test:
        splits.append('test')
    if args.avalon:
        splits.append('avalon')

    for strategy in Strategies:
        logger.info(f"Training for strategy {strategy}")
        args.output_dir = os.path.join(output_dir, strategy)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if len(args.gpu) > 0:
            torch.cuda.manual_seed_all(args.seed)
        model = model_class.from_pretrained(args.model_name, num_labels=2)
        model.to(args.device)
        tokenizer = tokenizer_class.from_pretrained(args.model_name)
        if args.context_size != 0:
            tokenizer.add_tokens(['<end of text>'], special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))

        train_dataset = read_data(args, logger, strategy, tokenizer, mode='train')
        dev_dataset = read_data(args, logger, strategy, tokenizer, mode='dev')

        if not args.no_train:
            global_step, tr_loss, best_f1 = train(model, train_dataset, dev_dataset)
            logger.info(" global_step = %s, average loss = %s, best eval f1 = %s", global_step, tr_loss, best_f1)
            logger.info("Reloading best model")

        model = model_class.from_pretrained(os.path.join(args.output_dir, 'best'))
        model.to(args.device)

        if not args.no_eval:
            results, utterance_encoding = evaluate(model, dev_dataset, mode="dev", prefix='final')
            filename = os.path.join(args.output_dir, 'results_dev.json')
            with open(filename, 'w') as f:
                json.dump(results, f)

            if all_correct['dev'] is None:
                all_correct['dev'] = results['correct']
            else:
                all_correct['dev'] = [x + y for x, y in zip(all_correct['dev'], results['correct'])]

            if args.deduction:
                if utterance_encodings['dev'] is None:
                    utterance_encodings['dev'] = utterance_encoding
                else:
                    utterance_encodings['dev'] = np.sum([utterance_encodings['dev'], utterance_encoding], axis=0)

            preds['dev'][strategy] = results['preds']
            results.pop('correct')
            results.pop('preds')
            averaged_f1['dev'] += results['f1']
            all_result['dev'][strategy] = results

        if not args.no_test:
            test_dataset = read_data(args, logger, strategy, tokenizer, mode='test')
            results, utterance_encoding = evaluate(model, test_dataset, mode="test", prefix='final', output_dir=args.output_dir)
            # filename = os.path.join(args.output_dir, 'results_test.json')
            # with open(filename, 'w') as f:
            #     json.dump(results, f)
            #
            # if all_correct['test'] is None:
            #     all_correct['test'] = results['correct']
            # else:
            #     all_correct['test'] = [x + y for x, y in zip(all_correct['test'], results['correct'])]
            #
            # if args.deduction:
            #     if utterance_encodings['test'] is None:
            #         utterance_encodings['test'] = utterance_encoding
            #     else:
            #         utterance_encodings['test'] = np.sum([utterance_encodings['test'], utterance_encoding], axis=0)
            #
            # preds['test'][strategy] = results['preds']
            # results.pop('correct')
            # results.pop('preds')
            # averaged_f1['test'] += results['f1']
            # all_result['test'][strategy] = results

        if args.avalon:
            avalon_dataset = read_data(args, logger, strategy, tokenizer, mode='avalon')
            results, utterance_encoding = evaluate(model, avalon_dataset, mode="avalon", prefix='final', output_dir=args.output_dir)
            # filename = os.path.join(args.output_dir, 'results_avalon.json')
            # with open(filename, 'w') as f:
            #     json.dump(results, f)
            #
            # if all_correct['avalon'] is None:
            #     all_correct['avalon'] = results['correct']
            # else:
            #     all_correct['avalon'] = [x + y for x, y in zip(all_correct['avalon'], results['correct'])]
            #
            # preds['avalon'][strategy] = results['preds']
            # results.pop('correct')
            # results.pop('preds')
            # averaged_f1['avalon'] += results['f1']
            # all_result['avalon'][strategy] = results

        if args.deduction:
            results, utterance_encoding = evaluate(model, train_dataset, mode="train", prefix='final')
            if utterance_encodings['train'] is None:
                utterance_encodings['train'] = utterance_encoding
            else:
                utterance_encodings['train'] = np.sum([utterance_encodings['train'], utterance_encoding], axis=0)


if __name__ == "__main__":
    main()