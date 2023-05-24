import os
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score


def calculate_metric(text_pred_path, video_pred_path, label_path, strategy):
    text_logits = np.load(text_pred_path)
    video_probs = np.load(video_pred_path)
    label = np.load(label_path)

    text_preds = np.argmax(text_logits, axis=1)

    if strategy == "Identity Declaration":
        idx = 0
    elif strategy == "Accusation":
        idx = 1
    elif strategy == "Interrogation":
        idx = 2
    elif strategy == "Call for Action":
        idx = 3
    elif strategy == "Defense":
        idx = 4
    elif strategy == "Evidence":
        idx = 5
    else:
        raise ValueError
    video_accusation_preds = (video_probs[:, idx] > 0.5).astype(np.int64)

    # fuse
    text_prob = torch.tensor(text_logits).softmax(dim=1)
    text_prob = text_prob.numpy()[:, 1]
    video_prob = video_probs[:, idx]
    alpha = 0.2
    final_prob = alpha * video_prob + (1-alpha) * text_prob
    final_preds = (final_prob > 0.5).astype(np.int64)

    text_preds = text_preds.tolist()
    label = label.tolist()
    video_accusation_preds = video_accusation_preds.tolist()
    final_preds = final_preds.tolist()


    text_f1 = f1_score(label, text_preds)
    video_f1 = f1_score(label, video_accusation_preds)
    final_f1 = f1_score(label, final_preds)
    # print(text_f1)
    # print(video_f1)
    # print(final_f1)

    return text_f1, video_f1, final_f1


def calculate_metric_2(text_pred_path, video_pred_path, label_path, strategy):
    text_logits = np.load(text_pred_path)
    video_logits = np.load(video_pred_path)
    label = np.load(label_path)

    text_preds = np.argmax(text_logits, axis=1)
    video_preds = np.argmax(video_logits, axis=1)

    # fuse
    text_prob = torch.tensor(text_logits).softmax(dim=1)
    text_prob = text_prob.numpy()[:, 1]
    video_prob = torch.tensor(video_logits).softmax(dim=1)
    video_prob = video_prob.numpy()[:, 1]
    alpha = 0.8
    final_prob = alpha * video_prob + (1-alpha) * text_prob
    final_preds = (final_prob > 0.5).astype(np.int64)

    text_correct = (text_preds == label).astype(np.int64)
    video_correct = (video_preds == label).astype(np.int64)
    final_correct = (final_preds == label).astype(np.int64)

    text_preds = text_preds.tolist()
    label = label.tolist()
    video_preds = video_preds.tolist()
    final_preds = final_preds.tolist()


    text_f1 = f1_score(label, text_preds)
    video_f1 = f1_score(label, video_preds)
    final_f1 = f1_score(label, final_preds)
    # print(text_f1)
    # print(video_f1)
    # print(final_f1)

    return text_f1, video_f1, final_f1, text_correct, video_correct, final_correct


def aggregate(text_pred_dir, video_pred_dir):
    all_text_f1, all_video_f1, all_final_f1 = list(), list(), list()
    all_text_acc, all_video_acc, all_final_acc = list(), list(), list()
    video_seed = ['227', '624', '817']
    text_seed = ['227', '624', '817']
    for v_seed, t_seed in zip(video_seed, text_seed):
        all_text_correct, all_video_correct, all_final_correct = None, None, None
        for strategy in ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]:
            video_pred_path = os.path.join(video_pred_dir, v_seed, strategy, 'avalon_logits_test.npy')
            text_pred_path = os.path.join(text_pred_dir, t_seed, strategy, 'avalon_logits_test.npy')
            label_path = os.path.join(text_pred_dir, t_seed, strategy, 'avalon_labels_test.npy')
            # text_f1, video_f1, final_f1 = calculate_metric(text_pred_path=text_pred_path, video_pred_path=video_pred_path, label_path=label_path, strategy=strategy)
            text_f1, video_f1, final_f1, text_correct, video_correct, final_correct\
                = calculate_metric_2(text_pred_path=text_pred_path, video_pred_path=video_pred_path, label_path=label_path, strategy=strategy)
            all_text_f1.append(text_f1)
            all_video_f1.append(video_f1)
            all_final_f1.append(final_f1)

            all_text_correct = text_correct if all_text_correct is None else all_text_correct * text_correct
            all_video_correct = video_correct if all_video_correct is None else all_video_correct * video_correct
            all_final_correct = final_correct if all_final_correct is None else all_final_correct * final_correct

        all_text_acc.append(all_text_correct.mean())
        all_video_acc.append(all_video_correct.mean())
        all_final_acc.append(all_final_correct.mean())

    all_text_f1 = np.array(all_text_f1).reshape((3, -1))
    all_video_f1 = np.array(all_video_f1).reshape((3, -1))
    all_final_f1 = np.array(all_final_f1).reshape((3, -1))
    all_text_acc = np.array(all_text_acc)
    all_video_acc = np.array(all_video_acc)
    all_final_acc = np.array(all_final_acc)
    print(all_text_f1.mean(axis=0))
    print(all_video_f1.mean(axis=0))
    print(all_final_f1.mean(axis=0))
    print(all_text_f1.std(axis=0))
    print(all_video_f1.std(axis=0))
    print(all_final_f1.std(axis=0))
    print('text_f1:', all_text_f1.mean(), all_text_f1.mean(axis=1).std())
    print('video_f1:', all_video_f1.mean(), all_video_f1.mean(axis=1).std())
    print('final_f1:', all_final_f1.mean(), all_final_f1.mean(axis=1).std())
    print('text_acc:', all_text_acc.mean(), all_text_acc.std())
    print('video_acc:', all_video_acc.mean(), all_video_acc.std())
    print('final_acc:', all_final_acc.mean(), all_final_acc.std())


if __name__ == '__main__':
    aggregate(text_pred_dir='out/Ego4D/roberta/context/5',
              video_pred_dir='out/Ego4D/roberta/video_32x3_K400_3crop')
