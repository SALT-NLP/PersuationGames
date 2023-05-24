import argparse
import logging as log

import numpy as np
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from read_data import *

# import pingouin as pg


logger = log.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--kernel", default='rbf', type=str)
parser.add_argument("--dataset", nargs='+', default=('Ego4D', 'Youtube'), type=str, help="Name of dataset, Ego4D or Youtube")
parser.add_argument("--role_embed", default=False, type=bool, help="Use starting role in features or not")
parser.add_argument("--output_dir", default=None, type=str)
parser.add_argument("--log_dir", default='log.txt', type=str)
parser.add_argument("--svc", action="store_true")
parser.add_argument("--svc_bag", action="store_true")
parser.add_argument("--lr", action='store_true')
parser.add_argument("--lr_bag", action='store_true')
parser.add_argument("--lr_boost", action='store_true')
parser.add_argument("--random_forest", action='store_true')
args = parser.parse_args()

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


train_data, train_labels = read_data_for_deduction_simple_paired(args, logger, 'train', role_embed=args.role_embed, organize_in_dataset=False)
val_data, val_labels = read_data_for_deduction_simple_paired(args, logger, 'dev', role_embed=args.role_embed, organize_in_dataset=False)
test_data, test_labels = read_data_for_deduction_simple_paired(args, logger, 'test', role_embed=args.role_embed, organize_in_dataset=False)
train_data, train_labels = train_data.numpy(), train_labels.numpy()
val_data, val_labels = val_data.numpy(), val_labels.numpy()
test_data, test_labels = test_data.numpy(), test_labels.numpy()


if args.svc:
    print("================== SVC ====================")
    # for weight in [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]:  # grid search
    #     for C in [0.6, 0.8, 1.0, 1.2, 1.4]:
    svc_weight = {0: 1.0, 1: 4.8}  # best hyperparameters
    C = 1.0
    # svc_weight = {0: 1.0, 1: weight}
    svc = svm.SVC(kernel=args.kernel, class_weight=svc_weight, C=C, verbose=False)
    svc.fit(train_data, train_labels)
    train_pred_svc = svc.predict(train_data)
    val_pred_svc = svc.predict(val_data)
    test_pred_svc = svc.predict(test_data)

    train_f1_svc = f1_score(y_true=train_labels, y_pred=train_pred_svc)
    val_f1_svc = f1_score(y_true=val_labels, y_pred=val_pred_svc)
    test_f1_svc = f1_score(y_true=test_labels, y_pred=test_pred_svc)
    recall_svc = (test_pred_svc * test_labels).sum() / test_labels.sum()
    precision_svc = (test_pred_svc * test_labels).sum() / test_pred_svc.sum()
    auc_svc = roc_auc_score(y_true=test_labels, y_score=test_pred_svc)
    # print("weight:", weight, "C:", C)
    print('train f1:', train_f1_svc)
    print('val f1:', val_f1_svc)
    print('test f1:', test_f1_svc, 'test recall:', recall_svc, 'test precision:', precision_svc)
    print('test auc:', auc_svc)


if args.svc_bag:
    print("================== SVC Bagging ====================")
    # for weight in [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]:  # grid search
    #     for C in [0.6, 0.8, 1.0, 1.2, 1.4]:
    # for m_sample in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #     for m_feature in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    svc_weight = {0: 1.0, 1: 4.8}
    C = 1.0
    m_sample = 0.7
    m_feature = 0.9
    # svc_weight = {0: 1.0, 1: weight}
    svc = svm.SVC(kernel=args.kernel, class_weight=svc_weight, C=C, verbose=False)
    bag_svc = BaggingClassifier(base_estimator=svc, n_estimators=20, max_samples=m_sample, max_features=m_feature, random_state=13)
    bag_svc.fit(train_data, train_labels)
    train_pred_svc = bag_svc.predict(train_data)
    val_pred_svc = bag_svc.predict(val_data)
    test_pred_svc = bag_svc.predict(test_data)

    train_f1_svc = f1_score(y_true=train_labels, y_pred=train_pred_svc)
    val_f1_svc = f1_score(y_true=val_labels, y_pred=val_pred_svc)
    test_f1_svc = f1_score(y_true=test_labels, y_pred=test_pred_svc)
    recall_svc = (test_pred_svc * test_labels).sum() / test_labels.sum()
    precision_svc = (test_pred_svc * test_labels).sum() / test_pred_svc.sum()
    auc_svc = roc_auc_score(y_true=test_labels, y_score=test_pred_svc)
    # print("weight:", weight, "C:", C)
    # print("max_sample:", m_sample, "max_feature:", m_feature)
    print('train f1:', train_f1_svc)
    print('val f1:', val_f1_svc)
    print('test f1:', test_f1_svc, 'test recall:', recall_svc, 'test precision:', precision_svc)
    print('test auc:', auc_svc)


if args.lr:
    print("================== Logistic Regression ====================")
    # for weight in [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]:  # grid search
    #     for C in [0.6, 0.8, 1.0, 1.2, 1.4]:
    #         lr_weight = {0: 1.0, 1: weight}

    if not args.role_embed:
        lr_weight = {0: 1.0, 1: 4.2}  # only strategy distribution
        C = 1.4
    else:
        # lr_weight = {0: 1.0, 1: 4.0}  # add starting role
        # C = 1.4
        lr_weight = {0: 1.0, 1: 4.8}  # add starting role on for voter
        C = 0.8
    # lr_weight = {0: 1.0, 1: 3.0}  # add ending role
    # C = 1.2

    logistic_regression = LogisticRegression(class_weight=lr_weight, C=C)
    logistic_regression.fit(train_data, train_labels)
    train_pred_lr = logistic_regression.predict(train_data)
    val_pred_lr = logistic_regression.predict(val_data)
    test_pred_lr = logistic_regression.predict(test_data)

    train_f1_lr = f1_score(y_true=train_labels, y_pred=train_pred_lr)
    val_f1_lr = f1_score(y_true=val_labels, y_pred=val_pred_lr)
    test_f1_lr = f1_score(y_true=test_labels, y_pred=test_pred_lr)
    recall_lr = (test_pred_lr * test_labels).sum() / test_labels.sum()
    precision_lr = (test_pred_lr * test_labels).sum() / test_pred_lr.sum()
    auc_lr = roc_auc_score(y_true=test_labels, y_score=test_pred_lr)

    # print("weight:", weight, "C:", C)
    print('train f1:', train_f1_lr)
    print('val f1:', val_f1_lr)
    print('test f1:', test_f1_lr, 'test recall:', recall_lr, 'test precision:', precision_lr)
    print('test auc:', auc_lr)
    print(logistic_regression.coef_)

    # for i in range(test_data.shape[1]):
    #     stat_val, p_val = stats.ttest_ind(test_data[:, i], test_labels, equal_var=False)
    #     # res = pg.ttest(test_data[:, i], test_labels)
    #     print(i)
    #     print(stat_val, p_val)
    #     # print(res)


if args.lr_bag:
    print("================== Logistic Regression Bagging ====================")
    # for weight in [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]:  # grid search
    #     for C in [0.6, 0.8, 1.0, 1.2, 1.4]:
    # for m_sample in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #     for m_feature in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    lr_weight = {0: 1.0, 1: 4.2}
    C = 1.4
    m_sample = 0.7
    m_feature = 0.6
    # lr_weight = {0: 1.0, 1: weight}
    logistic_regression = LogisticRegression(class_weight=lr_weight, C=C)
    bag_lr = BaggingClassifier(base_estimator=logistic_regression, n_estimators=20, max_samples=m_sample, max_features=m_feature, random_state=13)
    bag_lr.fit(train_data, train_labels)
    train_pred_lr = bag_lr.predict(train_data)
    val_pred_lr = bag_lr.predict(val_data)
    test_pred_lr = bag_lr.predict(test_data)

    train_f1_lr = f1_score(y_true=train_labels, y_pred=train_pred_lr)
    val_f1_lr = f1_score(y_true=val_labels, y_pred=val_pred_lr)
    test_f1_lr = f1_score(y_true=test_labels, y_pred=test_pred_lr)
    recall_lr = (test_pred_lr * test_labels).sum() / test_labels.sum()
    precision_lr = (test_pred_lr * test_labels).sum() / test_pred_lr.sum()
    auc_lr = roc_auc_score(y_true=test_labels, y_score=test_pred_lr)

    # print("weight:", weight, "C:", C)
    # print("max_sample:", m_sample, "max_feature:", m_feature)
    print('train f1:', train_f1_lr)
    print('val f1:', val_f1_lr)
    print('test f1:', test_f1_lr, 'test recall:', recall_lr, 'test precision:', precision_lr)
    print('test auc:', auc_lr)


if args.lr_boost:
    print("================== Logistic Regression Boosting ====================")
    # for weight in [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]:  # grid search
    #     for C in [0.6, 0.8, 1.0, 1.2, 1.4]:
    # for learning_rate in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
    lr_weight = {0: 1.0, 1: 4.2}
    C = 1.4
    # lr_weight = {0: 1.0, 1: weight}
    logistic_regression = LogisticRegression(class_weight=lr_weight, C=C)
    lr_boost = AdaBoostClassifier(base_estimator=logistic_regression, learning_rate=1.2, random_state=13)
    lr_boost.fit(train_data, train_labels)
    train_pred_lr = lr_boost.predict(train_data)
    val_pred_lr = lr_boost.predict(val_data)
    test_pred_lr = lr_boost.predict(test_data)

    train_f1_lr = f1_score(y_true=train_labels, y_pred=train_pred_lr)
    val_f1_lr = f1_score(y_true=val_labels, y_pred=val_pred_lr)
    test_f1_lr = f1_score(y_true=test_labels, y_pred=test_pred_lr)
    recall_lr = (test_pred_lr * test_labels).sum() / test_labels.sum()
    precision_lr = (test_pred_lr * test_labels).sum() / test_pred_lr.sum()
    auc_lr = roc_auc_score(y_true=test_labels, y_score=test_pred_lr)

    # print("weight:", weight, "C:", C)
    # print('learning rate:', learning_rate)
    print('train f1:', train_f1_lr)
    print('val f1:', val_f1_lr)
    print('test f1:', test_f1_lr, 'test recall:', recall_lr, 'test precision:', precision_lr)
    print('test auc:', auc_lr)


if args.random_forest:
    print("================== Random Forest ====================")
    # for weight in [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]:  # grid search
    # for max_depth in [2, 3, 4, 5, 6, 7, 8]:
        # rf_weight = {0: 1.0, 1: weight}
    rf_weight = {0: 1.0, 1: 5.6}
    max_depth = 4
    random_forest = RandomForestClassifier(n_estimators=20, max_depth=max_depth, class_weight=rf_weight, random_state=13)
    random_forest.fit(train_data, train_labels)
    train_pred_rf = random_forest.predict(train_data)
    val_pred_rf = random_forest.predict(val_data)
    test_pred_rf = random_forest.predict(test_data)

    train_f1_rf = f1_score(y_true=train_labels, y_pred=train_pred_rf)
    val_f1_rf = f1_score(y_true=val_labels, y_pred=val_pred_rf)
    test_f1_rf = f1_score(y_true=test_labels, y_pred=test_pred_rf)
    recall_rf = (test_pred_rf * test_labels).sum() / test_labels.sum()
    precision_rf = (test_pred_rf * test_labels).sum() / test_pred_rf.sum()
    auc_rf = roc_auc_score(y_true=test_labels, y_score=test_pred_rf)

    # print("weight:", weight)
    print("max_depth:", max_depth)
    print('train f1:', train_f1_rf)
    print('val f1:', val_f1_rf)
    print('test f1:', test_f1_rf, 'test recall:', recall_rf, 'test precision:', precision_rf)
    print('test auc:', auc_rf)

