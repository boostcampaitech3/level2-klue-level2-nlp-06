import random
import wandb
import pickle as pickle
import os
from random import shuffle
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import (
    AutoTokenizer, AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
    RobertaModel,
    RobertaForCausalLM
)
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from load_data import *
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold


import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("cuda empty cache!!")


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
                  'org:product', 'per:title', 'org:alternate_names',
                  'per:employee_of', 'org:place_of_headquarters', 'per:product',
                  'org:number_of_employees/members', 'per:children',
                  'per:place_of_residence', 'per:alternate_names',
                  'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                  'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                  'org:member_of', 'per:parents', 'org:dissolved',
                  'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                  'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                  'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    wandb.log({"f1": f1, "auprc": auprc, "acc": acc})

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


###################### Basic Train #######################
def train():
    seed_everything(42)
    # load model and tokenizer
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    dataset = load_data(args.dataset)
    target = label_to_num(dataset["label"].values)
    train_dataset, dev_dataset = train_test_split(
        dataset, test_size=0.15, shuffle=True, stratify=target,
    )

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=args.epoch,              # total number of training epochs
        learning_rate=1e-4,               # learning_rate
        # batch size per device during training
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        warmup_ratio=0.1,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps',  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,            # evaluation step.
        report_to="wandb",
        metric_for_best_model='eval_micro f1 score',  # eval_micro f1 score
        load_best_model_at_end=True,
        lr_scheduler_type=args.lr_sch,  # default: linear
    )

    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # trainer.hyperparameter_search(direction="maximize", hp_space=my_hp_space_ray)
    # train model
    trainer.train()

    model.save_pretrained(f'./best_model/{args.case_name}')


def my_hp_space_ray(trial):
    from ray import tune

    return {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "num_train_epochs": tune.choice(range(5, 15)),
        "seed": tune.choice(range(41, 512)),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32]),
    }


############################### K_FOLD #####################################
def k_fold(args):
    print()
    print("I'm k_fold Trainer!")

    kfold = StratifiedKFold(n_splits=args.fold_num, shuffle=True)
    k_dataset = load_data(args.dataset)
    k_label = label_to_num(k_dataset['label'].values)

    for n_iter, (train_ind, test_ind) in enumerate(kfold.split(k_dataset, k_label)):

        print("k_fold no : ", n_iter+1)
        print("# train : ", len(train_ind), " # test : ", len(test_ind))

        train_dataset = k_dataset.iloc[train_ind]
        dev_dataset = k_dataset.iloc[test_ind]

        train_label = label_to_num(train_dataset['label'].values)
        dev_label = label_to_num(dev_dataset['label'].values)

        MODEL_NAME = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # for gpt2

        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

        # make dataset for pytorch.
        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("cuda empty cache!")

        print(device)
        # setting model hyperparameter
        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = 30

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=model_config)
        # model.resize_token_embeddings(len(tokenizer)) # 모델(토크나이저)를 바꿨으니 리사이즈 해줍시다...
        # model.config.pad_token_id = tokenizer.pad_token_id # modelp에도 pad 토큰 id를 전달

        model.parameters
        model.to(device)

        # 사용한 option 외에도 다양한 option들이 있습니다.
        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            save_total_limit=5,              # number of total save model.
            save_steps=500,                 # model saving step.
            num_train_epochs=args.epoch,              # total number of training epochs
            learning_rate=2e-5,               # learning_rate
            # batch size per device during training 32
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,   # batch size for evaluation 32
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=100,              # log saving step.
            evaluation_strategy='steps',  # evaluation strategy to adopt during training
            # `no`: No evaluation during training.
                                             # `steps`: Evaluate every `eval_steps`.
                                             # `epoch`: Evaluate every end of epoch.
            eval_steps=500,            # evaluation step.
            load_best_model_at_end=True,
            report_to="wandb",
            metric_for_best_model='eval_micro f1 score',
            lr_scheduler_type=args.lr_sch,  # default: linear
        )

        trainer = Trainer(
            model=model,      # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=RE_train_dataset,         # training dataset
            eval_dataset=RE_dev_dataset,             # evaluation dataset
            compute_metrics=compute_metrics  # define metrics function

        )

       # trainer.hyperparameter_search(direction="maximize", hp_space=my_hp_space_ray)

        # train model
        trainer.train()
        model.save_pretrained(
            f'./best_model/{args.case_name}/{args.model_name}')

    print("Done!")


def main(args):
    wandb.login()
    wandb.init(project="KLUE_MODEL", entity=args.wandb_entity, name=args.name)
    torch.cuda.empty_cache()
    if args.k_fold:
        k_fold(args)
    else:
        train()
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="./dataset/train.csv",
                        help='학습 데이터셋을 선택합니다. default: ./dataset/train.csv')
    parser.add_argument('--model_name', type=str,
                        default="klue/roberta-large", help='default: klue/roberta-large')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='default: 16')
    parser.add_argument('--k_fold', type=bool,
                        default=False, help='default: False')
    parser.add_argument('--fold_num', type=int, default=5, help='default: 5')
    parser.add_argument('--epoch', type=int, default=5,
                        help='epoch(default:5)')
    parser.add_argument('--lr_sch', type=str, default='linear',
                        help='LR 스케쥴러 cosine, linear, cosine_with_restarts default:linear')
    parser.add_argument('--wandb_entity', type=str, default="user_name",
                        help='wandb.init()의 entity를 입력해주세요. default: user_name')
    parser.add_argument('--name', type=str, default="new test",
                        help='wandb에 표시할 실험 이름입니다. 기본규칙: 모델명-kfold유무-사용데이터셋-기타')
    parser.add_argument('--case_name', type=str, default="last",
                        help='모델을 저장할 로컬 폴더의 이름! default: last')

    args = parser.parse_args()

    main(args)
