import pickle as pickle
import os
from pickletools import optimize
from random import shuffle
from sched import scheduler
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
from sklearn.model_selection import train_test_split , StratifiedKFold

#### ray tune #####
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from my_model import *

import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("cuda empty cache!!")


import wandb

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
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation??? ?????? metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # ???????????? ???????????? ???????????? ????????????.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

import random 
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
    MODEL_NAME = "klue/roberta-small"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # load dataset
    dataset = load_data("../dataset/train/train.csv")
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

    model =AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)
    
    
    # ????????? option ????????? ????????? option?????? ????????????.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments ??????????????????.
    wandb.config = {

    "save_total_limit": 5,              # number of total save model.
    "save_steps": 500,                 # model saving step.
    "num_train_epochs" : 15,              # total number of training epochs
    "learning_rate" : 1e-4,               # learning_rate
    "per_device_train_batch_size" : 4,  # batch size per device during training
    "per_device_eval_batch_size" : 4,   # batch size for evaluation
    "warmup_steps" : 500,                # number of warmup steps for learning rate scheduler
    "weight_decay" : 0.4,      

    }
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=5,              # total number of training epochs
        learning_rate=1e-4,               # learning_rate
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.4,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 500,            # evaluation step.
        report_to="wandb",
        metric_for_best_model = 'eval_micro f1 score',
        load_best_model_at_end = True 
    )

    trainer = Trainer(
        model=model,                         # the instantiated ???? Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
        #optimizers= (optimizer, scheduler)
    )

    # trainer.hyperparameter_search(direction="maximize", hp_space=my_hp_space_ray)
    # train model
    trainer.train()

    model.save_pretrained('./best_model')

   
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

  kfold = StratifiedKFold(n_splits=args.fold_num, shuffle = True)
  k_dataset = load_data("./train_backup.csv")
  k_label = label_to_num(k_dataset['label'].values)
  MODEL_NAME = args.model_name
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  my_defined = ['[UNU]',"?","!"]

  for i in range(1,200,3):
      my_defined.append(f'[UNU]')
      
  special_tokens_dict = {'additional_special_tokens': my_defined}
  tokenizer.add_special_tokens(special_tokens_dict)
  
  for n_iter, (train_ind, test_ind) in enumerate(kfold.split(k_dataset,k_label)):
    
    wandb.init(
                entity="boostcamp_nlp06",
                project="KLUE_MODEL",
                name=args.wandb_name  + f'_{n_iter}-fold',
  
            )
    wandb.config.update(args)
        
    print("------------------------k_fold no : ",n_iter+1)
    print("# train : ",len(train_ind)," # test : ",len(test_ind))

    train_dataset = k_dataset.iloc[train_ind]
    dev_dataset = k_dataset.iloc[test_ind]

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values) 

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
        # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    print(model.config)
    model.parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)    
   
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
      print("cuda empty cache!")

    
    # ??? ????????? ??? ????????? ??? ???????????? ????????? ????????? ?????? ???????????? ??? ??? ????????????. 
    #??????????????????  trainer?????? ??? optimizers ?????? ?????? ????????????.

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)

   # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.02)

   
    # ????????? option ????????? ????????? option?????? ????????????.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments ??????????????????.
    training_args = TrainingArguments(
      output_dir='./results',          # output directory
      save_total_limit=5,              # number of total save model.
      save_steps=500,                 # model saving step.
      num_train_epochs=7,              # total number of training epochs
      learning_rate=3e-5,               # learning_rate
      per_device_train_batch_size=8,  # batch size per device during training
      per_device_eval_batch_size=8,   # batch size for evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.001,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=100,              # log saving step.
      evaluation_strategy='steps', # evaluation strategy to adopt during training
                                         # `no`: No evaluation during training.
                                         # `steps`: Evaluate every `eval_steps`.
                                         # `epoch`: Evaluate every end of epoch.
      eval_steps = 500,            # evaluation step.
      load_best_model_at_end = True ,
      report_to="wandb",
      metric_for_best_model = 'eval_micro f1 score'
    )
  #  def model_init(): 
   #     return model

    trainer = Trainer(    
        model = model,      # the instantiated ???? Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        optimizers=(optimizer, scheduler)
            
)
       
   # trainer.hyperparameter_search(direction="maximize", hp_space=my_hp_space_ray)
    
    # train model
    trainer.train()
    model.save_pretrained(f'./best_model/k_fold/{args.wandb_name}/{n_iter}')

  print("Done!")



def main(args):
    torch.cuda.empty_cache()  
    if args.k_fold:
        k_fold(args)
    else:
        train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="klue/roberta-large")
    parser.add_argument('--k_fold', type=bool, default=False)
    parser.add_argument('--fold_num', type=int, default=5)
    parser.add_argument('--wandb_name', type=str, default="roberta-large_kfold6_Adam_stepLR")
    args = parser.parse_args()
    main(args)

