# π§ μ, μ°μ΄ νμ μ's Second P-Stage Project π§

BoostCamp AI Tech 3th. KLUE λ¬Έμ₯ λΆλ₯ Task

## Index

- [π§ μ, μ°μ΄ νμ μ's Second P-Stage Project π§](#-μ-μ°μ΄-νμ μs-second-p-stage-project-)
  - [Index](#index)
  - [Competition Overview](#competition-overview)
    - [Background](#background)
    - [Target](#target)
    - [Dataset](#dataset)
  - [Dev Environment](#dev-environment)
  - [Project Tree](#project-tree)
  - [Usage](#usage)
    - [Train option](#train-option)
    - [Inference](#inference)
    - [Ensemble](#ensemble)
  - [Wrap Up Report](#wrap-up-report)
  - [Reference & License](#reference--license)

## Competition Overview

### Background

> λ¬Έμ₯ μμμ λ¨μ΄κ° κ΄κ³μ±μ νμνλ κ²μ μλ―Έλ μλλ₯Ό ν΄μν¨μ μμ΄μ λ§μ λμμ μ€λλ€. μμ½λ μ λ³΄λ₯Ό νμ©νμ¬ QA μμ€ν κ΅¬μΆμ νκ±°λ, μμ°μ€λ¬μ΄ λνκ° μ΄μ΄μ§ μ μλλ‘ νλ λ°νμ΄ λκΈ°λ ν©λλ€. 
μ΄μ²λΌ κ΄κ³ μΆμΆ(Relation Extraction)μ λ¬Έμ₯μ λ¨μ΄(Entity)μ λν μμ±κ³Ό κ΄κ³λ₯Ό μμΈ‘νλ Taskμλλ€. κ΄κ³ μΆμΆμ μ§μ κ·Έλνλ₯Ό κ΅¬μΆνκΈ° μν ν΅μ¬μΌλ‘ κ΅¬μ‘°νλ κ²μ, κ°μ λΆμ, μ§μμλ΅, μμ½κ³Ό κ°μ λ€μν NLP taskμ κΈ°λ°μ΄ λ©λλ€.

### Target

> μ΄λ² competition μμλ νκ΅­μ΄ λ¬Έμ₯κ³Ό subject_entity, object_entityκ° μ£Όμ΄μ‘μ λ, entity κ°μ κ΄κ³λ₯Ό μΆλ‘ νλ λͺ¨λΈμ νμ΅μν€κ² λ©λλ€. μλλ μμμλλ€.

```
sentence: μ€λΌν΄(κ΅¬ μ¬ λ§μ΄ν¬λ‘μμ€νμ¦)μμ μ κ³΅νλ μλ° κ°μ λ¨Έμ  λ§κ³ λ κ° μ΄μ μ²΄μ  κ°λ°μ¬κ° μ κ³΅νλ μλ° κ°μ λ¨Έμ  λ° μ€νμμ€λ‘ κ°λ°λ κ΅¬ν λ²μ μ μ¨μ ν μλ° VMλ μμΌλ©°, GNUμ GCJλ μνμΉ μννΈμ¨μ΄ μ¬λ¨(ASF: Apache Software Foundation)μ νλͺ¨λ(Harmony)μ κ°μ μμ§μ μμ νμ§ μμ§λ§ μ§μμ μΈ μ€ν μμ€ μλ° κ°μ λ¨Έμ λ μ‘΄μ¬νλ€.
subject_entity: μ¬ λ§μ΄ν¬λ‘μμ€νμ¦
object_entity: μ€λΌν΄

relation: λ¨μ²΄:λ³μΉ­ (org:alternate_names)
```

- input: sentence, subject_entity, object_entity
- output: pred_label, probs

### Evaluation
> KLUE-RE evaluation metricκ³Ό λμΌν micro F1-scoreμ AUPRCλ₯Ό evaluation metricμΌλ‘ μ¬μ©ν©λλ€.
- Micro F1 score
  - micro-precisionκ³Ό micro-recallμ μ‘°ν νκ· μ΄λ©°, κ° μνμ λμΌν importanceλ₯Ό λΆμ¬ν΄, <u>μνμ΄ λ§μ ν΄λμ€μ λ λ§μ κ°μ€μΉλ₯Ό λΆμ¬</u>ν©λλ€. 
  - λ°μ΄ν° λΆν¬μ λ§μ λΆλΆμ μ°¨μ§νκ³  μλ **no_relation classλ μ μΈ**νκ³  F1 scoreκ° κ³μ° λ©λλ€.
  ![](src/micro_f1_precision.png)
  ![](src/micro_f1_recall.png)
  ![](src/micro_f1_score.png)
  ![](src/micro_f1_t.png)
- AUPRC
  - xμΆμ **Recall**, yμΆμ **Precision**μ΄λ©°, λͺ¨λ  classμ λν νκ· μ μΈ AUPRCλ‘ κ³μ°ν΄ scoreλ₯Ό μΈ‘μ  ν©λλ€. imbalanceν λ°μ΄ν°μ μ μ©ν metric μλλ€.
  ![](src/auprc.png)
  - μ κ·Έλνμ μμλ scikit-learnμ Precision-Recall κ·Έλνμ [μμ](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py) μλλ€. κ·Έλ¦Όμ μμμ κ°μ΄ class 0, 1, 2μ area(λ©΄μ  κ°)μ κ°κ° κ΅¬ν ν, νκ· μ κ³μ°ν κ²°κ³Όλ₯Ό AUPRC scoreλ‘ μ¬μ©ν©λλ€.

### Dataset

- train.csv: μ΄ 32470κ°
- test_data.csv: μ΄ 7765κ° (label blind)
- dict_label_to_num.pkl: λ¬Έμ labelκ³Ό μ«μ labelλ‘ ννλ dictionary
- dict_num_to_label.pkl: μ«μ labelκ³Ό λ¬Έμ labelλ‘ ννλ dictionary
- data μμ
  ![](src/data_ex.png)
- relation κ΅¬μ± μ λ³΄
  ![](src/dataset.png)

λν νκ° λ°μ΄ν°μ κ²½μ° Public(μ§ν μ€)κ³Ό Private(μ’λ£ ν)λ‘ κ΅¬μ±λμ΄ μμΌλ©°, κ°κ° 50% λΉμ¨λ‘ λ¬΄μμλ‘ μ μ λμμ΅λλ€.

## Dev Environment

- CPU: Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- GPU: Tesla V100-PCIE-32GB
- OS: Ubuntu 18.04.5 LTS
- IDE: Jupyter notebook, VSCode
- Tools: Github, Notion, Zoom, wanDB

## Project Tree

```
level2-klue-level2-nlp-06
βββ EDA
β βββ 001_EDA.ipynb
β βββ 002_cm_EDA.ipynb
βββ dataset
β βββ test_data.csv
β βββ train.csv
β βββ train_btl_eng_pos400_comb300.csv
β βββ train_entity850.csv
βββ src
βββ utils
β βββ back_translation
β β   βββ backTraslation.ipynb
β β   βββ index_modify.ipynb
β βββ postag
β     βββ aug_postage.py
β     βββ pos_scaling.ipynb
βββ dict_label_to_num.pkl
βββ dict_num_to_label.pkl
βββ ensemble.py
βββ inference_gpt.py
βββ inference.py
βββ load_data.py
βββ README.md
βββ requirements.txt
βββ train.py
```

- `EDA/`: μ§νν λ°μ΄ν° λΆμμ μ μ₯ν λλ ν λ¦¬
- `dataset/`: μ΄λ²μ μ°μΈ μλ³Έ λ°μ΄ν°μμ ν¬ν¨, augmentationμ΄ μ μ©λ λ°μ΄ν°μμ΄ μ μ₯λ λλ ν λ¦¬
  - `train_pos400`: pos-taggingμ 400κ° λ―Έλ§μ λ°μ΄ν°μ μ μ©
  - `train_btl_eng_pos400_comb300.csv`: Back translationκ³Ό pos-tagging μ μ©
  - `train_entity850.csv`: λ°μ΄ν° κ°―μκ° 850κ° λ―Έλ§μΈ κ²½μ° augmentationμ μ§ν
- `utils/`: λ°μ΄ν° μ μ²λ¦¬ λ° augmentationκ³Ό κ΄λ ¨λ λͺ¨λ
  - `bacTranslation.ipynb`: μΉ ν¬λ‘€λ§μ μ΄μ©ν back translation
  - `index_modify.ipynb`: back translation κ²°κ³Ό μμ λ entityμ idexλ₯Ό μ¬μμ 
  - `aug_postage.py`: ν ν°ν λ° pos-tagging
  - `pos_scaling.ipynb`: pos-taggingμ μ΄μ©ν νμ©μ¬/λΆμ¬ μ κ±° λ¬Έμ₯ (RD)
- `ensemble.py`: soft-voting ensemble
- `inference_gpt.py`: gpt κ³μ΄ λͺ¨λΈμ inference
- `inference.py`: robert κ³μ΄ λͺ¨λΈμ inference
- `load_data.py`: dataset ν΄λμ€
- `train.py`: λͺ¨λΈ νμ΅ μ½λ. k-fold ν¬ν¨

## Usage

### Train option

- λͺ¨λΈμ νμ΅μν΅λλ€.
- k-fold μ λ¬΄λ₯Ό μ ννμ¬ μ§νν  μ μμΌλ©°, wandbμ μ°κ²°μ΄ λμ΄μμ΅λλ€.
  - wandb μ€μ  λ³κ²½μ μ½λλ₯Ό λ³κ²½ν΄μ£ΌμΈμ
- κΈ°λ³Έμ μΌλ‘ `./dataset/` λλ ν λ¦¬μ csvνμΌμ μλ ₯μΌλ‘ λ°μΌλ©°, best modelμ `./best_model`μ μ μ₯ν©λλ€.
  - check-pointμ κ²½μ°μλ 500 stepλ§λ€ evalμ κ±°μΉ ν μ μ₯λ©λλ€.

```
python train.py -h
```

|   argument   | description                                                                  | default             |
| :----------: | :--------------------------------------------------------------------------- | :------------------ |
|   dataset    | νμ΅ λ°μ΄ν°μ μ ν                                                           | ./dataset/train.csv |
|  model_name  | μ¬μ©ν  λͺ¨λΈ μ ν(checkpoint)                                                 | klue/roberta-large  |
|  batch_size  | batch size μ€μ                                                               | 16                  |
|    k_fold    | K Fold μ¬μ© μ λ¬΄                                                             | False               |
|   fold_num   | K Fold number μ€μ                                                            | 5                   |
|    epoch     | epochs μ€μ                                                                   | 5                   |
|    lr_sch    | LR μ€μΌμ€λ¬ μ€μ                                                              | linear              |
| wandb_entity | wandb μ¬μ© μ entity                                                         | user-name           |
|     name     | wandb μ¬μ© μ νμν  μ€ν μ΄λ¦. κΈ°λ³Έκ·μΉ: λͺ¨λΈλͺ-kfoldμ λ¬΄-μ¬μ©λ°μ΄ν°μ-κΈ°ν | new test            |
|  case_name   | λͺ¨λΈμ μ μ₯ν  λ‘μ»¬ ν΄λμ μ΄λ¦                                               | last                |

### Inference

- λͺ¨λΈμ inferenceλ₯Ό μ§νν©λλ€.
- κ²°κ³Όλ `./prediction/submission.csv` ννλ‘ μ μ₯λ©λλ€.
- gptμ κ²½μ°μλ inference_gpt.pyλ₯Ό μ΄μ©ν©λλ€.

```
python inference.py -h
```

| argument  | description           | default            |
| :-------: | :-------------------- | :----------------- |
| model_dir | inferenceν  λͺ¨λΈ μμΉ | ./best_model       |
|   model   | μ¬μ©ν  λͺ¨λΈ μ ν      | klue/roberta-large |

### Ensemble

- soft-voting Ensembleμ μ§νν©λλ€.
- λͺ¨λΈμ κ° inference κ²°κ³Όλ₯Ό μ·¨ν©νμ¬ `./prediction/ensemble.csv`μ μ μ₯ν©λλ€.

```
python ensemble.py -h
```

|    argument    | description                                 | default      |
| :------------: | :------------------------------------------ | :----------- |
| prediction_dir | soft-voting μ μ©ν  predictionμ΄ μ μ₯λ μμΉ | ./collection |

## Wrap Up Report

Taskλ₯Ό μννλ©΄μ μλνλ μ€νκ³Ό κ²°κ³Ό, νκ³ λ [WrapUp Report](https://github.com/boostcampaitech3/level2-klue-level2-nlp-06/blob/main/src/KLUE_NLP_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(06%EC%A1%B0).pdf)μμ νμΈν  μ μμ΅λλ€.

## Reference & License

- Paper
  - [KLUE: Korean Language Understanding Evaluation](https://arxiv.org/abs/2105.09680)
- Dataset
  ![](src/license.png)
