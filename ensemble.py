import pandas as pd
import os
import pickle as pickle
from glob import glob
import argparse


def ensemble(prediction_dir):
  """
  모델들의 예측 결과(prediction)들을 취합하여 앙상블 합니다.
  soft-voting 방식을 사용했습니다.
  """
  predictions = glob(f"{prediction_dir}/*.csv")
  
  for index, file_name in enumerate(predictions, 1):
    print(f"{index}번 csv파일: {os.path.basename(file_name)}")
    for i, prediction in enumerate(predictions):
      df = pd.read_csv(prediction)
      if i == 0:
        probs = pd.DataFrame(df.probs.map(eval).to_list())
      else:
        probs += pd.DataFrame(df.probs.map(eval).to_list())

  probs = probs.div(probs.sum(axis=1), axis=0)
  #print(probs)
  probs_argmax = probs.idxmax(axis=1).values.tolist()
  #print(probs_argmax)
  pred_answer = num_to_label(probs_argmax)
  output_prob = probs.values.tolist()

  #test_dataset_dir = "../dataset/test/test_data.csv"
  #test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  #print(test_id)
  
  output = pd.DataFrame({"id": [i for i in range(7765)],"pred_label": pred_answer,"probs": output_prob,})
  output.to_csv("./prediction/ensemble.csv", index=False)
  print("./prediction/ensemble.csv에 저장 완료...")

def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label

def main(args):
  print(f"{'='*15}Ensemble을 시작합니다!{'='*15}")
  ensemble(args.prediction_dir)
  print(f"{'='*15}Done!!!!{'='*15}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--prediction_dir', type=str, default='./collection', help='모델의 prediction 위치. 기본값: ./collection')
  
  args = parser.parse_args()
  print(args)
  main(args)