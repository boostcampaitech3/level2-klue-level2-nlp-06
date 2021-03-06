import pandas as pd
import os
import pickle as pickle
from glob import glob
import argparse

from inference import num_to_label


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
  probs_argmax = probs.idxmax(axis=1).values.tolist()
  pred_answer = num_to_label(probs_argmax)
  output_prob = probs.values.tolist()
  
  output = pd.DataFrame({"id": [i for i in range(7765)],"pred_label": pred_answer,"probs": output_prob,})
  output.to_csv("./prediction/ensemble.csv", index=False)
  print("./prediction/ensemble.csv에 저장 완료...")


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