import os
import sys
import urllib.request
import json
import pandas as pd
import hugging


# train.csv 읽기 (baseline code)
def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = i[1:-1].split(',')[0].split(':')[1]
        j = j[1:-1].split(',')[0].split(':')[1]
        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': dataset['sentence'],
                                'subject_entity': subject_entity, 'object_entity': object_entity,
                                'label': dataset['label'], })

    return out_dataset

# Google Cloud Translation API
client_id = "18msuF1RoiqoAmqipM_8"  # 개발자센터에서 발급받은 Client ID 값
client_secret = "GYthker0Kf"  # 개발자센터에서 발급받은 Client Secret 값
dataset_dir = '../dataset/train/train.csv'  # csv 파일 위치

# 번역결과를 저장할 DataFrame 구축
pd_dataset = pd.read_csv(dataset_dir)
dataset = preprocessing_dataset(pd_dataset)
convert_data = {
    '원본': [],
    '번역': [],
    '재역': []
}
convert_result_set = pd.DataFrame(convert_data)

# API request
try:
    for sen in dataset['sentence']:
        sen_en = hugging.translate_to_eng(sen)
        sen_en_ko = hugging.translate_to_ko(sen_en)
        convert_sentence = {
            '원본': sen,
            '번역': sen_en,
            '재역': sen_en_ko
        }
        convert_result_set = convert_result_set.append(
        convert_sentence, ignore_index=True)
        
        '''
        encText = sen
        data = "source=ko&target=en&text=" + encText

        url = "https://openapi.naver.com/v1/papago/n2mt"

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
        rescode = response.getcode()

        if(rescode == 200):
            response_body = response.read()
            trans_dat = json.loads(response_body)
            print(trans_dat['message']['result']['translatedText'])
            convert_sentence = {
                '원본': sen,
                '결과': trans_dat['message']['result']['translatedText']
            }
            # 결과 append
            convert_result_set = convert_result_set.append(
                convert_sentence, ignore_index=True)
        else:
            print("Error Code:" + rescode)
        '''
except Exception as e:
    print(e)
    
    


# 결과물 csv로 저장
convert_result_set.to_csv('./convert_result.csv')
print('convert done..!')
