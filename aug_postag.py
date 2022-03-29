import os
import pandas as pd
import numpy as np
from ast import literal_eval
from collections import Counter
import warnings

from tqdm.auto import trange
from time import sleep

import argparse

# from pykospacing import Spacing
from hanspell import spell_checker
import konlpy
from konlpy.tag import Okt, Komoran, Kkma, Hannanum

os.environ['JAVA_HOME'] = '/opt/conda/bin/java'

warnings.filterwarnings('ignore')


# limit 이하의 개수인 label만 포함하는 데이터프레임 출력
def accept_label(dataset,limit=200):
    """except label이 아닌 label(relation) 문장만 포함한 데이터프레임 출력"""
    # ver 1. dataframe
    # df_cnt = pd.DataFrame(dataset['label'].value_counts())
    # df_cnt = df_cnt.reset_index(drop=False)
    # df_cnt.columns = ['label','count']
    # # print(df_cnt.head())
    # acc_label = df_cnt.loc[df_cnt['count']<=limit,'label']
    # acc_label = list(acc_label)
    
    # ver 2. counter
    df_cnt = Counter(dataset['label'])
    acc_label = [label for idx,label in enumerate(df_cnt) if df_cnt[label]<=limit]
    print(f"Data Augmentation:{len(acc_label)} labels")
    # print(acc_label)

    # labels = dataset.loc[:,'label']
    # idx_label = [idx for idx,label in enumerate(labels) if label not in acc_label]
    # print(len(dataset),len(idx_label))
    
    df = dataset.loc[dataset['label'].isin(acc_label),:]
    df = df.reset_index(drop=True)
    return df

# entity-->>word, stt_idx, end_idx, type으로 쪼개기
def eval_entity(df):
    """딕셔너리 형태의 entity를 쪼개 word, stt_idx, end_idx, type을 각각 칼럼으로 추가하는 함수"""

    df['subject_word'] = [literal_eval(entity)['word'] for entity in df['subject_entity']]
    df['subject_stt_idx'] = [literal_eval(entity)['start_idx'] for entity in df['subject_entity']]
    df['subject_end_idx'] = [literal_eval(entity)['end_idx'] for entity in df['subject_entity']]
    df['subject_type'] = [literal_eval(entity)['type'] for entity in df['subject_entity']]

    df['object_word'] = [literal_eval(entity)['word'] for entity in df['object_entity']]
    df['object_stt_idx'] = [literal_eval(entity)['start_idx'] for entity in df['object_entity']]
    df['object_end_idx'] = [literal_eval(entity)['end_idx'] for entity in df['object_entity']]
    df['object_type'] = [literal_eval(entity)['type'] for entity in df['object_entity']]
    print(df.columns)
    return df
    
# subject_entity의 stt_idx와 object entity의 stt_idx에 각각 토큰 집어넣기
def tokenize_entity(sentence,sub_idx,ob_idx):
    """문장을 subject, object의 토큰으로 치환하는 함수
    sentence: str |-->input 문장
    sub_idx: list(stt,end) |-->subject 인덱스
    ob_idx: list(stt,end) |-->object 인덱스"""
    sub_stt, sub_end = sub_idx
    ob_stt, ob_end = ob_idx
    
    if sub_stt < ob_stt: # sub가 ob보다 앞에 있을 때
        sentence = sentence[:sub_stt]+'subject'+sentence[sub_end+1:]
        l = 7 - (sub_end-sub_stt+1)
        ob_stt += l
        ob_end += l
        sentence = sentence[:ob_stt]+'object'+sentence[ob_end+1:]
    else:
        sentence = sentence[:ob_stt]+'subject'+sentence[ob_end+1:]
        l = 7 - (ob_end-ob_stt+1)
        sub_stt += l
        sub_end += l
        sentence = sentence[:sub_stt]+'object'+sentence[sub_end+1:]
    return sentence

# 토큰화 문장을 태깅하고 형용사/부사를 제거하는 함수
def PosTag_token(sentence,tag='okt',drop_pos=['Adjective','Adverb'],norm=True,pacing=True):
    """문장을 태깅 후 성분 제거하는 함수
    sentence: pd.Series |-->idx기반 dataframe의 행"""

    taggers = {'okt':Okt(), 'kom':Komoran(), 'kkm':Kkma(), 'han':Hannanum()}
    tagg = taggers[tag]
    sentence_pos = tagg.pos(sentence,norm=norm)
    result = []
    drop = False

    for idx,(word,pos) in enumerate(sentence_pos):
        if pos in drop_pos:
            drop=True
            continue
        else:
            result.append(word)

    if not drop:
        return result, drop

    #띄어쓰기 보정
    elif pacing:
        result = ''.join(result)
        # spacing = Spacing()
        # result = spacing(result)
        
        spelled_sent = spell_checker.check(result)
        result = spelled_sent.checked

        #subject, object 보정
        if result.find('subject')<0:
            for i in range(1,7):
                sub = 'subject'
                sub = sub[:i] + ' ' + sub[i:]
                if result.find(sub)!=-1:
                    result = result.replace(sub,'subject')
        if result.find('object')<0:
            for i in range(1,6):
                ob = 'object'
                ob = ob[:i] + ' ' +ob[i:]
                if result.find(ob)!=-1:
                    result = result.replace(ob,'object')
        
    else:
        result = ' '.join(result)
    return result,drop

# 토큰 다시 치환하기
def back_tokenize(sentence,subject_word,object_word):
    """subject, object 토큰을 원래 단어로 치환하는 함수"""
    if sentence.find('subject')<sentence.find('object'): #sub가 앞에 있을 때
        sub_stt_idx = sentence.find('subject')
        sub_end_idx = sub_stt_idx+len(subject_word)-1
        ob_stt_idx = sentence.find('object') - (7-len(subject_word))
        ob_end_idx = ob_stt_idx+len(object_word)-1
    else:
        ob_stt_idx = sentence.find('object')
        ob_end_idx = ob_stt_idx+len(object_word)-1
        sub_stt_idx = sentence.find('subject') - (7-len(object_word))
        sub_end_idx = sub_stt_idx+len(subject_word)-1
        
    # subject = sentence.split('subject')
    # if len(subject)!=2:
    #     return print(subject)
    # else:
    #     subject_pre,subject_end = subject#sentence.split('subject')
    #     result = subject_pre+subject_word+subject_end
    subject_pre,subject_end = sentence.split('subject')
    result = subject_pre+subject_word+subject_end

    # object = result.split('object')
    # if len(object)!=2:
    #     return print(object)
    # else:
    #     object_pre,object_end = object #result.split('object')
    #     result = object_pre+object_word+object_end
    object_pre,object_end = result.split('object')
    result = object_pre+object_word+object_end

    return result, sub_stt_idx, sub_end_idx, ob_stt_idx, ob_end_idx

# 데이터프레임 처리
def pos_augmentation(data,exc_limit=200,tag='okt',drop_pos=['Adjective','Adverb'],norm=True,pacing=True):
    """데이터프레임 전처리"""
    
    # exc_label = except_label(data,limit=exc_limit)
    dataset = accept_label(data,limit=exc_limit)
    dataset = eval_entity(dataset)

    new_sentences = []
    new_sub_entity = []
    new_ob_entity = []
    new_labels = []
    new_sources = []

    #'id', 'sentence', 'subject_entity', 'object_entity', 'label', 'source',
    #    'subject_word', 'subject_stt_idx', 'subject_end_idx', 'subject_type',
    #    'object_word', 'object_stt_idx', 'object_end_idx', 'object_type'

    for i in trange(len(dataset)):
        _,sent,_,_,label,source,sub_wd,sub_stt,sub_end,sub_type,ob_wd,ob_stt,ob_end,ob_type = dataset.iloc[i,:]

        new_sent = tokenize_entity(sent,[sub_stt,sub_end],[ob_stt,ob_end])
        new_sent,drop = PosTag_token(new_sent,tag=tag,drop_pos=drop_pos,norm=norm,pacing=pacing)
        if drop: #drop_pos 적용한 케이스
            new_sent,sub_stt,sub_end,ob_stt,ob_end = back_tokenize(new_sent,sub_wd,ob_wd)
            new_sentences.append(new_sent)
            new_sub_entity.append(str({'word':sub_wd,'start_idx':sub_stt,'end_idx':sub_end,'type':sub_type}))
            new_ob_entity.append(str({'word':ob_wd,'start_idx':ob_stt,'end_idx':ob_end,'type':ob_type}))
            new_labels.append(label)
            new_sources.append(source)
        sleep(0.01)

    add_rows = len(new_sentences)
    ids = list(range(len(data),len(data)+add_rows))
    
    new_df = pd.DataFrame({ 'id':ids,
                            'sentence':new_sentences,
                            'subject_entity':new_sub_entity,
                            'object_entity':new_ob_entity,
                            'label':new_labels,
                            'source':new_sources})

    return new_df  

def main():
    dataset = pd.read_csv(args.data_dir)
    result = pos_augmentation(dataset,
                                exc_limit = args.limit,
                                tag = args.tag,
                                drop_pos = args.drop,
                                norm = args.norm,
                                pacing = args.pacing)
    result.to_csv(os.path.join('./dataset/train',args.name+'.csv'),index=False,encoding='uft-8')
    print("---------Complete---------")
  # data,exc_limit=200,tag='okt',drop_pos=['Adjective','Adverb'],norm=True,pacing=True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  
    parser.add_argument('--data_dir',type=str, default='./dataset/train/train.csv',help='target dataset directory')
    parser.add_argument('--name',type=str, default='train_pos_200',help='new file name')
    parser.add_argument('--limit',type=int, default=200,help='target label counts maximum(default:200)')
    parser.add_argument('--tag',type=str, default='okt',help='korean pos-tagger(default:okt)')
    parser.add_argument('--drop',type=list, default=['Adjective','Adverb'],help="drop target pos(default:['Adjective','Adverb'])")
    parser.add_argument('--norm',type=bool, default=True,help='pos-normalization(default:True)')
    parser.add_argument('--pacing',type=bool, default=False,help='pacing(default:False)')

    args = parser.parse_args()
    print(args)

    main()