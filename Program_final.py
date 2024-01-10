# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:21:35 2023

@author: Oracle Java Team 1

※ 사용방법 (READ ME)
4개의 단락으로 구성되어있음
1. 모듈 import
2. 프로그램 실행 함수
3. main 함수 실행 전 실행해야하는 코드 (다양한 머신러닝 모델 훈련)
4. main 함수 (머신러닝 모델의 뉴스 기사 예측 프로그램)

1번 2번 3번 4번 순서로 실행하면 되고
1번 2번 3번은 처음 프로그램 켰을 시 한 번만 실행
3번의 경우 모델을 fitting 하기 때문에 시간이 오래걸림

"""
#%%
import os
import re
import pandas as pd
from konlpy.tag import Okt
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import classification_report
import nvapi_module as nvapi
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거
pd.set_option('mode.chained_assignment', None)
#%%
# main함수 실행 시 필요한 변수 (실행 순서에 따라 프로그램이 진행됨)
flag = [0, 0] 

# 웹 데이터를 크롤링 하는 함수
def crawling_data():
    web_data_file_name = nvapi.webcrawler()

    web_data_json = pd.read_json(web_data_file_name, orient='str')

    web_data_json['document'] = web_data_json['description']

    web_data_json = data_cleansing(web_data_json)

    return web_data_json, web_data_file_name

# 혼동(오차)행렬과 classification_report
def get_clf_eval(y_test, pred, pred_proba):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    # ROC-AUC 추가
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))

# 한글을 형태소 별로 분리하는 함수
def okt_tokenizer(text):
    okt = Okt()
    tokens = okt.morphs(text)
    return tokens

# 데이터를 train과 test set으로 분리하는 함수
def data_tts(data, tfidf):
    df_nlp = data

    x_data = df_nlp['document']
    y_data = df_nlp['label']

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=y_data)
    tfidf.fit(x_train)

    train_nlp_tfidf = tfidf.transform(x_train)
    test_nlp_tfidf = tfidf.transform(x_test)

    return train_nlp_tfidf, test_nlp_tfidf, y_train, y_test

# 데이터를 정제하는 함수
def data_cleansing(data):
    # null값 제거
    df_nlp = data[data['document'].notnull()]

    # 정규식으로 한글만 남김
    df_nlp['document'] = df_nlp['document'].apply(
        lambda x: re.sub(r'[^ㄱ-ㅣ가-힣]+', ' ', x))
    # 1글자 단어 제거
    df_nlp['document'] = df_nlp['document'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 1]))

    return df_nlp

# 모델을 훈련시키는 함수
def model_training(model, params, x_train, x_test, y_train, y_test):
    model_name = str(model).split('(')[0]

    grid_cv = GridSearchCV(model, param_grid=params, cv=3,
                           scoring='accuracy', verbose=1, n_jobs=-1)
    grid_cv.fit(x_train, y_train)

    estimator = grid_cv.best_estimator_
    score = grid_cv.best_score_
    pred = estimator.predict(x_test)
    # grid_cv.best_params_ : 좋은 모델 선정 기준
    print("=" * 7, model_name, "=" * 7, "\n", grid_cv.best_params_, "\n=== Best Score ===\n", score)
    print("=== Train ===\n", estimator.score(x_train, y_train))
    print("=== Test ===\n", estimator.score(x_test, y_test), "\n")
    
    if model_name == "SVC":
        get_clf_eval(y_test, pred, pred)
    else:
        pred_proba = estimator.predict_proba(x_test)[:, 1]
        get_clf_eval(y_test, pred, pred_proba)

    print(classification_report(y_test, pred, target_names=['negative', 'positive']))

    return estimator, score

# 다양한 머신러닝 기법을 이용하여 가장 좋은 3가지 모델을 반환하는 함수
def search_best_estimator(data):
    estimator_columns = ['model', 'score']
    tr_model = pd.DataFrame(columns=estimator_columns)

    # 데이터 정제
    data = data_cleansing(data)
    
    # 자연어 데이터를 숫자 데이터로 바꾸기 위해 TfidfVectorizer 사용
    # 한글을 형태소 별로 나누기 위해 Okt 사용
    tfidf = TfidfVectorizer(tokenizer=okt_tokenizer,
                            ngram_range=(1, 2),  # 단어크기 1~2개 단어
                            min_df=3,          # 출현빈도 최소 3
                            max_df=0.9)        # 너무 높은 출현빈도 제외
    # 데이터를 훈련과 검증용으로 나눔
    x_train, x_test, y_train, y_test = data_tts(data, tfidf)

    using_model_names = ['Logistic', 'naive_bayes', 'LGBM', 'SVM', 'RandomForest',
                         'DecisionTree', 'XGBoost']

    # 모델 생성 후 훈련 및 검증
    # LogisticRegression model
    model = LogisticRegression(random_state=0, solver='saga')
    params = {'C': [2.5, 3.0, 3.5]}
    estimator, score = model_training(
        model, params, x_train, x_test, y_train, y_test)
    # 상위 모델 3개를 추출하기 위해 다른 df에 저장
    temp = pd.DataFrame({'model': [estimator], 'score': [score]})
    tr_model = tr_model.append(temp)

    '''
    LogisticRegression model 결과 (accuracy_score)
    {'C': 2.5, 'solver': 'saga'}
    === Best Score ===
    0.8406296984169099
    === Train ===
    0.9370301583090122
    === Test ===
    0.8486027591085957
    
    {'C': 3.0, 'solver': 'saga'}
    === Best Score ===
    0.8410719023613691
    === Train ===
    0.9401255859202264
    === Test ===
    0.8457729041386629
    
    {'C': 3.5, 'solver': 'saga'}
    === Best Score ===
    0.8385955602723977
    === Train ===
    0.9442823029981428
    === Test ===
    0.8478952953661125
    '''

    # NaiveBayes model
    model = MultinomialNB()
    params = {'alpha': [1.0, 1.5, 2.0]}
    estimator, score = model_training(
        model, params, x_train, x_test, y_train, y_test)
    # 상위 모델 3개를 추출하기 위해 다른 df에 저장
    temp = pd.DataFrame({'model': [estimator], 'score': [score]})
    tr_model = tr_model.append(temp)

    '''
    NaiveBayes model 결과 (accuracy_score)
    {'alpha': 1.0}
    === Best Score ===
    0.8457592641726365
    === Train ===
    0.9097903953303264
    === Test ===
    0.8560311284046692
    
    {'alpha': 1.5}
    === Best Score ===
    0.8460245865393118
    === Train ===
    0.9042186256301407
    === Test ===
    0.8546162009197029
    
    {'alpha': 2.0}
    === Best Score ===
    0.8444326523392589
    === Train ===
    0.9010347572300345
    === Test ===
    0.853555005305978
    '''

    # LGB
    model = LGBMClassifier(random_state=0)
    # 파라미터 값에 의해 훈련 시간이 많이 소요되기 때문에 하나씩만 대입하였음
    params = {'n_estimators': [200, 500, 750, 1000],
              'learning_rate': [0.05, 0.1, 0.5],
              'max_depth': [3, 5, 7]
              }
    estimator, score = model_training(
        model, params, x_train, x_test, y_train, y_test)
    # 상위 모델 3개를 추출하기 위해 다른 df에 저장
    temp = pd.DataFrame({'model': [estimator], 'score': [score]})
    tr_model = tr_model.append(temp)

    '''
    LGBMClassifier model 결과 (accuracy_score)
    {'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 2000}
    === Best Score ===
    0.7568762713363403
    === Train ===
    0.8066684354824445
    === Test ===
    0.7789175804740007
    
    {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200}
    === Best Score ===
    0.7571415937030158
    === Train ===
    0.8041920933934731
    === Test ===
    0.7775026529890343
    
    {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 750}
    === Best Score ===
    0.763509330503228
    === Train ===
    0.8677810206067038
    === Test ===
    0.794481782808631
    
    {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 500}
    === Best Score ===
    0.7704077120367914
    === Train ===
    0.8251525603608384
    === Test ===
    0.7845772904138663
    '''

    # SVC model
    model = SVC(random_state=0)
    params = {'C': [0.1, 0.5, 1],
              'gamma': ['auto', 0.1, 0.01],
              'kernel': ['linear', 'rbf', 'poly']}
    estimator, score = model_training(
        model, params, x_train, x_test, y_train, y_test)
    # 상위 모델 3개를 추출하기 위해 다른 df에 저장
    temp = pd.DataFrame({'model': [estimator], 'score': [score]})
    tr_model = tr_model.append(temp)

    '''
    SVC model 결과 (accuracy_score)
    {'C': 0.5, 'kernel': 'linear'}
    === Best Score ===
    0.8364729813389936
    === Train ===
    0.9149199610860529
    === Test ===
    0.8457729041386629
    
    {'C': 0.4, 'kernel': 'linear'}
    === Best Score ===
    0.834085080038914
    === Train ===
    0.9087291058636243
    === Test ===
    0.8436505129112133
    
    {'C': 0.6, 'kernel': 'linear'}
    === Best Score ===
    0.8348810471389405
    === Train ===
    0.9187229150084019
    === Test ===
    0.8457729041386629
    '''

    # RandomForest model
    model = RandomForestClassifier(random_state=0)
    params = {'n_estimators': [90, 100, 110],
              'max_depth': [40, 50, 60],
              'min_samples_leaf': [1, 2],
              'min_samples_split': [12, 14, 16]}
    estimator, score = model_training(
        model, params, x_train, x_test, y_train, y_test)
    # 상위 모델 3개를 추출하기 위해 다른 df에 저장
    temp = pd.DataFrame({'model': [estimator], 'score': [score]})
    tr_model = tr_model.append(temp)

    '''
    RandomForestClassifier model 결과 (accuracy_score)
    {'max_depth': 60, 'min_samples_leaf': 2, 'min_samples_split': 16, 'n_estimators': 100}
    === Best Score ===
    0.7794286725037587
    === Train ===
    0.8061377907490935
    === Test ===
    0.7796250442164839
    
    {'max_depth': 60, 'min_samples_leaf': 2, 'min_samples_split': 16, 'n_estimators': 90}
    === Best Score ===
    0.7785442646148404
    === Train ===
    0.8051649420712833
    === Test ===
    0.7785638486027591
    
    {'max_depth': 60, 'min_samples_leaf': 2, 'min_samples_split': 16, 'n_estimators': 110}
    === Best Score ===
    0.7785442646148404
    === Train ===
    0.8076412841602547
    === Test ===
    0.7803325079589671
    
    '''
    # DecisionTree model
    model = DecisionTreeClassifier(random_state=0, criterion="entropy")
    params = {'max_depth': [60, 100, 160],
              'min_samples_leaf': [1, 2],
              'min_samples_split': [12, 14, 20]}
    estimator, score = model_training(
        model, params, x_train, x_test, y_train, y_test)
    # 상위 모델 3개를 추출하기 위해 다른 df에 저장
    temp = pd.DataFrame({'model': [estimator], 'score': [score]})
    tr_model = tr_model.append(temp)

    '''
    DecisionTreeClassifier model 결과 (accuracy_score)
    {'max_depth': 60, 'min_samples_leaf': 1, 'min_samples_split': 12}
    === Best Score ===
    0.6968249756787831
    === Train ===
    0.7617405147253914
    === Test ===
    0.6950831269897417
    
    {'max_depth': 100, 'min_samples_leaf': 1, 'min_samples_split': 14}
    === Best Score ===
    0.714247811090475
    === Train ===
    0.8147165472716017
    === Test ===
    0.7191368942341705
    
    {'max_depth': 160, 'min_samples_leaf': 1, 'min_samples_split': 20}
    === Best Score ===
    0.722384363668524
    === Train ===
    0.8647740337843813
    === Test ===
    0.7265652635302441
    '''
    # XGBoost model
    model = xgb.XGBClassifier(random_state=0)
    params = {'max_depth': [6, 8, 10],  # 3 ~ 10 적정
              'gamma': [0.5, 1, 2]}
    estimator, score = model_training(
        model, params, x_train, x_test, y_train, y_test)
    # 상위 모델 3개를 추출하기 위해 다른 df에 저장
    temp = pd.DataFrame({'model': [estimator], 'score': [score]})
    tr_model = tr_model.append(temp)

    '''
    XGBClassifier model 결과 (accuracy_score)
    {'gamma': 1, 'max_depth': 10}
    === Best Score ===
    0.7818165738038383
    === Train ===
    0.8621208101176262
    === Test ===
    0.7976653696498055
    
    {'gamma': 2, 'max_depth': 10}
    === Best Score ===
    0.7834969487927833
    === Train ===
    0.8581409746174936
    === Test ===
    0.7955429784223559
    '''
    
    tr_model = tr_model.reset_index(drop=True)

    model_df = pd.DataFrame(columns=using_model_names, index=['train', 'test', 'roc'])

    # 표로 확인
    for idx in range(len(tr_model['model'])):
        train = tr_model['model'][idx].score(x_train, y_train)
        test = tr_model['model'][idx].score(x_test, y_test)

        predict = tr_model['model'][idx].predict(x_test)
        roc = roc_auc_score(y_test, predict)

        # model_df.iloc[:, idx] = [train, test, roc]
        model_df[model_df.columns[idx]] = [train, test, roc]

    print(model_df)

    # 좋은 모델을 판별하기 위해 시각화
    plt.rc('font', family='Malgun Gothic')

    model_df.plot()

    plt.title('감성분석 모델 성능 비교')
    plt.xlabel('성능지표')
    plt.ylabel('정확도')
    plt.savefig('MachineLearning_Performance_Indicator.png')
    plt.pause(0.01)
    # 가장 좋은 모델 3가지를 반환
    best_models = tr_model.sort_values(by='score', ascending=False).head(3)
    
    best_models = best_models.reset_index(drop = True)
    
    return best_models, tfidf

# 머신 러닝 적용한 모델로 웹 크롤링 뉴스 데이터를 추측하는 함수
def predict_data(best_models, text, size):
    best_models['predict'] = ''
    model_num = len(best_models)
    
    # 뉴스 데이터 예측
    for i in range(model_num):
        model = best_models['model'][i]
        best_models['predict'][i] = model.predict(text)
        
    # 3개의 모델끼리 예측값 비교      
    predict_same = 0
    predict_different = 0                    
    for i in range(size):
        if (best_models['predict'][0][i] == 
            best_models['predict'][1][i] == 
            best_models['predict'][2][i]):
            predict_same += 1
        else:
            predict_different += 1
    
    # 모든 모델과 같은 값으로 예측하는 예측률
    print("모델들의 예측률")
    print(str(predict_same / (predict_same + predict_different) * 100) + "%")
    
def main(best_models, tifdf, flag):
    flag[1] = 0

    while(True):
        menu_items = [str(i + 1) for i in range (3)]

        print("=" * 3, "시작화면", "=" * 3)
        print("1. 웹 데이터 크롤링")
        print("2. 모델 예측")
        print("3. 프로그램 종료")
        menu = input("메뉴 선택: ")
        
        if menu in menu_items:
            if menu == '1':
                if flag[0] == 1:
                    flag[1] = 1
                    web_data, file = crawling_data()
                    size = len(web_data)
                    text = tfidf.transform(web_data['document'])
                else:
                    print("모델 훈련을 먼저 진행해주세요.\n")
            elif menu == '2':
                if flag[1] == 1:
                    predict_data(best_models, text, size)
                    if os.path.isfile(file):
                        os.remove(file)
                    flag[1] = 0
                else :
                    print("모델 훈련과 웹 데이터 크롤링을 먼저 진행해주세요.\n")
            elif menu == '3':
                print("프로그램을 종료합니다.\n")
                break
        else:
            print("다시 입력해주세요.\n")
        

#%% main 전 꼭 실행
if __name__ == '__main__':
    data = pd.read_csv('data/nlp.csv', encoding='utf-8')
    flag = [0, 0]    
    flag[0] = 1
    # 모델 훈련 (많은 시간이 소요됩니다.)
    best_models, tfidf = search_best_estimator(data)

#%% main()
if __name__ == '__main__':
    main(best_models, tfidf, flag)









