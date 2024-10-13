

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import os


from sklearn.model_selection import KFold, cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,OrdinalEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.inspection import PartialDependenceDisplay
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# Sliding window를 사용해 X와 y 구성 (다음 3개 시점을 예측)
def create_sliding_window(X_data, y_data, window_size=9, output_size=3):
    X, y = [], []
    for i in range(len(X_data) - window_size - output_size + 1):
        X.append(X_data[i:i + window_size, :])  # 9개 시점의 X 변수 데이터
        y.append(y_data[i + window_size:i + window_size + output_size])  # 다음 3개 시점의 '평균가격(원)' 값
    X = np.array(X)  # (샘플 수, window_size, num_features)
    y = np.array(y)  # (샘플 수, output_size)
    return X, y
    
def add_rolling_features_for_multiple_windows(df, column, window_sizes):
    rolling_features = {}

    # 여러 윈도우 사이즈에 대한 rolling mean 및 std 계산
    for window_size in window_sizes:
        rolling_features[f'{column}_MA_{window_size}'] = df[column].rolling(window=window_size).mean()
        rolling_features[f'{column}_MSTD_{window_size}'] = df[column].rolling(window=window_size).std()

    # 딕셔너리로 모은 rolling feature들을 한 번에 concat
    rolling_df = pd.DataFrame(rolling_features)

    # 원래 df와 합치기
    df = pd.concat([df, rolling_df], axis=1)

    return df

def replace_outliers_iqr(df, columns, factor=1.5):
    df_copy = df.copy()  # 원본 데이터프레임을 수정하지 않도록 복사
    for col in columns:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1

        # 이상치 조건 설정
        lower_bound = Q1 - (IQR * factor)
        upper_bound = Q3 + (IQR * factor)

        # IQR 밖의 이상치를 0으로 대체
        df_copy.loc[(df_copy[col] < lower_bound) | (df_copy[col] > upper_bound), col] = 0

    return df_copy


def create_ratio_column_and_drop(df, numerator_col, denominator_col, new_col_name):
    df_copy = df.copy()  # 원본 DataFrame을 보호하기 위해 복사

    # 분모가 0인 경우 대비하여 비율 계산할 때 분모에 1e-8 추가
    df_copy[new_col_name] = df_copy[numerator_col] / (df_copy[denominator_col] + 1e-8)

    # 기존의 두 열 삭제
    df_copy = df_copy.drop(columns=[numerator_col, denominator_col])

    return df_copy

####################################################################




def process_양파_for_train(raw_file, 산지공판장_file, 전국도매_file, scaler = None):
    raw_data_양파 = pd.read_csv(raw_file)
    산지공판장_양파 = pd.read_csv(산지공판장_file)
    전국도매_양파 = pd.read_csv(전국도매_file)
    품목명 = '양파'

    전국도매_양파['봄'] = None

    # '봄배추' 데이터 추출
    spring_cabbage = 전국도매_양파[전국도매_양파['품종명'] == '봄배추'][['시점', '총반입량(kg)']]

    # 동일한 시점에 '봄배추'의 '총반입량(kg)'을 '봄' 칼럼에 넣음
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '봄배추':
            matching_spring_cabbage = spring_cabbage[spring_cabbage['시점'] == row['시점']]
            if not matching_spring_cabbage.empty:
                전국도매_양파.at[index, '봄'] = matching_spring_cabbage['총반입량(kg)'].values[0]

    전국도매_양파['봄_1'] = None
    spring_radish = 전국도매_양파[전국도매_양파['품종명'] == '저장무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '저장무':
            matching_spring_radish = spring_radish[spring_radish['시점'] == row['시점']]
            if not matching_spring_radish.empty:
                전국도매_양파.at[index, '봄_1'] = matching_spring_radish['총반입량(kg)'].values[0]

    전국도매_양파['여름'] = None
    summer_cabbage = 전국도매_양파[전국도매_양파['품종명'] == '여름배추'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '여름배추':
            matching_summer_cabbage = summer_cabbage[summer_cabbage['시점'] == row['시점']]
            if not matching_summer_cabbage.empty:
                전국도매_양파.at[index, '여름'] = matching_summer_cabbage['총반입량(kg)'].values[0]

    전국도매_양파['여름_1'] = None
    summer_radish = 전국도매_양파[전국도매_양파['품종명'] == '여름무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '여름무':
            matching_summer_radish = summer_radish[summer_radish['시점'] == row['시점']]
            if not matching_summer_radish.empty:
                전국도매_양파.at[index, '여름_1'] = matching_summer_radish['총반입량(kg)'].values[0]

    전국도매_양파['가을'] = None
    fall_apple = 전국도매_양파[전국도매_양파['품종명'] == '미얀마'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '미얀마':
            matching_fall_apple = fall_apple[fall_apple['시점'] == row['시점']]
            if not matching_fall_apple.empty:
                전국도매_양파.at[index, '가을'] = matching_fall_apple['총반입량(kg)'].values[0]

    전국도매_양파['가을_1'] = None
    fall_daeji = 전국도매_양파[전국도매_양파['품종명'] == '대지'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '대지':
            matching_fall_daeji = fall_daeji[fall_daeji['시점'] == row['시점']]
            if not matching_fall_daeji.empty:
                전국도매_양파.at[index, '가을_1'] = matching_fall_daeji['총반입량(kg)'].values[0]

    전국도매_양파['겨울'] = None
    winter_cabbage = 전국도매_양파[전국도매_양파['품종명'] == '김장(가을)배추'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '김장(가을)배추':
            matching_winter_cabbage = winter_cabbage[winter_cabbage['시점'] == row['시점']]
            if not matching_winter_cabbage.empty:
                전국도매_양파.at[index, '겨울'] = matching_winter_cabbage['총반입량(kg)'].values[0]

    전국도매_양파['겨울_1'] = None
    winter_radish = 전국도매_양파[전국도매_양파['품종명'] == '고냉지무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '고냉지무':
            matching_winter_radish = winter_radish[winter_radish['시점'] == row['시점']]
            if not matching_winter_radish.empty:
                전국도매_양파.at[index, '겨울_1'] = matching_winter_radish['총반입량(kg)'].values[0]

    전국도매_양파['겨울_2'] = None
    winter_2 = 전국도매_양파[전국도매_양파['품종명'] == '월동배추'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '월동배추':
            matching_winter_2 = winter_2[winter_2['시점'] == row['시점']]
            if not matching_winter_2.empty:
                전국도매_양파.at[index, '겨울_2'] = matching_winter_2['총반입량(kg)'].values[0]

    전국도매_양파['겨울_3'] = None
    winter_3 = 전국도매_양파[전국도매_양파['품종명'] == '봄무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '봄무':
            matching_winter_3 = winter_3[winter_3['시점'] == row['시점']]
            if not matching_winter_3.empty:
                전국도매_양파.at[index, '겨울_3'] = matching_winter_3['총반입량(kg)'].values[0]


    양파_condition = {'양파': {
            'target': lambda df: (df['품종명'] == '양파') & (df['거래단위'] == '1키로') & (df['등급'] == '상'),
            '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['양파'], '품종명': ['기타양파'], '등급명': ['상']},
            '도매': {'시장명': ['*전국도매시장'], '품목명': ['양파'], '품종명': ['양파(일반)']}
        }}
    # 타겟 데이터 필터링
    raw_품목_양파 = raw_data_양파[raw_data_양파['품목명'] == '양파']
    target_mask_양파 = 양파_condition['양파']['target'](raw_품목_양파)
    filtered_data_양파 = raw_품목_양파[target_mask_양파]


    # 다른 품종에 대한 파생변수 생성
    other_data_양파 = raw_품목_양파[~target_mask_양파]
    unique_combinations_양파 = other_data_양파[['품종명', '거래단위', '등급']].drop_duplicates()
    for _, row in unique_combinations_양파.iterrows():
        품종명, 거래단위, 등급 = row['품종명'], row['거래단위'], row['등급']
        mask_양파 = (other_data_양파['품종명'] == 품종명) & (other_data_양파['거래단위'] == 거래단위) & (other_data_양파['등급'] == 등급)
        temp_df_양파 = other_data_양파[mask_양파]
        for col in ['평년 평균가격(원)', '평균가격(원)']:
            new_col_name = f'{품종명}_{거래단위}_{등급}_{col}'
            filtered_data_양파 = filtered_data_양파.merge(temp_df_양파[['시점', col]], on='시점', how='left', suffixes=('', f'_{new_col_name}'))
            filtered_data_양파.rename(columns={f'{col}_{new_col_name}': new_col_name}, inplace=True)


    # 공판장 데이터 처리
    if 양파_condition['양파']['공판장']:
        filtered_공판장_양파 = 산지공판장_양파
        for key, value in 양파_condition['양파']['공판장'].items():
            filtered_공판장_양파 = filtered_공판장_양파[filtered_공판장_양파[key].isin(value)]

        filtered_공판장_양파 = filtered_공판장_양파.add_prefix('공판장_').rename(columns={'공판장_시점': '시점'})
        filtered_data_양파 = filtered_data_양파.merge(filtered_공판장_양파, on='시점', how='left')


    if 양파_condition['양파']['도매']:
        filtered_도매_양파 = 전국도매_양파
        for key, value in 양파_condition['양파']['도매'].items():
            filtered_도매_양파 = filtered_도매_양파[filtered_도매_양파[key].isin(value)]

        filtered_도매_양파 = filtered_도매_양파.add_prefix('도매_').rename(columns={'도매_시점': '시점'})
        filtered_data_양파 = filtered_data_양파.merge(filtered_도매_양파, on='시점', how='left')

    if not 양파_condition[품목명]['도매']:
        filtered_data_양파.loc[:, '도매_봄'] = 0
        filtered_data_양파.loc[:, '도매_여름'] = 0
        filtered_data_양파.loc[:, '도매_가을'] = 0
        filtered_data_양파.loc[:, '도매_겨울'] = 0
        filtered_data_양파.loc[:, '도매_봄_1'] = 0
        filtered_data_양파.loc[:, '도매_여름_1'] = 0
        filtered_data_양파.loc[:, '도매_가을_1'] = 0
        filtered_data_양파.loc[:, '도매_겨울_1'] = 0
        filtered_data_양파.loc[:, '도매_겨울_2'] = 0
        filtered_data_양파.loc[:, '도매_겨울_3'] = 0

    # 수치형 컬럼 처리
    numeric_columns_양파 = filtered_data_양파.select_dtypes(include=[np.number]).columns
    filtered_data_양파 = filtered_data_양파[['시점'] + list(numeric_columns_양파)]
    filtered_data_양파[numeric_columns_양파] = filtered_data_양파[numeric_columns_양파].fillna(0)

    양파_train_data_copy = filtered_data_양파.copy()

    ## 주석
    zero_counts = (양파_train_data_copy == 0).sum()
    zero_counts_over_one = zero_counts[zero_counts > 0]

    threshold = 50  # 값이 너무 크다고 판단하는 기준
    columns_to_drop = zero_counts_over_one[zero_counts_over_one > threshold].index

    # 원본 DataFrame에서 해당 열들을 삭제
    양파_train_data_copy_raw = 양파_train_data_copy.drop(columns=columns_to_drop)

    # 삭제할 칼럼 목록
    columns_to_drop = ['공판장_공판장코드', '공판장_품목코드', '공판장_연도',
                        '도매_시장코드', '도매_품목코드', '도매_연도']

    # 해당 칼럼 삭제
    양파_train_data_copy = 양파_train_data_copy_raw.drop(columns=columns_to_drop, errors='ignore')

    numeric_columns = filtered_data_양파.select_dtypes(include=[np.number]).columns

    # IQR을 이용한 이상치 처리
    양파_train_data_copy_cleaned_iqr = replace_outliers_iqr(양파_train_data_copy, ['도매_전순 평균가격(원) PreVious SOON',
                                                                                '도매_전달 평균가격(원) PreVious MMonth','도매_전년 평균가격(원) PreVious YeaR',
                                                                                '도매_최고가(원/kg)'])

    # 퍼센트 변화율 계산
    양파_train_data_copy = 양파_train_data_copy_cleaned_iqr.copy()  # 복사본 생성
    양파_train_data_copy['평균가격_비율변화'] = 양파_train_data_copy['평균가격(원)'].pct_change()* 100

    # fillna 사용
    양파_train_data_copy['평균가격_비율변화'] = 양파_train_data_copy['평균가격_비율변화'].fillna(0)

    양파_train_data_copy = create_ratio_column_and_drop(양파_train_data_copy,
                                                    '공판장_총반입량(kg)',
                                                    '공판장_총거래금액(원)',
                                                    '공판장_반입량_대비_거래금액_비율')

    양파_train_data_copy = create_ratio_column_and_drop(양파_train_data_copy,
                                                    '도매_총반입량(kg)',
                                                    '도매_총거래금액(원)',
                                                    '도매_반입량_대비_거래금액_비율')
    # 퍼센트 변화율 계산
    양파_train_data_copy = 양파_train_data_copy.copy()  # 복사본 생성

    rolling_features = {}
    window_sizes = [3, 6]  # 예시 윈도우 사이즈
    for window_size in window_sizes:
        양파_train_data_copy[f'평균가격_MA_{window_size}'] = 양파_train_data_copy['평균가격(원)'].rolling(window=window_size).mean()
        양파_train_data_copy[f'평균가격_MSTD_{window_size}'] = 양파_train_data_copy['평균가격(원)'].rolling(window=window_size).std()

    # 새로 계산한 rolling features를 한 번에 추가
    rolling_df = pd.DataFrame(rolling_features)

    양파_train_data_copy = pd.concat([양파_train_data_copy.reset_index(drop=True), rolling_df.reset_index(drop=True)], axis=1)

    양파_train_data_copy = 양파_train_data_copy.fillna(0)

    return 양파_train_data_copy, scaler, 양파_train_data_copy_raw


#################################################################################################

def process_양파_for_test(raw_file, 산지공판장_file, 전국도매_file, scaler = None):
    raw_data_양파 = pd.read_csv(raw_file)
    산지공판장_양파 = pd.read_csv(산지공판장_file)
    전국도매_양파 = pd.read_csv(전국도매_file)
    품목명 = '양파'

    전국도매_양파['봄'] = None

    # '봄배추' 데이터 추출
    spring_cabbage = 전국도매_양파[전국도매_양파['품종명'] == '봄배추'][['시점', '총반입량(kg)']]

    # 동일한 시점에 '봄배추'의 '총반입량(kg)'을 '봄' 칼럼에 넣음
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '봄배추':
            matching_spring_cabbage = spring_cabbage[spring_cabbage['시점'] == row['시점']]
            if not matching_spring_cabbage.empty:
                전국도매_양파.at[index, '봄'] = matching_spring_cabbage['총반입량(kg)'].values[0]

    전국도매_양파['봄_1'] = None
    spring_radish = 전국도매_양파[전국도매_양파['품종명'] == '저장무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '저장무':
            matching_spring_radish = spring_radish[spring_radish['시점'] == row['시점']]
            if not matching_spring_radish.empty:
                전국도매_양파.at[index, '봄_1'] = matching_spring_radish['총반입량(kg)'].values[0]

    전국도매_양파['여름'] = None
    summer_cabbage = 전국도매_양파[전국도매_양파['품종명'] == '여름배추'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '여름배추':
            matching_summer_cabbage = summer_cabbage[summer_cabbage['시점'] == row['시점']]
            if not matching_summer_cabbage.empty:
                전국도매_양파.at[index, '여름'] = matching_summer_cabbage['총반입량(kg)'].values[0]

    전국도매_양파['여름_1'] = None
    summer_radish = 전국도매_양파[전국도매_양파['품종명'] == '여름무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '여름무':
            matching_summer_radish = summer_radish[summer_radish['시점'] == row['시점']]
            if not matching_summer_radish.empty:
                전국도매_양파.at[index, '여름_1'] = matching_summer_radish['총반입량(kg)'].values[0]

    전국도매_양파['가을'] = None
    fall_apple = 전국도매_양파[전국도매_양파['품종명'] == '미얀마'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '미얀마':
            matching_fall_apple = fall_apple[fall_apple['시점'] == row['시점']]
            if not matching_fall_apple.empty:
                전국도매_양파.at[index, '가을'] = matching_fall_apple['총반입량(kg)'].values[0]

    전국도매_양파['가을_1'] = None
    fall_daeji = 전국도매_양파[전국도매_양파['품종명'] == '대지'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '대지':
            matching_fall_daeji = fall_daeji[fall_daeji['시점'] == row['시점']]
            if not matching_fall_daeji.empty:
                전국도매_양파.at[index, '가을_1'] = matching_fall_daeji['총반입량(kg)'].values[0]

    전국도매_양파['겨울'] = None
    winter_cabbage = 전국도매_양파[전국도매_양파['품종명'] == '김장(가을)배추'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '김장(가을)배추':
            matching_winter_cabbage = winter_cabbage[winter_cabbage['시점'] == row['시점']]
            if not matching_winter_cabbage.empty:
                전국도매_양파.at[index, '겨울'] = matching_winter_cabbage['총반입량(kg)'].values[0]

    전국도매_양파['겨울_1'] = None
    winter_radish = 전국도매_양파[전국도매_양파['품종명'] == '고냉지무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '고냉지무':
            matching_winter_radish = winter_radish[winter_radish['시점'] == row['시점']]
            if not matching_winter_radish.empty:
                전국도매_양파.at[index, '겨울_1'] = matching_winter_radish['총반입량(kg)'].values[0]

    전국도매_양파['겨울_2'] = None
    winter_2 = 전국도매_양파[전국도매_양파['품종명'] == '월동배추'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '월동배추':
            matching_winter_2 = winter_2[winter_2['시점'] == row['시점']]
            if not matching_winter_2.empty:
                전국도매_양파.at[index, '겨울_2'] = matching_winter_2['총반입량(kg)'].values[0]

    전국도매_양파['겨울_3'] = None
    winter_3 = 전국도매_양파[전국도매_양파['품종명'] == '봄무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_양파.iterrows():
        if row['품종명'] != '봄무':
            matching_winter_3 = winter_3[winter_3['시점'] == row['시점']]
            if not matching_winter_3.empty:
                전국도매_양파.at[index, '겨울_3'] = matching_winter_3['총반입량(kg)'].values[0]



    양파_condition = {'양파': {
            'target': lambda df: (df['품종명'] == '양파') & (df['거래단위'] == '1키로') & (df['등급'] == '상'),
            '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['양파'], '품종명': ['기타양파'], '등급명': ['상']},
            '도매': {'시장명': ['*전국도매시장'], '품목명': ['양파'], '품종명': ['양파(일반)']}
        }}
    # 타겟 데이터 필터링
    raw_품목_양파 = raw_data_양파[raw_data_양파['품목명'] == '양파']
    target_mask_양파 = 양파_condition['양파']['target'](raw_품목_양파)
    filtered_data_양파 = raw_품목_양파[target_mask_양파]

    # 다른 품종에 대한 파생변수 생성
    other_data_양파 = raw_품목_양파[~target_mask_양파]
    unique_combinations_양파 = other_data_양파[['품종명', '거래단위', '등급']].drop_duplicates()
    for _, row in unique_combinations_양파.iterrows():
        품종명, 거래단위, 등급 = row['품종명'], row['거래단위'], row['등급']
        mask_양파 = (other_data_양파['품종명'] == 품종명) & (other_data_양파['거래단위'] == 거래단위) & (other_data_양파['등급'] == 등급)
        temp_df_양파 = other_data_양파[mask_양파]
        for col in ['평년 평균가격(원)', '평균가격(원)']:
            new_col_name = f'{품종명}_{거래단위}_{등급}_{col}'
            filtered_data_양파 = filtered_data_양파.merge(temp_df_양파[['시점', col]], on='시점', how='left', suffixes=('', f'_{new_col_name}'))
            filtered_data_양파.rename(columns={f'{col}_{new_col_name}': new_col_name}, inplace=True)


    # 공판장 데이터 처리
    if 양파_condition['양파']['공판장']:
        filtered_공판장_양파 = 산지공판장_양파
        for key, value in 양파_condition['양파']['공판장'].items():
            filtered_공판장_양파 = filtered_공판장_양파[filtered_공판장_양파[key].isin(value)]

        filtered_공판장_양파 = filtered_공판장_양파.add_prefix('공판장_').rename(columns={'공판장_시점': '시점'})
        filtered_data_양파 = filtered_data_양파.merge(filtered_공판장_양파, on='시점', how='left')


    if 양파_condition['양파']['도매']:
        filtered_도매_양파 = 전국도매_양파
        for key, value in 양파_condition['양파']['도매'].items():
            filtered_도매_양파 = filtered_도매_양파[filtered_도매_양파[key].isin(value)]

        filtered_도매_양파 = filtered_도매_양파.add_prefix('도매_').rename(columns={'도매_시점': '시점'})
        filtered_data_양파 = filtered_data_양파.merge(filtered_도매_양파, on='시점', how='left')

    if not 양파_condition[품목명]['도매']:
        filtered_data_양파.loc[:, '도매_봄'] = 0
        filtered_data_양파.loc[:, '도매_여름'] = 0
        filtered_data_양파.loc[:, '도매_가을'] = 0
        filtered_data_양파.loc[:, '도매_겨울'] = 0
        filtered_data_양파.loc[:, '도매_봄_1'] = 0
        filtered_data_양파.loc[:, '도매_여름_1'] = 0
        filtered_data_양파.loc[:, '도매_가을_1'] = 0
        filtered_data_양파.loc[:, '도매_겨울_1'] = 0
        filtered_data_양파.loc[:, '도매_겨울_2'] = 0
        filtered_data_양파.loc[:, '도매_겨울_3'] = 0


    # 수치형 컬럼 처리
    numeric_columns_양파 = filtered_data_양파.select_dtypes(include=[np.number]).columns
    filtered_data_양파 = filtered_data_양파[['시점'] + list(numeric_columns_양파)]
    filtered_data_양파[numeric_columns_양파] = filtered_data_양파[numeric_columns_양파].fillna(0)

    rolling_features = {}
    window_sizes = [3, 6]  # 예시 윈도우 사이즈
    for window_size in window_sizes:
        rolling_features[f'평균가격_MA_{window_size}'] = filtered_data_양파['평균가격(원)'].rolling(window=window_size).mean()
        rolling_features[f'평균가격_MSTD_{window_size}'] = filtered_data_양파['평균가격(원)'].rolling(window=window_size).std()

    # 새로 계산한 rolling features를 한 번에 추가
    rolling_df = pd.DataFrame(rolling_features)
    filtered_data_양파 = pd.concat([filtered_data_양파.reset_index(drop=True), rolling_df.reset_index(drop=True)], axis=1)

    filtered_data_양파 = create_ratio_column_and_drop(filtered_data_양파,
                                                    '공판장_총반입량(kg)',
                                                    '공판장_총거래금액(원)',
                                                    '공판장_반입량_대비_거래금액_비율')

    filtered_data_양파 = create_ratio_column_and_drop(filtered_data_양파,
                                                    '도매_총반입량(kg)',
                                                    '도매_총거래금액(원)',
                                                    '도매_반입량_대비_거래금액_비율')

    # 퍼센트 변화율 계산
    filtered_data_양파 = filtered_data_양파.copy()  # 복사본 생성

    filtered_data_양파['평균가격_비율변화'] = filtered_data_양파['평균가격(원)'].pct_change() * 100

    # fillna 사용
    filtered_data_양파['평균가격_비율변화'] = filtered_data_양파['평균가격_비율변화'].fillna(0)


    filtered_data_양파 = filtered_data_양파.fillna(0)

    return filtered_data_양파











################################
   # '봄' 칼럼 추가 및 초기화
    전국도매_data['봄'] = None

    # '봄배추' 데이터 추출
    spring_cabbage = 전국도매_data[전국도매_data['품종명'] == '봄배추'][['시점', '총반입량(kg)']]

    # 동일한 시점에 '봄배추'의 '총반입량(kg)'을 '봄' 칼럼에 넣음
    for index, row in 전국도매_data.iterrows():
        if row['품종명'] != '봄배추':
            matching_spring_cabbage = spring_cabbage[spring_cabbage['시점'] == row['시점']]
            if not matching_spring_cabbage.empty:
                전국도매_data.at[index, '봄'] = matching_spring_cabbage['총반입량(kg)'].values[0]

    전국도매_data['봄_1'] = None
    spring_radish = 전국도매_data[전국도매_data['품종명'] == '저장무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_data.iterrows():
        if row['품종명'] != '저장무':
            matching_spring_radish = spring_radish[spring_radish['시점'] == row['시점']]
            if not matching_spring_radish.empty:
                전국도매_data.at[index, '봄_1'] = matching_spring_radish['총반입량(kg)'].values[0]

    전국도매_data['여름'] = None
    summer_cabbage = 전국도매_data[전국도매_data['품종명'] == '여름배추'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_data.iterrows():
        if row['품종명'] != '여름배추':
            matching_summer_cabbage = summer_cabbage[summer_cabbage['시점'] == row['시점']]
            if not matching_summer_cabbage.empty:
                전국도매_data.at[index, '여름'] = matching_summer_cabbage['총반입량(kg)'].values[0]

    전국도매_data['여름_1'] = None
    summer_radish = 전국도매_data[전국도매_data['품종명'] == '여름무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_data.iterrows():
        if row['품종명'] != '여름무':
            matching_summer_radish = summer_radish[summer_radish['시점'] == row['시점']]
            if not matching_summer_radish.empty:
                전국도매_data.at[index, '여름_1'] = matching_summer_radish['총반입량(kg)'].values[0]

    전국도매_data['가을'] = None
    fall_apple = 전국도매_data[전국도매_data['품종명'] == '미얀마'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_data.iterrows():
        if row['품종명'] != '미얀마':
            matching_fall_apple = fall_apple[fall_apple['시점'] == row['시점']]
            if not matching_fall_apple.empty:
                전국도매_data.at[index, '가을'] = matching_fall_apple['총반입량(kg)'].values[0]

    전국도매_data['가을_1'] = None
    fall_daeji = 전국도매_data[전국도매_data['품종명'] == '대지'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_data.iterrows():
        if row['품종명'] != '대지':
            matching_fall_daeji = fall_daeji[fall_daeji['시점'] == row['시점']]
            if not matching_fall_daeji.empty:
                전국도매_data.at[index, '가을_1'] = matching_fall_daeji['총반입량(kg)'].values[0]

    전국도매_data['겨울'] = None
    winter_cabbage = 전국도매_data[전국도매_data['품종명'] == '김장(가을)배추'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_data.iterrows():
        if row['품종명'] != '김장(가을)배추':
            matching_winter_cabbage = winter_cabbage[winter_cabbage['시점'] == row['시점']]
            if not matching_winter_cabbage.empty:
                전국도매_data.at[index, '겨울'] = matching_winter_cabbage['총반입량(kg)'].values[0]

    전국도매_data['겨울_1'] = None
    winter_radish = 전국도매_data[전국도매_data['품종명'] == '고냉지무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_data.iterrows():
        if row['품종명'] != '고냉지무':
            matching_winter_radish = winter_radish[winter_radish['시점'] == row['시점']]
            if not matching_winter_radish.empty:
                전국도매_data.at[index, '겨울_1'] = matching_winter_radish['총반입량(kg)'].values[0]

    전국도매_data['겨울_2'] = None
    winter_2 = 전국도매_data[전국도매_data['품종명'] == '월동배추'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_data.iterrows():
        if row['품종명'] != '월동배추':
            matching_winter_2 = winter_2[winter_2['시점'] == row['시점']]
            if not matching_winter_2.empty:
                전국도매_data.at[index, '겨울_2'] = matching_winter_2['총반입량(kg)'].values[0]

    전국도매_data['겨울_3'] = None
    winter_3 = 전국도매_data[전국도매_data['품종명'] == '봄무'][['시점', '총반입량(kg)']]
    for index, row in 전국도매_data.iterrows():
        if row['품종명'] != '봄무':
            matching_winter_3 = winter_3[winter_3['시점'] == row['시점']]
            if not matching_winter_3.empty:
                전국도매_data.at[index, '겨울_3'] = matching_winter_3['총반입량(kg)'].values[0]