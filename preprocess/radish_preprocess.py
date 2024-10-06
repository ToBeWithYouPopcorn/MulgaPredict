
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


def process_무_for_train(raw_file, 산지공판장_file, 전국도매_file, scaler = None):
  raw_data_무 = pd.read_csv(raw_file)
  산지공판장_무 = pd.read_csv(산지공판장_file)
  전국도매_무 = pd.read_csv(전국도매_file)
  품목명 = '무'
  무_condition = {'무': {
        'target': lambda df: (df['거래단위'] == '20키로상자') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['무'], '품종명': ['기타무'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['무'], '품종명': ['무']}
    }}
  # 타겟 데이터 필터링
  raw_품목_무 = raw_data_무[raw_data_무['품목명'] == '무']
  target_mask_무 = 무_condition['무']['target'](raw_품목_무)
  filtered_data_무 = raw_품목_무[target_mask_무]

# 다른 품종에 대한 파생변수 생성
  other_data_무 = raw_품목_무[~target_mask_무]
  unique_combinations_무 = other_data_무[['품종명', '거래단위', '등급']].drop_duplicates()
  for _, row in unique_combinations_무.iterrows():
      품종명, 거래단위, 등급 = row['품종명'], row['거래단위'], row['등급']
      mask_무 = (other_data_무['품종명'] == 품종명) & (other_data_무['거래단위'] == 거래단위) & (other_data_무['등급'] == 등급)
      temp_df_무 = other_data_무[mask_무]
      for col in ['평년 평균가격(원)', '평균가격(원)']:
          new_col_name = f'{품종명}_{거래단위}_{등급}_{col}'
          filtered_data_무 = filtered_data_무.merge(temp_df_무[['시점', col]], on='시점', how='left', suffixes=('', f'_{new_col_name}'))
          filtered_data_무.rename(columns={f'{col}_{new_col_name}': new_col_name}, inplace=True)


  # 공판장 데이터 처리
  if 무_condition['무']['공판장']:
      filtered_공판장_무 = 산지공판장_무
      for key, value in 무_condition['무']['공판장'].items():
          filtered_공판장_무 = filtered_공판장_무[filtered_공판장_무[key].isin(value)]

      filtered_공판장_무 = filtered_공판장_무.add_prefix('공판장_').rename(columns={'공판장_시점': '시점'})
      filtered_data_무 = filtered_data_무.merge(filtered_공판장_무, on='시점', how='left')


  if 무_condition['무']['도매']:
      filtered_도매_무 = 전국도매_무
      for key, value in 무_condition['무']['도매'].items():
          filtered_도매_무 = filtered_도매_무[filtered_도매_무[key].isin(value)]

      filtered_도매_무 = filtered_도매_무.add_prefix('도매_').rename(columns={'도매_시점': '시점'})
      filtered_data_무 = filtered_data_무.merge(filtered_도매_무, on='시점', how='left')

  # 수치형 컬럼 처리
  numeric_columns_무 = filtered_data_무.select_dtypes(include=[np.number]).columns
  filtered_data_무 = filtered_data_무[['시점'] + list(numeric_columns_무)]
  filtered_data_무[numeric_columns_무] = filtered_data_무[numeric_columns_무].fillna(0)


  무_train_data_copy = filtered_data_무.copy()

  zero_counts = (무_train_data_copy == 0).sum()
  zero_counts_over_one = zero_counts[zero_counts > 0]

  threshold = 50  # 값이 너무 크다고 판단하는 기준
  columns_to_drop = zero_counts_over_one[zero_counts_over_one > threshold].index

  # 원본 DataFrame에서 해당 열들을 삭제
  무_train_data_copy = 무_train_data_copy.drop(columns=columns_to_drop)

  # 삭제할 칼럼 목록
  columns_to_drop = ['공판장_공판장코드', '공판장_품목코드', '공판장_연도',
                    '도매_시장코드', '도매_품목코드', '도매_연도']

  # 해당 칼럼 삭제
  무_train_data_copy = 무_train_data_copy.drop(columns=columns_to_drop, errors='ignore')

  scaler = StandardScaler()
  numeric_columns_무 = 무_train_data_copy.select_dtypes(include=[np.number]).columns
  무_train_data_copy[numeric_columns_무] = scaler.fit_transform(무_train_data_copy[numeric_columns_무])

  return 무_train_data_copy, scaler

def process_무_for_test(raw_file, 산지공판장_file, 전국도매_file, scaler = None):
  raw_data_무 = pd.read_csv(raw_file)
  산지공판장_무 = pd.read_csv(산지공판장_file)
  전국도매_무 = pd.read_csv(전국도매_file)
  품목명 = '무'
  무_condition = {'무': {
        'target': lambda df: (df['거래단위'] == '20키로상자') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['무'], '품종명': ['기타무'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['무'], '품종명': ['무']}
    }}
  # 타겟 데이터 필터링
  raw_품목_무 = raw_data_무[raw_data_무['품목명'] == '무']
  target_mask_무 = 무_condition['무']['target'](raw_품목_무)
  filtered_data_무 = raw_품목_무[target_mask_무]

# 다른 품종에 대한 파생변수 생성
  other_data_무 = raw_품목_무[~target_mask_무]
  unique_combinations_무 = other_data_무[['품종명', '거래단위', '등급']].drop_duplicates()
  for _, row in unique_combinations_무.iterrows():
      품종명, 거래단위, 등급 = row['품종명'], row['거래단위'], row['등급']
      mask_무 = (other_data_무['품종명'] == 품종명) & (other_data_무['거래단위'] == 거래단위) & (other_data_무['등급'] == 등급)
      temp_df_무 = other_data_무[mask_무]
      for col in ['평년 평균가격(원)', '평균가격(원)']:
          new_col_name = f'{품종명}_{거래단위}_{등급}_{col}'
          filtered_data_무 = filtered_data_무.merge(temp_df_무[['시점', col]], on='시점', how='left', suffixes=('', f'_{new_col_name}'))
          filtered_data_무.rename(columns={f'{col}_{new_col_name}': new_col_name}, inplace=True)


  # 공판장 데이터 처리
  if 무_condition['무']['공판장']:
      filtered_공판장_무 = 산지공판장_무
      for key, value in 무_condition['무']['공판장'].items():
          filtered_공판장_무 = filtered_공판장_무[filtered_공판장_무[key].isin(value)]

      filtered_공판장_무 = filtered_공판장_무.add_prefix('공판장_').rename(columns={'공판장_시점': '시점'})
      filtered_data_무 = filtered_data_무.merge(filtered_공판장_무, on='시점', how='left')


  if 무_condition['무']['도매']:
      filtered_도매_무 = 전국도매_무
      for key, value in 무_condition['무']['도매'].items():
          filtered_도매_무 = filtered_도매_무[filtered_도매_무[key].isin(value)]

      filtered_도매_무 = filtered_도매_무.add_prefix('도매_').rename(columns={'도매_시점': '시점'})
      filtered_data_무 = filtered_data_무.merge(filtered_도매_무, on='시점', how='left')

  # 수치형 컬럼 처리
  numeric_columns_무 = filtered_data_무.select_dtypes(include=[np.number]).columns
  filtered_data_무 = filtered_data_무[['시점'] + list(numeric_columns_무)]
  filtered_data_무[numeric_columns_무] = filtered_data_무[numeric_columns_무].fillna(0)

  return filtered_data_무
