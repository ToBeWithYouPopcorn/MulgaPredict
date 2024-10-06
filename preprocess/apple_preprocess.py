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

def process_사과_for_train(raw_file, 산지공판장_file, 전국도매_file, scaler = None):
  raw_data_사과 = pd.read_csv(raw_file)
  산지공판장_사과 = pd.read_csv(산지공판장_file)
  전국도매_사과 = pd.read_csv(전국도매_file)
  품목명 = '사과'
  사과_condition = {'사과': {
          'target': lambda df: (df['품종명'].isin(['홍로', '후지'])) & (df['거래단위'] == '10 개') & (df['등급'] == '상품'),
          '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['사과'], '품종명': ['후지'], '등급명': ['상']},
          '도매': {'시장명': ['*전국도매시장'], '품목명': ['사과'], '품종명': ['후지']}}}
  # 타겟 데이터 필터링
  raw_품목_사과 = raw_data_사과[raw_data_사과['품목명'] == '사과']
  target_mask_사과 = 사과_condition['사과']['target'](raw_품목_사과)
  filtered_data_사과 = raw_품목_사과[target_mask_사과]

# 다른 품종에 대한 파생변수 생성
  other_data_사과 = raw_품목_사과[~target_mask_사과]
  unique_combinations_사과 = other_data_사과[['품종명', '거래단위', '등급']].drop_duplicates()
  for _, row in unique_combinations_사과.iterrows():
      품종명, 거래단위, 등급 = row['품종명'], row['거래단위'], row['등급']
      mask_사과 = (other_data_사과['품종명'] == 품종명) & (other_data_사과['거래단위'] == 거래단위) & (other_data_사과['등급'] == 등급)
      temp_df_사과 = other_data_사과[mask_사과]
      for col in ['평년 평균가격(원)', '평균가격(원)']:
          new_col_name = f'{품종명}_{거래단위}_{등급}_{col}'
          filtered_data_사과 = filtered_data_사과.merge(temp_df_사과[['시점', col]], on='시점', how='left', suffixes=('', f'_{new_col_name}'))
          filtered_data_사과.rename(columns={f'{col}_{new_col_name}': new_col_name}, inplace=True)


  # 공판장 데이터 처리
  if 사과_condition['사과']['공판장']:
      filtered_공판장_사과 = 산지공판장_사과
      for key, value in 사과_condition['사과']['공판장'].items():
          filtered_공판장_사과 = filtered_공판장_사과[filtered_공판장_사과[key].isin(value)]

      filtered_공판장_사과 = filtered_공판장_사과.add_prefix('공판장_').rename(columns={'공판장_시점': '시점'})
      filtered_data_사과 = filtered_data_사과.merge(filtered_공판장_사과, on='시점', how='left')


  if 사과_condition['사과']['도매']:
      filtered_도매_사과 = 전국도매_사과
      for key, value in 사과_condition['사과']['도매'].items():
          filtered_도매_사과 = filtered_도매_사과[filtered_도매_사과[key].isin(value)]

      filtered_도매_사과 = filtered_도매_사과.add_prefix('도매_').rename(columns={'도매_시점': '시점'})
      filtered_data_사과 = filtered_data_사과.merge(filtered_도매_사과, on='시점', how='left')

  # 수치형 컬럼 처리
  numeric_columns_사과 = filtered_data_사과.select_dtypes(include=[np.number]).columns
  filtered_data_사과 = filtered_data_사과[['시점'] + list(numeric_columns_사과)]
  filtered_data_사과[numeric_columns_사과] = filtered_data_사과[numeric_columns_사과].fillna(0)

  사과_train_data_copy = filtered_data_사과.copy()

  zero_counts = (사과_train_data_copy == 0).sum()
  zero_counts_over_one = zero_counts[zero_counts > 0]

  threshold = 100  # 값이 너무 크다고 판단하는 기준
  columns_to_drop = zero_counts_over_one[zero_counts_over_one > threshold].index
  사과_train_data_copy = 사과_train_data_copy.drop(columns=columns_to_drop)

    # 정규화 적용
  scaler = StandardScaler()
  numeric_columns_사과 = 사과_train_data_copy.select_dtypes(include=[np.number]).columns
  사과_train_data_copy[numeric_columns_사과] = scaler.fit_transform(사과_train_data_copy[numeric_columns_사과])

  return 사과_train_data_copy, scaler


def process_사과_for_test(raw_file, 산지공판장_file, 전국도매_file, scaler = None):
  raw_data_사과 = pd.read_csv(raw_file)
  산지공판장_사과 = pd.read_csv(산지공판장_file)
  전국도매_사과 = pd.read_csv(전국도매_file)
  품목명 = '사과'
  사과_condition = {'사과': {
          'target': lambda df: (df['품종명'].isin(['홍로', '후지'])) & (df['거래단위'] == '10 개') & (df['등급'] == '상품'),
          '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['사과'], '품종명': ['후지'], '등급명': ['상']},
          '도매': {'시장명': ['*전국도매시장'], '품목명': ['사과'], '품종명': ['후지']}}}
  # 타겟 데이터 필터링
  raw_품목_사과 = raw_data_사과[raw_data_사과['품목명'] == '사과']
  target_mask_사과 = 사과_condition['사과']['target'](raw_품목_사과)
  filtered_data_사과 = raw_품목_사과[target_mask_사과]

# 다른 품종에 대한 파생변수 생성
  other_data_사과 = raw_품목_사과[~target_mask_사과]
  unique_combinations_사과 = other_data_사과[['품종명', '거래단위', '등급']].drop_duplicates()
  for _, row in unique_combinations_사과.iterrows():
      품종명, 거래단위, 등급 = row['품종명'], row['거래단위'], row['등급']
      mask_사과 = (other_data_사과['품종명'] == 품종명) & (other_data_사과['거래단위'] == 거래단위) & (other_data_사과['등급'] == 등급)
      temp_df_사과 = other_data_사과[mask_사과]
      for col in ['평년 평균가격(원)', '평균가격(원)']:
          new_col_name = f'{품종명}_{거래단위}_{등급}_{col}'
          filtered_data_사과 = filtered_data_사과.merge(temp_df_사과[['시점', col]], on='시점', how='left', suffixes=('', f'_{new_col_name}'))
          filtered_data_사과.rename(columns={f'{col}_{new_col_name}': new_col_name}, inplace=True)


  # 공판장 데이터 처리
  if 사과_condition['사과']['공판장']:
      filtered_공판장_사과 = 산지공판장_사과
      for key, value in 사과_condition['사과']['공판장'].items():
          filtered_공판장_사과 = filtered_공판장_사과[filtered_공판장_사과[key].isin(value)]

      filtered_공판장_사과 = filtered_공판장_사과.add_prefix('공판장_').rename(columns={'공판장_시점': '시점'})
      filtered_data_사과 = filtered_data_사과.merge(filtered_공판장_사과, on='시점', how='left')


  if 사과_condition['사과']['도매']:
      filtered_도매_사과 = 전국도매_사과
      for key, value in 사과_condition['사과']['도매'].items():
          filtered_도매_사과 = filtered_도매_사과[filtered_도매_사과[key].isin(value)]

      filtered_도매_사과 = filtered_도매_사과.add_prefix('도매_').rename(columns={'도매_시점': '시점'})
      filtered_data_사과 = filtered_data_사과.merge(filtered_도매_사과, on='시점', how='left')

  # 수치형 컬럼 처리
  numeric_columns_사과 = filtered_data_사과.select_dtypes(include=[np.number]).columns
  filtered_data_사과 = filtered_data_사과[['시점'] + list(numeric_columns_사과)]
  filtered_data_사과[numeric_columns_사과] = filtered_data_사과[numeric_columns_사과].fillna(0)

  return filtered_data_사과