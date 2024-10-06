
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

def process_건고추_for_train(raw_file, 산지공판장_file, 전국도매_file, scaler = None):
  raw_data_건고추 = pd.read_csv(raw_file)
  산지공판장_건고추 = pd.read_csv(산지공판장_file)
  전국도매_건고추 = pd.read_csv(전국도매_file)
  품목명 = '건고추'
  건고추_condition = {'건고추': {
        'target': lambda df: (df['품종명'] == '화건') & (df['거래단위'] == '30 kg') & (df['등급'] == '상품'),
        '공판장': None,
        '도매': None
    }}
  # 타겟 데이터 필터링
  raw_품목_건고추 = raw_data_건고추[raw_data_건고추['품목명'] == '건고추']
  target_mask_건고추 = 건고추_condition['건고추']['target'](raw_품목_건고추)
  filtered_data_건고추 = raw_품목_건고추[target_mask_건고추]

# 다른 품종에 대한 파생변수 생성
  other_data_건고추 = raw_품목_건고추[~target_mask_건고추]
  unique_combinations_건고추 = other_data_건고추[['품종명', '거래단위', '등급']].drop_duplicates()
  for _, row in unique_combinations_건고추.iterrows():
      품종명, 거래단위, 등급 = row['품종명'], row['거래단위'], row['등급']
      mask_건고추 = (other_data_건고추['품종명'] == 품종명) & (other_data_건고추['거래단위'] == 거래단위) & (other_data_건고추['등급'] == 등급)
      temp_df_건고추 = other_data_건고추[mask_건고추]
      for col in ['평년 평균가격(원)', '평균가격(원)']:
          new_col_name = f'{품종명}_{거래단위}_{등급}_{col}'
          filtered_data_건고추 = filtered_data_건고추.merge(temp_df_건고추[['시점', col]], on='시점', how='left', suffixes=('', f'_{new_col_name}'))
          filtered_data_건고추.rename(columns={f'{col}_{new_col_name}': new_col_name}, inplace=True)


  # 공판장 데이터 처리
  if 건고추_condition['건고추']['공판장']:
      filtered_공판장_건고추 = 산지공판장_건고추
      for key, value in 건고추_condition['건고추']['공판장'].items():
          filtered_공판장_건고추 = filtered_공판장_건고추[filtered_공판장_건고추[key].isin(value)]

      filtered_공판장_건고추 = filtered_공판장_건고추.add_prefix('공판장_').rename(columns={'공판장_시점': '시점'})
      filtered_data_건고추 = filtered_data_건고추.merge(filtered_공판장_건고추, on='시점', how='left')


  if 건고추_condition['건고추']['도매']:
      filtered_도매_건고추 = 전국도매_건고추
      for key, value in 건고추_condition['건고추']['도매'].items():
          filtered_도매_건고추 = filtered_도매_건고추[filtered_도매_건고추[key].isin(value)]

      filtered_도매_건고추 = filtered_도매_건고추.add_prefix('도매_').rename(columns={'도매_시점': '시점'})
      filtered_data_건고추 = filtered_data_건고추.merge(filtered_도매_건고추, on='시점', how='left')

  # 수치형 컬럼 처리
  numeric_columns_건고추 = filtered_data_건고추.select_dtypes(include=[np.number]).columns
  filtered_data_건고추 = filtered_data_건고추[['시점'] + list(numeric_columns_건고추)]
  filtered_data_건고추[numeric_columns_건고추] = filtered_data_건고추[numeric_columns_건고추].fillna(0)



  건고추_train_data_copy = filtered_data_건고추.copy()

  zero_counts = (건고추_train_data_copy == 0).sum()
  zero_counts_over_one = zero_counts[zero_counts > 0]

  threshold = 100  # 값이 너무 크다고 판단하는 기준
  columns_to_drop = zero_counts_over_one[zero_counts_over_one > threshold].index

  # 원본 DataFrame에서 해당 열들을 삭제
  건고추_train_data_copy = 건고추_train_data_copy.drop(columns=columns_to_drop)

  # 삭제할 칼럼 목록
  columns_to_drop = ['공판장_공판장코드', '공판장_품목코드', '공판장_연도',
                    '도매_시장코드', '도매_품목코드', '도매_연도']

  # 해당 칼럼 삭제
  건고추_train_data_copy = 건고추_train_data_copy.drop(columns=columns_to_drop, errors='ignore')

  scaler = StandardScaler()
  numeric_columns_건고추 = 건고추_train_data_copy.select_dtypes(include=[np.number]).columns
  건고추_train_data_copy[numeric_columns_건고추] = scaler.fit_transform(건고추_train_data_copy[numeric_columns_건고추])

  return 건고추_train_data_copy, scaler

def process_건고추_for_test(raw_file, 산지공판장_file, 전국도매_file, scaler = None):
  raw_data_건고추 = pd.read_csv(raw_file)
  산지공판장_건고추 = pd.read_csv(산지공판장_file)
  전국도매_건고추 = pd.read_csv(전국도매_file)
  품목명 = '건고추'
  건고추_condition = {'건고추': {
        'target': lambda df: (df['품종명'] == '화건') & (df['거래단위'] == '30 kg') & (df['등급'] == '상품'),
        '공판장': None,
        '도매': None
    }}
  # 타겟 데이터 필터링
  raw_품목_건고추 = raw_data_건고추[raw_data_건고추['품목명'] == '건고추']
  target_mask_건고추 = 건고추_condition['건고추']['target'](raw_품목_건고추)
  filtered_data_건고추 = raw_품목_건고추[target_mask_건고추]

# 다른 품종에 대한 파생변수 생성
  other_data_건고추 = raw_품목_건고추[~target_mask_건고추]
  unique_combinations_건고추 = other_data_건고추[['품종명', '거래단위', '등급']].drop_duplicates()
  for _, row in unique_combinations_건고추.iterrows():
      품종명, 거래단위, 등급 = row['품종명'], row['거래단위'], row['등급']
      mask_건고추 = (other_data_건고추['품종명'] == 품종명) & (other_data_건고추['거래단위'] == 거래단위) & (other_data_건고추['등급'] == 등급)
      temp_df_건고추 = other_data_건고추[mask_건고추]
      for col in ['평년 평균가격(원)', '평균가격(원)']:
          new_col_name = f'{품종명}_{거래단위}_{등급}_{col}'
          filtered_data_건고추 = filtered_data_건고추.merge(temp_df_건고추[['시점', col]], on='시점', how='left', suffixes=('', f'_{new_col_name}'))
          filtered_data_건고추.rename(columns={f'{col}_{new_col_name}': new_col_name}, inplace=True)


  # 공판장 데이터 처리
  if 건고추_condition['건고추']['공판장']:
      filtered_공판장_건고추 = 산지공판장_건고추
      for key, value in 건고추_condition['건고추']['공판장'].items():
          filtered_공판장_건고추 = filtered_공판장_건고추[filtered_공판장_건고추[key].isin(value)]

      filtered_공판장_건고추 = filtered_공판장_건고추.add_prefix('공판장_').rename(columns={'공판장_시점': '시점'})
      filtered_data_건고추 = filtered_data_건고추.merge(filtered_공판장_건고추, on='시점', how='left')


  if 건고추_condition['건고추']['도매']:
      filtered_도매_건고추 = 전국도매_건고추
      for key, value in 건고추_condition['건고추']['도매'].items():
          filtered_도매_건고추 = filtered_도매_건고추[filtered_도매_건고추[key].isin(value)]

      filtered_도매_건고추 = filtered_도매_건고추.add_prefix('도매_').rename(columns={'도매_시점': '시점'})
      filtered_data_건고추 = filtered_data_건고추.merge(filtered_도매_건고추, on='시점', how='left')

  # 수치형 컬럼 처리
  numeric_columns_건고추 = filtered_data_건고추.select_dtypes(include=[np.number]).columns
  filtered_data_건고추 = filtered_data_건고추[['시점'] + list(numeric_columns_건고추)]
  filtered_data_건고추[numeric_columns_건고추] = filtered_data_건고추[numeric_columns_건고추].fillna(0)


  return filtered_data_건고추


