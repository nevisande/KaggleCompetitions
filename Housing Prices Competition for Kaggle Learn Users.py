import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from category_encoders import CountEncoder
from itertools import combinations

pd.set_option('display.max_columns', 500)
np.random.seed(40)


# baseline mae = 15624.61491598887
# loading data
data = pd.read_csv('data/Housing Prices Competition for Kaggle Learn Users/train.csv', index_col='Id')
# dropping object columns with null values
null_threshold = data.shape[0] * .05
object_columns_with_null = [col for col in data.select_dtypes(include=['object']).columns if
                            data[col].isna().sum() > null_threshold]
data.drop(object_columns_with_null, axis=1, inplace=True)
# dropping columns with low correlation
corr = data.corr()['SalePrice']
low_correlation = [col for col in corr.index if abs(corr[col]) < .07]
data.drop(low_correlation, axis=1, inplace=True)
# add interaction features
object_col = [col for col in data.select_dtypes('object')]
for col1, col2 in combinations(object_col, 2):
    data[col1 + '_' + col2] = data[col1] + '_' + data[col2]
# creating pipeline
numeric_columns = [col for col in data.select_dtypes(include=np.number).columns if data[col].isnull().any()]
object_columns_with_low_cardinality = [col for col in data.select_dtypes(include=['object']).columns if
                                       data[col].nunique() < 10]
object_columns_with_high_cardinality = [col for col in data.select_dtypes(include=['object']).columns if
                                        data[col].nunique() >= 10]
numeric_transformer = SimpleImputer(strategy='mean')
object_imputer = SimpleImputer(strategy='constant', fill_value='missing')
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ce = CountEncoder(min_group_size=0.01)
object_transformer_with_low_cardinality = Pipeline(steps=[('imputer', object_imputer), ('ohe', ohe)])
object_transformer_with_high_cardinality = Pipeline(steps=[('imputer', object_imputer), ('ce', ce)])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_columns),
                                               ('object_low_cardinality', object_transformer_with_low_cardinality,
                                                object_columns_with_low_cardinality),
                                               ('object_high_cardinality', object_transformer_with_high_cardinality,
                                                object_columns_with_high_cardinality)],
                                 remainder='passthrough')
model = XGBRegressor(n_estimators=1000, learning_rate=.05)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
# creating train and test data
X = data.drop('SalePrice', axis=1)
y = data.SalePrice
mae = -1 * cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f'MAE= {mae.mean()}')
