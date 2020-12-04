import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_columns', 500)

# baseline mae = 17707.492064579255
# loading data
data = pd.read_csv('data/Housing Prices Competition for Kaggle Learn Users/train.csv', index_col='Id')
# dropping object columns with null value and high cardinality
null_threshold = data.shape[0] * .05
object_columns_with_null = [col for col in data.select_dtypes(include=['object']).columns if data[col].isna().sum() >
                            null_threshold or data[col].nunique() > 10]
data.drop(object_columns_with_null, axis=1, inplace=True)
# dropping columns with low correlation
corr = data.corr()['SalePrice']
low_correlation = [col for col in corr.index if abs(corr[col]) < .07]
data.drop(low_correlation, axis=1, inplace=True)
# creating pipeline
numeric_columns = [col for col in data.select_dtypes(include=np.number).columns if data[col].isnull().any()]
object_columns = [col for col in data.select_dtypes(include=['object']).columns]
numeric_transformer = SimpleImputer(strategy='mean')
object_imputer = SimpleImputer(strategy='constant', fill_value='missing')
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
object_transformer = Pipeline(steps=[('imputer', object_imputer), ('ohe', ohe)])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_columns),
                                               ('object', object_transformer, object_columns)], remainder='passthrough')
model = RandomForestRegressor(random_state=1)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
# creating train and test data
X = data.drop('SalePrice', axis=1)
y = data.SalePrice
mae = -1 * cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f'MAE= {mae.mean()}')
