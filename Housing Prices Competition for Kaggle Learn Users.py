import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_columns', 500)

# baseline mae = 16533.507283757335
# loading data
data = pd.read_csv('data/Housing Prices Competition for Kaggle Learn Users/train.csv', index_col='Id')
# dropping object columns with null value and high cardinality
object_columns_with_null = [col for col in data.select_dtypes(include=['object']).columns if data[col].isnull().any()
                            or data[col].nunique() > 10]
data.drop(object_columns_with_null, axis=1, inplace=True)
# creating pipeline
numeric_columns = [col for col in data.select_dtypes(include=np.number).columns if data[col].isnull().any()]
object_columns = [col for col in data.select_dtypes(include=['object']).columns]
numeric_transformer = SimpleImputer(strategy='mean')
object_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_columns),
                                               ('object', object_transformer, object_columns)], remainder='passthrough')
model = RandomForestRegressor(random_state=1)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
# creating train and test data
X = data.drop('SalePrice', axis=1)
y = data.SalePrice
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# fit and predict
pipeline.fit(train_X, train_y)
predicted_y = pipeline.predict(val_X)
mae = mean_absolute_error(val_y, predicted_y)
print(f'MAE= {mae}')
