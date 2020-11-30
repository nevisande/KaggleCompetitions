import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
pd.set_option('display.max_columns', 500)


# baseline mae = 16433.14142074364
def imputation(strategy, column_list, df):
    columns_must_be_imputed = df[column_list]
    df.drop(column_list, axis=1, inplace=True)
    imputer = SimpleImputer(strategy=strategy)
    imputed_columns = pd.DataFrame(imputer.fit_transform(columns_must_be_imputed))
    imputed_columns.columns = columns_must_be_imputed.columns
    imputed_columns.index = columns_must_be_imputed.index
    return pd.concat([df, imputed_columns], axis=1)


# loading data
data = pd.read_csv('data/Housing Prices Competition for Kaggle Learn Users/train.csv', index_col='Id')
# finding numeric and null columns to impute
numeric_and_null_columns = [col for col in data.select_dtypes(include=np.number).columns if data[col].isnull().any()]
correlation = data[numeric_and_null_columns + ['SalePrice']].corr()
print(f'correlation between numeric and null columns and sales price\n{correlation.SalePrice}\n{"-" * 30}')
percentage_of_null = data[numeric_and_null_columns].isnull().sum() * 100 / data.shape[0]
print(f'percentage of nulls\n{percentage_of_null}\n{"-" * 30}')
# imputing numeric and null columns
data = imputation('mean', numeric_and_null_columns, data)
# dropping null columns
not_na_data = data.dropna(axis=1)
X = not_na_data.drop('SalePrice', axis=1)
obj_col = list(X.select_dtypes('object').columns)
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_obj_col = pd.DataFrame(ohe.fit_transform(X[obj_col]))
encoded_obj_col.index = X.index
X.drop(obj_col, axis=1, inplace=True)
X = pd.concat([X, encoded_obj_col], axis=1)
y = not_na_data.SalePrice
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rfr = RandomForestRegressor(random_state=1)
rfr.fit(train_X, train_y)
predicted_y = rfr.predict(val_X)
mae = mean_absolute_error(val_y, predicted_y)
print(f'MAE= {mae}')
