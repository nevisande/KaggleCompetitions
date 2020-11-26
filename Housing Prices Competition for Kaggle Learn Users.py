import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/Housing Prices Competition for Kaggle Learn Users/train.csv')
not_na_data = data.dropna(axis=1)
X = not_na_data.drop('SalePrice', axis=1)
X = pd.get_dummies(X)
y = not_na_data.SalePrice
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rfr = RandomForestRegressor(random_state=1)
rfr.fit(train_X, train_y)
predicted_y = rfr.predict(val_X)
mae = mean_absolute_error(val_y, predicted_y)
print(mae)
