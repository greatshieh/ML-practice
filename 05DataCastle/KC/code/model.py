import pandas as pd
import numpy as np

# import sys
# sys.path.append('.')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

columns = [
    'date', 'price', 'bedroom', 'bathroom', 'roomarea', 'parkarea', 'floornum',
    'score', 'buildingarea', 'baseroomarea', 'buildingdate', 'fixdate', 'lat',
    'long'
]
df_train = pd.read_csv('../dataset/kc_train.csv', header=None, names=columns)
columns.remove('price')
df_test = pd.read_csv('../dataset/kc_test.csv', header=None, names=columns)

min_max = StandardScaler()
columns = [
    'bedroom', 'bathroom', 'roomarea', 'parkarea', 'floornum', 'score',
    'buildingarea', 'baseroomarea'
]

#df_train[columns] = df_train[columns].apply(lambda x: np.log(x+1))
#df_test[columns] = df_test[columns].apply(lambda x: np.log(x+1))

min_max.fit(df_train[columns])
df_train[columns] = min_max.transform(df_train[columns])
df_test[columns] = min_max.transform(df_test[columns])

X_train, X_test, y_train, y_test = train_test_split(
    df_train[columns], df_train['price'], test_size=0.3, random_state=7)

linear = LinearRegression()
model = linear.fit(X_train, y_train)
result = model.predict(X_test)
mse = sum((result-y_test)**2)/(10000*len(y_test))
print(mse)

predictor = model.predict(df_test[columns])
result = pd.DataFrame({'price': predictor})
result.to_csv('result.csv', index=False)
