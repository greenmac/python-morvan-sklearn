# https://morvanzhou.github.io/tutorials/machine-learning/sklearn/2-4-model-attributes/
# https://github.com/MorvanZhou/tutorials/blob/master/sklearnTUT/sk6_model_attribute_method.py
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

## load_boston
loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

# print(model.predict(data_X[:4, :]))
# print(data_y[:4])

# print(model.coef_) # y=0.1x+0.3, 顯示0.1的部分
# print(model.intercept_) # y=0.1x+0.3, 顯示0.3的部分

# print(model.get_params()) # 之前定義的參數

print(model.score(data_X, data_y)) # R^2 coefficient of determination 確定係數 百分率