# https://morvanzhou.github.io/tutorials/machine-learning/sklearn/2-3-database/
# https://github.com/MorvanZhou/tutorials/blob/master/sklearnTUT/sk5_datasets.py
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

## load_boston
# loaded_data = datasets.load_boston()
# data_X = loaded_data.data
# data_y = loaded_data.target

# model = LinearRegression()
# model.fit(data_X, data_y)
# print(model.predict(data_X[:4, :]))
# print(data_y[:4])


## plt
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
plt.scatter(X, y)
plt.show()