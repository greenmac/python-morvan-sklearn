# https://morvanzhou.github.io/tutorials/machine-learning/sklearn/2-2-general-pattern/
# https://github.com/MorvanZhou/tutorials/blob/master/sklearnTUT/sk4_learning_pattern.py
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
# print(iris_X[:2, :])
# print(iris_y)

# 把数据集分为训练集和测试集，其中 test_size=0.3，即测试集占总数据的 30%
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
# print(y_train)
print(iris_X)


knn = KNeighborsClassifier()
knn.fit(X_train, y_train) # fit函式, 丟進去就幫我們訓練
# print(knn.predict(X_test))
# print(y_test)