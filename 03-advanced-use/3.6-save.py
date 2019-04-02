# https://morvanzhou.github.io/tutorials/machine-learning/sklearn/3-5-save/
# https://github.com/MorvanZhou/tutorials/blob/master/sklearnTUT/sk11_save.py
from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris['data'], iris['target']
clf.fit(X, y)

"""
使用 pickle 保存
"""
import pickle #pickle模块

#保存Model(注:save文件夹要预先建立，否则会报错)
# with open('save/clf.pickle', 'wb') as f:
#     pickle.dump(clf, f)

#读取Model
with open('save/clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    # 测试读取后的Model
    # predict返回的是一个大小为n的一维数组，一维数组中的第i个值为模型预测第i个预测样本的标签
    # predict_proba返回的是一个n行k列的数组，第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率。此时每一行的和应该等于1
    print(clf2.predict(X[0:1]))
    # output=>[0]

"""
使用 joblib 保存
joblib是sklearn的外部模块
"""
from sklearn.externals import joblib #jbolib模块

#保存Model(注:save文件夹要预先建立，否则会报错)
joblib.dump(clf, 'save/clf.pkl')

#读取Model
clf3 = joblib.load('save/clf.pkl')

#测试读取后的Model
print(clf3.predict(X[0:1]))
# output=>[0]

# 最后可以知道joblib在使用上比较容易，读取速度也相对pickle快