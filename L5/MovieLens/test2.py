# coding:UTF-8
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.context import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel

# ----------------线性回归--------------

import numpy as np

sc = SparkContext(master='local', appName='Regression')
data = [
    LabeledPoint(1.0, [1.0, 1.0]),
    LabeledPoint(2.0, [1.0, 1.4]),
    LabeledPoint(4.0, [2.0, 1.9]),
    LabeledPoint(6.0, [3.0, 4.0])]  # 训练集
lrm = LinearRegressionWithSGD.train(sc.parallelize(data), iterations=100, initialWeights=np.array([1.0, 1.0]))
print(lrm.predict(np.array([2.0, 1.0])) ) # 利用训练出的回归模型进行预测

import os, tempfile
from pyspark.mllib.regression import LinearRegressionModel
from pyspark.mllib.linalg import SparseVector

path = tempfile.mkdtemp()
lrm.save(sc, path)  # 将模型保存至外存
sameModel = LinearRegressionModel.load(sc, path)  # 读取模型
print(sameModel.predict(SparseVector(2, {0: 100.0, 1: 150})) ) # 利用稀疏向量作为数据结构,返回单个预测值
test_set = []
for i in range(100):
    for j in range(100):
        test_set.append(SparseVector(2, {0: i, 1: j}))
print(sameModel.predict(sc.parallelize(test_set)).collect())  # 预测多值，返回一个RDD数据集
print(sameModel.weights)  # 返回参数

# -----------------岭回归------------------

from pyspark.mllib.regression import RidgeRegressionWithSGD

data = [
    LabeledPoint(1.0, [1.0, 1.0]),
    LabeledPoint(4.0, [1.0, 3.0]),
    LabeledPoint(8.0, [2.0, 3.0]),
    LabeledPoint(10.0, [3.0, 4.0])]
train_set = sc.parallelize(data)
rrm = RidgeRegressionWithSGD.train(train_set, iterations=100, initialWeights=np.array([1.0, 1.0]))
test_set = []
for i in range(100):
    for j in range(100):
        test_set.append(np.array([i, j]))
print(rrm.predict(sc.parallelize(test_set)).collect())
print(rrm.weights)