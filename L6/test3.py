import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
# 数据加载
data = data = pd.read_csv('./train.csv')
train_data = data
print(data.info())
print(data.describe())
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
# 使用票价的均值填充票价中的nan值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)

print(train_data['Embarked'].value_counts())
# 使用登录最多的港口来填充登录港口的nan值
train_data['Embarked'].fillna('S', inplace=True)


train_data.drop(['Cabin','Ticket','Name','PassengerId'], axis=1, inplace = True)
print(train_data.info())

label = train_data['Survived'].values

columns = data.columns.tolist()
columns.remove('Survived')
feature = pd.get_dummies(train_data[columns])

train_x, test_x, train_y, test_y = train_test_split(feature,label,test_size=0.2,random_state=1)

# 构造各种分类器
classifiers = [
    SVC(random_state=1, kernel='rbf'),
    DecisionTreeClassifier(random_state=1, criterion='gini'),
    LogisticRegression(penalty='l1'),
    KNeighborsClassifier(metric='minkowski'),
]
# 分类器名称
classifier_names = [
    'svc',
    'decisiontreeclassifier',
    'lg',
    'kneighborsclassifier',
]
# 分类器参数
classifier_param_grid = [
    {'svc__C': np.arange(0.5,1,0.1), 'svc__gamma': np.arange(0.001,0.05,0.01)},
    {'decisiontreeclassifier__max_depth': [6, 9, 11]},
    {'lg__C': np.arange(0.001,0.05,0.001)},
    {'kneighborsclassifier__n_neighbors': [4, 6, 8]},
]


# 对具体的分类器进行GridSearchCV参数调优
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score='accuracy'):
    response = {}
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=score, n_jobs=8)
    # 寻找最优的参数 和最优的准确率分数
    search = gridsearch.fit(train_x, train_y)
    print("GridSearch最优参数：", search.best_params_)
    print("GridSearch最优分数： %0.4lf" % search.best_score_)
    predict_y = gridsearch.predict(test_x)
    print("准确率 %0.4lf" % accuracy_score(test_y, predict_y))
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y, predict_y)
    return response


for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid, score='accuracy')