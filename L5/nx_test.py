from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly, KNNBasic, NormalPredictor
from surprise import accuracy
from surprise.model_selection import KFold
import pandas as pd
import numpy as np

data = pd.read_csv('./combined_data_1.txt',header = None, names=['users' ,'rate'],usecols=[0,1])
data = data.iloc[0:100000]

df_nan = pd.DataFrame(pd.isnull(data.rate))
df_nan = df_nan[df_nan['rate'] == True]
df_nan = df_nan.reset_index()

df_nan = pd.DataFrame(pd.isnull(data.rate))
df_nan = df_nan[df_nan['rate'] == True]
df_nan = df_nan.reset_index()

movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    # numpy approach
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

# Account for last record and corresponding length
# numpy approach
last_record = np.full((1,len(data) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

df = data[pd.notnull(data['rate'])]
df['movie_id'] = movie_np.astype(int)
df['movie_id_t'] = movie_np.astype(int)
df['users'] = df['users'].astype(int)

cols = list(df)
cols.insert(1,cols.pop(cols.index('movie_id')))
df = df.loc[:,cols]

data = Dataset.load_from_df(df[['users', 'movie_id', 'rate','movie_id_t']][:], Reader)
train_set = data.build_full_trainset()
'''
# 数据读取
reader = Reader(line_format='user rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./probe.txt', reader=reader)
train_set = data.build_full_trainset()

'''

# ALS优化
#bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
# SGD优化
bsl_options = {'method': 'sgd','n_epochs': 5}
algo = BaselineOnly(bsl_options=bsl_options)
#algo = BaselineOnly()
#algo = NormalPredictor()

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)

#uid = str(196)
#iid = str(302)
# 输出uid对iid的预测结果
#pred = algo.predict(uid, iid, r_ui=4, verbose=True)



