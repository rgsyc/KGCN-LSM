# KGCN-LSM
This is the code to support the paper
运行环境：
tensorflow=1.15.0
numpy=1.21

ns_lstm是LSTM输出的用户特征向量（不用运行，已经生成存好了，生成一次用户特征向量大概需要7个小时）


点开src，运行main.py即可
model是加了特征融合模块的KGCN，user_features是ns_lstm生成的用户特征向量
