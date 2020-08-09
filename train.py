from sklearn.model_selection import StratifiedKFold
from models import *
from dataset import  Make_data
import pandas as pd
from utils import *
import time
import os
import warnings
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import initializers
import matplotlib.pyplot as plt
mk_data = Make_data(data_path='./dataset/')
x, y, t = mk_data.load_dataset()


begin_time = time.time()

pre = {}
scores_all = []
val_train = np.zeros((7500,19))
nums = 1
cv= 20
for num in range(nums):
    seed = np.random.randint(0,10000)
    print('='*15,'第{}次'.format(num),'随机因子',seed,'='*15)
#     kfold = StratifiedKFold(n_splits=cv, shuffle=True,random_state=seed)
    kfold = StratifiedKFold(n_splits=cv, shuffle=True)
    proba_t = np.zeros((7500, 19))
    train_pred= np.zeros((7292, 19))
    cvscores = []
    valid_fold = []

    for fold, (xx, yy) in enumerate(kfold.split(x, y)):#返回索引xx,yy
        print('='*15,'第{}次'.format(num),'fold=',fold,'='*15)
        
        y_ = one_hot(y, num_classes=19)# 转换成二进制矩阵 
        # 样本平衡
        
        x1,y1= unba_randomos(x[xx],y[xx])
        x1,y1= part_replacment(x[xx],y[xx])
        y1 = one_hot(y1, num_classes=19)
        
        

        # x1 =  np.vstack((x[xx],new_x[valid_fold[fold]]))
        # y1 =  np.hstack((y[xx],new_y.values[valid_fold[fold]]))
        # y1 = to_categorical(y1, num_classes=19)
        model = Net()
        model.compile(optimizer=optimizers.Adam(),
                 loss='categorical_crossentropy',#编译网络
                 metrics=['acc'])
        plateau = ReduceLROnPlateau(monitor="val_acc",
                                    verbose=0,
                                    mode='max',
                                    factor=0.10,
                                    patience=6)
        early_stopping = EarlyStopping(monitor='val_acc',
                                       verbose=0,
                                       mode='max',
                                       patience=25)
        checkpoint = ModelCheckpoint(f'fold{fold}.h5',
                                     monitor='val_acc',
                                     verbose=0,
                                     mode='max',
                                     save_best_only=True)
        model.fit(x1,y1,
                  epochs=1,
                  batch_size=32,
                  verbose=1,
                  shuffle=True,
                  validation_data=(x[yy], y_[yy]),
                  callbacks=[plateau, early_stopping, checkpoint])
        
        scores = model.evaluate(x[yy], y_[yy], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        model.load_weights(f'fold{fold}.h5')
        proba_t += model.predict(t,verbose=0, batch_size=1024) /cv/nums#最终的预测，5折交叉验证的平均
        train_pred[yy] += model.predict(x[yy],verbose=0, batch_size=1024) #最终的预测，5折交叉验证的平均
        
    
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))) 
    scores_all.append(np.mean(cvscores))
print("综合分数：%.2f%% (+/- %.2f%%)" % (np.mean(scores_all), np.std(scores_all)))
# 计算执行时间
end_time = time.time()
run_time = (end_time-begin_time)/60
print ('该循环程序运行时间：',run_time,'分钟') #该循环程序运行时间： 1.4201874732

## 保存文件成提交
sub = pd.read_csv('./dataset/result.csv')
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('submit.csv', index=False)

