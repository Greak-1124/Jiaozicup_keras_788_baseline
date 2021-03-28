## 0 比赛背景介绍：
https://www.kesci.com/home/competition/5ece30cc73a1b3002c9f1bf5/content

## 1 Baseline方案：Jiaozicup_keras_788_baseline
2020年“创青春·交子杯”新网银行金融科技挑战赛-AI算法赛道keras版本baseline，线上成绩788。

### 1.1 线上分数
线上分数0.788。

### 1.2 运行
python train.py

## 2 最终方案
最终方案采用的是一个2D CNN主模型（0.788）；一个1D CNN模型（0.765），一个Lgb模型（0.72），一个CNN-LSTM模型（0.760）。由于模型的差异性不够大，因此，对于每个模型使用的数据增强策略均有所不同，如：对于2D CNN网络，主要采用的是SMOTE过采样+Cutout进行数据增强；对于1D CNN网络，则是高斯噪声+伪标签以及CutMix方法进行数据增强。体框架图如下：    
![image](https://github.com/Greak-1124/Jiaozicup_keras_788_baseline/blob/master/Last_code/pic/Overall.png)   
解释：最终四个模型的预测的概率作为新的训练样本，投喂给多个学习器学习（包括LR、KNN、Lgb等），其思想类似于Stacking+Voting结合的思想，先进行Stacking，再将各模型Stacking的结果进行Voting。我对Stacking的理解就是一个对预测标签的结果做修正的过程，但为了避免过修正导致过拟合，我们将各学习器的结果再经过一个Voting得到最终的提交结果。


### 2.1 文件说明
data：文件夹为数据集。  
model：为训练时产生的模型放到这个文件夹里面。  
npy_file：是每个模型训练完毕产生的npy文件会放里面。  
pseudo_labels：存放伪标签的文件夹。  
最终生成submit文件在code/目录下。

### 2.2 线上分数
经过stacking以后可以到0.797，初赛排名排名42，最终经过方案筛选排名Top20。

### 2.3 运行
#### 一、进入最终方案文件目录
cd Last_code

#### 二、执行CNN.py文件，生成伪标签文件，oneD_CNN.py文件需要用到
python CNN.py

#### 三、执行oneD_CNN.py文件
python oneD_CNN.py

#### 四、执行CNN_LSTM.py文件
python CNN_LSTM.py

#### 五、执行lgb.py文件
python lgb.py

#### 六、执行Stacking.py文件
python Stacking.py



