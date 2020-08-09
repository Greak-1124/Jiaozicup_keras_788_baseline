from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.utils import to_categorical 
import random
from tensorflow import random
import os
import numpy as np
from tqdm import tqdm
################### 设随机数 ############
# 设置随机数

def set_seed(SEED =2031):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED']=str(SEED)
    np.random.seed(SEED)
    random.set_seed(SEED)

def one_hot(labels, num_classes):
    return to_categorical(labels, num_classes)
  
##################### 数据增强 ################################

def part_replacment(batch_x, batch_y):#局部随机替换
    import numpy as np
    batch_y = batch_y.values
    np.random.seed(2031)

    index = [i for i in range(len(batch_x))]
    np.random.shuffle(index)
    r_index1=index.copy()
    np.random.shuffle(index)
    r_index2 = index.copy()
    n = int(2/3*batch_x.shape[0])
    x_new = np.zeros((n,batch_x.shape[1],batch_x.shape[2],batch_x.shape[3]))
    y_new = np.zeros((n,))
    # np.random.sample(list_sample2, 1)
    for i in tqdm(range(n)):
        a =  np.random.randint(0,58)
        b =  np.random.randint(0,6)
        r_index11=np.random.choice(r_index1)
        r_index22=np.random.choice(r_index2)

    #     batch_x_mixup = (1-rd) * batch_x[r_index1] + rd * batch_x[r_index2f]
    #     batch_y_mixup = (1-rd) * batch_y[r_index1] + rd * batch_y[r_index2]
        x_new[i] = batch_x[r_index11]
        x_new[i,a:a+2,b:b+2,0] = batch_x[r_index22,a:a+2,b:b+2,0]
    #         [:,a:a+2,b:b+2,0] + batch_x[r_index2][:,a:a+2,b:b+2,]
        y_new[i] = batch_y[r_index22]
    x = np.vstack((batch_x,x_new))
    y = np.hstack((batch_y,y_new))
    return x,y



# 随机下采样# 随机下采样
def unba_randomus(x,y):
    x1 = x.reshape(x.shape[0],-1)# 7259*480
    ros = RandomUnderSampler(random_state=0) # 建立ros模型对象
    x1,y1 = ros.fit_resample(x1,y)# 扩增以后*480
    x2 = np.zeros((x1.shape[0],x.shape[1],x.shape[2],1))
    for i in tqdm(range(x1.shape[0])):
        x2[i,:,:,0] = np.reshape(x1[i],(60,8))
    return x2,y1
# 不平衡标签SMOTE过采样
def unba_smote(x,y):
    x1 = x.reshape(x.shape[0],-1)# 7259*480
    model_smote = SMOTE() # 建立SMOTE模型对象
    x1,y1 = model_smote.fit_sample(x1,y)# 扩增以后*480
    x2 = np.zeros((x1.shape[0],x.shape[1],x.shape[2],1))
    for i in tqdm(range(x1.shape[0])):
        x2[i,:,:,0] = np.reshape(x1[i],(60,8))
    return x2,y1
 # 不平衡标签随机过采样
def unba_randomos(x,y):
    x1 = x.reshape(x.shape[0],-1)# 7259*480
    ros = RandomOverSampler(random_state=0) # 建立ros模型对象
    x1,y1 = ros.fit_resample(x1,y)# 扩增以后*480
    x2 = np.zeros((x1.shape[0],x.shape[1],x.shape[2],1))
    for i in tqdm(range(x1.shape[0])):
        x2[i,:,:,0] = np.reshape(x1[i],(60,8))
    return x2,y1
# adasyn过采样
def unba_adasyn(x,y):
    x1 = x.reshape(x.shape[0],-1)# 7259*480
    adasyn = ADASYN(sampling_strategy='minority') # 建立ros模型对象
    x1,y1 = adasyn.fit_resample(x1,y)# 扩增以后*480
    x2 = np.zeros((x1.shape[0],x.shape[1],x.shape[2],1))
    for i in tqdm(range(x1.shape[0])):
        x2[i,:,:,0] = np.reshape(x1[i],(60,8))
    return x2,y1
# 组合采样
from imblearn.combine import SMOTEENN
def unba_smoteenn(x,y):
    x1 = x.reshape(x.shape[0],-1)# 7259*480
    smoteenn = SMOTEENN(random_state=0) # 建立smoteenn模型对象
    x1,y1 = smoteenn.fit_resample(x1,y)# 扩增以后*480
    x2 = np.zeros((x1.shape[0],x.shape[1],x.shape[2],1))
    for i in tqdm(range(x1.shape[0])):
        x2[i,:,:,0] = np.reshape(x1[i],(60,8))
    return x2,y1




    # 竖直翻转
def data_HorizontalFlip(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.HorizontalFlip(p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 水平翻转
def data_VerticalFlip(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.VerticalFlip(p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 水平竖直翻转
def data_Flip(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.Flip(p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 随机伽马变换
def data_RandomGamma(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.RandomGamma(gamma_limit=(8,60), eps=1e-07, always_apply=False, p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 随机图片旋转
def data_Rotate(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 图像均值平滑滤波。
def data_Blur(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.Blur(blur_limit = 7,always_apply = False,p = 0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
def data_CenterCrop(x,y):# 中心剪切
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.CenterCrop(height=1,width=1,always_apply = False,p = 0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
def data_RandomCrop(x,y):# 随机剪切
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.RandomCrop(height=1,width=1,always_apply = False,p = 0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 图像中生成正方形黑块   测试过
def data_Cutout(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.Cutout(num_holes=8, max_h_size=1, max_w_size=1, fill_value=0, always_apply=False, p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 网格失真
def data_GridDistortion(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 弹性变换
def data_ElasticTransform(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.ElasticTransform(alpha = 1,sigma = 50,alpha_affine = 50,interpolation = 1,border_mode = 4,value = None,mask_value = None,always_apply = False,approximate = False,p = 0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 随机网格洗牌
def data_RandomGridShuffle(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.RandomGridShuffle(grid=(3, 3), always_apply=False, p=1.0)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 反转图片，255-像素值
def data_InvertImg(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.InvertImg(always_apply=False, p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 高斯噪声
def data_GaussianBlur(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.GaussNoise(var_limit=(10.0, 50.0), mean=None, always_apply=False, p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 高斯平滑滤波
def data_GaussianBlur(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.GaussianBlur(blur_limit=7, always_apply=False, p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 中值滤波
def data_MedianBlur(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.MedianBlur(blur_limit=7, always_apply=False, p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y
# 运动模糊
def data_MotionBlur(x,y):
    x1 = np.zeros((x.shape[0],x.shape[1],x.shape[2],1))
    for i in range(x.shape[0]):
        transform = A.MotionBlur(blur_limit=7, always_apply=False, p=0.5)
        x1[i,:,:,0] = transform(image=x[i,:,:,0])['image']
    x = np.vstack((x,x1))
    y = np.hstack((y,y))
    return x,y