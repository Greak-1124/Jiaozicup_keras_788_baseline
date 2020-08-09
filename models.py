from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
# 线上78.8分模型 

def Net():
    input = Input(shape=(60, 8, 1))
    X = Conv2D(filters=64,
               kernel_size=(3, 3),
#                activation='relu',
               padding='same')(input)
  
    X = Activation('relu')(X)
    X = Conv2D(filters=128,
               kernel_size=(3, 3),
#                activation='relu',
               padding='same')(X)
    
    X = Activation('relu')(X)
    X = AveragePooling2D()(X)
#     X = Dropout(0.4)(X)
    X = Conv2D(filters=256,
               kernel_size=(3, 3),
#                activation='relu',
               padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=512,
               kernel_size=(3, 3),
#                activation='relu',
               padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = GlobalAveragePooling2D()(X)
    #     X = Dropout(0.4)(X)
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
#     X = GaussianNoise(0.01)(X)
#     X = Dropout(0.3)(X)
# kernel_regularizer=regularizers.l2(0.002)
    X = Dense(19,activation='softmax')(X)
    return Model([input], X)

