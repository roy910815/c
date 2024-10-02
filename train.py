import tensorflow as tf
import pandas as pd
from tensorflow import keras
import tensorflow.keras.utils as np_utils
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential() #建立MODEL
#CN layer convolution
model.add(Conv2D(filters=16,
        kernel_size=(5,5),
        padding='same',
        input_shape=(28,28,1), #轉換影像單位
        activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_1')) # Max-pooling
#CN layer convolution
model.add(Conv2D(filters=16,
        kernel_size=(5,5),
        padding='same',
        activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_2')) # Max-pooling
#CN layer convolution
model.add(Conv2D(filters=16,
        kernel_size=(5,5),
        padding='same',
        activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_3')) # Max-pooling

model.add(Dropout(0.25)) # dropout 減少overfit 不會過度依賴特定特徵
model.add(Flatten()) #fully connect network
#Hidden layer 目前只有一層
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.summary()
print("")
#訓練模型 次數多 增加模型完整性
predictions = model(x_train[:1]).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
#test 正確率預估
model.evaluate(x_test,  y_test, verbose=2)
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])


