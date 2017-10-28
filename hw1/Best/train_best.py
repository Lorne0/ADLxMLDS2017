import numpy as np
import pickle as pk
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import tensorflow as tf

def get_session(gpu_fraction=0.4):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

#####
batch_size = 20
timestep = 60 
epoch_num = 50
#####

def backend_reshape1(x):
    return K.reshape(x, (1, -1, 17, 69, 1))
def backend_reshape2(x):
    return K.reshape(x, (-1, timestep, 512))


model = Sequential()
### CNN Model ###
# reshape
model.add(Lambda(backend_reshape1, input_shape=(timestep, 17, 69, 1)))
# cnn1
model.add(TimeDistributed(Conv2D(32, (3,7), activation='relu')))
model.add(TimeDistributed(Conv2D(32, (3,7), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
# cnn2
model.add(TimeDistributed(Conv2D(64, (3,7), activation='relu')))
model.add(TimeDistributed(Conv2D(64, (3,7), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
# flat
model.add(TimeDistributed(Flatten()))
### LSTM Model ###
model.add(Lambda(backend_reshape2))
model.add(Bidirectional(LSTM(512, return_sequences=True), merge_mode='concat'))
model.add(Bidirectional(LSTM(512, return_sequences=True), merge_mode='concat'))
model.add(Bidirectional(LSTM(512, return_sequences=True), merge_mode='concat'))
model.add(Bidirectional(LSTM(512, return_sequences=True), merge_mode='concat'))
model.add(Bidirectional(LSTM(512, return_sequences=True), merge_mode='concat'))

model.add(TimeDistributed(Dense(128, activation='relu')))
model.add(TimeDistributed(Dense(48, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])
print(model.summary())


# Read Data
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t50_5D_train_data.pk", "rb") as fp:
    X_train = pk.load(fp)
    Y_train = pk.load(fp)
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t50_5D_valid_data.pk", "rb") as fp:
    X_valid = pk.load(fp)
    Y_valid = pk.load(fp)
## to fit batch size 20 3794->3800
X_valid = np.append(X_valid, X_valid[:6], axis=0)
Y_valid = np.append(Y_valid, Y_valid[:6], axis=0)

print("===== Read data done =====")

train_len = X_train.shape[0]
train_batch_num = int(train_len/batch_size)
print("train_batch_num:", train_batch_num)
valid_len = X_valid.shape[0]
valid_batch_num = int(valid_len/batch_size)
print("valid_batch_num:", valid_batch_num)
order = np.arange(train_len)

for epoch in range(1, epoch_num+1):
    print("Epoch:", epoch)
    np.random.shuffle(order)
    X = X_train[order]
    Y = Y_train[order]
    for b in range(train_batch_num):
        if b%200==0:
            print(b)
        b_x = X[b*batch_size:b*batch_size+batch_size] # (20, 60, 17, 69, 1)
        b_y = Y[b*batch_size:b*batch_size+batch_size] # (20, 60, 48)
        model.train_on_batch(b_x, b_y)
    
    acc = 0.0
    count = 0
    for b in range(valid_batch_num): 
        b_x = X_valid[b*batch_size:b*batch_size+batch_size] # (20, 60, 17, 69, 1)
        b_y = Y_valid[b*batch_size:b*batch_size+batch_size] # (20, 60, 48)

        y = model.predict_on_batch(b_x) # (20, 60, 48)
        #print(y.shape)
        y = np.reshape(y, (batch_size*timestep, 48))
        b_y = np.reshape(b_y, (batch_size*timestep, 48))
        y = np.argmax(y, axis=1)
        b_y = np.argmax(b_y, axis=1)
        acc += sum(y==b_y)
        count += len(y)
    print("Accuracy:", str(acc/count))
    model.save("/tmp2/b02902030/ADLxMLDS/hw1/data/model/best_large_5/e_"+str(epoch)+".h5")















