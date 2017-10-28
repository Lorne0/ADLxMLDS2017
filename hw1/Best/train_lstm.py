import os, sys
import numpy as np
import pickle as pk
import keras
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import tensorflow as tf

def get_session(gpu_fraction=0.4):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

#####
batch_size = 40
timestep = 50 
epoch_num = 70
temperature = 1.0
#####
model = Sequential()
model.add(Bidirectional(LSTM(400, return_sequences=True), merge_mode='sum', input_shape=(timestep, 108)))
model.add(Bidirectional(LSTM(400, return_sequences=True), merge_mode='sum'))
model.add(Bidirectional(LSTM(400, return_sequences=True, dropout=0.2), merge_mode='sum'))
model.add(Bidirectional(LSTM(400, return_sequences=True), merge_mode='sum'))
model.add(Bidirectional(LSTM(400, return_sequences=True), merge_mode='sum'))
model.add(Bidirectional(LSTM(400, return_sequences=True, dropout=0.2), merge_mode='sum'))
#model.add(Bidirectional(LSTM(400, return_sequences=True, dropout=0.4), merge_mode='sum'))


model.add(Dense(48))
model.add(Lambda(lambda x: x/temperature))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0004), metrics=['accuracy'])
print(model.summary())

if sys.argv[1] != 'start':
    # e_ #
    print("e_"+sys.argv[1])
    temperature = 3.0
    model = load_model("/dhome/b02902030/ADLxMLDS/hw1/model/bilstm_sum_mix_400x6_cd/e_"+sys.argv[1]+".h5", custom_objects={"timestep":timestep, "temperature":temperature})



# Read Data
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t50_mix_train_data.pk", "rb") as fp:
    X_train = pk.load(fp)
    Y_train = pk.load(fp)
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t50_mix_valid_data.pk", "rb") as fp:
    X_valid = pk.load(fp)
    Y_valid = pk.load(fp)

print("===== Read data done =====")

train_len = X_train.shape[0]
train_batch_num = int(train_len/batch_size)+1
print("train_batch_num:", train_batch_num)
valid_len = X_valid.shape[0]
valid_batch_num = int(valid_len/batch_size)+1
print("valid_batch_num:", valid_batch_num)
order = np.arange(train_len)

e = int(sys.argv[1])+1 if sys.argv[1]!='start' else 1
for epoch in range(e, epoch_num+1):

    # Cooling down
    temperature = max(1.0, temperature-0.2)
    model.pop() # pop activation
    model.pop() # pop lambda temperature
    model.add(Lambda(lambda x: x/temperature))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0004), metrics=['accuracy'])

    print("Epoch:", epoch)
    np.random.shuffle(order)
    X = X_train[order]
    Y = Y_train[order]
    train_loss = 0.0
    train_acc = 0.0
    for b in range(train_batch_num):
        if b%200==0:
            print(b)
        b_x = X[b*batch_size:b*batch_size+batch_size] # (b, 50, 69)
        b_y = Y[b*batch_size:b*batch_size+batch_size] # (b, 50, 48)
        loss, acc = model.train_on_batch(b_x, b_y)
        train_loss += loss*b_x.shape[0]
        train_acc += acc*b_x.shape[0]
    train_loss /= train_len
    train_acc /= train_len
    
    valid_loss = 0.0
    valid_acc = 0.0
    #count = 0
    for b in range(valid_batch_num): 
        b_x = X_valid[b*batch_size:b*batch_size+batch_size] # (b, 50, 69)
        b_y = Y_valid[b*batch_size:b*batch_size+batch_size] # (b, 50, 48)

        loss, acc = model.test_on_batch(b_x, b_y)
        valid_loss += loss*b_x.shape[0]
        valid_acc += acc*b_x.shape[0]
    valid_loss /= valid_len
    valid_acc /= valid_len
        
    #y = model.predict_on_batch(b_x) # (b, 50, 48)
    #print(y.shape)
    #y = np.reshape(y, (-1, 48))
    #b_y = np.reshape(b_y, (-1, 48))
    #y = np.argmax(y, axis=1)
    #b_y = np.argmax(b_y, axis=1)
    #acc += sum(y==b_y)
    #count += len(y)
    #print("Accuracy:", str(acc/count))

    print("%.4f, %.4f, %.4f, %.4f" %(train_acc, train_loss, valid_acc, valid_loss))
    model.save("/dhome/b02902030/ADLxMLDS/hw1/model/bilstm_sum_mix_400x6_cd/e_"+str(epoch)+".h5")















