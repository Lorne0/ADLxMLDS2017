import numpy as np
import pickle as pk
import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential, load_model
from keras import backend as K
import tensorflow as tf
from edit_distance import edit_distance

def get_session(gpu_fraction=0.2):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

data_dir = "/tmp2/b02902030/ADLxMLDS/hw1/data/"
with open(data_dir+"t50_num_mix.pk", "rb") as fp:
    t50_num = pk.load(fp)

with open(data_dir+"t50_mix_valid_data.pk", "rb") as fp:
    X_valid = pk.load(fp)
    Y_valid = pk.load(fp)
with open(data_dir+"t50_mix_train_data.pk", "rb") as fp:
    X_train = pk.load(fp)
    Y_train = pk.load(fp)

timestep = 50
temperature = 5.0

model = "model/bilstm_sum_mix_50_400x6_drop/e_39.h5"
print(model)
model = load_model(data_dir+model, custom_objects={'timestep':timestep})
weights = model.layers[-1].get_weights()
model.pop()
model.add(Dense(48))
model.layers[-1].set_weights(weights)
model.add(Lambda(lambda x: x/temperature))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0004), metrics=['accuracy'])
print(model.summary())

train_len = X_train.shape[0]
valid_len = X_valid.shape[0]

X_train_soft = np.zeros((train_len, 50, 48), dtype=np.float32)
X_valid_soft = np.zeros((valid_len, 50, 48), dtype=np.float32)

batch_size = 200

batch_num = int(train_len/batch_size)+1
for b in range(batch_num):
    if b%200==0:
        print(b)
    X_train_soft[b*batch_size:(b+1)*batch_size] = model.predict_on_batch(X_train[b*batch_size:(b+1)*batch_size])

batch_num = int(valid_len/batch_size)+1
for b in range(batch_num):
    X_valid_soft[b*batch_size:(b+1)*batch_size] = model.predict_on_batch(X_valid[b*batch_size:(b+1)*batch_size])

with open(data_dir+"soft_t50_mix_train_data.pk", "wb") as fw:
    pk.dump(X_train_soft, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_train, fw, protocol=pk.HIGHEST_PROTOCOL)

with open(data_dir+"soft_t50_mix_valid_data.pk", "wb") as fw:
    pk.dump(X_valid_soft, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_valid, fw, protocol=pk.HIGHEST_PROTOCOL)




