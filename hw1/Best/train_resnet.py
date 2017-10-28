import numpy as np
import pickle as pk
import keras
from keras.models import Model
from keras.models import Sequential
from keras import layers
from keras.layers import *
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import tensorflow as tf

def get_session(gpu_fraction=0.2):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

def res_unit(input_tensor, filters, base_name, cnn_shortcut):
    # Not use Bottleneck
    f1, f2 = filters

    conv1 = Conv2D(f1, (3,3), padding='same', name=base_name+"_conv1")(input_tensor)
    bn1 = BatchNormalization(axis=3, name=base_name+"_bn1")(conv1)
    r1 = Activation('relu')(bn1)

    if cnn_shortcut:
        conv2 = Conv2D(f2, (3,3), name=base_name+"_conv2")(r1)
        bn2 = BatchNormalization(axis=3, name=base_name+"_bn2")(conv2)

        shortcut = Conv2D(f2, (3,3), name=base_name+"_CNNshortcut")(input_tensor)
        out = layers.add([bn2, shortcut])
    else:
        conv2 = Conv2D(f2, (3,3), padding='same', name=base_name+"_conv2")(r1)
        bn2 = BatchNormalization(axis=3, name=base_name+"_bn2")(conv2)

        out = layers.add([bn2, input_tensor])

    r2 = Activation('relu')(out)
    return r2

#===========================================#

with open("/tmp2/b02902030/ADLxMLDS/hw1/data/pic_train_data.pk", "rb") as fp:
    X_train = pk.load(fp)
    Y_train = pk.load(fp)
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/pic_valid_data.pk", "rb") as fp:
    X_valid = pk.load(fp)
    Y_valid = pk.load(fp)

print("Read Data Done.")

#H, W, channel = 41, 69, 1
H, W, channel = X_train[0].shape
img_input = Input(shape=(H, W, channel))
x = Conv2D(32, (5,5), name='conv1')(img_input)
x = BatchNormalization(axis=3, name='bn_conv1')(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2))(x)

x = res_unit(x, (16, 32), "block1_1", True)
x = res_unit(x, (16, 32), "block1_2", False)
x = res_unit(x, (16, 32), "block1_3", False)

x = res_unit(x, (32, 64), "block2_1", True)
x = res_unit(x, (32, 64), "block2_2", False)
x = res_unit(x, (32, 64), "block2_3", False)
x = res_unit(x, (32, 64), "block2_4", False)

x = AveragePooling2D((5,5), name='avg_pool')(x)
x = Flatten()(x)
x = Dense(256, activation='relu', name='fc256')(x)
out = Dense(48, activation='softmax', name='fc48')(x)

Resnet = Model(img_input, out)
print(Resnet.summary())
Resnet.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

print("Make Model Done.")

#===========================================#

batch_size = 100
epoch_num = 50

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
        if b%4000==0:
            print(b)
        b_x = X[b*batch_size:b*batch_size+batch_size] # (batch_size, 41, 69, 1)
        b_y = Y[b*batch_size:b*batch_size+batch_size] # (batch_size, 48)
        Resnet.train_on_batch(b_x, b_y)
    
    acc = 0.0
    count = 0
    for b in range(valid_batch_num): 
        b_x = X_valid[b*batch_size:b*batch_size+batch_size] 
        b_y = Y_valid[b*batch_size:b*batch_size+batch_size] 

        y = Resnet.predict_on_batch(b_x) # (batch_size, 48)
        y = np.argmax(y, axis=1)
        b_y = np.argmax(b_y, axis=1)
        acc += sum(y==b_y)
        count += len(y)
    print("Accuracy:", str(acc/count))
    Resnet.save("/tmp2/b02902030/ADLxMLDS/hw1/data/model/resnet/e_"+str(epoch)+".h5")















