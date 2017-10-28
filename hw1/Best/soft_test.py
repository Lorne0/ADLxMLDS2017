import numpy as np
import pickle as pk
import os, sys
import keras
from keras.models import load_model, Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf

def get_session(gpu_fraction=0.25):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

def make_map(data_dir):
    if data_dir[-1]!="/":
        data_dir += "/"
    map_num2phone = {} #num(48) -> phone(48)
    map_48to39 = {} # 48 -> 39
    map_phone2alpha = {} # phone 48(39) -> alphabet 48(39)
    with open(data_dir+"48phone_char.map") as fp:
        for f in fp:
            s = f.strip('\n').split('\t')
            map_num2phone[int(s[1])] = s[0]
    with open(data_dir+"phones/48_39.map") as fp:
        for f in fp:
            s = f.strip('\n').split('\t')
            map_48to39[s[0]] = s[1]
    with open(data_dir+"48phone_char.map") as fp:
        for f in fp:
            s = f.strip('\n').split('\t')
            map_phone2alpha[s[0]] = s[2]
    return map_num2phone, map_48to39, map_phone2alpha

# Make Answer
def string_compress(s0):
    s0 = s0.strip("L")    
    prev, s = "", ""
    for c in s0:
        if c!=prev:
            prev = c
            s += c
    return s

def string_trim(s0):
    ss = list(s0)
    '''
    # Trim AABBAA -> AAAAAA
    for i in range(2, len(ss)-3):
        if ss[i]==ss[i+1] and ss[i]!=ss[i-1] and ss[i-1]==ss[i-2] and ss[i+2]==ss[i+3] and ss[i-1]==ss[i+2]:
            ss[i] = ss[i-1]
            ss[i+1] = ss[i-1]
    '''
    # Trim AABAA -> AAAAA
    for i in range(2, len(ss)-2):
        if ss[i-2]==ss[i-1] and ss[i-2]==ss[i+1] and ss[i-2]==ss[i+2]:
            ss[i] = ss[i-2]
    # Trim AABAC -> AAAAC
    for i in range(2, len(ss)-2):
        if ss[i-2]==ss[i-1] and ss[i-2]!=ss[i] and ss[i-2]==ss[i+1] and ss[i+2]!=ss[i]:
            ss[i] = ss[i-2]
    # Trim CABAA -> CAAAA
    for i in range(2, len(ss)-2):
        if ss[i+1]==ss[i+2] and ss[i+2]==ss[i-1] and ss[i-2]!=ss[i]  and ss[i]!=ss[i+2]:
            ss[i] = ss[i+2]
    # Trim AABCC -> AAACC or AACCC
    for i in range(2, len(ss)-2):
        if ss[i-1]==ss[i-2] and ss[i+1]==ss[i+2]:
            ss[i] = ss[i-1]

    s0 = ''.join(ss)
    return s0
    
def string_trim2(s0):
    ss = list(s0)
    for i in range(1, len(ss)-1):
        if ss[i-1]==ss[i+1] and ss[i-1]!=ss[i]:
            ss[i] = ss[i-1]
    s0 = ''.join(ss)
    return s0

def window_slide(s0, num):
    s = ""
    for i in range(len(s0)):
        w = num[s0[i]]
        if w%2==1:
            w1, w2 = int(w/2), int(w/2)
        else:
            w1, w2 = int(w/2), int(w/2)-1
        h = max(0, i-w1)
        t = min(len(s0), i+w2+1)
        ss = s0[h:t]
        if ss.count(s0[i]) == ss.count(max(ss, key=ss.count)):
            s += s0[i]
    return s

X = {} # key->2D numpy array of feature
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/fbank/test.ark") as fp:
    for f in fp:
        s = f.strip('\n').split(' ')
        ins = s[0].split('_')
        ins_id = ins[0]+'_'+ins[1]
        x = np.array(s[1:], dtype=np.float32)
        if ins_id not in X:
            X[ins_id] = x
        else: # in X,Y
            X[ins_id] = np.vstack((X[ins_id], x))
X_m = {} # key->2D numpy array of feature
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/mfcc/test.ark") as fp:
    for f in fp:
        s = f.strip('\n').split(' ')
        ins = s[0].split('_')
        ins_id = ins[0]+'_'+ins[1]
        x = np.array(s[1:], dtype=np.float32)
        if ins_id not in X_m:
            X_m[ins_id] = x
        else: # in X,Y
            X_m[ins_id] = np.vstack((X_m[ins_id], x))
ins_list = list(X.keys())
for ins in ins_list:
    X[ins] = np.hstack((X[ins], X_m[ins]))
t50_num = []
for ins in ins_list:
    t50_num.append(int(X[ins].shape[0]))
t50_num = np.array(t50_num)
#------------------------------------------------------#
def get_overlap_chunks(array, chunk_size, overlap):
    result, a = [], 0
    while a<len(array):
        r = array[a:a+chunk_size]
        if len(r)<chunk_size:
            r = array[-chunk_size:]
            result.append(r)
            return np.array(result)
        result.append(r)
        a += chunk_size-overlap
#------------------------------------------------------#
X_test, count = np.zeros((7000, 50, 108), dtype=np.float32), 0 
for ins in ins_list:
    sl = X[ins].shape[0]
    idx = get_overlap_chunks(np.arange(sl), 50, 10)
    for ii in idx:
        X_test[count] = X[ins][ii]
        count += 1
X_test = X_test[:count]
print(X_test.shape)

data_dir = "/tmp2/b02902030/ADLxMLDS/hw1/data/"
timestep = 50
temperature = 5.0
model = "/dhome/b02902030/ADLxMLDS/hw1/model/bilstm_sum_mix_50_400x6_drop/e_39.h5"
#print(model)
model = load_model(model, custom_objects={'timestep':timestep})
weights = model.layers[-1].get_weights()
model.pop()
model.add(Dense(48))
model.layers[-1].set_weights(weights)
model.add(Lambda(lambda x: x/temperature))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0004), metrics=['accuracy'])
#print(model.summary())

test_len = X_test.shape[0]
X_test_soft = np.zeros((test_len, 50, 48), dtype=np.float32)
batch_size = 200
batch_num = int(test_len/batch_size)+1
for b in range(batch_num):
    X_test_soft[b*batch_size:(b+1)*batch_size] = model.predict_on_batch(X_test[b*batch_size:(b+1)*batch_size])

print("Soft done.")


##################### Read Data ####################
data_dir = "/tmp2/b02902030/ADLxMLDS/hw1/data/"

batch_size = 60
timestep = 50
temperature = 3.0
model = "/dhome/b02902030/ADLxMLDS/hw1/model/soft_bilstm_sum_mix_50_400x5_drop_cd/e_8.h5"
print(model)
model = load_model(model, custom_objects={'timestep':timestep, 'temperature':temperature})
test_len = X_test_soft.shape[0]
batch_num = int(test_len/batch_size)+1

print("===== Read Data done =====")

Y = np.zeros((X_test_soft.shape[0], timestep), dtype=np.int32)
Y_value = np.zeros((X_test_soft.shape[0], timestep), dtype=np.float32)
for b in range(batch_num):
    b_x = X_test_soft[b*batch_size:b*batch_size+batch_size]

    y = model.predict_on_batch(b_x) # (batch_size, 50, 48)
    yy = np.argmax(y, axis=2) # batch_size, 50
    Y[b*batch_size:b*batch_size+batch_size] = yy
    yy = np.max(y, axis=2) # batch_size, 50
    Y_value[b*batch_size:b*batch_size+batch_size] = yy

print("===== Predict done =====")
print(Y.shape)
print(Y_value.shape)

map_num2phone, map_48to39, map_phone2alpha = make_map(data_dir)
with open("phone_mean_num.pk", "rb") as fp:
    phone_mean_num = pk.load(fp)
#threshold = [0.55, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.73, 0.75, 0.77, 0.8]
threshold = 0.0
count = 0
with open("output_best_lstm_soft.csv", "w") as fw:
    fw.write("id,phone_sequence\n")
    for i in range(len(t50_num)):
        idx = get_overlap_chunks(np.arange(t50_num[i]), 50, 10)
        y_pred = np.zeros((t50_num[i]), dtype=np.int32)
        y_v = np.zeros((t50_num[i]), dtype=np.float32)
        for ii in idx:
            y_pred[ii] = Y[count]
            y_v[ii] = Y_value[count]
            count+=1

        s1=""
        for k in range(len(y_pred)):
            if y_v[k]>threshold:
                s1 += map_phone2alpha[map_48to39[map_num2phone[y_pred[k]]]]
        s1 = window_slide(s1, phone_mean_num)
        s1 = string_trim(s1)
        s1 = string_trim2(s1)
        s1 = string_compress(s1)
        fw.write(ins_list[i]+','+str(s1)+'\n')


