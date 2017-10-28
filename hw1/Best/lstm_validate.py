import os, sys
import numpy as np
import pickle as pk
import keras
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from edit_distance import edit_distance

def get_session(gpu_fraction=0.08):
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

##################### Read Data ####################
data_dir = "/tmp2/b02902030/ADLxMLDS/hw1/data/"
with open(data_dir+"t50_num_mix.pk", "rb") as fp:
    t50_num = pk.load(fp)

with open(data_dir+"t50_mix_valid_data.pk", "rb") as fp:
    X_valid = pk.load(fp)
    Y_valid = pk.load(fp)
Y_valid = np.argmax(Y_valid, axis=2) # len, 50

batch_size = 60
timestep = 50
model = "/dhome/b02902030/ADLxMLDS/hw1/model/bilstm_sum_mix_50_400x6_drop/e_39.h5"
print(model)
model = load_model(model, custom_objects={'timestep':timestep})
valid_len = X_valid.shape[0]
batch_num = int(valid_len/batch_size)+1

print("===== Read Data done =====")

Y = np.zeros((Y_valid.shape))
Y_value = np.zeros((Y_valid.shape), dtype=np.float32)
for b in range(batch_num):
    b_x = X_valid[b*batch_size:b*batch_size+batch_size]
    b_y = Y_valid[b*batch_size:b*batch_size+batch_size]

    y = model.predict_on_batch(b_x) # (batch_size, 50, 48)
    yy = np.argmax(y, axis=2) # batch_size, 50
    Y[b*batch_size:b*batch_size+batch_size] = yy
    yy = np.max(y, axis=2) # batch_size, 50
    Y_value[b*batch_size:b*batch_size+batch_size] = yy

print("===== Predict done =====")
print(Y.shape)
print(Y_valid.shape)
print(Y_value.shape)

def get_overlap_chunks(array, chunk_size, overlap):
    result = []
    a = 0
    while a<len(array):
        r = array[a:a+chunk_size]
        if len(r)<chunk_size:
            r = array[-chunk_size:]
            result.append(r)
            return np.array(result)
        result.append(r)
        a += chunk_size-overlap

map_num2phone, map_48to39, map_phone2alpha = make_map(data_dir)
with open("phone_mean_num.pk", "rb") as fp:
    phone_mean_num = pk.load(fp)
threshold = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.51, 0.53, 0.55, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.73, 0.75, 0.77, 0.8, 0.81, 0.82, 0.83, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
dis = np.zeros((len(threshold)))
acc = 0.0
cc = 0
count = 0
for i in range(len(t50_num)):
    idx = get_overlap_chunks(np.arange(t50_num[i]), 50, 10)
    y_true = np.zeros((t50_num[i]), dtype=np.int32)
    y_pred = np.zeros((t50_num[i]), dtype=np.int32)
    y_v = np.zeros((t50_num[i]), dtype=np.float32)
    for ii in idx:
        y_true[ii] = Y_valid[count]
        y_pred[ii] = Y[count]
        y_v[ii] = Y_value[count]
        count+=1
    acc += sum(y_true==y_pred)
    cc += len(y_true)

    s0=""
    for k in range(len(y_true)):
        s0 += map_phone2alpha[map_48to39[map_num2phone[y_true[k]]]]
    s0 = string_compress(s0)

    for it, t in enumerate(threshold):
        s1=""
        for k in range(len(y_pred)):
            if y_v[k]>t:
                s1 += map_phone2alpha[map_48to39[map_num2phone[y_pred[k]]]]
        s1 = window_slide(s1, phone_mean_num)
        s1 = string_trim(s1)
        s1 = string_trim2(s1)
        s1 = string_compress(s1)
        dis[it] += edit_distance(s0, s1)

print("Acc:", acc/cc)
dis = dis/len(t50_num)
for i in range(len(threshold)):
    print(threshold[i], dis[i])

