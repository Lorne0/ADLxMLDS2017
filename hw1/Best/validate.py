import os, sys
import numpy as np
import pickle as pk
import keras
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from edit_distance import edit_distance

def get_session(gpu_fraction=0.1):
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

##################### Read Data ####################
data_dir = sys.argv[1]
if data_dir[-1] != "/":
    data_dir += "/"
with open(data_dir+"valid_data.pk", "rb") as fp:
    X_valid = pk.load(fp) # 370
    Y_valid = pk.load(fp) # 370
instance_list = X_valid.keys()
X_len = {}
for k in instance_list:
    X_len[k] = int(X_valid[k].shape[0])

with open(data_dir+"t70_5D_valid_data.pk", "rb") as fp:
    X_valid = pk.load(fp)
    Y_valid = pk.load(fp)
X_valid = np.append(X_valid, X_valid[:2], axis=0)
Y_valid = np.append(Y_valid, Y_valid[:2], axis=0)
Y_valid = np.argmax(Y_valid, axis=2) # len, 70

batch_size = 16
timestep = 70
model = "model/best_drop_4/e_16.h5"
print(model)
model = load_model(data_dir+model, custom_objects={'timestep':timestep})
valid_len = X_valid.shape[0]
batch_num = int(valid_len/batch_size)

print("===== Read Data done =====")


#K.set_learning_phase(0)



Y = np.zeros((Y_valid.shape))
Y_value = np.zeros((Y_valid.shape), dtype=np.float32)
for b in range(batch_num):
    b_x = X_valid[b*batch_size:b*batch_size+batch_size]
    b_y = Y_valid[b*batch_size:b*batch_size+batch_size]

    y = model.predict_on_batch(b_x) # (batch_size, 70, 48)
    yy = np.argmax(y, axis=2) # batch_size, 70
    Y[b*batch_size:b*batch_size+batch_size] = yy
    yy = np.max(y, axis=2) # batch_size, 70
    Y_value[b*batch_size:b*batch_size+batch_size] = yy
    

Y = Y[:-2]
Y_valid = Y_valid[:-2]
Y_value = Y_value[:-2]

print("===== Predict done =====")
print(Y.shape)
print(Y_valid.shape)
print(Y_value.shape)
print(Y_value[0])

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

Y_dic = {}
Y_valid_dic = {}
Y_value_dic = {}
count = 0
for k in instance_list:
    y = np.zeros(X_len[k]-10, dtype=np.int32)
    y_ = np.zeros(X_len[k]-10, dtype=np.int32)
    yy = np.zeros(X_len[k]-10, dtype=np.float32)

    idx = get_overlap_chunks(np.arange(X_len[k]-10), 70, 10)
    for ii in idx:
        y[ii] = Y[count]
        y_[ii] = Y_valid[count]
        yy[ii] = Y_value[count]
        count+=1
    Y_dic[k] = y
    Y_valid_dic[k] = y_
    Y_value_dic[k] = yy
print("count:", count)

acc = 0.0
cc = 0
for k in instance_list:
    acc += sum(Y_dic[k]==Y_valid_dic[k])
    cc += len(Y_dic[k])
print("Acc: ", acc/cc)


map_num2phone, map_48to39, map_phone2alpha = make_map(data_dir)
dis = 0
for k in instance_list:
    s0 = ""
    for n in Y_valid_dic[k]:
        s0 += map_phone2alpha[map_48to39[map_num2phone[n]]]
    s0 = string_compress(s0)

    s1 = ""
    for n in Y_dic[k]:
        s1 += map_phone2alpha[map_48to39[map_num2phone[n]]]
    s1 = string_trim(s1)
    s1 = string_trim2(s1)
    s1 = string_compress(s1)

    dis += edit_distance(s0, s1)

print(dis/len(instance_list))

print("No trim:")
for t in [0.55, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.73, 0.75, 0.77, 0.8]:
    dis = 0
    for k in instance_list:
        s0 = ""
        for n in Y_valid_dic[k]:
            s0 += map_phone2alpha[map_48to39[map_num2phone[n]]]
        s0 = string_compress(s0)
        s1 = ""
        for i in range(len(Y_dic[k])):
            if Y_value_dic[k][i]>t:
                s1 += map_phone2alpha[map_48to39[map_num2phone[Y_dic[k][i]]]]
        #s1 = string_trim(s1)
        #s1 = string_trim2(s1)
        s1 = string_compress(s1)
        dis += edit_distance(s0, s1)
    print(t, dis/len(instance_list))

print("Trim1:")
for t in [0.55, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.73, 0.75, 0.77, 0.8]:
    dis = 0
    for k in instance_list:
        s0 = ""
        for n in Y_valid_dic[k]:
            s0 += map_phone2alpha[map_48to39[map_num2phone[n]]]
        s0 = string_compress(s0)
        s1 = ""
        for i in range(len(Y_dic[k])):
            if Y_value_dic[k][i]>t:
                s1 += map_phone2alpha[map_48to39[map_num2phone[Y_dic[k][i]]]]
        s1 = string_trim(s1)
        #s1 = string_trim2(s1)
        s1 = string_compress(s1)
        dis += edit_distance(s0, s1)
    print(t, dis/len(instance_list))

print("Trim1+2:")
for t in [0.55, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.73, 0.75, 0.77, 0.8]:
    dis = 0
    for k in instance_list:
        s0 = ""
        for n in Y_valid_dic[k]:
            s0 += map_phone2alpha[map_48to39[map_num2phone[n]]]
        s0 = string_compress(s0)
        s1 = ""
        for i in range(len(Y_dic[k])):
            if Y_value_dic[k][i]>t:
                s1 += map_phone2alpha[map_48to39[map_num2phone[Y_dic[k][i]]]]
        s1 = string_trim(s1)
        s1 = string_trim2(s1)
        s1 = string_compress(s1)
        dis += edit_distance(s0, s1)
    print(t, dis/len(instance_list))





