import os, sys
import numpy as np
import pickle as pk
import keras
from keras.models import load_model
from keras import backend as K
import tensorflow as tf

'''
def get_session(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())
'''

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

#print(sys.argv[1])
#print(sys.argv[2])
#print(sys.argv[3])

# Make Test data
X_test = {}
with open(data_dir+"mfcc/test.ark") as fp:
    for f in fp:
        s = f.strip('\n').split(' ')
        instance = s[0]
        instance_id = s[0].split('_')
        instance_id = instance_id[0]+'_'+instance_id[1]
        k = np.array(s[1:], dtype=np.float32)
        if instance_id not in X_test:
            X_test[instance_id] = k
        else:
            X_test[instance_id] = np.vstack((X_test[instance_id], k))

instance_list = X_test.keys()
X_len = {}
for k in instance_list:
    X_len[k] = int(X_test[k].shape[0])


# Make 2D test data

X_5D = np.zeros((4000, 70, 11, 39, 1), dtype=np.float32)
count = 0
for k in X_test.keys():
    X = X_test[k]
    X_collect = np.zeros((1, X.shape[0]-10, 11, 39, 1), dtype=np.float32)
    for i in range(5, X.shape[0]-5):
        x = X[i-5:i+5+1] # -> (11,39)
        x = np.expand_dims(x, axis=2) # (11,39,1)
        #x = np.expand_dims(x, axis=0) # (1,11,39,1)
        X_collect[0][i-5] = x
    idx = get_overlap_chunks(np.arange(X.shape[0]-10), 70, 10)
    for ii in idx:
        X_5D[count] = X_collect[0][ii]
        count+=1
X_5D = X_5D[:count]
X_test = X_5D
#print(X_test.shape)


batch_size = 16
timestep = 70
#model = "/tmp2/b02902030/ADLxMLDS/hw1/data/model/CNNbiLSTM/e_11.h5"
model = "e_12.h5"
#print(model)
model = load_model(model, custom_objects={'timestep':timestep})
test_len = X_test.shape[0]
batch_num = int(test_len/batch_size)

#print("===== Read Data done =====")

Y = np.zeros((test_len, timestep))
Y_value = np.zeros((test_len, timestep))
for b in range(batch_num):
    b_x = X_test[b*batch_size:b*batch_size+batch_size]

    y = model.predict_on_batch(b_x) # (batch_size, 70, 48)
    Y[b*batch_size:b*batch_size+batch_size] = np.argmax(y, axis=2) # batch_size, 70
    Y_value[b*batch_size:b*batch_size+batch_size] = np.max(y, axis=2) # batch_size, 70

#print("===== Predict done =====")
#print(Y.shape)
#print(Y_value.shape)

Y_dic = {}
Y_value_dic = {}
count = 0
for k in instance_list:
    y = np.zeros(X_len[k]-10, dtype=np.int32)
    yy = np.zeros(X_len[k]-10, dtype=np.float32)
    idx = get_overlap_chunks(np.arange(X_len[k]-10), 70, 10)
    for ii in idx:
        y[ii] = Y[count]
        yy[ii] = Y_value[count]
        count+=1
    Y_dic[k] = y
    Y_value_dic[k] = yy
#print("count:", count)

map_num2phone, map_48to39, map_phone2alpha = make_map(data_dir)
dis = 0

with open(sys.argv[2], "w") as fw:
    fw.write("id,phone_sequence\n")
    for k in instance_list:
        s1 = ""
        for i in range(len(Y_dic[k])):
            if Y_value_dic[k][i]>0.6:
                s1 += map_phone2alpha[map_48to39[map_num2phone[Y_dic[k][i]]]]
        s1 = string_trim(s1)
        s1 = string_trim2(s1)
        s1 = string_compress(s1)
        fw.write(k+','+s1+'\n')
    










