import numpy as np
import pickle as pk

########### Part 0. Make map  ##########################################
data_dir = "/tmp2/b02902030/ADLxMLDS/hw1/data/"
map_48to39 = {} # 48 -> 39
map_phone2num = {} # 48 -> 48, phone->num
map_id2label = {} # for Y, instance id to label
with open(data_dir+"phones/48_39.map") as fp:
    for f in fp:
        s = f.strip('\n').split('\t')
        map_48to39[s[0]] = s[1]
with open(data_dir+"48phone_char.map") as fp:
    for f in fp:
        s = f.strip('\n').split('\t')
        map_phone2num[s[0]] = int(s[1])
with open(data_dir+"label/train.lab") as fp:
    for f in fp:
        s = f.strip("\n").split(",")
        map_id2label[s[0]] = s[1]
print("===== Part 0. Make map done ======")
########### Part 1. Read Fbank feature, 69 ####################
X = {} # key->2D numpy array of feature
Y = {}
Zeros = np.zeros(48, dtype=np.int32)
speaker_list = []
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/fbank/train.ark") as fp:
    for f in fp:
        s = f.strip('\n').split(' ')
        ins = s[0].split('_')
        ins_id = ins[0]+'_'+ins[1]
        if ins[0] not in speaker_list:
            speaker_list.append(ins[0])
        x = np.array(s[1:], dtype=np.float32)
        y = Zeros.copy()
        y[map_phone2num[map_48to39[map_id2label[s[0]]]]]=1
        if ins_id not in X:
            X[ins_id] = x
            Y[ins_id] = y
        else: # in X,Y
            X[ins_id] = np.vstack((X[ins_id], x))
            Y[ins_id] = np.vstack((Y[ins_id], y))
X_m = {} # key->2D numpy array of feature
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/mfcc/train.ark") as fp:
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
print("===== Part1. Read Mixure feature done =====")
########### Part 2. Split into sub-sentences ####################
### Part 2-1. Split Train and Valid set
ins_list = np.array(ins_list)
d = np.arange(len(ins_list))
np.random.shuffle(d)
d = d[:100]
ins_list_valid = ins_list[d].copy()
ins_list_train = np.delete(ins_list, d) 
print(len(ins_list_train))
print(len(ins_list_valid))
X_train = {}
Y_train = {}
for ins in ins_list_train:
    X_train[ins] = X[ins]
    Y_train[ins] = Y[ins]    
X_valid = {}
Y_valid = {}
for ins in ins_list_valid:
    X_valid[ins] = X[ins]
    Y_valid[ins] = Y[ins]
del X
del Y
t50_num = []
for ins in ins_list_valid:
    t50_num.append(int(X_valid[ins].shape[0]))
t50_num = np.array(t50_num)
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t30_num_mix.pk", "wb") as fw:
    pk.dump(t50_num, fw, protocol=pk.HIGHEST_PROTOCOL)
print("===== Part 2-1. Split Train and Valid set =====")
### Part 2-2. Split sentences
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
X, Y, count = np.zeros((70000, 30, 108), dtype=np.float32), np.zeros((70000, 30, 48), dtype=np.int32), 0 
for ins in ins_list_train:
    sl = X_train[ins].shape[0]
    idx = get_overlap_chunks(np.arange(sl), 30, 10)
    for ii in idx:
        X[count] = X_train[ins][ii]
        Y[count] = Y_train[ins][ii]
        count += 1
        if count%3000==0:
            print(count)
X = X[:count]
Y = Y[:count]
print(X.shape, Y.shape)
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t30_mix_train_data.pk", "wb") as fw:
    pk.dump(X, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y, fw, protocol=pk.HIGHEST_PROTOCOL)
del X_train
del Y_train
X, Y, count = np.zeros((10000, 30, 108), dtype=np.float32), np.zeros((10000, 30, 48), dtype=np.int32), 0 
for ins in ins_list_valid:
    sl = X_valid[ins].shape[0]
    idx = get_overlap_chunks(np.arange(sl), 30, 10)
    for ii in idx:
        X[count] = X_valid[ins][ii]
        Y[count] = Y_valid[ins][ii]
        count += 1
        if count%1000==0:
            print(count)
X = X[:count]
Y = Y[:count]
print(X.shape, Y.shape)
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t30_mix_valid_data.pk", "wb") as fw:
    pk.dump(X, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y, fw, protocol=pk.HIGHEST_PROTOCOL)
print("===== Part 2-2. Split sentences =====")
print("===== Part 2. Split into sub-sentences done =====")




