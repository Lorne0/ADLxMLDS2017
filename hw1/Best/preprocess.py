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
print("===== Part1. Read Fbank feature done =====")
########### Part 2. Transfer to Picture ####################
### Part 2-1. Head & Tail add 8 element
ins_list = list(X.keys())
for ins in ins_list:
    head = np.tile(X[ins][:2], (4,1))
    tail = np.tile(X[ins][-2:], (4,1))
    X[ins] = np.vstack((head, X[ins]))
    X[ins] = np.vstack((X[ins], tail))
    head = np.tile(Y[ins][:2], (4,1))
    tail = np.tile(Y[ins][-2:], (4,1))
    Y[ins] = np.vstack((head, Y[ins]))
    Y[ins] = np.vstack((Y[ins], tail))
print("===== Part 2-1. Head & Tail add 8 element done =====")
### Part 2-2. Reshape to Pictures
X_P = {}
Y_P = {}
for ins in ins_list:
    sl = X[ins].shape[0]
    X_P[ins] = np.zeros((sl, 17, 69, 1))
    for i in range(8, sl-8):
        x = X[ins][i-8:i+8+1]
        X_P[ins][i-8] = np.expand_dims(x, axis=2)
    Y_P[ins] = np.zeros((sl, 48), dtype=np.int32)
    for i in range(8, sl-8):
        Y_P[ins][i-8] = Y[ins][i] 
del X
del Y
print("===== Part 2-2. Reshape to Pictures done =====")
print("===== Part 2. Transfer to Picture done =====")
########### Part 3. Split into sub-sentences ####################
### Part 3-1. Split Train and Valid set
#np.random.shuffle(ins_list)
ins_list_valid = []
for speaker in speaker_list:
    for ins in ins_list:
        if ins.split('_')[0] == speaker:
            ins_list_valid.append(ins)
            ins_list.remove(ins)
            break
ins_list_train = ins_list
print(len(ins_list_train))
print(len(ins_list_valid))
X_P_train = {}
Y_P_train = {}
for ins in ins_list_train:
    X_P_train[ins] = X_P[ins]
    Y_P_train[ins] = Y_P[ins]    
X_P_valid = {}
Y_P_valid = {}
for ins in ins_list_valid:
    X_P_valid[ins] = X_P[ins]
    Y_P_valid[ins] = Y_P[ins]
del X_P
del Y_P
print("===== Part 3-1. Split Train and Valid set =====")
### Part 3-2. Split sentences
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
X_train, Y_train, count = np.zeros((29000, 60, 17, 69, 1), dtype=np.float32), np.zeros((29000, 60, 48), dtype=np.int32), 0 
for ins in ins_list_train:
    sl = X_P_train[ins].shape[0]
    idx = get_overlap_chunks(np.arange(sl), 60, 15)
    for ii in idx:
        X_train[count] = X_P_train[ins][ii]
        Y_train[count] = Y_P_train[ins][ii]
        count += 1
        if count%2000==0:
            print(count)
X_train = X_train[:count]
Y_train = Y_train[:count]
print(X_train.shape, Y_train.shape)
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t50_5D_train_data.pk", "wb") as fw:
    pk.dump(X_train, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_train, fw, protocol=pk.HIGHEST_PROTOCOL)
del X_train
del Y_train
del X_P_train
del Y_P_train
X_valid, Y_valid, count = np.zeros((10000, 60, 17, 69, 1), dtype=np.float32), np.zeros((10000, 60, 48), dtype=np.int32), 0 
for ins in ins_list_valid:
    sl = X_P_valid[ins].shape[0]
    idx = get_overlap_chunks(np.arange(sl), 60, 15)
    for ii in idx:
        X_valid[count] = X_P_valid[ins][ii]
        Y_valid[count] = Y_P_valid[ins][ii]
        count += 1
        if count%1000==0:
            print(count)
X_valid = X_valid[:count]
Y_valid = Y_valid[:count]
print(X_valid.shape, Y_valid.shape)
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t50_5D_valid_data.pk", "wb") as fw:
    pk.dump(X_valid, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_valid, fw, protocol=pk.HIGHEST_PROTOCOL)
print("===== Part 3-2. Split sentences =====")
print("===== Part 3. Split into sub-sentences done =====")




