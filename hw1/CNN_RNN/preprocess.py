import numpy as np
import pickle as pk

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

data_dir = "/tmp2/b02902030/ADLxMLDS/hw1/data/"
with open('/tmp2/b02902030/ADLxMLDS/hw1/data/train_data.pk', 'rb') as fp:
    X_train = pk.load(fp)
    Y_train = pk.load(fp)
with open('/tmp2/b02902030/ADLxMLDS/hw1/data/valid_data.pk', 'rb') as fp:
    X_valid = pk.load(fp)
    Y_valid = pk.load(fp)
with open('/tmp2/b02902030/ADLxMLDS/hw1/data/test_data.pk', 'rb') as fp:
    X_test = pk.load(fp)

'''
Make tensors: 
X:
5-D tensor: (batch_size/all_data_size, timestep, H(combined samples), W(feature_size), channel(1))
Y:
3-D tensor: (batch_size/all_data_size, timestep, 48)
'''

##### Train #####
X_5D, Y_5D = np.zeros((20000, 70, 11, 39, 1), dtype=np.float32), np.zeros((20000, 70, 48), dtype=np.int32)
count = 0
for k in X_train.keys():
    X = X_train[k]
    Y = Y_train[k]
    X_collect = np.zeros((1, X.shape[0]-10, 11, 39, 1), dtype=np.float32)
    Y_collect = np.zeros((1, X.shape[0]-10, 48), dtype=np.int32)
    for i in range(5, X.shape[0]-5):
        x = X[i-5:i+5+1] # -> (11,39)
        x = np.expand_dims(x, axis=2) # (11,39,1)
        #x = np.expand_dims(x, axis=0) # (1,11,39,1)
        X_collect[0][i-5] = x
        y = Y[i] # -> (48, )
        #y = np.expand_dims(y, axis=0) # (1, 48)
        #y = np.expand_dims(y, axis=0) # (1, 1, 48)
        Y_collect[0][i-5] = y
    idx = get_overlap_chunks(np.arange(X.shape[0]-10), 70, 10)
    for ii in idx:
        X_5D[count] = X_collect[0][ii]
        Y_5D[count] = Y_collect[0][ii]
        count+=1
        if count%1000==0:
            print(count)
X_5D = X_5D[:count]
Y_5D = Y_5D[:count]
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t70_5D_train_data.pk", "wb") as fw:
    pk.dump(X_5D, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_5D, fw, protocol=pk.HIGHEST_PROTOCOL)
print(X_5D.shape)
print(Y_5D.shape)

##### Valid #####
X_5D, Y_5D = np.zeros((3000, 70, 11, 39, 1), dtype=np.float32), np.zeros((3000, 70, 48), dtype=np.int32)
count = 0
for k in X_valid.keys():
    X = X_valid[k]
    Y = Y_valid[k]
    X_collect = np.zeros((1, X.shape[0]-10, 11, 39, 1), dtype=np.float32)
    Y_collect = np.zeros((1, X.shape[0]-10, 48), dtype=np.int32)
    for i in range(5, X.shape[0]-5):
        x = X[i-5:i+5+1] # -> (11,39)
        x = np.expand_dims(x, axis=2) # (11,39,1)
        #x = np.expand_dims(x, axis=0) # (1,11,39,1)
        X_collect[0][i-5] = x
        y = Y[i] # -> (48, )
        #y = np.expand_dims(y, axis=0) # (1, 48)
        #y = np.expand_dims(y, axis=0) # (1, 1, 48)
        Y_collect[0][i-5] = y
    idx = get_overlap_chunks(np.arange(X.shape[0]-10), 70, 10)
    for ii in idx:
        X_5D[count] = X_collect[0][ii]
        Y_5D[count] = Y_collect[0][ii]
        count+=1
        if count%1000==0:
            print(count)
X_5D = X_5D[:count]
Y_5D = Y_5D[:count]
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t70_5D_valid_data.pk", "wb") as fw:
    pk.dump(X_5D, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y_5D, fw, protocol=pk.HIGHEST_PROTOCOL)
print(X_5D.shape)
print(Y_5D.shape)

##### Test #####
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
        if count%1000==0:
            print(count)
X_5D = X_5D[:count]
with open("/tmp2/b02902030/ADLxMLDS/hw1/data/t70_5D_test_data.pk", "wb") as fw:
    pk.dump(X_5D, fw, protocol=pk.HIGHEST_PROTOCOL)
print(X_5D.shape)


