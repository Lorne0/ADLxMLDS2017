import numpy as np
import json
import re
import time
from collections import Counter
############# Read label and data ###################

def read_data():
    tS = time.time()
    data_file = "/dhome/b02902030/ADLxMLDS/hw2/MLDS_hw2_data/"
    regex = re.compile('[^a-zA-Z]')
    Vocab = []
    vocab_size = 4000 - 3
    unk = '<UNK>'
    bos = '<BOS>'
    eos = '<EOS>'
    #' '.join(regex.sub(' ', line).split())
    # Read Label
    # Training
    train_label = {}
    with open(data_file+'training_label.json') as fp:
        d = json.load(fp)
    for i in range(len(d)):
        train_label[d[i]['id']] = []
        for c in d[i]['caption']:
            s = regex.sub(' ', c).lower().split()
            Vocab += s
            if len(s)>=5:
                train_label[d[i]['id']].append(' '.join(s))

    # Testing
    test_label = {}
    with open(data_file+'testing_label.json') as fp:
        d = json.load(fp)
    for i in range(len(d)):
        test_label[d[i]['id']] = []
        for c in d[i]['caption']:
            s = regex.sub(' ', c).lower().split()
            Vocab += s
            if len(s)>=5:
                test_label[d[i]['id']].append(' '.join(s))

    train_list = list(train_label.keys())
    test_list = list(test_label.keys())
    print("Num of training data: %d" %(len(train_list)))
    print("Num of testing data: %d" %(len(test_list)))

    train_cap_len = []
    for v in train_list:
        for c in train_label[v]:
            train_cap_len.append(len(c.split()))
    max_train_cap_len = max(train_cap_len)
    min_train_cap_len = min(train_cap_len)

    test_cap_len = []
    for v in test_list:
        for c in test_label[v]:
            test_cap_len.append(len(c.split()))
    max_test_cap_len = max(test_cap_len)
    min_test_cap_len = min(test_cap_len)

    print("Train max/min caption length: %d/%d" %(max_train_cap_len, min_train_cap_len))
    print("Test max/min caption length: %d/%d" %(max_test_cap_len, min_test_cap_len))

    print("Read Label Done.")
    # Make vocab set
    Vocab = dict(Counter(Vocab))
    Vocab = sorted(Vocab, key=Vocab.get, reverse=True)
    print("Vocab size before delete: %d" %(len(Vocab)))
    Vocab = Vocab[:vocab_size]
    Vocab.extend([unk, bos, eos])
    print("Vocab size after delete: %d" %(len(Vocab)))

    tE = time.time()
    print("Time: %fs" %(tE-tS))
    # Training
    #train_list = os.listdir('MLDS_hw2_data/training_data/feat')
    #for t in train_list:
    #    pass



# func: change word to one-hot vector + padding 0

#####################################################

read_data()
