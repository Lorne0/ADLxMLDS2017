import numpy as np
import scipy.misc

attributes = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair', 'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

def make_tags():
    X = np.zeros((33431, 23))
    cnt = 0
    with open("./data/tags_clean.csv") as fp:
        for f in fp:
            s = f.strip('\n').split(',')[1].split('\t')
            for i in range(len(s)):
                k = s[i].split(':')[0].strip(' ')
                if k in attributes:
                    X[cnt][attributes.index(k)] = 1
            cnt += 1
    return X

def load_image():
    imgs = np.zeros((33431, 64, 64, 3), dtype=np.float32)
    for i in range(33431):
        file_name = './data/faces/'+str(i)+'.jpg'
        img = scipy.misc.imread(file_name)
        img = scipy.misc.imresize(img, [64, 64, 3])
        imgs[i] = img.astype(np.float32)
    imgs = imgs/127.5 -1
    return imgs
        





