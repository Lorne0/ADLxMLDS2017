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

def make_better():
    X = np.zeros((100000, 23))
    imgs = np.zeros((100000, 64, 64, 3), dtype=np.uint8)
    cnt = 0
    global_cnt = 0
    with open("./data/tags_clean.csv") as fp:
        for f in fp:
            s = f.strip('\n').split(',')[1].split('\t')
            c = 0
            for i in range(len(s)):
                k = s[i].split(':')[0].strip(' ')
                if k in attributes:
                    c += 1
            #if c==2:
            for i in range(len(s)):
                k = s[i].split(':')[0].strip(' ')
                if k in attributes:
                    X[cnt][attributes.index(k)] = 1
            # imgs
            file_name = './data/faces/'+str(global_cnt)+'.jpg'
            img = scipy.misc.imread(file_name)
            img = scipy.misc.imresize(img, [64, 64, 3])
            imgs[cnt] = img
            cnt += 1
            global_cnt += 1
    print(cnt)
    print(global_cnt)
    X = X[:cnt]
    imgs = imgs[:cnt]

    # fliplr, rotate 5, -5 -> 4x data
    flr_imgs = np.zeros_like(imgs, dtype=np.uint8)
    r5_imgs = np.zeros_like(imgs, dtype=np.uint8)
    rn5_imgs = np.zeros_like(imgs, dtype=np.uint8)
    for i in range(cnt):
        flr_imgs[i] = np.fliplr(imgs[i])
        r5_imgs[i] = scipy.misc.imrotate(imgs[i], 5)
        rn5_imgs[i] = scipy.misc.imrotate(imgs[i], -5)
    imgs = np.vstack((imgs,flr_imgs))
    imgs = np.vstack((imgs,r5_imgs))
    imgs = np.vstack((imgs,rn5_imgs))
    imgs = imgs.astype(np.float32)
    imgs = imgs/127.5 -1
    X = np.tile(X, (4,1))
    print(imgs.shape)
    print(X.shape)

    return X, imgs

def load_image():
    imgs = np.zeros((33431, 64, 64, 3), dtype=np.float32)
    for i in range(33431):
        file_name = './data/faces/'+str(i)+'.jpg'
        img = scipy.misc.imread(file_name)
        img = scipy.misc.imresize(img, [64, 64, 3])
        imgs[i] = img.astype(np.float32)
    imgs = imgs/127.5 -1
    return imgs
        





