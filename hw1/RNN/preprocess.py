import sys, os
import numpy as np
import pickle as pk

########### Make Map dictionary ###########
def make_map(data_dir):
    if data_dir[-1]!="/":
        data_dir += "/"
    map1 = {} # 48 -> 39
    map_num = {} # 48 -> 48, phone->num
    with open(data_dir+"phones/48_39.map") as fp:
        for f in fp:
            s = f.strip('\n').split('\t')
            map1[s[0]] = s[1]
    with open(data_dir+"48phone_char.map") as fp:
        for f in fp:
            s = f.strip('\n').split('\t')
            map_num[s[0]] = int(s[1])
    return map1, map_num

########### Training data preprocess ###########

# List all the instance
def list_instances(data_dir):
    if data_dir[-1]!="/":
        data_dir += "/"
    instances = []
    with open(data_dir+"mfcc/train.ark") as fp:
        for i, f in enumerate(fp):
            instance = f.strip('\n').split(' ')[0].split('_')
            instance = instance[0]+"_"+instance[1]
            instances.append(instance)
            if i%100000==0:
                print(i)

    unique, counts = np.unique(instances, return_counts=True)
    print("Length of instances list: ", len(unique))

    # Random shuffle
    np.random.shuffle(unique)

    with open(data_dir+"instances_train_list.txt", "w") as fw:
        for i in range(3326):
            fw.write(str(unique[i])+'\n')
    with open(data_dir+"instances_valid_list.txt", "w") as fw:
        for i in range(3326, len(unique)):
            fw.write(str(unique[i])+'\n')


# Since label's instance_id order is different from feature's, map it first
def make_label_map(data_dir):
    label_map = {}
    with open(data_dir+"label/train.lab") as fp:
        for f in fp:
            s = f.strip("\n").split(",")
            label_map[s[0]] = s[1]
    with open(data_dir+"label_map.pk", "wb") as fw:
        pk.dump(label_map, fw, protocol=pk.HIGHEST_PROTOCOL)
# make Training feature
def make_feature(data_dir):
    if data_dir[-1]!="/":
        data_dir += "/"
    if os.path.isfile(data_dir+"label_map.pk")==False:
        make_label_map(data_dir)
        print("make label map done.")
    with open(data_dir+"label_map.pk", "rb") as fp:
        label_map = pk.load(fp)

    map1, map_num = make_map(data_dir)

    X = {}
    Y = {}
    prev_instance_id = ""
    Zero = np.zeros(48, dtype=np.int32)
    with open(data_dir+"mfcc/train.ark") as fp:
        for e, f in enumerate(fp):
            if e % 100000==0:
                print(e)
            s = f.strip("\n").split(" ")
            instance = s[0]
            instance_id = instance.split('_')
            instance_id = instance_id[0]+"_"+instance_id[1]
            if prev_instance_id == instance_id:
                x = np.array(s[1:], dtype=np.float32)
                hot = map_num[map1[label_map[instance]]]
                y = Zero.copy()
                y[hot] = 1
                X[instance_id] = np.vstack((X[instance_id], x))
                Y[instance_id] = np.vstack((Y[instance_id], y))
            else:
                prev_instance_id = instance_id
                hot = map_num[map1[label_map[instance]]]
                X[instance_id] = np.array(s[1:], dtype=np.float32)
                Y[instance_id] = Zero.copy()
                Y[instance_id][hot] = 1
    X_train = {}
    y_train = {}
    X_valid = {}
    y_valid = {}
    with open(data_dir+"instances_train_list.txt") as fp:
        for f in fp:
            instance = f.strip('\n')
            X_train[instance] = X[instance]
            y_train[instance] = Y[instance]
    with open(data_dir+"instances_valid_list.txt") as fp:
        for f in fp:
            instance = f.strip('\n')
            X_valid[instance] = X[instance]
            y_valid[instance] = Y[instance]

    with open(data_dir+"train_data.pk", "wb") as fw:
        pk.dump(X_train, fw, protocol=pk.HIGHEST_PROTOCOL)
        pk.dump(y_train, fw, protocol=pk.HIGHEST_PROTOCOL)
    with open(data_dir+"valid_data.pk", "wb") as fw:
        pk.dump(X_valid, fw, protocol=pk.HIGHEST_PROTOCOL)
        pk.dump(y_valid, fw, protocol=pk.HIGHEST_PROTOCOL)


########### Main ###########
def main():
    #data_dir = "/tmp2/b02902030/ADLxMLDS/hw1/data/"
    data_dir = sys.argv[1]
    list_instances(data_dir)
    make_feature(data_dir)


if __name__ == "__main__":
    main()


            





