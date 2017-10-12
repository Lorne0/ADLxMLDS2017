import os, sys
import numpy as np
import pickle as pk
import tensorflow as tf

'''
def set_config(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    #config = tf.ConfigProto(gpu_options=gpu_options)
    return config
'''

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

def string_trim(s0):
    s0 = s0.strip("L")
    prev, s = "", ""
    for c in s0:
        if c!=prev:
            prev = c
            s += c
    return s

data_dir = sys.argv[1]
if data_dir[-1] != "/":
    data_dir += "/"

with tf.Session() as sess:

    model = sys.argv[2]
    model_file = tf.train.latest_checkpoint(data_dir+model)
    print("Model file:", model_file)
    saver = tf.train.import_meta_graph(model_file+'.meta')
    saver.restore(sess, model_file)
    output = tf.get_collection('output')[0]
    tf_x = tf.get_collection('output')[1]

    map_num2phone, map_48to39, map_phone2alpha = make_map(data_dir)

    X = {}
    with open(data_dir+"mfcc/test.ark") as fp:
        for f in fp:
            s = f.strip('\n').split(' ')
            instance = s[0]
            instance_id = s[0].split('_')
            instance_id = instance_id[0]+'_'+instance_id[1]
            k = np.array(s[1:], dtype=np.float32)

            if instance_id not in X:
                X[instance_id] = k
            else:
                X[instance_id] = np.vstack((X[instance_id], k))

        with open("output.csv", "w") as fw:
            fw.write("id,phone_sequence\n")
            for instance in X.keys():
                x_ = X[instance]
                out = sess.run(output, {tf_x: x_})
                out = np.argmax(out, 1)
                s1 = ""
                for k in out:
                    s1 += map_phone2alpha[map_48to39[map_num2phone[k]]]
                s1 = string_trim(s1)
                fw.write(instance+','+s1+'\n')


