import os, sys
import numpy as np
import pickle as pk
import tensorflow as tf
from edit_distance import edit_distance

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

##################### Read Data ####################
data_dir = sys.argv[1]
if data_dir[-1] != "/":
    data_dir += "/"
with open(data_dir+"valid_data.pk", "rb") as fp:
    X_valid = pk.load(fp) # 370
    y_valid = pk.load(fp) # 370

instances_valid_list = []
with open(data_dir+"instances_valid_list.txt") as fp:
    for f in fp:
        s = f.strip('\n')
        instances_valid_list.append(s)

instances_valid_list = np.array(instances_valid_list)

print("========== Read data done ============")
####################################################

# Make Answer
def string_trim(s0):
    s0 = s0.strip("L")
    prev, s = "", ""
    for c in s0:
        if c!=prev:
            prev = c
            s += c
    return s

##################################################

#config = set_config()
with tf.Session() as sess:

    model = sys.argv[2]
    model_file = tf.train.latest_checkpoint(data_dir+model)
    print("Model file:", model_file)
    saver = tf.train.import_meta_graph(model_file+'.meta')
    saver.restore(sess, model_file)
    output = tf.get_collection('output')[0]
    tf_x = tf.get_collection('output')[1]
    #tf_y = tf.get_collection('output')[2]

    #all_vars = tf.trainable_variables()
    #for v in all_vars:
    #    print("%s with value %s" % (v.name, sess.run(v)))

    map_num2phone, map_48to39, map_phone2alpha = make_map(data_dir)
    dist = 0.0
    for instance in instances_valid_list:
        X = X_valid[instance]
        y = y_valid[instance]
        out = sess.run(output, {tf_x: X})

        y = np.argmax(y_valid[instance], 1)
        s0 = ""
        for k in y:
            s0 += map_phone2alpha[map_48to39[map_num2phone[k]]]
        s0 = string_trim(s0)

        out = np.argmax(out, 1)
        s1 = ""
        for k in out:
            s1 += map_phone2alpha[map_48to39[map_num2phone[k]]]
        s1 = string_trim(s1)
    
        dist += edit_distance(s0, s1)

    print(dist/370)
            


