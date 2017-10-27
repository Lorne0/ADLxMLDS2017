import os, sys
import numpy as np
import pickle as pk
import tensorflow as tf
from edit_distance import edit_distance

def set_config(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    #config = tf.ConfigProto(gpu_options=gpu_options)
    return config

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

##################################################

config = set_config()
with tf.Session(config=config) as sess:

    #model = sys.argv[2]
    #model_file = tf.train.latest_checkpoint(data_dir+model)
    model_file = "/dhome/b02902030/ADLxMLDS/hw1/epoch-25"
    print("Model file:", model_file)
    saver = tf.train.import_meta_graph(model_file+'.meta')
    saver.restore(sess, model_file)
    output = tf.get_collection('output')[0]
    tf_x = tf.get_collection('output')[1]

    threshold = 0.0

    map_num2phone, map_48to39, map_phone2alpha = make_map(data_dir)

    output_result = {}
    for instance in instances_valid_list:
        X = X_valid[instance]
        y = y_valid[instance]
        out = sess.run(output, {tf_x: X})
        output_result[instance] = out


    #for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
    print("Threshold:", threshold)
    dist = 0.0
    for instance in instances_valid_list:
        y = np.argmax(y_valid[instance], 1)
        s0 = ""
        for k in y:
            s0 += map_phone2alpha[map_48to39[map_num2phone[k]]]
        s0 = string_compress(s0)

        out = output_result[instance]
        out_v = np.max(out, 1)
        out = np.argmax(out, 1)
        s1 = ""
        for i in range(len(out)):
            if out_v[i]>threshold:
                k = out[i]
                s1 += map_phone2alpha[map_48to39[map_num2phone[k]]]
        s1 = string_trim(s1)
        s1 = string_trim2(s1)
        s1 = string_compress(s1)
    
        dist += edit_distance(s0, s1)

        #print(s0)
        #print(s1)
        #print()

    print(dist/370)
            


