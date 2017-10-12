import os, sys
import numpy as np
import pickle as pk
import tensorflow as tf

def set_config(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    #config = tf.ConfigProto(gpu_options=gpu_options)
    return config

##################### Read Data ####################
data_dir = sys.argv[1]
if data_dir[-1] != "/":
    data_dir += "/"
with open(data_dir+"train_data.pk", "rb") as fp:
    X_train = pk.load(fp) # 3326
    y_train = pk.load(fp) # 3326
with open(data_dir+"valid_data.pk", "rb") as fp:
    X_valid = pk.load(fp) # 370
    y_valid = pk.load(fp) # 370

instances_train_list = []
with open(data_dir+"instances_train_list.txt") as fp:
    for f in fp:
        s = f.strip('\n')
        instances_train_list.append(s)
instances_valid_list = []
with open(data_dir+"instances_valid_list.txt") as fp:
    for f in fp:
        s = f.strip('\n')
        instances_valid_list.append(s)

instances_train_list = np.array(instances_train_list)
instances_valid_list = np.array(instances_valid_list)

print("========== Read data done ============")
####################################################

###### Parameters ######
lstm_num_units = int(sys.argv[2])
feature_size = 39
batch_size = 1 # one sentence one batch
epoch_numbers = int(sys.argv[3])
#LR =  [1.0*1e-4, 3.0*1e-4, 6.0*1e-4, 1.0*1e-3, 1.0*1e-5, 3.0*1e-5, 6.0*1e-5, 1.0*1e-6, 3.0*1e-6, 6.0*1e-6]
lr = float(sys.argv[4])
print(lr)
restore = sys.argv[5] # "true" or "false" 
last_epoch = int(sys.argv[6])
########################

with tf.device('/gpu:0'):
    tf_x = tf.placeholder(tf.float32, [None, feature_size])  # 1*timestep, 39
    tf_x_r = tf.reshape(tf_x, [batch_size, -1, feature_size])  # 1 ,timestep, 39 
    tf_y = tf.placeholder(tf.int32, [None, 48]) # 1*timestep, 48

    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_num_units)
    states, (h_c, h_n) = tf.nn.dynamic_rnn(rnn_cell, tf_x_r, initial_state=None, dtype=tf.float32, time_major=False)
    # states: 1, timestep, 128(lstm_num_units)
    states = tf.reshape(states, [-1, lstm_num_units]) # -> 1*timestep, 128
    output = tf.layers.dense(states, 48)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output, reduction="none"))
    train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)
    #acc_op = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))[1]
    tf.add_to_collection('output', output)
    tf.add_to_collection('output', tf_x)
    tf.add_to_collection('output', tf_y)
    tf.add_to_collection('train_op', train_op)


config = set_config()
with tf.Session(config=config) as sess:

    saver = tf.train.Saver(max_to_keep=None)
    if restore=="true":
        saver.restore(sess, data_dir+"model/units_"+str(lstm_num_units)+"_lr_"+str(lr)+"/epoch_"+str(last_epoch))
    else:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

    for epoch in range(1+last_epoch, epoch_numbers+1+last_epoch):
        np.random.shuffle(instances_train_list)
        loss_all = 0
        for i in range(3326): #training
            if i%1000==0:
                print(i)
            instance = instances_train_list[i]
            X = X_train[instance]
            y = y_train[instance]
            _, loss_ = sess.run([train_op, loss], {tf_x: X, tf_y: y})
            loss_all += loss_

        acc_all = 0
        count_all = 0
        for instance in instances_valid_list:
            X = X_valid[instance]
            y = y_valid[instance]
            out = sess.run(output, {tf_x: X, tf_y: y})
            out = np.argmax(out, 1)
            y = np.argmax(y, 1)
            acc_all += sum(out==y)
            count_all += len(y)

        print("Epoch:", epoch, " loss: ", loss_all/3326.0, " accuracy: ", acc_all/float(count_all))

        saver.save(sess, data_dir+"model/units_"+str(lstm_num_units)+"_lr_"+str(lr)+"/epoch", global_step=epoch)

            


