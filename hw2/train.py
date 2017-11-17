import numpy as np
import json, re, time, random, os, sys
from time import gmtime, strftime
from collections import Counter
import tensorflow as tf
from tensorflow.python.layers.core import Dense
#from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from bleu_eval import BLEU
############# Read label and data ###################
def read_data(data_path, vocab_size):
    regex = re.compile('[^a-zA-Z]')
    Vocab = []
    bos = '<BOS>'
    eos = '<EOS>'
    unk = '<UNK>'
    # Training
    train_label = {}
    with open(data_path+'training_label.json') as fp:
        d = json.load(fp)
    for i in range(len(d)):
        train_label[d[i]['id']] = []
        for c in d[i]['caption']:
            s = regex.sub(' ', c).lower().split()
            Vocab += s
            if len(s)>=5 and len(s)<=35:
                train_label[d[i]['id']].append(' '.join(s))

    # Testing
    test_label = {}
    with open(data_path+'testing_label.json') as fp:
        d = json.load(fp)
    for i in range(len(d)):
        test_label[d[i]['id']] = []
        for c in d[i]['caption']:
            test_label[d[i]['id']].append(c.lower())

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
    #print("Vocab size before delete: %d" %(len(Vocab)))
    #Vocab = Vocab[:vocab_size-3]
    '''    
    with open("20k.txt") as fp:
        for f in fp:
            w = f.strip('\n')
            if w not in Vocab:
                Vocab.append(w)
    '''    
    print("Vocab size before delete: %d" %(len(Vocab)))
    Vocab = Vocab[:vocab_size-3]
    Vocab = [bos, eos, unk] + Vocab
    print("Vocab size after delete: %d" %(len(Vocab)))

    # Read training data
    X_train = {}
    for v in train_list:
        X_train[v] = np.load(data_path+'training_data/feat/'+v+'.npy').astype(np.float32)

    X_test = {}
    for v in test_list:
        X_test[v] = np.load(data_path+'testing_data/feat/'+v+'.npy').astype(np.float32)
        
    print("Size of training data: %d" %(len(X_train)))
    print("Size of testing data: %d" %(len(X_test)))

    return Vocab, train_list, test_list, X_train, X_test, train_label, test_label

#####################################################
def sentences2ids(sentences, Vocab, batch_size, decoder_max_len):
    decoder_input = np.ones((batch_size, decoder_max_len), dtype=np.int32) # make every element as eos
    decoder_target = np.ones((batch_size, decoder_max_len), dtype=np.int32) # make every element as eos
    for b in range(batch_size):
        sen_target = sentences[b].split()
        sen_input = sentences[b].split()
        sen_input.insert(0, Vocab[0]) # add bos at first
        for i in range(len(sen_input)):
            idx = 2 if sen_input[i] not in Vocab else Vocab.index(sen_input[i])
            decoder_input[b][i] = idx
            if i>0:
                decoder_target[b][i-1] = idx

    return decoder_input, decoder_target

def gen_test_data(test_list, X_test, test_label, Vocab):
    test_size = len(test_list)
    encoder_input_test = np.zeros((test_size, 80, 4096), dtype=np.float32) #(batch, time, feature)
    for t in range(test_size):
        encoder_input_test[t] = X_test[test_list[t]]
    #----------#
    sentences = []
    decoder_lens_test = []
    for t in range(test_size):
        sentences.append(random.choice(test_label[test_list[t]]))
        decoder_lens_test.append(len(sentences[t].split())+1)
    decoder_max_len = max(decoder_lens_test)
    decoder_input_test, decoder_target_test = sentences2ids(sentences, Vocab, test_size, decoder_max_len)

    return encoder_input_test, decoder_input_test, decoder_target_test, decoder_lens_test

def gen_batch(batch_list, X_train, train_label, Vocab):
    batch_size = len(batch_list)
    encoder_input = np.zeros((batch_size, 80, 4096), dtype=np.float32)
    for b in range(batch_size):
        encoder_input[b] = X_train[batch_list[b]]
    #----------#
    sentences = []
    decoder_lens = []
    for b in range(batch_size):
        sentences.append(random.choice(train_label[batch_list[b]]))
        decoder_lens.append(len(sentences[b].split())+1)
    decoder_max_len = max(decoder_lens)
    decoder_input, decoder_target = sentences2ids(sentences, Vocab, batch_size, decoder_max_len)
    
    return encoder_input, decoder_input, decoder_target, decoder_lens

################# Setting ######################
#data_path = "/dhome/b02902030/ADLxMLDS/hw2/MLDS_hw2_data/"
data_path = sys.argv[1] + '/' if sys.argv[1][-1]!='/' else sys.argv[1]
save_path = sys.argv[2] + '/' if sys.argv[2][-1]!='/' else sys.argv[2]

vocab_size = 4000

t1 = time.time()
Vocab, train_list, test_list, X_train, X_test, train_label, test_label = read_data(data_path, vocab_size)
t2 = time.time()
print("Read Data Time: %fs" %(t2-t1))
print("========= Preprocessing done =========")


############ Build model ##########################################
def build_multi_cell(num_units, num_layers):
    layers = [tf.contrib.rnn.LSTMCell(num_units) for _ in range(num_layers)]
    return tf.contrib.rnn.MultiRNNCell(layers)

num_units = 512
num_layers = 3
embedding_size = 300
lr = 1e-4*0.8

print("num_units: %d | num_layers: %d | embedding_size: %d| lr: %f" %(num_units, num_layers, embedding_size, lr))

# placeholder
tf_encoder_input = tf.placeholder(tf.float32, shape=[None, 80, 4096]) # (batch_size, 80, 4096)
tf_decoder_input = tf.placeholder(tf.int32, shape=[None, None]) # (batch_size, decocer_max_len)
tf_decoder_target = tf.placeholder(tf.int32, shape=[None, None]) # (batch_size, decoder_max_len)
tf_decoder_seq_len = tf.placeholder(tf.int32, shape=[None]) # (batch_size), each length of sentences 
tf_decoder_max_len = tf.reduce_max(tf_decoder_seq_len)
tf_prob = tf.placeholder(tf.float32, shape=())

# Encoder
encoder_fw_cell = build_multi_cell(num_units, num_layers)
encoder_bw_cell = build_multi_cell(num_units, num_layers)
(fw_out,bw_out), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(encoder_fw_cell, encoder_bw_cell, tf_encoder_input, dtype=tf.float32)

encoder_state = []
for i in range(num_layers):
    if isinstance(fw_state[i], tf.contrib.rnn.LSTMStateTuple):
        state_c = tf.add(fw_state[i].c, bw_state[i].c)
        state_h = tf.add(fw_state[i].h, bw_state[i].h)
        state = tf.contrib.rnn.LSTMStateTuple(c=state_c, h=state_h)
    elif isinstance(fw_state[i], tf.Tensor):
        state = tf.add(fw_state[i], bw_state[i])
    encoder_state.append(state)
encoder_state = tuple(encoder_state)
encoder_output = tf.add(fw_out, bw_out)

# Decoder
# Embedding
embedding_decoder = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size], stddev=0.1))
with tf.device('/cpu:0'):
    emb_decoder_input = tf.nn.embedding_lookup(embedding_decoder, tf_decoder_input)
# decoder
# attention
attention_state = tf.transpose(encoder_output, [0,1,2]) # (batch_size, max_time, num_units)
attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, attention_state)
decoder_cell = build_multi_cell(num_units, num_layers)
attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=num_units)
tf_batch_size = tf.shape(tf_encoder_input)[0]
decoder_initial_state = attention_cell.zero_state(tf_batch_size, tf.float32).clone(cell_state=encoder_state)
#decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32)

#train_helper = tf.contrib.seq2seq.TrainingHelper(emb_decoder_input, tf_decoder_seq_len)
train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(emb_decoder_input, tf_decoder_seq_len, embedding_decoder, tf_prob)
projection_layer = Dense(vocab_size)
#decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, encoder_state, output_layer=projection_layer)
decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, train_helper, decoder_initial_state, output_layer=projection_layer)
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder) #output_time_major default=False, (batch, time, feature) ?
logits = outputs.rnn_output

# loss and train_op
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_decoder_target, logits=logits)
decoder_mask = tf.sequence_mask(tf_decoder_seq_len, tf_decoder_max_len, dtype=logits.dtype)
#loss = (tf.reduce_sum(cross_entropy*decoder_mask)/tf.to_float(batch_size))
loss = tf.reduce_sum(cross_entropy*decoder_mask)
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

### Inference
tf_bos = tf.placeholder(tf.int32, shape=[None]) # (batch_size), [0,0,0...]
tf_eos = 1

'''
tiled_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=5)
tiled_encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier=5)
attention_state = tf.transpose(tiled_encoder_output, [0, 1, 2])
attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, attention_state)
attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=num_units)
decoder_initial_state_inf = attention_cell.zero_state(tf_batch_size*5, tf.float32).clone(cell_state=tiled_encoder_state)
decoder_inf = tf.contrib.seq2seq.BeamSearchDecoder(attention_cell, embedding_decoder, tf_bos, tf_eos, decoder_initial_state_inf, 5, projection_layer)
output_inf, _, _, = tf.contrib.seq2seq.dynamic_decode(decoder_inf, maximum_iterations=50)
output_bs = output_inf.beam_search_decoder_output
print(output_bs.predicted_ids.get_shape().as_list())
print(output_bs.parent_ids.get_shape().as_list())
'''
#print(dir(outputs_inf))
#outputs_inf_words = outputs_inf.predicted_ids # (batch_size, time, beam)
#outputs_inf_words = outputs_inf.beam_search_decoder_output# (batch_size, time, beam)
#output_inf_words = tf.transpose(outputs_inf_words, perm=[0,2,1]) # -> (batch, beam, time)

inf_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, tf_bos, tf_eos)
decoder_inf = tf.contrib.seq2seq.BasicDecoder(attention_cell, inf_helper, decoder_initial_state, output_layer=projection_layer)
outputs_inf, _, _, = tf.contrib.seq2seq.dynamic_decode(decoder_inf, maximum_iterations=50)
outputs_inf_words = outputs_inf.sample_id


tf.add_to_collection('for_test', outputs_inf_words)
tf.add_to_collection('for_test', loss)
tf.add_to_collection('for_test', tf_encoder_input)
tf.add_to_collection('for_test', tf_decoder_input)
tf.add_to_collection('for_test', tf_decoder_target)
tf.add_to_collection('for_test', tf_decoder_seq_len)
tf.add_to_collection('for_test', tf_bos)
tf.add_to_collection('for_test', tf_prob)

t3 = time.time()
print("Build Graph Time: %fs" %(t3-t2))
###################################################################################

def evaluate(output_id):
    bleu = []
    bleu1 = []
    for i, v in enumerate(test_list):
        bleu_v = []
        s = []
        for idx in output_id[i]:
            if idx > 2:
                s.append(Vocab[idx])
        s = ' '.join(s)
        if i==0 or i==5 or i==10:
            print(s + " / " + test_label[v][0])
        for cap in test_label[v]:
            bleu_v.append(BLEU(s, cap.rstrip('.')))
        bleu.append(np.mean(bleu_v))
    for i, v in enumerate(test_list):
        s = []
        for idx in output_id[i]:
            if idx > 2:
                s.append(Vocab[idx])
        s = ' '.join(s)
        caps = [x.rstrip('.') for x in test_label[v]]
        bleu1.append(BLEU(s, caps, True))

    return np.mean(bleu), np.mean(bleu1)

'''
def beam_path(parent_id, predict_id):
    beam = parent_id.shape[2]
    ans = predict_id.copy() # 100,50,5/10
    for batch in range(100):
        for b in range(beam):
            pid = parent_id[batch][49][b]
            for i in range(48, 0, -1):
                ans[batch][i][b] = predict_id[batch][i][pid]
                pid = parent_id[batch][i][pid]
    return ans
'''

epoch_num = 150
batch_size = 32
print("epoch_num: %d | batch_size: %d" %(epoch_num, batch_size))
train_len = len(X_train)
test_len = len(X_test)
train_batch_num = int(train_len/batch_size)+1
test_batch_num = int(test_len/batch_size)+1
print("Training batch number: %d" %(train_batch_num))
print("Testing batch number: %d" %(test_batch_num))

encoder_input_test, decoder_input_test, decoder_target_test, decoder_lens_test = gen_test_data(test_list, X_test, test_label, Vocab)

#save_path = "/home/aria0/ADLxMLDS2017/hw2/model/lstm_256_simple/"
saver = tf.train.Saver(max_to_keep=None)
with tf.Session() as sess:
    datetime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    with open(save_path+"train_log.txt", "a+") as fw:
        fw.write('\n'+str(datetime)+'\n')

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        best_bleu = 0.0
        best_bleu1 = 0.0
        num_save = 0
        sample_prob = 1 # prob of input, need to use 1-prob to be prob of output
        step = 0
        for epoch in range(1, epoch_num+1):
            print("Epoch: %d" %(epoch))
            # Train
            train_total_loss = 0.0
            for b in range(train_batch_num):
                encoder_input, decoder_input, decoder_target, decoder_lens = \
                    gen_batch(train_list[b*batch_size:(b+1)*batch_size], X_train, train_label, Vocab)

                step += 1
                #sample_prob = 0 if epoch<50 else 0
                sample_prob = 0
                _, train_loss = sess.run([train_op, loss], {tf_encoder_input: encoder_input,
                                                            tf_decoder_input: decoder_input,
                                                            tf_decoder_target: decoder_target,
                                                            tf_decoder_seq_len: decoder_lens,
                                                            tf_prob: sample_prob})
                train_total_loss += train_loss

            train_total_loss /= len(train_list)
            
            # Inference
            output_ids, test_loss = sess.run([outputs_inf_words, loss], 
                                                        {tf_encoder_input: encoder_input_test,
                                                         tf_decoder_input: decoder_input_test,
                                                         tf_decoder_target: decoder_target_test,
                                                         tf_decoder_seq_len: decoder_lens_test,
                                                         tf_prob: 1,
                                                         tf_bos: np.zeros(100, dtype=np.int32)})
            test_loss /= len(test_list)
            bleu_score, bleu1_score = evaluate(output_ids)

            if best_bleu<bleu_score:
                best_bleu = bleu_score
                if bleu_score>=0.3:
                    saver.save(sess, save_path+"epoch_"+str(epoch))
                    num_save+=1

            if best_bleu1<bleu1_score:
                best_bleu1 = bleu1_score

            # log
            print("Train loss: %.4f | Test loss: %.4f | BLEU: %.4f | Best: %.4f | BLEU1: %.4f | Best1: %.4f" %(train_total_loss, test_loss, bleu_score, best_bleu, bleu1_score, best_bleu1))
            fw.write("Epoch: %d | Train loss: %.4f | Test loss: %.4f | BLEU: %.4f | Best: %.4f | BLEU1: %.4f | Best1: %.4f\n" %(epoch, train_total_loss, test_loss, bleu_score, best_bleu, bleu1_score, best_bleu1))
            fw.flush()

        print("num_save: %d" %(num_save))

