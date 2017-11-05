import numpy as np
import json, re, time, random, os, sys
from time import gmtime, strftime
from collections import Counter
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from bleu_eval import BLEU
############# Read label and data ###################
def read_data(data_path, vocab_size):
    #t1 = time.time()
    regex = re.compile('[^a-zA-Z]')
    Vocab = []
    bos = '<BOS>'
    eos = '<EOS>'
    unk = '<UNK>'
    #' '.join(regex.sub(' ', line).split())
    # Read Label
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
            s = regex.sub(' ', c).lower().split()
            Vocab += s
            if len(s)>=5 and len(s)<=35:
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
    Vocab = Vocab[:vocab_size-3]
    #Vocab.extend([unk, bos, eos])
    Vocab = [bos, eos, unk] + Vocab
    print("Vocab size after delete: %d" %(len(Vocab)))

    #t2 = time.time()
    #print("Time: %fs" %(t2-t1))

    # Read training data
    X_train = {}
    for v in train_list:
        X_train[v] = np.load(data_path+'training_data/feat/'+v+'.npy').astype(np.float32)

    X_test = {}
    for v in test_list:
        X_test[v] = np.load(data_path+'testing_data/feat/'+v+'.npy').astype(np.float32)
        
    print("Size of training data: %d" %(len(X_train)))
    print("Size of testing data: %d" %(len(X_test)))

    #t3 = time.time()
    #print("Time: %fs" %(t3-t2))

    return Vocab, train_list, test_list, X_train, X_test, train_label, test_label

#####################################################
def word2onehot(word, Vocab, vocab_size):
    z = np.zeros((vocab_size))
    if word not in Vocab:
        z[2] = 1.0 # unknown
    else:
        z[Vocab.index(word)] = 1.0
    return z

'''
def sentences2onehot(sentences, Vocab, batch_size, decoder_max_len):
    vocab_size = len(Vocab)
    decoder_target = np.zeros((batch_size, decoder_max_len, vocab_size))
    decoder_target[:, :, 1] = 1.0 # make every element as eos
    decoder_input = decoder_target.copy()
    for b in range(batch_size):
        sen_target = sentences[b].split()
        sen_input = sentences[b].split()
        sen_input.insert(0, Vocab[0]) # add bos at first
        for d in range(len(sen_target)):
            # wrong
            #decoder_target[b][d] = word2onehot(sen_target[d], Vocab, vocab_size)
            #if d>0:
            #    decoder_input[b][d] = word2onehot(sen_input[d], Vocab, vocab_size)
    #decoder_input[:, 0, 1] = 0.0 # make first element as bos
    #decoder_input[:, 0, 0] = 1.0 # make first element as bos
    return decoder_input, decoder_target
'''

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
    encoder_input_test = np.zeros((test_size, 80, 4096), dtype=np.float32)
    for t in range(test_size):
        encoder_input_test[t] = X_test[test_list[t]]
    #----------#
    sentences = []
    decoder_lens_test = []
    for t in range(test_size):
        sentences.append(random.choice(test_label[test_list[t]]))
        decoder_lens_test.append(len(sentences[t].split())+1)
    #decoder_input, decoder_target = sentences2onehot(sentences, Vocab, batch_size, decoder_max_len)
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
    #decoder_input, decoder_target = sentences2onehot(sentences, Vocab, batch_size, decoder_max_len)
    decoder_max_len = max(decoder_lens)
    decoder_input, decoder_target = sentences2ids(sentences, Vocab, batch_size, decoder_max_len)
    
    return encoder_input, decoder_input, decoder_target, decoder_lens

#data_path = "/dhome/b02902030/ADLxMLDS/hw2/MLDS_hw2_data/"
data_path = sys.argv[1]
if data_path[-1] != '/':
    data_path += '/'
vocab_size = 4000

t1 = time.time()
Vocab, train_list, test_list, X_train, X_test, train_label, test_label = read_data(data_path, vocab_size)
t2 = time.time()
print("Read Data Time: %fs" %(t2-t1))
print("========= Preprocessing done =========")


############ Build model ##########################################
num_units = 256
embedding_size = 300
lr = 1e-4

print("num_units: %d, embedding_size: %d, lr: %f" %(num_units, embedding_size, lr))

# placeholder
tf_encoder_input = tf.placeholder(tf.float32, shape=[None, 80, 4096]) # (batch_size, 80, 4096)
tf_decoder_input = tf.placeholder(tf.int32, shape=[None, None]) # (batch_size, decoder_max_len)
tf_decoder_target = tf.placeholder(tf.int32, shape=[None, None]) # (batch_size, decoder_max_len)
tf_decoder_seq_len = tf.placeholder(tf.int32, shape=[None]) # (batch_size), each length of sentences 
tf_decoder_max_len = tf.reduce_max(tf_decoder_seq_len)

# Encoder
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, tf_encoder_input, dtype=tf.float32)

# Decoder
# Embedding
embedding_decoder = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size], stddev=0.1))
with tf.device('/cpu:0'):
    emb_decoder_input = tf.nn.embedding_lookup(embedding_decoder, tf_decoder_input)
# decoder
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
train_helper = tf.contrib.seq2seq.TrainingHelper(emb_decoder_input, tf_decoder_seq_len)
projection_layer = Dense(vocab_size)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, encoder_state, output_layer=projection_layer)
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output

# loss and train_op
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_decoder_target, logits=logits)
decoder_mask = tf.sequence_mask(tf_decoder_seq_len, tf_decoder_max_len, dtype=logits.dtype)
#loss = (tf.reduce_sum(cross_entropy*decoder_mask)/tf.to_float(batch_size))
loss = tf.reduce_sum(cross_entropy*decoder_mask)
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Inference
tf_bos = tf.placeholder(tf.int32, shape=[None]) # (batch_size), [0,0,0...]
tf_eos = 1
inf_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, tf_bos, tf_eos) # no embedding
decoder_inf = tf.contrib.seq2seq.BasicDecoder(decoder_cell, inf_helper, encoder_state, output_layer=projection_layer)
outputs_inf, _, _, = tf.contrib.seq2seq.dynamic_decode(decoder_inf, maximum_iterations=50)
outputs_inf_words = outputs_inf.sample_id

t3 = time.time()
print("Build Graph Time: %fs" %(t3-t2))
###################################################################################

def evaluate(output_id):
    bleu = []
    for i, v in enumerate(test_list):
        bleu_v = []
        s = []
        for idx in output_id[i]:
            s.append(Vocab[idx])
        s = ' '.join(s)
        for cap in test_label[v]:
            bleu_v.append(BLEU(s, cap))
        bleu.append(np.mean(bleu_v))
    return np.mean(bleu)

epoch_num = 100
train_len = len(X_train)
test_len = len(X_test)
batch_size = 32
train_batch_num = int(train_len/batch_size)+1
test_batch_num = int(test_len/batch_size)+1
print("Training batch number: %d" %(train_batch_num))
print("Testing batch number: %d" %(test_batch_num))

encoder_input_test, decoder_input_test, decoder_target_test, decoder_lens_test = gen_test_data(test_list, X_test, test_label, Vocab)

save_path = "/home/aria0/ADLxMLDS2017/hw2/model/lstm_256_simple/"
saver = tf.train.Saver(max_to_keep=None)
with tf.Session() as sess:
    datetime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    with open(save_path+"train_log.txt", "a+") as fw:
        fw.write('\n'+str(datetime)+'\n')

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for epoch in range(1, epoch_num+1):
            print("Epoch: %d" %(epoch))
            # Train
            train_total_loss = 0.0
            for b in range(train_batch_num):
                encoder_input, decoder_input, decoder_target, decoder_lens = \
                    gen_batch(train_list[b*batch_size:(b+1)*batch_size], X_train, train_label, Vocab)
            
                _, train_loss = sess.run([train_op, loss], {tf_encoder_input: encoder_input,
                                                            tf_decoder_input: decoder_input,
                                                            tf_decoder_target: decoder_target,
                                                            tf_decoder_seq_len: decoder_lens})
                train_total_loss += train_loss

            train_total_loss /= len(train_list)
            
            # Inference
            output_sentences_id, test_loss = sess.run([outputs_inf_words, loss], 
                                                        {tf_encoder_input: encoder_input_test,
                                                         tf_decoder_input: decoder_input_test,
                                                         tf_decoder_target: decoder_target_test,
                                                         tf_decoder_seq_len: decoder_lens_test,
                                                         tf_bos: np.zeros(100, dtype=np.int32)})
            #print(output_sentences.shape)
            # (100, 50)??
            test_loss /= len(test_list)
            bleu_score = evaluate(output_sentences_id)
            
            '''
            s = []
            for idx in output_sentences_id[0]:
                s.append(Vocab[idx])
            s = ' '.join(s)
            print(s)
            '''

            # log
            print("Train loss: %.4f | Test loss: %.4f | BLEU: %.4f" %(train_total_loss, test_loss, bleu_score))
            fw.write("Epoch: %d | Train loss: %.4f | Test loss: %.4f | BLEU: %.4f\n" %(epoch, train_total_loss, test_loss, bleu_score))
            fw.flush()

            saver.save(sess, save_path+"epoch", global_step=epoch)




