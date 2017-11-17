import numpy as np
import json, re, time, random, os, sys
from time import gmtime, strftime
from collections import Counter
import tensorflow as tf
from tensorflow.python.layers.core import Dense
############# Read label and data ###################
def read_data(data_path, vocab_size):
    #t1 = time.time()
    regex = re.compile('[^a-zA-Z]')
    Vocab = []
    bos = '<BOS>'
    eos = '<EOS>'
    unk = '<UNK>'
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
            if len(s)>=5 and len(s)<=35:
                test_label[d[i]['id']].append(' '.join(s))

    train_list = list(train_label.keys())
    test_list = list(test_label.keys())

    Vocab = dict(Counter(Vocab))
    Vocab = sorted(Vocab, key=Vocab.get, reverse=True)
    Vocab = Vocab[:vocab_size-3]
    Vocab = [bos, eos, unk] + Vocab

    # Read training data
    X_test = {}
    for v in test_list:
        X_test[v] = np.load(data_path+'testing_data/feat/'+v+'.npy').astype(np.float32)

    peer_list = []
    with open(data_path+"peer_review_id.txt") as fp:
        for f in fp:
            peer_list.append(f.strip('\n'))
    X_peer = {}
    for v in peer_list:
        X_peer[v] = np.load(data_path+'peer_review/feat/'+v+'.npy').astype(np.float32)
        
    return Vocab, test_list, X_test, test_label, peer_list, X_peer

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
    encoder_input_test = np.zeros((test_size, 80, 4096), dtype=np.float32)
    for t in range(test_size):
        encoder_input_test[t] = X_test[test_list[t]]
    sentences = []
    decoder_lens_test = []
    for t in range(test_size):
        sentences.append(random.choice(test_label[test_list[t]]))
        decoder_lens_test.append(len(sentences[t].split())+1)
    decoder_max_len = max(decoder_lens_test)
    decoder_input_test, decoder_target_test = sentences2ids(sentences, Vocab, test_size, decoder_max_len)

    return encoder_input_test, decoder_input_test, decoder_target_test, decoder_lens_test

def gen_peer_data(peer_list, X_peer):
    peer_size = len(peer_list)
    encoder_input_peer = np.zeros((peer_size, 80, 4096), dtype=np.float32)
    for t in range(peer_size):
        encoder_input_peer[t] = X_peer[peer_list[t]]

    return encoder_input_peer

################# Setting ######################
data_path = sys.argv[1] + '/' if sys.argv[1][-1]!='/' else sys.argv[1]
#model_path = sys.argv[2] + '/' if sys.argv[2][-1]!='/' else sys.argv[2]

vocab_size = 4000

t1 = time.time()
Vocab, test_list, X_test, test_label, peer_list, X_peer = read_data(data_path, vocab_size)
t2 = time.time()


############ Build model ##########################################
num_units = 512
num_layers = 3
embedding_size = 300
lr = 1e-4*0.8
###################################################################################
mode = sys.argv[2] # all, sp
model_file = sys.argv[3]
output_file = sys.argv[4]
pr_output_file = sys.argv[5]

encoder_input_test, decoder_input_test, decoder_target_test, decoder_lens_test = gen_test_data(test_list, X_test, test_label, Vocab)
encoder_input_peer = gen_peer_data(peer_list, X_peer)

with tf.Session() as sess:

    #model_file = './test_model/epoch_61'
    saver = tf.train.import_meta_graph(model_file+'.meta')
    saver.restore(sess, model_file)
    outputs_inf_words = tf.get_collection('for_test')[0]
    loss = tf.get_collection('for_test')[1]
    tf_encoder_input = tf.get_collection('for_test')[2]
    tf_decoder_input = tf.get_collection('for_test')[3]
    tf_decoder_target = tf.get_collection('for_test')[4]
    tf_decoder_seq_len = tf.get_collection('for_test')[5]
    tf_bos = tf.get_collection('for_test')[6]
    tf_prob = tf.get_collection('for_test')[7]
    
    output_sentences_id, test_loss = sess.run([outputs_inf_words, loss], 
                                                {tf_encoder_input: encoder_input_test,
                                                 tf_decoder_input: decoder_input_test,
                                                 tf_decoder_target: decoder_target_test,
                                                 tf_decoder_seq_len: decoder_lens_test,
                                                 tf_prob: 1,
                                                 tf_bos: np.zeros(100, dtype=np.int32)})

    sens = []
    for i in range(100):
        s = []
        for idx in output_sentences_id[i]:
            if idx > 2:
                s.append(Vocab[idx])
        s = ' '.join(s)
        sens.append(s)

    if mode == 'all':
        with open(output_file, "w") as fw:
            for i in range(100):
                fw.write(test_list[i]+','+sens[i]+'\n')
    elif mode == 'sp':
        with open(output_file, "w") as fw:
            sp_video = ["klteYv1Uv9A_27_33.avi", "5YJaS2Eswg0_22_26.avi", "UbmZAe5u5FI_132_141.avi", "JntMAcTlOF0_50_70.avi", "tJHUH9tpqPg_113_118.avi"]
            for v in sp_video:
                fw.write(v+','+sens[test_list.index(v)]+'\n')

    ######## peer review #########
    output_sentences_id = sess.run(outputs_inf_words, 
                                    {tf_encoder_input: encoder_input_peer,
                                     tf_prob: 1,
                                     tf_bos: np.zeros(len(peer_list), dtype=np.int32)})
    sens = []
    for i in range(len(peer_list)):
        s = []
        for idx in output_sentences_id[i]:
            if idx > 2:
                s.append(Vocab[idx])
        s = ' '.join(s)
        sens.append(s)

    with open(pr_output_file, "w") as fw:
        for i in range(len(peer_list)):
            fw.write(peer_list[i]+','+sens[i]+'\n')
            







