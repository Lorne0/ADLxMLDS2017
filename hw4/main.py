import os, sys
import pickle as pk
from utils import *
from gd_models import *

class DCGAN(object):
    def __init__(self, tags, imgs, mode):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
        self.sess = tf.Session(config=config)

        self.lr = 0.0002
        
        if mode=='train':
            self.build_model()
            self.sess.run(tf.global_variables_initializer())

            self.tags = tags
            self.imgs = imgs

        else: # test
            self.build_model()
            saver = tf.train.Saver()
            saver.restore(self.sess, "./save_model/dcgan")
            

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [None, 123]) # 100 z~N(0,1) + 23 tag vectors
        self.img = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.seq = tf.placeholder(tf.float32, [None, 23])
        self.wrong_img = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.wrong_seq = tf.placeholder(tf.float32, [None, 23])

        self.fake_img, self.g_vars = build_generator(self.inputs)
        # d0: real img, right seq  -> 1
        # d1: fake img, right seq  -> 0
        # d2: real img, wrong seq  -> 0
        # d3: wrong img, right seq -> 0
        self.d0, self.d_vars = build_discriminator(self.img, self.seq, reuse=False)
        self.d1, _ = build_discriminator(self.fake_img, self.seq)
        self.d2, _ = build_discriminator(self.img, self.wrong_seq)
        self.d3, _ = build_discriminator(self.wrong_img, self.seq)
        
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d1, labels=tf.ones_like(self.d1))) 

        d0_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d0, labels=tf.ones_like(self.d0)))
        d1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d1, labels=tf.zeros_like(self.d1)))
        d2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d2, labels=tf.zeros_like(self.d2)))
        d3_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d3, labels=tf.zeros_like(self.d3)))
        self.d_loss = d0_loss + (d1_loss+d2_loss+d3_loss)/3

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_updates = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.d_vars)
            self.g_updates = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=self.g_vars)

    def test(self):
        f = [('pink hair', 'aqua eyes'), ('white hair', 'red eyes'), ('purple hair', 'black eyes')]
        #f = [('red hair', 'green eyes')]
        np.random.seed(1229)
        for k in range(3):
            z = np.random.randn(1, 100) # generate 32x100 normal random vector
            #z = np.random.random((1, 100)) # generate 32x100 normal random vector
            for i in range(len(f)):
                test_seq = np.zeros((1, 23))
                test_seq[0][attributes.index(f[i][0])] = 1
                test_seq[0][attributes.index(f[i][1])] = 1
                inputs = np.hstack((z, test_seq)) # generator inputs, 32x(100+23)
                f_img = self.sess.run(self.fake_img, feed_dict={self.inputs: inputs})
                f_img = (f_img+1.0)/2*255.0
                f_img = f_img.astype(np.uint8)[0]
                fn = '_'.join(f[i][0].split(' '))+'_'+'_'.join(f[i][1].split(' '))
                scipy.misc.imsave('./samples/s_'+str(k)+'_'+fn+'.jpg', f_img)
                #scipy.misc.imsave('./early/sample_'+str(i+1)+'_'+str(k+1)+'.jpg', f_img)
        

    def train(self):
        data_size = 33431
        epochs = 500
        batch_size = 64
        B = int(data_size/batch_size)
        update_times = 0
        saver = tf.train.Saver()
        for e in range(1, epochs+1):
            print('Epoch: %d' %(e))
            g_cost, d_cost = 0, 0
            ids = np.arange(data_size)
            np.random.shuffle(ids)
            for b in range(B):
                seq = self.tags[ids[b*batch_size:(b+1)*batch_size]].copy() # right seq
                img = self.imgs[ids[b*batch_size:(b+1)*batch_size]].copy() # real img

                z = np.random.randn(batch_size, 100) # generate 32x100 normal random vector
                inputs = np.hstack((z, seq)) # generator inputs, 32x(100+23)
                
                wrong_seq = np.zeros_like(seq)
                wrong_img = np.zeros_like(img)
                # wrong img, wrong text
                for i in range(batch_size):
                    while True: # sample until they're different
                        k = np.random.randint(data_size)
                        a = self.tags[k].copy()
                        if np.array_equal(a, seq[i])==False:
                            wrong_seq[i] = a
                            wrong_img[i] = self.imgs[k].copy()
                            break

                # update discriminator
                feed_dict = {self.inputs: inputs, self.seq: seq, self.img: img, self.wrong_seq: wrong_seq, self.wrong_img: wrong_img}
                _, loss = self.sess.run([self.d_updates, self.d_loss], feed_dict=feed_dict)
                d_cost += loss

                # update generator
                z = np.random.randn(batch_size, 100) # generate 32x100 normal random vector
                inputs = np.hstack((z, seq)) # generator inputs, 32x(100+23)
                feed_dict = {self.inputs: inputs, self.seq: seq, self.img: img, self.wrong_seq: wrong_seq, self.wrong_img: wrong_img}
                _, loss = self.sess.run([self.g_updates, self.g_loss], feed_dict=feed_dict)
                g_cost += loss

            print("G loss: %f | D loss: %f" %(g_cost, d_cost))
            # save model
            save_path = saver.save(self.sess, "./save_model/dcgan")
            # test some images and save
            f = [('pink hair', 'aqua eyes'), ('white hair', 'red eyes'), ('purple hair', 'black eyes')]
            for i in range(3):
                test_seq = np.zeros((1, 23))
                test_seq[0][attributes.index(f[i][0])] = 1
                test_seq[0][attributes.index(f[i][1])] = 1
                z = np.random.randn(1, 100) # generate 32x100 normal random vector
                inputs = np.hstack((z, test_seq)) # generator inputs, 32x(100+23)
                f_img = self.sess.run(self.fake_img, feed_dict={self.inputs: inputs})
                f_img = (f_img+1.0)/2*255.0
                f_img = f_img.astype(np.uint8)[0]
                fn = '_'.join(f[i][0].split(' '))+'_'+'_'.join(f[i][1].split(' '))
                scipy.misc.imsave('./tmp_sample/s_'+str(e)+'_'+fn+'.jpg', f_img)


if __name__ == '__main__':
    if sys.argv[1]=='train': # train
        with open("./data/pretrained.pk", "rb") as fp:
            tags = pk.load(fp)
            imgs = pk.load(fp)
        print("Load data done")
        dcgan = DCGAN(tags=tags, imgs=imgs, mode='train')
        dcgan.train()
    else:
        dcgan = DCGAN(tags=None, imgs=None, mode='test')
        dcgan.test()
        






