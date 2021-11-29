from mlp_w import MLP
import numpy as np

import pickle, gzip

f = gzip.open('../data/mnist.pkl.gz', 'rb')
tset, vset, teset = pickle.load(f, encoding='latin1')
print(tset[0].shape, vset[0].shape, teset[0].shape)
f.close()

tread = 9000
train_in = tset[0][:tread, :]

# This is a little bit of work -- 1 of N encoding
# Make sure you understand how it does it
train_tgt = np.zeros((tread, 10))
for i in range(tread):
    train_tgt[i, tset[1][i]] = 1

# and use 1000 images for testing
teread = 1000
test_in = teset[0][:teread, :]
test_tgt = np.zeros((teread, 10))
for i in range(teread):
    test_tgt[i, teset[1][i]] = 1

sizes = [784, 15, 15, 10]  # 784 is the number of pixels of the images and 10 is the number of classes
classifier = MLP(sizes, beta=5, momentum=0.9)
classifier.train(train_in, train_tgt, 0.01, 10000)
