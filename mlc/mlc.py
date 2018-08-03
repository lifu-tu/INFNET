import torchfile
import numpy as np
import pickle
from mlc_model import MLC_model
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument("--eta", help="Learning rate", type=float, default=0.001)
parser.add_argument("--minibatchsize", help="Size of batch when training", type=int, default=32)
parser.add_argument("--L2", help="L2 regularization", type=float, default=0.01)
parser.add_argument("--hidden1", help="first layer hidden size of local energy", type=int, default=150)
parser.add_argument("--hidden2", help="second layer hidden size of local energy ", type=int, default=150)
parser.add_argument("--C1", help="hidden size of global energy ", type=int, default=16)
parser.add_argument("--hidden1_a", help="first layer hidden size of inference network", type=int, default=150)
parser.add_argument("--hidden2_a", help="second layer hidden size of inference network", type=int, default=150)
parser.add_argument("--trainfile", help="training file", default="icml_mlc_data/data/bibtex/bibtex-train.torch")
parser.add_argument("--devfile", help="dev file", default="icml_mlc_data/data/bibtex/bibtex-test.torch")
parser.add_argument("--testfile", help="test file", default="icml_mlc_data/data/bibtex/bibtex-test.torch")
parser.add_argument("--FeatureNet", help="feature network pretrain model", default="mlc_150_10.pickle")
parser.add_argument("--infNet", help="inference network pretrain model", default="mlc_150_10.pickle")
parser.add_argument("--margin_type", help="different traing method  0:margin rescaling, 1:contrastive, 2:perceptron, 3: slack rescaling", type=int, default=0)
parser.add_argument("--regu_pretrain", help="Learning rate", type=float, default=10)
params = parser.parse_args()

if 'bookmarks' not in params.trainfile:
                o = torchfile.load(params.trainfile)
                trainX = o['data']
                trainY = o['labels']
                print trainX.shape

		o = torchfile.load(params.devfile)
		devX = o['data'].astype('float32')
		devY = o['labels'].astype('int32')
	
                o = torchfile.load(params.testfile)
                testX = o['data'].astype('float32')
                testY = o['labels'].astype('int32')
else:
                f = open('icml_mlc_data/data/bookmarks/bookmarks.pickle')
                trainX, trainY, devX, devY, testX, testY = pickle.load(f)
                f.close()


num_lables = trainY.shape[1]
num_features = trainX.shape[1]
print num_features, num_lables

params.outfile = 'marginType_'+ str(params.margin_type) + '_hingefactor_' + str(params.eta)+ '_regu_pretrained_' +  str(params.regu_pretrain) +  '_L2_'  + str(params.L2)+'_minibatchsize_'  +str(params.minibatchsize)+ '_hidden1_' + str(params.hidden1)+'_hidden2_'  +str(params.hidden2)+'_hidden1_a_' +str(params.hidden1_a)+ '_hidden2a_' + str(params.hidden2_a) +'_C1_' + str(params.C1)+'_' + str(params.trainfile[-16:-6])
	
	
model = MLC_model(params, num_lables, num_features)
model.train(trainX, trainY, devX, devY, testX, testY, params)
