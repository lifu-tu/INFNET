import sys
import warnings
from utils import getWordmap
from utils import getData
from utils import getTagger
import random
import numpy as np
from build_CRF_inference_LM import CRF_model

random.seed(1)
np.random.seed(1)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--eta", help="Learning rate", type=float, default=0.02)
parser.add_argument("--l3", help="regularization hyperparater", type=float, default=5)
parser.add_argument("--lm", help="language model regularization hyperparater", type=float, default=0.1)
parser.add_argument("--batchsize", help="Size of batch when training", type=int, default=10)
parser.add_argument("--emb", help="0:fix embedding, 1:update embedding", type=int, default=0)
parser.add_argument("--dropout", help="dropout", type=float, default=0)
parser.add_argument("--regu_type", help="0:local cross entropy, 1:entropy 2:Regularization Toward Pretrained Inference Network", type=int, default=0)
parser.add_argument("--annealing", help="0:fixed weght, 1:weight annealing", type=int, default=0)


params = parser.parse_args()

params.dataf = '../pos_data/oct27.traindev.proc.cnn'
params.dev = '../pos_data/oct27.test.proc.cnn'
params.test = '../pos_data/daily547.proc.cnn'
params.hidden = 100
params.embedsize = 100
params.num_labels = 25
(words, We) = getWordmap('wordvects.tw100w5-m40-it2')
We = np.asarray(We).astype('float32')
tagger = getTagger('../pos_data/tagger')
params.outfile = "CRF_infnet_LM_Batchsize"+'_'+str(params.batchsize)+'_dropout_'+ str(params.dropout) + "_LearningRate"+'_'+str(params.eta)+ '_' + str(params.l3)+ str(params.lm) + '_emb_'+ str(params.emb)
traindata = getData(params.dataf, words, tagger)
trainx0, trainy0 = traindata
devdata = getData(params.dev, words, tagger)
devx0, devy0 = devdata
testdata = getData(params.test, words, tagger)
testx0, testy0 = testdata	

tm = CRF_model(We, params)
tm.train(trainx0, trainy0, devx0, devy0, testx0, testy0, params)
