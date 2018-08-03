import sys
import warnings
from utils import getWordmap
from params import params
from utils import getData
from utils import getTagger
import random
import numpy as np
from base_model import base_model
random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l2):
        params.outfile = 'Base_model'
        params.dataf = 'data/oct27.traindev.proc.cnn'
        params.dev = 'data/oct27.test.proc.cnn'
        params.test = 'data/daily547.proc.cnn'
        params.batchsize = 10
        params.hidden = 100
        params.embedsize = 100
        params.eta = eta
        params.L2 = l2
        params.dropout = 0
        params.frac = 0.1
        params.emb =0	

        (words, We) = getWordmap('wordvects.tw100w5-m40-it2')
        We = np.asarray(We).astype('float32')
        tagger = getTagger('data/tagger')
        print tagger
        params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_'+"LearningRate"+'_'+str(params.eta)+ '_' + str(params.hiddensize)+ '_' + str(l2) + '.pickle'
                      	
        traindata = getData(params.dataf, words, tagger)
        trainx0, trainy0 = traindata	
        devdata = getData(params.dev, words, tagger)
        devx0, devy0 = devdata
        testdata = getData(params.test, words, tagger)
        testx0, testy0 = testdata	

        tm = base_model(We, params)
        tm.train(traindata, devdata, testdata, params)

if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]) )
