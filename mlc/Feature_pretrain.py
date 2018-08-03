import numpy as np
import theano
from theano import tensor as T
import lasagne
import random as random
import pickle
import cPickle
import torchfile
import scipy.io
import numpy as np
import time
import sys
from random import sample

flag = 0
if ('bookmarks' not in sys.argv[1]) and ('.mat' not in sys.argv[1]):
	o = torchfile.load(sys.argv[1])
	trainX = o['data'].astype('float32')
	trainY = o['labels'].astype('int32')

	o = torchfile.load(sys.argv[2])
	devX = o['data'].astype('float32')
	devY = o['labels'].astype('int32')

elif '.mat' in sys.argv[1]:
       
        flag =1   
        o = scipy.io.loadmat(sys.argv[1])
        trainX = o['data']
        trainY = o['label']
        L = trainX.shape[0]

        random.seed(1)
        np.random.seed(1)
        indices = sample(range(L),int(L/5))
        devX = trainX[indices]
        devY = trainY[indices]
        indices_t = list(set(range(L))-set(indices))
        trainX = trainX[indices_t]
        trainY = trainY[indices_t]

else:
        f = open('icml_mlc_data/data/bookmarks/bookmarks.pickle')
        trainX, trainY, devX, devY, testX, testY = pickle.load(f)
        f.close()



hidden1 = int(sys.argv[3])
hidden2 = int(sys.argv[4])

random.seed(1)
np.random.seed(1)
eps = 0.0000001

def saveParams(para, fname):
        f = file(fname, 'wb')
        cPickle.dump(para, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


def get_minibatches_idx(n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")
        if shuffle:
            np.random.shuffle(idx_list)
        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)


def pretrain():
		eta = 0.001
		L2 = 0.000001
		num_lables = trainY.shape[1]
		num_features = trainX.shape[1]
                l_in = lasagne.layers.InputLayer((None, num_features))
                l_y1 = lasagne.layers.DenseLayer(l_in, hidden1)
                l_y2 = lasagne.layers.DenseLayer(l_y1, hidden2)
                l_y = lasagne.layers.DenseLayer(l_y2, num_lables, b = None, nonlinearity = lasagne.nonlinearities.linear)

                g1 = T.fmatrix()
                y = T.imatrix()
                y1 = T.ones_like(y)
                c_params = lasagne.layers.get_all_params(l_y, trainable=True)
                output = lasagne.layers.get_output(l_y, {l_in:g1})
                loss = -output*y + T.log(1+T.exp(output))
                loss = T.mean(loss)
                tmp_l = sum(lasagne.regularization.l2(x) for x in c_params)
		loss = loss + L2 * tmp_l
                pred = T.gt(output, 0)
                pg = T.eq(pred, y)
                prec = 1.0*(T.sum(pg*y, axis =1) + eps) / (T.sum(pred, axis =1) + eps)
                recall = 1.0 *(T.sum(pg*y, axis=1) + eps)/(T.sum(y, axis =1) + eps)
		
                updates = lasagne.updates.adam(loss, c_params, eta)
                train_function = theano.function([g1, y], loss, updates=updates)
                test_function = theano.function([g1, y], [loss, prec, recall, tmp_l])

                start_time = time.time()
                bestdev =0
                bestdev_time =0
                counter = 0
                try:
                       for eidx in xrange(int(sys.argv[5])):
                                start_time1 = time.time()
                                n_samples = 0
                                # Get new shuffled index for the training set.
                                kf = get_minibatches_idx(trainX.shape[0], 32, shuffle=True)
                                uidx = 0
                		for _, train_index in kf:
                                        uidx += 1
                                        if (flag ==0):
                                                x0 = trainX[train_index].astype('float32')
                                                y0 = trainY[train_index].astype('int32')
                                        else:
                                                x0 = np.asarray(trainX[train_index].todense()).astype('float32')
                                                y0 = np.asarray(trainY[train_index].todense()).astype('int32')
                                        n_samples += len(train_index)
                  
                    			traincost = train_function(x0, y0)
                    			if np.isnan(traincost) or np.isinf(traincost):
                        			print 'NaN detected'

                                end_time1 = time.time()		
                                start_time2 = time.time()
                                testacc = []
                                testrecall = []
                                devkf = get_minibatches_idx(devX.shape[0], 1024)
                                for _, dev_index in devkf:
                                        if (flag ==0):
                                                devx0 = devX[dev_index].astype('float32')
                                                devy0 = devY[dev_index].astype('int32')
                                        else:
                                                devx0 = np.asarray(devX[dev_index].todense()).astype('float32')
                                                devy0 = np.asarray(devY[dev_index].todense()).astype('int32')
                                        _, testprec0, testrecall0, _  = test_function(devx0, devy0)
                                        testacc += testprec0.tolist()
                                        testrecall += testrecall0.tolist()
                                testacc = np.asarray(testacc)
                                testrecall = np.asarray(testrecall)
                                a0 = 2 * testacc * testrecall
                                a1 = testacc + testrecall
                                testf1 =  np.divide(a0, a1)
                                testf1 = np.mean(testf1)
                                testacc = np.mean(testacc)
                                testrecall = np.mean(testrecall)				
                                end_time2 = time.time()
				
                                print 'Epoch ', (eidx+1), 'Cost ', traincost , 'testprec ', testacc, 'testrecall', testrecall, 'testf1', testf1, 'traing time', (end_time1 - start_time1), 'test time', (end_time2 - start_time2)
                                print ' '
                                if testf1 > bestdev:
                                        bestdev = testf1
                                        bestdev_time = eidx + 1
                                        tmp_para = lasagne.layers.get_all_param_values(l_y)
                                        saveParams( tmp_para , 'mlc_'+ sys.argv[4]+ '_'+ sys.argv[5] +'.pickle')
                     
                                print "Seen samples: " , n_samples

                except KeyboardInterrupt:
                       print "Training interupted"
                end_time = time.time()
                print "total time:", (end_time - start_time)    
                print "best dev acc:", bestdev, ' at time:  ', bestdev_time
		
if __name__ == "__main__":
                pretrain()
