import numpy as np
from params import params
import theano
from theano import tensor as T
import lasagne
import random as random
import pickle
import cPickle
import torchfile
import numpy as np
import time
import sys
import sparsemax_theano

eps = 0.0000001
random.seed(1)
np.random.seed(1)


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



class LM_model(object):

        def prepare_data(self, seqs):
                lengths = [len(s)+1 for s in seqs]
                n_samples = len(seqs)
                maxlen = np.max(lengths)
                sumlen = sum(lengths)
                x = np.zeros((n_samples, maxlen)).astype('int32')
                x_out = np.zeros((n_samples, maxlen)).astype('int32')
                x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
                for idx, s in enumerate(seqs):
                        x[idx,:lengths[idx]] = [45]+ s
                        x_mask[idx,:lengths[idx]] = 1.
                        x_out[idx, :lengths[idx]] = s + [45]
		
                tmp = x.flatten()
                y = np.zeros((n_samples*maxlen, 46))
                y[np.arange(n_samples*maxlen), tmp] = 1
                x_start = y.reshape((n_samples, maxlen, 46)).astype('float32')
                return x_start, x_mask ,x_out, maxlen

        def get_idxs(self, xmask):
                tmp = xmask.reshape(-1,1)
                idxs = []
                for i in range(len(tmp)):
                        if tmp[i] > 0:
                                idxs.append(i)
                return np.asarray(idxs).astype('int32')
	

        def __init__(self,   params):
		self.textfile = open(params.outfile+ 'lm', 'w')

        	hidden = params.hiddenlm


                l_in = lasagne.layers.InputLayer((None, None, 46))
                l_mask = lasagne.layers.InputLayer(shape=(None, None))
            
                l_lstm_f = lasagne.layers.LSTMLayer(l_in, hidden, mask_input=l_mask)
                #l_lstm_b = lasagne.layers.LSTMLayer(l_in, hidden, mask_input=l_mask, backwards = True)

                l_reshapef = lasagne.layers.ReshapeLayer(l_lstm_f,(-1,hidden))
                #l_reshapeb = lasagne.layers.ReshapeLayer(l_lstm_b,(-1,hidden))
                #concat2 = lasagne.layers.ConcatLayer([l_reshapef, l_reshapeb])
                #l_emb = lasagne.layers.DenseLayer(concat2, num_units=hidden, nonlinearity=lasagne.nonlinearities.tanh)
                l_local = lasagne.layers.DenseLayer(l_reshapef, num_units= 46, nonlinearity=lasagne.nonlinearities.softmax)
		
		

		lm_params = lasagne.layers.get_all_params(l_local, trainable=True)
		self.lm_params = lm_params
		print lm_params
             

		
			
		y_out = T.imatrix()		
		y_in = T.ftensor3()
            
                gmask = T.fmatrix()
		length = T.iscalar()	
	
	
		predy0 = lasagne.layers.get_output(l_local, {l_in:y_in, l_mask:gmask})
		y = y_out.flatten()
	
		predy0 = T.clip(predy0, 0.0000001, 1.0 - 0.0000001)
	
		cost =  lasagne.objectives.categorical_crossentropy(predy0, y)		
		cost = cost.reshape((-1, length))
		
		cost1 = T.sum(cost, axis=1)/gmask.sum(axis=1)
		
		cost = T.mean(cost1)

		predy = predy0.reshape((-1, length, 46))
		
	
		pred = T.argmax(predy, axis=2)	
		pg = T.eq(pred, y_out)		
		pg = pg*gmask

		acc = 1.0* T.sum(pg)/ T.sum(gmask)

		reg = sum(lasagne.regularization.l2(x) for x in lm_params)
		cost = cost + params.L2 * reg		

		updates = lasagne.updates.adam(cost, lm_params, params.eta)		


		self.train_lm = theano.function([y_in, gmask, y_out, length] , [cost, acc], updates=updates)
		self.test_lm = theano.function([y_in, gmask, y_out, length] , [cost, acc])
						

		
	def train(self, trainX, devX, params):	

		devx0, devx0mask, devy0,  devmaxlen = self.prepare_data(devX)	
		start_time = time.time()
        	bestdev = -1
        	bestdev_time =0
        	counter = 0
        	try:
            		for eidx in xrange(100):
                		n_samples = 0

                		start_time1 = time.time()
                		kf = get_minibatches_idx(len(trainX), params.batchsize, shuffle=True)
                		uidx = 0
				aa = 0
				bb = 0
                		for _, train_index in kf:

                    			uidx += 1

                    			x0 = [trainX[ii] for ii in train_index]
                    			n_samples += len(train_index)

					x0, x0mask, y0, maxlen = self.prepare_data(x0)					
					traincost, trainacc = self.train_lm(x0, x0mask, y0, maxlen)
                 							
					if np.isnan(traincost):
                        			#print 'NaN detected'
						self.textfile.write("NaN detected \n")
						self.textfile.flush()

				end_time1 = time.time()
				bestdev0 = 0
				best_t = 0

				
								
				start_time2 = time.time()
				devcost, devacc  = self.test_lm(devx0, devx0mask, devy0, devmaxlen)
				end_time2 = time.time()
				if bestdev < devacc:
					bestdev = devacc
					best_t = eidx
					tmp_a_para = [p.get_value() for p in self.lm_params]
                                	saveParams( tmp_a_para , params.outfile + 'lm.pickle')			

					
				#self.textfile.write("epoches %d energy_Cost %f devacc %f devrecall %f devf1 %f testf1 %f trainig time %f test time %f \n" %( eidx + 1, energy_cost ,  best_prec, best_recall,  bestdev0, testf1, end_time1 - start_time1, end_time2 - start_time2 ) )
				self.textfile.write("epoches %d devcost %f  devacc %f trainig time %f test time %f \n" %( eidx + 1, devcost, devacc, end_time1 - start_time1, end_time2 - start_time2 ) )
				self.textfile.flush()
				#self.textfile.write("epoches %d energy_Cost %f devacc %f devrecall %f devf1 %f  training time %f\n" %( eidx + 1, energy_cost ,  best_prec, best_recall,  bestdev0, end_time1 - start_time1) )
			       
        	except KeyboardInterrupt:
            		#print "Classifer Training interupted"
            		self.textfile.write( 'classifer training interrupt \n')
			self.textfile.flush()
        	end_time = time.time()
		#self.textfile.write("total time %f \n" % (end_time - start_time))
		self.textfile.write("best dev acc: %f  at time %d \n" % (bestdev, best_t))
        	self.textfile.close()
		
