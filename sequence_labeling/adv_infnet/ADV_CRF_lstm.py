import numpy as np
import theano
from theano import tensor as T
import lasagne
import random as random
import pickle
import cPickle

import time
import sys,os

from lasagne_embedding_layer_2 import lasagne_embedding_layer_2
from random import randint



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




class GAN_CRF_model(object):

	def prepare_data(self, seqs, labels):
		lengths = [len(s) for s in seqs]
                n_samples = len(seqs)
                maxlen = np.max(lengths)
                #sumlen = sum(lengths)

                x = np.zeros((n_samples, maxlen)).astype('int32')
                x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
                y = np.zeros((n_samples, maxlen)).astype('int32')
                for idx, s in enumerate(seqs):
                        x[idx,:lengths[idx]] = s
                        x_mask[idx,:lengths[idx]] = 1.
                        y[idx,:lengths[idx]] = labels[idx]


                tmp = y.flatten()
                ytmp = np.zeros((n_samples*maxlen, 25))
                ytmp[np.arange(n_samples*maxlen), tmp] = 1.0
                y_in = ytmp.reshape((n_samples, maxlen, 25)).astype('float32')


                return x, x_mask, y, y_in, maxlen
        

		

	def __init__(self,  We_initial,   params):
		self.textfile = open(params.outfile, 'w')
		We = theano.shared(We_initial)
        	embsize = We_initial.shape[1]
        	hidden = params.hidden
	

                l_in_word = lasagne.layers.InputLayer((None, None))
                l_mask_word = lasagne.layers.InputLayer(shape=(None, None))

		if params.emb ==1:
                        l_emb_word = lasagne.layers.EmbeddingLayer(l_in_word,  input_size= We_initial.shape[0] , output_size = embsize, W =We)
                else:
                        l_emb_word = lasagne_embedding_layer_2(l_in_word, embsize, We)

                l_lstm_wordf = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=l_mask_word)
                l_lstm_wordb = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=l_mask_word, backwards = True)

                l_reshapef = lasagne.layers.ReshapeLayer(l_lstm_wordf,(-1,hidden))
                l_reshapeb = lasagne.layers.ReshapeLayer(l_lstm_wordb,(-1,hidden))
		concat2 = lasagne.layers.ConcatLayer([l_reshapef, l_reshapeb])
		l_local = lasagne.layers.DenseLayer(concat2, num_units= 25, nonlinearity=lasagne.nonlinearities.linear)
                ### the above is for the uniary term energy
               
           
		
		if params.emb ==1:		
			f = open('F.pickle')
		else:
 			f = open('F0_new.pickle')



                para = pickle.load(f)
                f.close()
	
		f_params = lasagne.layers.get_all_params(l_local, trainable=True)
		
		for idx, p in enumerate(f_params):
      
                        p.set_value(para[idx])
            	

		
		Wyy0 = np.random.uniform(-0.02, 0.02, (26, 26)).astype('float32')
		
		Wyy = theano.shared(Wyy0)



		d_params = lasagne.layers.get_all_params(l_local, trainable=True)
		d_params.append(Wyy)
		self.d_params = d_params		
		
		

		
	
		l_in_word_a = lasagne.layers.InputLayer((None, None))
                l_mask_word_a = lasagne.layers.InputLayer(shape=(None, None))
	
		l_emb_word_a = lasagne_embedding_layer_2(l_in_word_a, embsize, l_emb_word.W)		

                #l_emb_word_a = lasagne.layers.EmbeddingLayer(l_in_word_a,  input_size=We_initial.shape[0] , output_size = embsize, W =We)
		if params.dropout:
			l_emb_word_a = lasagne.layers.DropoutLayer(l_emb_word_a, p=0.5)


                l_lstm_wordf_a = lasagne.layers.LSTMLayer(l_emb_word_a, hidden, mask_input=l_mask_word_a)
                l_lstm_wordb_a = lasagne.layers.LSTMLayer(l_emb_word_a, hidden, mask_input=l_mask_word_a, backwards = True)

                l_reshapef_a = lasagne.layers.ReshapeLayer(l_lstm_wordf_a ,(-1, hidden))
                l_reshapeb_a = lasagne.layers.ReshapeLayer(l_lstm_wordb_a ,(-1,hidden))
                concat2_a = lasagne.layers.ConcatLayer([l_reshapef_a, l_reshapeb_a])
		


		if params.dropout:
                	concat2_a = lasagne.layers.DropoutLayer(concat2_a, p=0.5)              

                l_local_a = lasagne.layers.DenseLayer(concat2_a, num_units= 25, nonlinearity=lasagne.nonlinearities.softmax)	
		
			
				
		a_params = lasagne.layers.get_all_params(l_local_a, trainable=True)
		
		self.a_params = a_params		
		
		
		if params.emb ==1:	
			f = open('F.pickle')
		else:
			f = open('F0_new.pickle')

                PARA = pickle.load(f)
                f.close()

                for idx, p in enumerate(a_params):
                        p.set_value(PARA[idx])		
	
		
		y_in = T.ftensor3()
	
                y = T.imatrix()
		g = T.imatrix()
                gmask = T.fmatrix()
		y_mask = T.fmatrix()
		length = T.iscalar()
		#y0 = T.ftensor3()
		# shape: n, L, 1
		#y1 = T.ftensor3()
		# shape: n, 1, 46

	
		predy0 = lasagne.layers.get_output(l_local_a, {l_in_word_a:g, l_mask_word_a:gmask})
		
		predy = predy0.reshape((-1, length, 25))
		#predy = predy * gmask[:,:,None]


		#newpredy = T.concatenate([predy, y0] , axis=2)

		# n , L, 46, 46
		# predy0: n, L, 25		

		# energy loss
		
		def inner_function( targets_one_step,  mask_one_step,  prev_label, tg_energy):
        		"""
        		:param targets_one_step: [batch_size, t]
        		:param prev_label: [batch_size, t]
        		:param tg_energy: [batch_size]
        		:return:
        		"""
			#newWyy = Wyy.dimshuffle('x', 0, 1)
			##prev_label0 = prev_label.dimshuffle(0, 'x', 1)	
			new_ta_energy = T.dot(prev_label, Wyy[:-1,:-1])
			#new_ta_energy = T.sum(new_ta_energy, axis=1)
			new_ta_energy = tg_energy + T.sum(new_ta_energy*targets_one_step, axis =1)
			tg_energy_t = T.switch(mask_one_step, new_ta_energy,  tg_energy)

        		return [targets_one_step, new_ta_energy]

    		# Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    		# but scan requires the iterable dimension to be first
    		# So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
		
		#U_shuffled = U.dimshuffle('x', 'x', 0)
		local_energy = lasagne.layers.get_output(l_local, {l_in_word: g, l_mask_word: gmask})
                local_energy = local_energy.reshape((-1, length, 25))
		local_energy = local_energy*gmask[:,:,None]

    		##feature_shuffled = pretrain_feature.dimshuffle(1, 0, 2, 3)
    		targets_shuffled = y_in.dimshuffle(1, 0, 2)
		masks_shuffled = gmask.dimshuffle(1, 0)
    		# initials should be energies_shuffles[0, :, -1, :]


    		target_time0 = targets_shuffled[0]
		initial_energy0 = T.dot(target_time0, Wyy[-1,:-1])


		length_index = T.sum(gmask, axis=1)-1
		length_index = T.cast(length_index, 'int32')



    		initials = [target_time0, initial_energy0]
    		[ _, target_energies], _ = theano.scan(fn=inner_function, outputs_info=initials, sequences=[targets_shuffled[1:], masks_shuffled[1:]])

	
		pos_end_target = y_in[T.arange(length_index.shape[0]), length_index]
		

    		pos_cost = target_energies[-1] + T.sum(T.sum(local_energy*y_in, axis=2)*gmask, axis=1) + T.dot( pos_end_target, Wyy[:-1,-1])
		##pos_cost = target_energies[-1]    
		check = T.sum(T.sum(local_energy*y_in, axis=2)*gmask, axis=1)



		negtargets_shuffled = predy.dimshuffle(1, 0, 2)
		negtarget_time0 = negtargets_shuffled[0]	
		
		

		
		#neginitial_energy0 = T.dot(y11, Wyy)
                neginitial_energy0 = T.dot(negtarget_time0, Wyy[-1,:-1])


                neginitials = [negtarget_time0, neginitial_energy0]
                [ _, negtarget_energies], _ = theano.scan(fn=inner_function, outputs_info=neginitials, sequences=[ negtargets_shuffled[1:], masks_shuffled[1:]])

		neg_end_target = predy[T.arange(length_index.shape[0]), length_index]
                neg_cost = negtarget_energies[-1] + T.sum(T.sum(local_energy*predy, axis=2)*gmask, axis=1) + T.dot(neg_end_target, Wyy[:-1,-1])
    	

	

		y_f = y.flatten()
                predy_f =  predy.reshape((-1, 25))
                ce_hinge = lasagne.objectives.categorical_crossentropy(predy_f+eps, y_f)
                ce_hinge = ce_hinge.reshape((-1, length))
		ce_hinge = T.sum(ce_hinge* gmask, axis=1)
              

                entropy_term = - T.sum(predy_f * T.log(predy_f + eps), axis=1)
                entropy_term = entropy_term.reshape((-1, length))
                entropy_term = T.sum(entropy_term*gmask, axis=1) 		

		
		delta0 = T.sum(abs((y_in - predy)), axis=2)*gmask
		delta0 = T.sum(delta0, axis=1)
	
		if (params.margin_type==1):
			hinge_cost = 1 + neg_cost  - pos_cost
		elif(params.margin_type==2):		
			hinge_cost =  neg_cost  - pos_cost

		elif (params.margin_type==0):
                        hinge_cost = delta0 + neg_cost  - pos_cost
                elif(params.margin_type==3):
                        hinge_cost =  delta0*(1.0 + neg_cost  - pos_cost)



		hinge_cost = hinge_cost * T.gt(hinge_cost, 0)

		d_cost = T.mean(hinge_cost)
		
		d_cost0 = d_cost
				
		l2_term = sum(lasagne.regularization.l2(x-PARA[index]) for index, x in enumerate(a_params))

		"""select different regulizer"""
                g_cost = -d_cost0 + params.l2* sum(lasagne.regularization.l2(x) for x in a_params) + params.l3*T.mean(ce_hinge)
               			
		d_cost = d_cost0 +  params.l2*sum(lasagne.regularization.l2(x) for x in d_params) 
	
		
			
	
		
		self.a_params = a_params	
		updates_g = lasagne.updates.sgd(g_cost, a_params, params.eta)
        	updates_g = lasagne.updates.apply_momentum(updates_g, a_params, momentum=0.9)

		self.train_g = theano.function([g, gmask, y, y_in, length], [g_cost, d_cost0, pos_cost, neg_cost, delta0, check], updates=updates_g, on_unused_input='ignore')	

	
		updates_d = lasagne.updates.adam(d_cost, d_params, 0.001)
                self.train_d = theano.function([g, gmask, y,  y_in, length], [d_cost, d_cost0, pos_cost, neg_cost, delta0, check], updates=updates_d, on_unused_input='ignore')
		
		
		




		# test the model and retuning the model

		predy_test = lasagne.layers.get_output(l_local_a, {l_in_word_a:g, l_mask_word_a:gmask}, deterministic=True)
		predy_test = predy_test.reshape((-1, length, 25))		
		pred = T.argmax(predy_test, axis=2)
                pg = T.eq(pred, y)
                pg = pg*gmask

                acc = 1.0* T.sum(pg)/ T.sum(gmask)
		
			
		negtargets_shuffled_test = predy_test.dimshuffle(1, 0, 2)
                negtarget_time0_test = negtargets_shuffled_test[0]


                
                neginitial_energy0_test = T.dot(negtarget_time0_test,  Wyy[-1,:-1])
                neginitials_test = [negtarget_time0_test, neginitial_energy0_test]
                [ _, negtarget_energies_test], _ = theano.scan(fn=inner_function, outputs_info=neginitials_test, sequences=[ negtargets_shuffled_test[1:], masks_shuffled[1:]])

		end_test_target = predy_test[T.arange(length_index.shape[0]), length_index]

                neg_cost_test = negtarget_energies_test[-1] + T.sum(T.sum(local_energy*predy_test, axis=2)*gmask, axis=1) + T.dot(end_test_target, Wyy[:-1,-1])

		"""ce regulizer"""
		test_cost = -T.mean(neg_cost_test)+ params.l3*T.mean(ce_hinge)
		
		test_updates = lasagne.updates.sgd(test_cost, a_params, params.eta)
		test_updates = lasagne.updates.apply_momentum(test_updates, a_params, momentum=0.9)
	
		self.test_time_turning = theano.function([g, gmask, y, length] , test_cost, updates = test_updates, on_unused_input='ignore')
		
		self.test_time1 = theano.function([g, gmask, y, y_in, length] , [acc, T.mean(neg_cost), T.mean(pos_cost), params.l3*T.mean(ce_hinge)], on_unused_input='ignore')
		self.test_time = theano.function([g, gmask, y, length] , acc)
						

		
	def train(self, trainX, trainY, devX, devY, testX, testY, params):	

		devx0, devx0mask, devy0, devy0_in, devmaxlen = self.prepare_data(devX, devY)
		testx0, testx0mask, testy0, testy0_in, testmaxlen = self.prepare_data(testX, testY)
		devacc  = self.test_time(devx0, devx0mask, devy0, devmaxlen)
		testacc  = self.test_time(testx0, testx0mask, testy0, testmaxlen)
		self.textfile.write("initial dev acc:%f  test acc: %f  \n" %(devacc, testacc)  )		
		self.textfile.flush()
	
		start_time = time.time()
        	bestdev = -1
		bestdev1 = -1
        	bestdev_time =0
        	counter = 0
        	try:
            		for eidx in xrange(200):
                        #for eidx in xrange(300)
                		n_samples = 0

                		start_time1 = time.time()
                		kf = get_minibatches_idx(len(trainX), params.batchsize, shuffle=True)
                		uidx = 0
				aa = 0
				bb = 0
                		for _, train_index in kf:

                    			uidx += 1

                    			x0 = [trainX[ii] for ii in train_index]
                    			y0 = [trainY[ii] for ii in train_index]
                    			n_samples += len(train_index)

					x0, x0mask, y0, y0_in, maxlen = self.prepare_data(x0, y0)					
			
                 			g_cost, hingeloss_g, pos_cost, neg_cost, delta, pred0 = self.train_g(x0, x0mask, y0,  y0_in, maxlen)
	
					d_cost, hingeloss_d, pos_cost, neg_cost, delta, pred0 = self.train_d(x0, x0mask, y0,  y0_in, maxlen)
					
                                        aa += hingeloss_g
					bb += hingeloss_d                                
										
					if np.isnan(d_cost) or np.isinf(d_cost) or np.isnan(g_cost) or np.isinf(g_cost):
                        			
						self.textfile.write("NaN detected \n")
						self.textfile.flush()



				self.textfile.write("hinge loss g:%f  hinge loss d: %f    \n" %(  aa, bb)  )
				self.textfile.flush()
				
				self.textfile.write("Seen samples:%d   \n" %( n_samples)  )
				self.textfile.flush()
			        end_time1 = time.time()
		

								
			

				start_time2 = time.time()
				devacc, negscore, posscore, margin  = self.test_time1(devx0, devx0mask, devy0,  devy0_in, devmaxlen)
			
				testacc = self.test_time(testx0, testx0mask, testy0, testmaxlen)
				end_time2 = time.time()
				if bestdev < devacc:
					bestdev = devacc
					best_t = eidx
														

					
				
				self.textfile.write("epoches %d  devacc %f  testacc %f trainig time %f test time %f \n" %( eidx + 1, devacc, testacc, end_time1 - start_time1, end_time2 - start_time2 ) )
				self.textfile.flush()
				
				#####################################################################################################
                		#returning step for the inference network
                		#####################################################################################################

				
						
				tmp_a_para = [p.get_value() for p in self.a_params]
                		saveParams( tmp_a_para , params.outfile + '_a_network.pickle')
				
                                i_index = 0	
				for i in xrange(1):
                                                n_samples = 0

                                                kf = get_minibatches_idx(len(trainX), params.batchsize, shuffle=True)


                                                for _, train_index in kf:
							ii += 1
                                                        x0 = [trainX[ii] for ii in train_index]
                                                        y0 = [trainY[ii] for ii in train_index]
                                                        x0, x0mask, y0, y0_in, maxlen = self.prepare_data(x0, y0)
                                                        turning_cost = self.test_time_turning(x0, x0mask, y0, maxlen)

							n_samples += len(train_index)
							
							
							if (i_index%10==0): 
                                                		devacc  = self.test_time(devx0, devx0mask, devy0, devmaxlen)
                                                		testacc  = self.test_time(testx0, testx0mask, testy0, testmaxlen)
                                                		if bestdev1 < devacc:
                                                        		bestdev1 = devacc
                                                        		besttest1 = testacc
									best_t1 = eidx +1
									
                                                			
									self.textfile.write("returning epoches %d  devacc %f  testacc %f \n" %( i_index, bestdev1, besttest1) )
									
				f = open(params.outfile + '_a_network.pickle', 'r')
                		PARA = pickle.load(f)
                		f.close()

       
                		for idx, p in enumerate(self.a_params):
                        		p.set_value(PARA[idx])
       			
        	except KeyboardInterrupt:
            		self.textfile.write( 'classifer training interrupt \n')
			self.textfile.flush()

        	end_time = time.time()
		
		self.textfile.write("best dev acc: %f  at time %d \n" % (bestdev, best_t))
		self.textfile.flush()
		print 'bestdev ', bestdev, 'bestdev1 ', bestdev1		

		os.remove(params.outfile + '_a_network.pickle')
                

		self.textfile.write("best dev acc: %f  at time %d  after returning step best dev acc: %f  at time %d     \n" % (bestdev, best_t, bestdev1, best_t1))
		self.textfile.flush()
        	self.textfile.close()
		
