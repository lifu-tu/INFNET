import numpy as np
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



class MLC_model(object):
        def __init__(self, params, num_lables, num_features):
                self.textfile = open(params.outfile, 'w')
                
                hidden1 = params.hidden1
                hidden2 = params.hidden2
                hidden1_a = params.hidden1_a
                hidden2_a = params.hidden2_a
                eta = params.eta
                L2 = params.L2
                C1 = params.C1
                
                ## for the local energy function
                l_in = lasagne.layers.InputLayer((None, num_features))
                l_y1 = lasagne.layers.DenseLayer(l_in, hidden1)
                l_y2 = lasagne.layers.DenseLayer(l_y1, hidden2)
                l_local = lasagne.layers.DenseLayer(l_y2, num_lables, b = None, nonlinearity = lasagne.nonlinearities.linear)		
		
                g1 = T.fmatrix()
                y1 = T.fmatrix()
		
                c_params0 = lasagne.layers.get_all_params(l_y2, trainable=True)		
                c_params1 = lasagne.layers.get_all_params(l_local, trainable=True)
                f = open(params.FeatureNet, 'rb')
                para = pickle.load(f)
                f.close()

                for idx, p in enumerate(c_params1):
                        if idx < (len(c_params1) -1):
                                p.set_value(para[idx])
                        else:
                                p.set_value(-para[idx])		
                local_cost = lasagne.layers.get_output(l_local, {l_in:g1})
                local_cost = T.sum(local_cost*y1, axis =1)
		
                ## for the global energy function
                l_in1 = lasagne.layers.InputLayer((None, num_lables))
                l_label1 = lasagne.layers.DenseLayer(l_in1, C1, nonlinearity = lasagne.nonlinearities.softplus)
                l_label2 = lasagne.layers.DenseLayer(l_label1, 1, b = None,  nonlinearity = lasagne.nonlinearities.linear)
                global_cost = lasagne.layers.get_output(l_label2, {l_in1:y1})
                global_cost = T.sum(global_cost, axis =1)
                d_params = lasagne.layers.get_all_params(l_label2)
                d_params.append(l_local.W)
		
                self.d_params = d_params		
                energy_cost = local_cost + global_cost
                self.cost_function = theano.function([g1, y1], energy_cost)	
	        """
                for the inference network
                """
                g2 = T.fmatrix()
                l_in_a = lasagne.layers.InputLayer((None, num_features))
                l_y1_a = lasagne.layers.DenseLayer(l_in_a, hidden1_a) 
                l_y2_a = lasagne.layers.DenseLayer(l_y1_a, hidden2_a)
                l_local_a = lasagne.layers.DenseLayer(l_y2_a, num_lables, b = None, nonlinearity = lasagne.nonlinearities.sigmoid)		
                a_params = lasagne.layers.get_all_params(l_local_a, trainable=True)
                self.a_params = a_params		
                f = open(params.infNet, 'rb')
                PARA = pickle.load(f)
                f.close()
                for idx, p in enumerate(a_params):
                        p.set_value(PARA[idx])
		
                train_y = lasagne.layers.get_output(l_local_a, {l_in_a:g2})		
                self.a_function = theano.function([g2], train_y)
	        
                g = T.fmatrix()
                y = T.fmatrix()
                predy = lasagne.layers.get_output(l_local_a, {l_in_a:g})
                local_cost = lasagne.layers.get_output(l_local, {l_in:g})
                pos_local_cost = T.sum(local_cost*y, axis =1)
                neg_local_cost = T.sum(local_cost*predy, axis =1)		
                pos_global_cost = lasagne.layers.get_output(l_label2, {l_in1:y})
                neg_global_cost = lasagne.layers.get_output(l_label2, {l_in1:predy})

                yy = T.cast(y, 'int32')
                delta0 = T.sum((y - predy)**2, axis =1)

                margin_type = params.margin_type
                if(margin_type ==0):
                        hinge_cost = delta0 - (neg_local_cost + T.sum(neg_global_cost, axis =1) ) + (pos_local_cost + T.sum(pos_global_cost, axis=1))
                elif(margin_type ==1):
                        hinge_cost = 1 - (neg_local_cost + T.sum(neg_global_cost, axis =1) ) + (pos_local_cost + T.sum(pos_global_cost, axis=1))
                elif(margin_type ==2):
                        hinge_cost = - (neg_local_cost + T.sum(neg_global_cost, axis =1) ) + (pos_local_cost + T.sum(pos_global_cost, axis=1))
                elif(margin_type ==3):
                        hinge_cost = delta0*(1 - (neg_local_cost + T.sum(neg_global_cost, axis =1) ) + (pos_local_cost + T.sum(pos_global_cost, axis=1)))		

                hinge_cost = hinge_cost * T.gt(hinge_cost, 0)
                d_cost = T.mean(hinge_cost)
                d_cost0 = d_cost		
                margin_pred_y_loss = - T.mean(predy*T.log(predy)+ (1-predy)*T.log(1-predy))		
                g_cost = -d_cost + L2* sum(lasagne.regularization.l2(x) for x in a_params) + params.regu_pretrain*sum(lasagne.regularization.l2(x-PARA[index]) for index, x in enumerate(a_params)) +  margin_pred_y_loss
                d_cost = d_cost + L2 * sum(lasagne.regularization.l2(x) for x in d_params)
		
                self.a_params = a_params	
                updates_g = lasagne.updates.adam(g_cost, a_params, eta)
                self.train_g = theano.function([g,y], [g_cost, d_cost0, margin_pred_y_loss], updates=updates_g)	

                updates_d = lasagne.updates.adam(d_cost, d_params, eta)
                self.train_d = theano.function([g,y], [d_cost, d_cost0], updates=updates_d)
			
                t0 = T.fscalar()		
                g0 = T.fmatrix()
                y00 = T.imatrix()
                local_cost0 = lasagne.layers.get_output(l_local, {l_in:g0})
                predy0 = lasagne.layers.get_output(l_local_a, {l_in_a:g0}, deterministic=True)
                pred_test = T.gt(predy0, t0)		

                neg_local_cost0 = T.sum(local_cost0*predy0, axis =1)
                neg_global_cost0 = lasagne.layers.get_output(l_label2, {l_in1:predy0})
	
                energy_cost20 = T.mean(neg_local_cost0 + T.sum(neg_global_cost0, axis =1))
                energy_cost2 = energy_cost20 - T.mean(predy0*T.log(predy0)+(1-predy0)*T.log(1-predy0))
                #############
                ## optimizer for final returning of the inference network
                updates_test = lasagne.updates.adam(energy_cost2, a_params, 0.00001)
                y0 = T.imatrix()
                neg_local_cost_test = T.sum(local_cost0*pred_test, axis =1)
                neg_global_cost_test = lasagne.layers.get_output(l_label2, {l_in1:pred_test})
                energy_cost_test = T.mean(neg_local_cost_test + T.sum(neg_global_cost_test, axis =1))		
                pg = T.eq(pred_test, y0)
                prec = 1.0*(T.sum(pg*y0, axis =1)+ eps) /(T.sum(pred_test, axis =1)+ eps)
                recall = 1.0 *(T.sum(pg*y0, axis=1) + eps)/(T.sum(y0, axis =1)+ eps)
                f1 = 2*prec*recall/(prec + recall)
                f1 = T.mean(f1)
		
                prec = T.mean(prec)
                recall = T.mean(recall)
                
                self.test = theano.function([g0] ,[energy_cost20, energy_cost2], updates=updates_test)	
                self.test_a = theano.function([g0, y0, t0] ,[ energy_cost20, prec, recall, f1, T.sum(pred_test, axis=1), energy_cost_test])
                self.test_time = theano.function([g0] , predy0)

        def train(self, trainX, trainY, devX, devY, testX, testY, params):
                minibatchsize = params.minibatchsize		
                start_time = time.time()
                bestdev = -1
                bestdev1 = -1
                bestdev_time =0
                counter = 0
                try:
                      for eidx in xrange(500):
                                n_samples = 0
                                start_time1 = time.time()
                                kf = get_minibatches_idx(len(trainX), minibatchsize, shuffle=True)
                                uidx = 0
                                for _, train_index in kf:

                                        uidx += 1
                    			x0 = trainX[train_index].astype('float32')
                                        y0 = trainY[train_index].astype('float32')
                                        n_samples += len(train_index)
                                        g_cost, hingeloss_g, margin_y0 = self.train_g(x0, y0)
                                        d_cost, hingeloss_d = self.train_d(x0, y0)
	                                if np.isnan(d_cost) or np.isinf(d_cost) or np.isnan(g_cost) or np.isinf(g_cost):
		                                 self.textfile.write("NaN detected \n")
                                self.textfile.write("Seen samples:%d   \n" %( n_samples)  )
                                end_time1 = time.time()
                                bestdev0 = 0
                                best_t0 = 0
                                bestdev0_0 = 0
                                besttestf0 = 0
                                best_t1 = 0
                                """
                                evaluated on the dev set
                                """
                                start_time2 = time.time()
                                Pred = []
                                devkf = get_minibatches_idx(len(devX), 32)
                                for _, dev_index in devkf:
                                        devx0 = devX[dev_index].astype('float32')
                                        devy0 = devY[dev_index].astype('int32')
                                        predy0  = self.test_time(devx0)
                                        Pred += predy0.tolist()
                                Pred = np.asarray(Pred)
			
                                Threshold = [0,0.01, 0.02, 0.03, 0.04, 0.05,0.10,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.60,0.65,0.70,0.75]	
                                for t in Threshold:
                                        pred_test = np.greater(Pred, t)
                                        pred = np.equal(pred_test, devY)
                                        aa = np.sum(pred*devY, axis =1) + eps
                                        prec =  1.0 * np.divide(aa, np.sum(pred_test, axis=1) + eps)
                                        recall = 1.0*np.divide(aa, np.sum(devY, axis=1) + eps)
                                        a0 = 2 * prec * recall
                                        a1 = prec + recall
                                        f1 =  np.divide(a0, a1)
                                        f1 = np.mean(f1) 
                                        prec = np.mean(prec)
                                        recall = np.mean(recall)			
                                        if f1> bestdev0:
                                                bestdev0 = f1
                                                best_prec = prec
                                                best_recall = recall
                                                best_t0 = t
                                ### final retuning the parameter of inference network from the cost-augment inference network
                                tmp_a_para = [p.get_value() for p in self.a_params]
                                saveParams( tmp_a_para , params.outfile + '_cost_augment_infnet.pickle')
                                for i in range(20):
                                        devx0 = devX.astype('float32')
                                        devy0 = devY.astype('int32')
                                        cost0, cost1  = self.test(devx0)
                                        Pred = []
                                        bestdev00 = 0
                                        for _, dev_index in devkf:
                                                devx0 = devX[dev_index].astype('float32')
                                                devy0 = devY[dev_index].astype('int32')
                                                predy0  = self.test_time(devx0)
                                                Pred += predy0.tolist()
                                        Pred = np.asarray(Pred)	

                                        for t in Threshold:
                                                pred_test = np.greater(Pred, t)
                                                pred = np.equal(pred_test, devY)
                                                aa = np.sum(pred*devY, axis =1) + eps
                                                prec =  1.0 * np.divide(aa, np.sum(pred_test, axis=1) + eps)
                                                recall = 1.0*np.divide(aa, np.sum(devY, axis=1) + eps)
                                                a0 = 2 * prec * recall
                                                a1 = prec + recall
                                                f1 =  np.divide(a0, a1)
                                                f1 = np.mean(f1)
                                                prec = np.mean(prec)
                                                recall = np.mean(recall)
                                                if f1> bestdev00:
                                                        bestdev00 = f1
                                                        best_t1 = t
                                                        _, _ , _ , testf1_more, _ , _  = self.test_a(testX, testY, best_t0)
                                        if bestdev0_0 < bestdev00:
                                                bestdev0_0 = bestdev00
                                                besttestf0 = testf1_more 
					        self.textfile.write(" best test threshold %f  dev f1 score %f  test f1 score%f\n "%(best_t1, bestdev00, testf1_more))
                                """
                                load the previous parameters from the cost-augment inference network
                                and continue to train the cost-augment inference network
                                """
                                f = open(params.outfile+ '_cost_augment_infnet.pickle', 'rb')
                                para = pickle.load(f)
                                f.close()
                                for idx, p in enumerate(self.a_params):
                                        p.set_value(para[idx])

                                if bestdev0 > bestdev:
                                        bestdev = bestdev0
                                        bestdev_time = eidx + 1
                                        best_t = best_t0
                                if bestdev0_0 > bestdev1:
                                        bestdev1 = bestdev0_0
                                        bestdev_time1 = eidx + 1
                        
                                end_time2 = time.time()	
                                _, _ , _ , testf1, _ , _  = self.test_a(testX, testY, best_t0)								
                                self.textfile.write("epoches %d devacc %f devrecall %f devf1 %f moredevf1 %f  testf1  %f  moretest %f trainigtime %f testtime %f \n" %( eidx + 1, best_prec, best_recall,  bestdev0, bestdev0_0, testf1, besttestf0, end_time1 - start_time1, end_time2 - start_time2 ) )			       
                except KeyboardInterrupt:
                      self.textfile.write( 'classifer training interrupt \n')
                end_time = time.time()
                self.textfile.write("total time %f \n" % (end_time - start_time))
                self.textfile.write("best dev acc: %f  best_thres %f  at time %d   retuning devf1 %f  at time %d \n" % (bestdev, best_t,bestdev_time , bestdev1 , bestdev_time1))
                self.textfile.close()
