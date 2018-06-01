
from crf import CRFLayer
from utils import crf_loss, crf_accuracy

def build_bi_lstm_crf(l_emb_word,  hidden, mask=None, num_labels):
	
	l_lstm_wordf = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=mask)
        l_lstm_wordb = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=mask, backwards = True)
	
	concat = lasagne.layers.concat([l_lstm_wordf, l_lstm_wordb], axis=2)
	
	return CRFLayer(concat, num_labels, mask_input=l_mask_word)



def main(We_initial,   params):
	
	We = theano.shared(We_initial)
        embsize = We_initial.shape[1]
        hidden = params.hidden
     
	input_var = T.imatrix(name='inputs')
	target_var = T.imatrix(name='targets')
    	mask_var = T.matrix(name='masks', dtype=theano.config.floatX)

	
	l_in_word = lasagne.layers.InputLayer((None, None))
        l_mask_word = lasagne.layers.InputLayer(shape=(None, None))

        if params.emb ==1:
         	l_emb_word = lasagne.layers.EmbeddingLayer(l_in_word,  input_size= We_initial.shape[0] , output_size = embsize, W =We)
        else:
                l_emb_word = lasagne_embedding_layer_2(l_in_word, embsize, We)

	
	bi_lstm_crf = build_bi_lstm_crf(l_emb_word, hidden, mask=l_mask_word, params.num_labels)

	energies_train = lasagne.layers.get_output(bi_lstm_crf, {l_in_word: input_var, l_mask_word: mask_var})
	
	loss_train = crf_loss(energies_train, target_var, mask_var).mean()

	prediction, corr = crf_accuracy(energies_train, target_var)


	corr_train = (corr * mask_var).sum(dtype=theano.config.floatX)
	num_tokens = mask_var.sum(dtype=theano.config.floatX)

	#prediction, corr = crf_accuracy(energies_eval, target_var)



	network_params = lasagne.layers.get_all_params(bi_lstm_crf, trainable=True)
	print network_params

	updates = utils.create_updates(loss_train, network_params, update_algo, learning_rate, momentum=momentum)
	
	train_fn = theano.function([input_var, target_var, mask_var], [loss_train, corr_train, num_train], updates=updates)
    
    	eval_fn = theano.function([input_var, target_var, mask_var], [loss_train, corr_train, num_train, prediction_eval])

	
	
