

import numpy as np
import theano
import theano.tensor as T


def theano_logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x), axis=axis) in a numerically stable
    fashion.
    Parameters
    ----------
    x : tensor_like
        A Theano tensor (any dimension will do).
    axis : int or symbolic integer scalar, or None
        Axis over which to perform the summation. `None`, the
        default, performs over all axes.
    Returns
    -------
    result : ndarray or scalar
        The result of the log(sum(exp(...))) operation.
    """

    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def crf_loss(uniaries, energies, targets, masks):
    """
    compute minus log likelihood of crf as crf loss.
    :param uniaries: Theano 2D tensor
	uniary energies of each step. the shape is [batch_size, n_time_steps, num_labels],
    :param energies: Theano 4D tensor
        energies of each step. the shape is [batch_size, n_time_steps, num_labels, num_labels],
        where the pad label index is at last.
    :param targets: Theano 2D tensor
        targets in the shape [batch_size, n_time_steps]
    :param masks: Theano 2D tensor
        masks in the shape [batch_size, n_time_steps]
    :return: Theano 1D tensor
        an expression for minus log likelihood loss.
    """

    assert energies.ndim == 4
    assert targets.ndim == 2
    assert masks.ndim == 2
    assert uniaries.ndim ==3

    def inner_function(uniaries_one_step, energies_one_step, targets_one_step, mask_one_step, prior_partition, prev_label, tg_energy):
        """
	:param uniaries_one_step: [batch_size, t]
        :param energies_one_step: [batch_size, t, t]
        :param targets_one_step: [batch_size]
	:param mask_one_step: [batch_size]
        :param prior_partition: [batch_size, t]
        :param prev_label: [batch_size]
        :param tg_energy: [batch_size]
	
        :return:
        """

        partition_shuffled = prior_partition.dimshuffle(0, 1, 'x')
	uniaries_one_step_shuffled = uniaries_one_step.dimshuffle(0, 'x', 1)

        partition_t = T.switch(mask_one_step.dimshuffle(0, 'x'),
                               theano_logsumexp(energies_one_step + uniaries_one_step_shuffled + partition_shuffled, axis=1),
                               prior_partition)
	tg_energy_t = T.switch(mask_one_step, tg_energy + uniaries_one_step[T.arange(uniaries_one_step.shape[0]), targets_one_step] + energies_one_step[T.arange(energies_one_step.shape[0]), prev_label, targets_one_step], tg_energy)		

        return [partition_t, targets_one_step, tg_energy_t]
	#return [partition_t, targets_one_step,
        #        tg_energy + energies_one_step[T.arange(energies_one_step.shape[0]), prev_label, targets_one_step]]
    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # but scan requires the iterable dimension to be first
    # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
    energies_shuffled = energies.dimshuffle(1, 0, 2, 3)
    
    targets_shuffled = targets.dimshuffle(1, 0)
    masks_shuffled = masks.dimshuffle(1, 0)
    uniaries_shuffled = uniaries.dimshuffle(1,0,2)
    # initials should be energies_shuffles[0, :, -1, :]
    init_label = T.cast(T.fill(energies[:, 0, 0, 0], -1), 'int32')
    energy_time0 = energies_shuffled[0]
    target_time0 = targets_shuffled[0]
    uniary_time0 = uniaries_shuffled[0]

    initials = [energy_time0[:, -1, :-1] + uniary_time0[:,:], target_time0, energy_time0[T.arange(energy_time0.shape[0]), init_label, target_time0] + uniary_time0[T.arange(energy_time0.shape[0]),target_time0]]

    #initials = [energy_time0[:, -1, :-1]+ uniary_time0[:,:], target_time0, energy_time0[T.arange(energy_time0.shape[0]), init_label, target_time0]]


    [partitions, _, target_energies], _ = theano.scan(fn=inner_function, outputs_info=initials,
                                                      sequences=[uniaries_shuffled[1:], energies_shuffled[1:,:,:-1,:-1], targets_shuffled[1:],masks_shuffled[1:]])
    partition = partitions[-1]
    target_energy = target_energies[-1]
    loss = theano_logsumexp(partition, axis=1) - target_energy
    return loss


def crf_accuracy( uniaries, energies, targets):
    """
    decode crf and compute accuracy
    :param uniaries:  [batch_size, n_time_steps,  num_labels]
    :param energies: Theano 4D tensor
        energies of each step. the shape is [batch_size, n_time_steps, num_labels, num_labels],
        where the pad label index is at last.
    :param targets: Theano 2D tensor
        targets in the shape [batch_size, n_time_steps]
    :return: Theano 1D tensor
        an expression for minus log likelihood loss.
    """

    assert energies.ndim == 4
    assert targets.ndim == 2
    assert uniaries.ndim ==3

    def inner_function(uniaries_one_step, energies_one_step, prior_pi, prior_pointer):
        """
	:param uniaries_one_step: [batch_size, t]
        :param energies_one_step: [batch_size, t, t]
        :param prior_pi: [batch_size, t]
        :param prior_pointer: [batch_size, t]
        :return:
        """
        prior_pi_shuffled = prior_pi.dimshuffle(0, 1, 'x')
	uniaries_one_step_shuffled = uniaries_one_step.dimshuffle(0, 'x', 1)
        pi_t = T.max(prior_pi_shuffled + uniaries_one_step_shuffled + energies_one_step, axis=1)
        pointer_t = T.argmax(prior_pi_shuffled + uniaries_one_step_shuffled + energies_one_step, axis=1)
	
	#pi_t = T.max(prior_pi_shuffled +  energies_one_step, axis=1)
        #pointer_t = T.argmax(prior_pi_shuffled + energies_one_step, axis=1)

        return [pi_t, pointer_t]

    def back_pointer(pointer, pointer_tp1):
        """
        :param pointer: [batch, t]
        :param point_tp1: [batch,]
        :return:
        """
        return pointer[T.arange(pointer.shape[0]), pointer_tp1]

    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # but scan requires the iterable dimension to be first
    # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
    energies_shuffled = energies.dimshuffle(1, 0, 2, 3)
    # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
    # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - 1.
    
    ###energies_shuffled = energies_shuffled[:, :, :-1, :-1]

    uniaries_shuffled = uniaries.dimshuffle(1,0,2)	
    #uniaries_shuffled = uniaries_shuffled[:,:,:-1]

    # pi at time 0 is the last rwo at time 0. but we need to remove the last column which is the pad symbol.
    pi_time0 = energies_shuffled[0, :, -1, :-1]+ uniaries_shuffled[0,:,:]
    #pi_time0 = energies_shuffled[0, :, -1, :]
	
    initials = [pi_time0, T.cast(T.fill(pi_time0, -1), 'int64')]

    [pis, pointers], _ = theano.scan(fn=inner_function, outputs_info=initials, sequences=[ uniaries_shuffled[1:], energies_shuffled[1:,:,:-1,:-1]])
    pi_n = pis[-1]
    pointer_n = T.argmax(pi_n, axis=1)

    back_pointers, _ = theano.scan(fn=back_pointer, outputs_info=pointer_n, sequences=[pointers], go_backwards=True)

    # prediction shape [batch_size, length]
    prediction_revered = T.concatenate([pointer_n.dimshuffle(0, 'x'), back_pointers.dimshuffle(1, 0)], axis=1)
    prediction = prediction_revered[:, T.arange(prediction_revered.shape[1] - 1, -1, -1)]
    return prediction, T.eq(prediction, targets)




#### the simple energy loss
def crf_loss0( uniaries, transition, targets, masks):
    """
    compute minus log likelihood of crf as crf loss.
    :param transition: Theano 3D tensor
	uniary energies of each step. the shape is [batch_size, n_time_steps, num_labels],

    :param transition: Theano 2D tensor
        pairwise energies of each step. the shape is [num_labels, num_labels],
        where the pad label index is at last.
    :param targets: Theano 2D tensor
        targets in the shape [batch_size, n_time_steps]
    :param masks: Theano 2D tensor
        masks in the shape [batch_size, n_time_steps]
    :return: Theano 1D tensor
        an expression for minus log likelihood loss.
    """

    assert transition.ndim == 2
    assert targets.ndim == 2
    assert masks.ndim == 2
    assert uniaries.ndim ==3

    def inner_function(uniaries_one_step, targets_one_step,  mask_one_step, prior_partition, prev_label, tg_energy, transition):
        """
	:param uniaries: [batch_size, t]
        :param targets_one_step: [batch_size]
        :param prior_partition: [batch_size, t]
        :param prev_label: [batch_size]
        :param tg_energy: [batch_size]
	:param transition: [t, t]
        :return:
        """

        partition_shuffled = prior_partition.dimshuffle(0, 1, 'x')
	uniaries_one_step_shuffled = uniaries_one_step.dimshuffle(0,'x', 1)	

        partition_t = T.switch(mask_one_step.dimshuffle(0, 'x'),
                               theano_logsumexp(uniaries_one_step_shuffled + transition.dimshuffle('x', 0, 1) + partition_shuffled, axis=1),
                               prior_partition)
	tg_energy_t = T.switch(mask_one_step, tg_energy + uniaries_one_step[ T.arange(uniaries_one_step.shape[0]), targets_one_step] + transition[prev_label, targets_one_step], tg_energy)

        return [partition_t, targets_one_step,tg_energy_t]

    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # but scan requires the iterable dimension to be first
    # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
   
    uniaries_shuffled = uniaries.dimshuffle(1,0,2) 
    ##energies_shuffled = energies.dimshuffle(1, 0, 2, 3)
    targets_shuffled = targets.dimshuffle(1, 0)
    masks_shuffled = masks.dimshuffle(1, 0)

    # initials should be energies_shuffles[0, :, -1, :]
    init_label = T.cast(T.fill(uniaries[:, 0, 0], -1), 'int32')
   
    #aa = T.cast(T.fill(uniaries[:,0,:],0), 'float32')
    #aa = aa.dimshuffle(0, 'x', 1) + transition.dimshuffle('x', 0, 1)

 
    target_time0 = targets_shuffled[0]
    uniary_time0 = uniaries_shuffled[0]
    energy_time0 = transition[-1, :-1]

    #initials = [uniary_time0[:, :]+ transition[-1, :].dimshuffle('x', 0), target_time0, uniary_time0[T.arange(target_time0.shape[0]),target_time0]+ aa[T.arange(target_time0.shape[0]), init_label, target_time0]]

    initials = [uniary_time0[:, :] + energy_time0.dimshuffle('x', 0), target_time0, uniary_time0[T.arange(target_time0.shape[0]),target_time0]+ transition[init_label, target_time0]]
    	
    #print (transition[-1, :].dimshuffle('x', 0)).ndim, (transition[init_label, target_time0]).ndim

    [partitions, _, target_energies], _ = theano.scan(fn=inner_function, outputs_info=initials,
                                                      sequences=[uniaries_shuffled[1:], targets_shuffled[1:],
                                                                 masks_shuffled[1:]], non_sequences = [transition[:-1,:-1]])
        
    partition = partitions[-1]
    target_energy = target_energies[-1]
    loss = theano_logsumexp(partition, axis=1) - target_energy
    return loss





def crf_loss0_energy( uniaries, transition, targets, masks):
    """
    compute minus log likelihood of crf as crf loss.
    :param transition: Theano 3D tensor
        uniary energies of each step. the shape is [batch_size, n_time_steps, num_labels],

    :param transition: Theano 2D tensor
        pairwise energies of each step. the shape is [num_labels, num_labels],
        where the pad label index is at last.
    :param targets: Theano 2D tensor
        targets in the shape [batch_size, n_time_steps]
    :param masks: Theano 2D tensor
        masks in the shape [batch_size, n_time_steps]
    :return: Theano 1D tensor
        an expression for minus log likelihood loss.
    """

    assert transition.ndim == 2
    assert targets.ndim == 2
    assert masks.ndim == 2
    assert uniaries.ndim ==3

    def inner_function(uniaries_one_step, targets_one_step,  mask_one_step, prior_partition, prev_label, tg_energy, transition):
        """
        :param uniaries: [batch_size, t]
        :param targets_one_step: [batch_size]
        :param prior_partition: [batch_size, t]
        :param prev_label: [batch_size]
        :param tg_energy: [batch_size]
        :param transition: [t, t]
        :return:
        """

        partition_shuffled = prior_partition.dimshuffle(0, 1, 'x')
        uniaries_one_step_shuffled = uniaries_one_step.dimshuffle(0,'x', 1)

        partition_t = T.switch(mask_one_step.dimshuffle(0, 'x'),
                               theano_logsumexp(uniaries_one_step_shuffled + transition.dimshuffle('x', 0, 1) + partition_shuffled, axis=1),
                               prior_partition)
        tg_energy_t = T.switch(mask_one_step, tg_energy + uniaries_one_step[ T.arange(uniaries_one_step.shape[0]), targets_one_step] + transition[prev_label, targets_one_step], tg_energy)

        return [partition_t, targets_one_step,tg_energy_t]

    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # but scan requires the iterable dimension to be first
    # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
    uniaries_shuffled = uniaries.dimshuffle(1,0,2)
    ##energies_shuffled = energies.dimshuffle(1, 0, 2, 3)
    targets_shuffled = targets.dimshuffle(1, 0)
    masks_shuffled = masks.dimshuffle(1, 0)

    # initials should be energies_shuffles[0, :, -1, :]
    init_label = T.cast(T.fill(uniaries[:, 0, 0], -1), 'int32')

    #aa = T.cast(T.fill(uniaries[:,0,:],0), 'float32')
    #aa = aa.dimshuffle(0, 'x', 1) + transition.dimshuffle('x', 0, 1)


    target_time0 = targets_shuffled[0]
    uniary_time0 = uniaries_shuffled[0]
    energy_time0 = transition[-1, :-1]

    #initials = [uniary_time0[:, :]+ transition[-1, :].dimshuffle('x', 0), target_time0, uniary_time0[T.arange(target_time0.shape[0]),target_time0]+ aa[T.arange(target_time0.shape[0]), init_label, target_time0]]

    initials = [uniary_time0[:, :] + energy_time0.dimshuffle('x', 0), target_time0, uniary_time0[T.arange(target_time0.shape[0]),target_time0]+ transition[init_label, target_time0]]

    #print (transition[-1, :].dimshuffle('x', 0)).ndim, (transition[init_label, target_time0]).ndim

    [partitions, _, target_energies], _ = theano.scan(fn=inner_function, outputs_info=initials,
                                                      sequences=[uniaries_shuffled[1:], targets_shuffled[1:],
                                                                 masks_shuffled[1:]], non_sequences = [transition[:-1,:-1]])

    partition = partitions[-1]
    target_energy = target_energies[-1]
    loss = theano_logsumexp(partition, axis=1) - target_energy
    return target_energy

   










def crf_accuracy0(uniaries, transition, targets, masks ):
    """
    decode crf and compute accuracy

    :param uniaries: Theano 2D tensor
        targets in the shape [batch_size, n_time_steps,  num_labels]

    :param transition: Theano 2D tensor
        energies of each step. the shape is [num_labels, num_labels],
        where the pad label index is at last.
    :param targets: Theano 2D tensor
        targets in the shape [batch_size, n_time_steps]
    :return: Theano 1D tensor
        an expression for minus log likelihood loss.
    """

    assert transition.ndim == 2
    assert targets.ndim == 2
    assert uniaries.ndim ==3

    def inner_function( uniaries_one_step, mask_one_step,  prior_pi, prior_pointer, transition0):
        """
	:param uniaries_one_step: [batch_size, t]
        :param transition: [t, t]
        :param prior_pi: [batch_size, t]
        :param prior_pointer: [batch_size, t]
        :return:
        """
	uniaries_one_step_shuffled = uniaries_one_step.dimshuffle(0, 'x', 1)
        energy_one_step_shuffled = transition0.dimshuffle('x', 0, 1)	
        prior_pi_shuffled = prior_pi.dimshuffle(0, 1, 'x')
	mask_one_step_shuffled = mask_one_step.dimshuffle(0,'x', 'x')	

        pi_t = T.max(prior_pi_shuffled + uniaries_one_step_shuffled + energy_one_step_shuffled*mask_one_step_shuffled, axis=1)
        pointer_t = T.argmax(prior_pi_shuffled + uniaries_one_step_shuffled + energy_one_step_shuffled*mask_one_step_shuffled , axis=1)

        return [pi_t, pointer_t]

    def back_pointer(pointer, pointer_tp1):
        """
        :param pointer: [batch, t]
        :param point_tp1: [batch,]
        :return:
        """
        return pointer[T.arange(pointer.shape[0]), pointer_tp1]

    
    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # but scan requires the iterable dimension to be first
    # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
    ##energies_shuffled = energies.dimshuffle(1, 0, 2, 3)


    # pi at time 0 is the last rwo at time 0. but we need to remove the last column which is the pad symbol.
    #pi_time0 = uniaries[:, 0, :-1]
    energies0 = transition[-1, :-1]
   
    pi_time0 = uniaries[:, 0, :] + energies0.dimshuffle('x',0)

    uniaries_shuffled = uniaries.dimshuffle(1,0,2)
    #uniaries_shuffled = uniaries_shuffled[:,:,:-1]
    masks_shuffled = masks.dimshuffle(1, 0)
	
    initials = [pi_time0, T.cast(T.fill(pi_time0, -1), 'int64')]

    [pis, pointers], _ = theano.scan(fn=inner_function, outputs_info=initials, sequences=[uniaries_shuffled[1:], masks_shuffled[1:]], non_sequences = transition[:-1,:-1])
    pi_n = pis[-1]
    pointer_n = T.argmax(pi_n, axis=1)

    back_pointers, _ = theano.scan(fn=back_pointer, outputs_info=pointer_n, sequences=[pointers], go_backwards=True)

    # prediction shape [batch_size, length]
    prediction_revered = T.concatenate([pointer_n.dimshuffle(0, 'x'), back_pointers.dimshuffle(1, 0)], axis=1)
    prediction = prediction_revered[:, T.arange(prediction_revered.shape[1] - 1, -1, -1)]
    return prediction, T.eq(prediction, targets)

