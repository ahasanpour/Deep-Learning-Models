"""
"""
from __future__ import print_function, division
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM


# start-snippet-1
class DBN(object):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10):

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.matrix('x')

        self.y = T.ivector('y')

        for i in range(self.n_layers):

            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
            sigmoid_layer = HiddenLayer(rng=numpy_rng,input=layer_input,n_in=input_size,n_out=hidden_layers_sizes[i],activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(input=self.sigmoid_layers[-1].output,n_in=hidden_layers_sizes[-1],n_out=n_outs)
        self.params.extend(self.logLayer.params)

        self.p_y_given_x = self.logLayer.p_y_given_x
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):

        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            cost, updates = rbm.get_cost_updates(learning_rate,persistent=None, k=k)
            fn = theano.function(inputs=[index, theano.In(learning_rate, value=0.1)],outputs=cost,updates=updates,
                givens={self.x: train_set_x[batch_begin:batch_end]})
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, train_set_x, train_set_y,valid_set_x, valid_set_y,test_set_x, test_set_y, batch_size, learning_rate):


        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        gparams = T.grad(self.finetune_cost, self.params)

        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(inputs=[index],outputs=self.finetune_cost,updates=updates,givens={
            self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
            self.y: train_set_y[index * batch_size: (index + 1) * batch_size]})

        predict_y_train = theano.function(inputs=[],outputs=self.p_y_given_x,givens={self.x: train_set_x})
        predict_y_test = theano.function(inputs=[],outputs=self.p_y_given_x,givens={self.x: test_set_x})
        predict_y_valid = theano.function(inputs=[],outputs=self.p_y_given_x,givens={self.x: valid_set_x})

        test_score_i = theano.function([index],self.errors,givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]})

        valid_score_i = theano.function([index],self.errors,givens={
                self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score,predict_y_train,predict_y_test,predict_y_valid


def test_DBN(finetune_lr=0.1, pretraining_epochs=10,pretrain_lr=0.01, k=1, training_epochs=10,
            train_set_x=None, train_set_y=None,
            valid_set_x=None, valid_set_y=None,
            test_set_x=None, test_set_y=None,
            batch_size=10,DBN=None):
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # numpy random generator

    print('... building the model')
    # construct the Deep Belief Network
    dbn = DBN

    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size, k=k)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    for i in range(dbn.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), end=' ')
            print(numpy.mean(c, dtype='float64'))

    end_time = timeit.default_timer()
    # end-snippet-2
    print('The pretraining code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model, test_model,predict_y_train,predict_y_test,predict_y_valid = dbn.build_finetune_functions(
        train_set_x=train_set_x, train_set_y=train_set_y,
        valid_set_x=valid_set_x, valid_set_y=valid_set_y,
        test_set_x=test_set_x, test_set_y=test_set_y,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetuning the model')
    # early-stopping parameters

    # look as this many examples regardless
    patience = 4 * n_train_batches

    # wait this much longer when a new best is found
    patience_increase = 2.

    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995

    # go through this many minibatches before checking the network on
    # the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    matrix = []
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch,minibatch_index + 1, n_train_batches,this_validation_loss * 100.))
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    train_prob = []
                    test_prob = []
                    valid_prob=[]
                    train_prob = predict_y_train()
                    test_prob=predict_y_test()
                    valid_prob = predict_y_valid()
                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses, dtype='float64')
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete with best validation score of %f %%, '
           'obtained at iteration %i, '
           'with test performance %f %%'
           ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The fine tuning code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)
    return train_prob,test_prob,valid_prob,dbn.rbm_layers


if __name__ == '__main__':
    dataset = 'mnist.pkl.gz'

    datasets = load_data(dataset)
    # train_set_x  shape is 500000 *784
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    numpy_rng = numpy.random.RandomState(123)
    train_prob,test_prob,valid_prob=test_DBN(finetune_lr=0.1, pretraining_epochs=10,pretrain_lr=0.01, k=1, training_epochs=10,
            train_set_x=train_set_x, train_set_y=train_set_y,
            valid_set_x=valid_set_x, valid_set_y=valid_set_y,
            test_set_x=test_set_x, test_set_y=test_set_y,
            batch_size=10, DBN=DBN(numpy_rng=numpy_rng, n_ins=28 * 28,hidden_layers_sizes=[1000,1000],n_outs=10))
    #print (train_prob)