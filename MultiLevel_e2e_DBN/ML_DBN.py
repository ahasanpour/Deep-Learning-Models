
from __future__ import print_function, division
import numpy
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data
import DBN_y_w
import DBN_y
from theano.sandbox.rng_mrg import MRG_RandomStreams
from mlp import HiddenLayer
def run_dsb (
            train_set_x=None, train_set_y=None,
            valid_set_x=None, valid_set_y=None,
            test_set_x=None, test_set_y=None,WW=None):
    #stack1
    train_prob, test_prob, valid_prob, rbm_layers = DBN_y.test_DBN(finetune_lr=0.1, pretraining_epochs=50, pretrain_lr=0.01, k=1, training_epochs=100,
                                     train_set_x=train_set_x, train_set_y=train_set_y,
                                     valid_set_x=valid_set_x, valid_set_y=valid_set_y,
                                     test_set_x=test_set_x, test_set_y=test_set_y,
                                     batch_size=10,
                                     DBN=DBN_y.DBN(numpy_rng=numpy_rng, n_ins=(train_set_x.get_value(borrow=True)).shape[1], hidden_layers_sizes=[1000],n_outs=10))

    train_set_x_DBN1=numpy.concatenate((train_prob,train_set_x.get_value(borrow=True)), axis=1)
    test_set_x_DBN1=numpy.concatenate((test_prob,test_set_x.get_value(borrow=True)), axis=1)
    valid_set_x_DBN1=numpy.concatenate((valid_prob,valid_set_x.get_value(borrow=True)), axis=1)

    rbm_w=(((rbm_layers[0]).W).get_value(borrow=True))
    numpy_rng1 = numpy.random.RandomState(123)

    Wadded = numpy.asarray(
        numpy_rng1.uniform(
            low=-numpy.sqrt(6. / (train_set_x_DBN1.shape[1]-rbm_w.shape[0]+ rbm_w.shape[1])),
            high=numpy.sqrt(6. / (train_set_x_DBN1.shape[1]-rbm_w.shape[0]+ rbm_w.shape[1])),
            size=(train_set_x_DBN1.shape[1]-rbm_w.shape[0], rbm_w.shape[1])
        ),
        dtype=theano.config.floatX
    )
    # if activation function be sigmoid is needed but if be tanh it is not
    Wadded *= 4
    newW=numpy.concatenate((rbm_w, Wadded), axis=0)
    WW = theano.shared(value=newW, borrow=True)
    #stack2
    train_prob1, test_prob1,valid_prob1,rbm_layers = DBN_y_w.test_DBN(finetune_lr=0.1, pretraining_epochs=50, pretrain_lr=0.01, k=1, training_epochs=100,
                                     train_set_x=theano.shared(train_set_x_DBN1), train_set_y=train_set_y,
                                     valid_set_x=theano.shared(valid_set_x_DBN1), valid_set_y=valid_set_y,
                                     test_set_x=theano.shared(test_set_x_DBN1), test_set_y=test_set_y,
                                     batch_size=10,
                                     DBN=DBN_y_w.DBN(numpy_rng=numpy_rng, n_ins=(train_set_x_DBN1).shape[1], hidden_layers_sizes=[1000],n_outs=10,W=WW))

    train_set_x_DBN1 = numpy.concatenate((train_prob,train_prob1, train_set_x.get_value(borrow=True)), axis=1)
    test_set_x_DBN1 = numpy.concatenate((test_prob,test_prob1, test_set_x.get_value(borrow=True)), axis=1)
    valid_set_x_DBN1 = numpy.concatenate((valid_prob,valid_prob1, valid_set_x.get_value(borrow=True)), axis=1)

    rbm_w = (((rbm_layers[0]).W).get_value(borrow=True))
    numpy_rng1 = numpy.random.RandomState(123)

    Wadded = numpy.asarray(
        numpy_rng1.uniform(
            low=-numpy.sqrt(6. / (train_set_x_DBN1.shape[1] - rbm_w.shape[0] + rbm_w.shape[1])),
            high=numpy.sqrt(6. / (train_set_x_DBN1.shape[1] - rbm_w.shape[0] + rbm_w.shape[1])),
            size=(train_set_x_DBN1.shape[1] - rbm_w.shape[0], rbm_w.shape[1])
        ),
        dtype=theano.config.floatX
    )
    # if activation function be sigmoid is needed but if be tanh it is not
    Wadded *= 4
    newW = numpy.concatenate((rbm_w, Wadded), axis=0)
    WW = theano.shared(value=newW, borrow=True)
    #stack3
    train_prob2, test_prob2, valid_prob2,rbm_layers = DBN_y_w.test_DBN(finetune_lr=0.1, pretraining_epochs=100, pretrain_lr=0.01, k=1,training_epochs=100,
                                    train_set_x=theano.shared(train_set_x_DBN1), train_set_y=train_set_y,
                                    valid_set_x=theano.shared(valid_set_x_DBN1), valid_set_y=valid_set_y,
                                    test_set_x=theano.shared(test_set_x_DBN1), test_set_y=test_set_y,
                                    batch_size=10,
                                    DBN=DBN_y_w.DBN(numpy_rng=numpy_rng, n_ins=(train_set_x_DBN1).shape[1],hidden_layers_sizes=[1000], n_outs=10,W=WW))

    train_set_x_DBN1 = numpy.concatenate((train_prob, train_prob1,train_prob2, train_set_x.get_value(borrow=True)), axis=1)
    test_set_x_DBN1 = numpy.concatenate((test_prob, test_prob1, test_prob2,test_set_x.get_value(borrow=True)), axis=1)
    valid_set_x_DBN1 = numpy.concatenate((valid_prob, valid_prob1,valid_prob2, valid_set_x.get_value(borrow=True)), axis=1)

    rbm_w = (((rbm_layers[0]).W).get_value(borrow=True))
    numpy_rng1 = numpy.random.RandomState(123)

    Wadded = numpy.asarray(
        numpy_rng1.uniform(
            low=-numpy.sqrt(6. / (train_set_x_DBN1.shape[1] - rbm_w.shape[0] + rbm_w.shape[1])),
            high=numpy.sqrt(6. / (train_set_x_DBN1.shape[1] - rbm_w.shape[0] + rbm_w.shape[1])),
            size=(train_set_x_DBN1.shape[1] - rbm_w.shape[0], rbm_w.shape[1])
        ),
        dtype=theano.config.floatX
    )
    # if activation function be sigmoid is needed but if be tanh it is not
    Wadded *= 4
    newW = numpy.concatenate((rbm_w, Wadded), axis=0)
    WW = theano.shared(value=newW, borrow=True)
    #stack4
    train_prob3, test_prob3, valid_prob3,rbm_layers = DBN_y_w.test_DBN(finetune_lr=0.1, pretraining_epochs=50, pretrain_lr=0.01,k=1, training_epochs=100,
                                                            train_set_x=theano.shared(train_set_x_DBN1),
                                                            train_set_y=train_set_y,
                                                            valid_set_x=theano.shared(valid_set_x_DBN1),
                                                            valid_set_y=valid_set_y,
                                                            test_set_x=theano.shared(test_set_x_DBN1),
                                                            test_set_y=test_set_y,
                                                            batch_size=10,
                                                            DBN=DBN_y_w.DBN(numpy_rng=numpy_rng,n_ins=(train_set_x_DBN1).shape[1],hidden_layers_sizes=[1000], n_outs=10,W=WW))


if __name__ == '__main__':
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    # train_set_x  shape is 500000 *784
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    numpy_rng = numpy.random.RandomState(123)
    run_dsb(train_set_x=train_set_x, train_set_y=train_set_y,
            valid_set_x=valid_set_x, valid_set_y=valid_set_y,
            test_set_x=test_set_x, test_set_y=test_set_y)
