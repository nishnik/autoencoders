import numpy as np
import theano as th
from theano import tensor as T
from numpy import random as rng
from theano import printing
import numpy
import theano
from theano.tensor.shared_randomstreams import RandomStreams
global_train_cost=[]# it has the training cost per epoch in list
global_valid_cost=[]# it has the validation cost per epoch in list

class AutoEncoder(object):
    def __init__(self, X,Y, hidden_size, activation_function,
                 output_function):
        #X is the data, an m x n numpy matrix
        #where rows correspond to datapoints
        #and columns correspond to features.
        assert type(X) is np.ndarray
        assert len(X.shape)==2# check it is 2 dimensional
        self.X=X #train data
        self.Y=Y #validation data
        #creating shared variable. Necessary for theano code run
        self.Y=th.shared(name='Y', value=np.asarray(self.Y, 
                         dtype=th.config.floatX),borrow=True) 
        self.X=th.shared(name='X', value=np.asarray(self.X, 
                         dtype=th.config.floatX),borrow=True)
        #storing the shapes of validation and train data
        self.Y_n = Y.shape[1]
        self.Y_m = Y.shape[0]
        self.n = X.shape[1]
        self.m = X.shape[0]
        #Hidden_size is the number of neurons in the hidden layer, an int.
        assert type(hidden_size) is int
        assert hidden_size > 0
        self.hidden_size=hidden_size
        #initialising weight matrix and creating shared variable
        #from a uniform distribution
        #shape of weight matrix is (number of features X hidden layer size)
        initial_W = np.asarray(rng.uniform(
                 low=-4 * np.sqrt(6. / (self.hidden_size + self.n)),
                 high=4 * np.sqrt(6. / (self.hidden_size + self.n)),
                 size=(self.n, self.hidden_size)), dtype=th.config.floatX)
        self.W = th.shared(value=initial_W, name='W', borrow=True)
        #initialising bias1 and bias2 matrix and creating shared variable
        #with all zeros
        #shape of b1 is (1 X hidden layer size)
        self.b1 = th.shared(name='b1', value=np.zeros(shape=(self.hidden_size,),
                            dtype=th.config.floatX),borrow=True)
        #shape of b1 is (1 X  number of features)
        self.b2 = th.shared(name='b2', value=np.zeros(shape=(self.n,),
                            dtype=th.config.floatX),borrow=True)
        #activation function and output function, we have taken it to be the same
        self.activation_function=activation_function
        self.output_function=output_function
                     
    def train(self, n_epochs=100, mini_batch_size=1, learning_rate=0.1):
        index = T.lscalar()
        x=T.matrix('x')
        is_dropout = T.dscalar('is_dropout')
        params = [self.W, self.b1, self.b2]
        # Calculating g(Wx + b1)
        hidden = self.activation_function(T.dot(x, self.W)+self.b1)
        # Generating mask (from binomial distribution)
        # for dropouts with retention probability of 0.5
        arr_n = self.get_mask(self.b1,0.5)
        hidden_tilde = hidden
        # Turning off hidden layers as per generated mask in hidden_tilde
        hidden_tilde = arr_n * hidden
        # Calculating hidden*W' + b2 for validation error calcuation
        output_without_drop = T.dot(hidden,T.transpose(self.W))+self.b2
        output_without_drop = self.output_function(output_without_drop)
        # Calculating actual output considering dropouts
        output_dropout = T.dot(hidden_tilde,T.transpose(self.W))+self.b2
        output_dropout = self.output_function(output_dropout)
        
        # Use cross-entropy loss.
        L = -T.sum(x*T.log(output_dropout) + (1-x)*T.log(1-output_dropout), axis=1)
        cost=L.mean()
        # Calculating validation cost and storing in cost2
        L_without_drops = -T.sum(x*T.log(output_without_drop) + (1-x)*T.log(1-output_without_drop), axis=1)
        cost2 = L_without_drops.mean()
        
        updates=[]
        #Return gradient with respect to W, b1, b2.
        gparams = T.grad(cost,params)

        # not updating hidden units which were dropped out
        gparams_shared = theano.shared(gparams,'gparams_shared')
        gparams[0] = gparams[0] * arr_n
        gparams[1] = gparams[1] * arr_n
        
        #Create a list of 2 tuples for updates.
        for param, gparam in zip(params, gparams):
            updates.append((param, param-learning_rate*gparam))
        
        # Train given a mini-batch of the data.
        train = th.function(inputs=[index], outputs=[cost], updates=updates,
                            givens={x:self.X[index:index+mini_batch_size,:]})

        # Return validation cost 
        valid = th.function(inputs=[index], outputs=[cost2],
                            givens={x:self.Y[index:index+mini_batch_size,:]}) 
        import time
        start_time = time.clock()
        for epoch in xrange(n_epochs):
            print "Epoch:",epoch
            cost_train = 0
            cost_valid = 0
            for row in xrange(0,self.m, mini_batch_size):
                cost_train= cost_train + train(row)[0]
            for row in xrange(0,self.Y_m, mini_batch_size):
                cost_valid = cost_valid + valid(row)[0]
            # for plotting the graph
            global_valid_cost.append((cost_valid/self.Y_m))
            global_train_cost.append((cost_train/self.m))
        
        end_time = time.clock()
        print "Average time per epoch=", (end_time-start_time)/n_epochs
                    
    def get_hidden(self,data):
        x=T.dmatrix('x')
        hidden = self.activation_function(T.dot(x,self.W)+self.b1)
        transformed_data = th.function(inputs=[x], outputs=[hidden])
        return transformed_data(data)
     
    def get_weights(self):
        return [self.W.get_value(), self.b1.get_value(), self.b2.get_value()]
        
    # it returns a random binomial distribution
    def get_mask(self, input, corruption_level):
    	 numpy_rng = numpy.random.RandomState(123)
    	 theano_rng = RandomStreams(numpy_rng.randint(2**30))
    	 return theano_rng.binomial(size=input.shape, n=1,p=1 - corruption_level,dtype=theano.config.floatX)



import cPickle
import gzip
import os
def load_data():
    #############
    # LOAD DATA #
    #############
 
    print '... loading data'
 
    # Load the dataset
    import scipy.io as sio
    mat = sio.loadmat('../octave_struct.mat')
    data = mat['DataMatrix']
    valid_set = data[30000:40000]
    test_set = data[40000:50000]
    train_set = data[0:30000]
    return (train_set, valid_set, test_set)
 
 
data=load_data()

def plot_first_k_numbers(X,k):
    from matplotlib import mpl,pyplot
    m=X.shape[0]
    k=min(m,k)
    pyplot.figure()
    j = int(round(k / 10.0))
     
    fig, ax = pyplot.subplots(j,10)
    
    for i in range(k):
 
        w=X[i,:]
 
         
        w=w.reshape(35,35)
        ax[i/10, i%10].imshow(w,cmap=pyplot.cm.gist_yarg,
                      interpolation='nearest', aspect='equal')
        ax[i/10, i%10].axis('off')
 
     
    pyplot.tick_params(\
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    pyplot.tick_params(\
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off', 
        right='off',    # ticks along the top edge are off
        labelleft='off')
    fig.savefig("dec12.jpg")
    fig.show()

def m_test(data):
    X=data[0]# train set
    Y=data[1]# valid set
    activation_function = T.nnet.sigmoid
    output_function=activation_function
    A = AutoEncoder(X,Y, 700, activation_function, output_function)
    A.train(1000,100)
    W=np.transpose(A.get_weights()[0])
    plot_first_k_numbers(W, 100)
    import csv
    with open('dec12.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(W)

    import scipy.io as sio
    sio.savemat('dec12.mat',{'weights':W})
 
m_test(data)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(len(global_train_cost)),global_train_cost)
plt.plot(range(len(global_valid_cost)),global_valid_cost)
plt.legend(['train data','validation data'],loc='upper left')
plt.savefig("dec11_cost.jpg")
plt.show()

