import numpy as np
import theano as th
from theano import tensor as T
from numpy import random as rng
from theano import printing
import numpy
import theano
from theano.tensor.shared_randomstreams import RandomStreams
global_train_cost=[]
global_valid_cost=[]

class AutoEncoder(object):
    def __init__(self, X,Y, hidden_size, activation_function,
                 output_function):
        #X is the data, an m x n numpy matrix
        #where rows correspond to datapoints
        #and columns correspond to features.
        assert type(X) is np.ndarray
        assert len(X.shape)==2
        self.X=X
        self.Y=Y
        self.Y=th.shared(name='Y', value=np.asarray(self.Y, 
                         dtype=th.config.floatX),borrow=True)
        self.X=th.shared(name='X', value=np.asarray(self.X, 
                         dtype=th.config.floatX),borrow=True)
        self.Y_n = Y.shape[1]
        self.Y_m = Y.shape[0]
        #The config.floatX and borrow=True stuff is to get this to run
        #fast on the gpu. I recommend just doing this without thinking about
        #it until you understand the code as a whole, then learning more
        #about gpus and theano.
        self.n = X.shape[1]
        self.m = X.shape[0]
        #Hidden_size is the number of neurons in the hidden layer, an int.
        assert type(hidden_size) is int
        assert hidden_size > 0
        self.hidden_size=hidden_size
        initial_W = np.asarray(rng.uniform(
                 low=-4 * np.sqrt(6. / (self.hidden_size + self.n)),
                 high=4 * np.sqrt(6. / (self.hidden_size + self.n)),
                 size=(self.n, self.hidden_size)), dtype=th.config.floatX)
        initial_W2 = np.asarray(rng.uniform(                                #un-tied weigths
                 low=-4 * np.sqrt(6. / (self.hidden_size + self.n)),
                 high=4 * np.sqrt(6. / (self.hidden_size + self.n)),
                 size=(self.n, self.hidden_size)), dtype=th.config.floatX)
        self.W2 = th.shared(value=initial_W2, name='W2', borrow=True)       #un-tied weigths
        self.W = th.shared(value=initial_W, name='W', borrow=True)
        self.b1 = th.shared(name='b1', value=np.zeros(shape=(self.hidden_size,),
                            dtype=th.config.floatX),borrow=True)
        self.b2 = th.shared(name='b2', value=np.zeros(shape=(self.n,),
                            dtype=th.config.floatX),borrow=True)
        self.activation_function=activation_function
        self.output_function=output_function
                     
    def train(self, n_epochs=100, mini_batch_size=1, learning_rate=0.1):
        index = T.lscalar()
        x=T.matrix('x')
        is_dropout = T.dscalar('is_dropout')
        params = [self.W,self.W2, self.b1, self.b2]                         #un-tied
        hidden = self.activation_function(T.dot(x, self.W)+self.b1)
        arr_n = self.get_mask(self.b1,0.5)
        hidden_tilde = hidden
        hidden_tilde = arr_n * hidden
        output_without_drop = T.dot(hidden,T.transpose(self.W2))+self.b2    #un-tied weigths
        output_without_drop = self.output_function(output_without_drop)
        output_dropout = T.dot(hidden_tilde,T.transpose(self.W2))+self.b2   #un-tied weigths
        output_dropout = self.output_function(output_dropout)
        
        #x_printed = theano.printing.Print('this is a very important value')(arr_n)

        #Use cross-entropy loss.
        L = -T.sum(x*T.log(output_dropout) + (1-x)*T.log(1-output_dropout), axis=1)
        cost=L.mean()
        L_without_drops = -T.sum(x*T.log(output_without_drop) + (1-x)*T.log(1-output_without_drop), axis=1)
        cost2 = L_without_drops.mean()
        
        updates=[]
        #Return gradient with respect to W,W2, b1, b2.      #un-tied
        gparams = T.grad(cost,params)
        
        gparams_shared = theano.shared(gparams,'gparams_shared')
        gparams[0] = gparams[0] * arr_n
        gparams[1] = gparams[1] * arr_n
        #x_printed1 = theano.printing.Print('this is a very important value')(gparams[0])
        #x_printed2 = theano.printing.Print('this is a very important value')(gparams[1])

        #gparams_shared = gparams_shared*mask
        #Create a list of 2 tuples for updates.
        for param, gparam in zip(params, gparams):
            updates.append((param, param-learning_rate*gparam))
        
        #Train given a mini-batch of the data.
        train = th.function(inputs=[index], outputs=[cost], updates=updates,
                            givens={x:self.X[index:index+mini_batch_size,:]})
         
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
        return [self.W.get_value(),self.W2.get_value(), self.b1.get_value(), self.b2.get_value()] #untied
        
    def get_mask(self, input, corruption_level):
    	 numpy_rng = numpy.random.RandomState(123)
    	 theano_rng = RandomStreams(numpy_rng.randint(2**30))
    	 return theano_rng.binomial(size=input.shape, n=1,p=1 - corruption_level,dtype=theano.config.floatX)



import cPickle
import gzip
import os
def load_data(dataset):
    ''' Loads the dataset
 
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
 
    #############
    # LOAD DATA #
    #############
 
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
 
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
 
    print '... loading data'
 
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #which row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
 
    return (train_set, valid_set, test_set)
         
 
path="/home/user/Desktop/work/tf/MNIST_data//mnist.pkl.gz"
 
data=load_data(path)

def plot_first_k_numbers(X,k):
    from matplotlib import mpl,pyplot
    m=X.shape[0]
    k=min(m,k)
    pyplot.figure()
    j = int(round(k / 10.0))
     
    fig, ax = pyplot.subplots(j,10)
    
    for i in range(k):
 
        w=X[i,:]
 
         
        w=w.reshape(28,28)
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
    
    fig.show()

def m_test(data):
    X=data[0][0]
    Y=data[1][0]
    activation_function = T.nnet.sigmoid
    output_function=activation_function
    A = AutoEncoder(X,Y, 100, activation_function, output_function)
    A.train(5,100)
    W=np.transpose(A.get_weights()[0])
    plot_first_k_numbers(W, 100)
    W2=np.transpose(A.get_weights()[1])
    plot_first_k_numbers(W2, 100)
  
m_test(data)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(len(global_train_cost)),global_train_cost)
plt.plot(range(len(global_valid_cost)),global_valid_cost)
plt.legend(['train data','validation data'],loc='upper left')
plt.show()

