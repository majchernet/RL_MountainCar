import tensorflow as tf
import numpy as np

# class to represent the Neural Network model
class NN:

    # build model of nn, layers define number of nodes in each layer 
    # first element of layers is numer of nodes in input layer
    # last element of layers is numer of nodes in output layer

    def __init__(self, layers):
        self.sess = tf.Session()

        self.inLayer  = layers.pop(0)
        self.outLayer = layers.pop(-1)
        self.hiddenLayers = layers

        #number of hidden layers
        self.nH = len(self.hiddenLayers)

        self.learning_rate = 0.005

        self.W = [0] * (self.nH+1)
        self.B = [0] * (self.nH+1)

        # output for each layers
        self.Y = [0] * (self.nH)
        self.Ylogits = [0] * (self.outLayer)

        # vector of input data
        self.X = tf.placeholder(tf.float32, [None, self.inLayer]) 

        # vector of labels
        self.Ylabels = tf.placeholder(tf.float32, [None, self.outLayer])

        # define model for input layer
        self.W[0] = tf.Variable(tf.truncated_normal([self.inLayer, self.hiddenLayers[0]], stddev=1.0))
        self.B[0] = tf.Variable(tf.zeros([self.hiddenLayers[0]]))
        self.Y[0] = tf.nn.sigmoid(tf.matmul(self.X, self.W[0]) + self.B[0])
	
        # iterate through the rest of layers
        for idx, val in enumerate(self.hiddenLayers):
	    # define model for output layer
            if  idx+1 == self.nH: 
                self.W[idx+1] = tf.Variable(tf.truncated_normal([val, self.outLayer], stddev=1.0))
                self.B[idx+1] = tf.Variable(tf.zeros(self.outLayer))
                self.Ylogits  = tf.matmul(self.Y[idx], self.W[idx+1]) + self.B[idx+1]
                break
            # define model for hidden layer
            self.W[idx+1] = tf.Variable(tf.truncated_normal([val, self.hiddenLayers[idx+1]], stddev=1.0))
            self.B[idx+1] = tf.Variable(tf.zeros(self.hiddenLayers[idx+1]))
            self.Y[idx+1] = tf.nn.sigmoid(tf.matmul(self.Y[idx], self.W[idx+1]) + self.B[idx+1])

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Ylogits, labels=self.Ylabels))
        self.train_step = tf.train.AdamOptimizer(0.005).minimize(self.cross_entropy)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()


    #return string with info about nn
    def info(self):
        info = "This is neural network object with nodes in layers\n"        
        info += "Input layer: {} nodes\n".format(self.inLayer)
        
        for idx, val in enumerate(self.hiddenLayers):
            info += "Hidden layer [{}]: {} nodes\n".format(idx+1, val)
            
        info += "Output layer: {} nodes\n".format(self.outLayer)
        return info


    # train nn with batch of data
    def train(self, batchX, batchY):
        if np.shape(batchX)[1] != self.inLayer:
            print ("Rows of X must be {} element vectors".format(self.inLayer))
            return        
        if np.shape(batchY)[1] != self.outLayer:
            print ("Rows of Y must be {} element vectors".format(self.outLayer))
            return
        return self.sess.run(self.train_step, feed_dict={self.X: batchX, self.Ylabels: batchY})


    # test nn with testset, return accuracy
    def test(self,testX,testY):
        self.correct_prediction = tf.equal(tf.argmax(self.Ylogits,1), tf.argmax(self.Ylabels,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        return self.sess.run(self.accuracy, feed_dict={self.X: testX, self.Ylabels: testY})

    def ask(self,askX):
        return self.sess.run(self.Ylogits, feed_dict={self.X: askX})
