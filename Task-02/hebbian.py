# Mohan, Shreyas
# 2019-10-07
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import math as Math
from numpy import array
from numpy import argmax
from keras.utils import to_categorical


def display_images(images):
	# This function displays images on a grid.
	# Farhad Kamangar Sept. 2019
	number_of_images=images.shape[0]
	number_of_rows_for_subplot=int(np.sqrt(number_of_images))
	number_of_columns_for_subplot=int(np.ceil(number_of_images/number_of_rows_for_subplot))
	for k in range(number_of_images):
		plt.subplot(number_of_rows_for_subplot,number_of_columns_for_subplot,k+1)
		plt.imshow(images[k], cmap=plt.get_cmap('gray'))
		# plt.imshow(images[k], cmap=pyplot.get_cmap('gray'))
	plt.show()

def display_numpy_array_as_table(input_array):
	# This function displays a 1d or 2d numpy array (matrix).
	# Farhad Kamangar Sept. 2019
	if input_array.ndim==1:
		num_of_columns,=input_array.shape
		temp_matrix=input_array.reshape((1, num_of_columns))
	elif input_array.ndim>2:
		print("Input matrix dimension is greater than 2. Can not display as table")
		return
	else:
		temp_matrix=input_array
	number_of_rows,num_of_columns = temp_matrix.shape
	plt.figure()
	tb = plt.table(cellText=np.round(temp_matrix,2), loc=(0,0), cellLoc='center')
	for cell in tb.properties()['child_artists']:
	    cell.set_height(1/number_of_rows)
	    cell.set_width(1/num_of_columns)

	ax = plt.gca()
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()
class Hebbian(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,transfer_function="Hard_limit",seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit" ,  "Sigmoid", "Linear".
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self.transfer_function=transfer_function
        self._initialize_weights()
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = np.random.randn(self.number_of_classes,self.input_dimensions+1)
    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = np.zeros([self.number_of_classes,self.input_dimensions+1])
    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]. This array is a numerical array.
        """
        #Insert first row of ones to X Matrix
        X=np.insert(X,0,np.ones(np.size(X,1)),0)
        #print(X)
        #Output matrix 'a' generated
        a=np.matmul(self.weights,X)
        #print(a)
        #Applying the hardlimit to 'a'
        if self.transfer_function == 'Hard_limit':

            for x in range(0, np.size(a,0)):
                for y in range(0,np.size(a,1)):
                    if a[x, y] >= 0:
                        a[x, y] = 1
                    else:
                        a[x,y]=0

        elif self.transfer_function == 'Sigmoid':

            for x in range(0, np.size(a,0)):
                for y in range(0,np.size(a,1)):
                    exponent=Math.exp(-a[x,y])
                    res=1/(1+exponent)
                    a[x, y] = res
            
        return a

    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)
    def train(self, X, y, batch_size=1,num_epochs=10,  alpha=0.1,gamma=0.9,learning="Delta"):
        print("")
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        
        X=np.insert(X,0,np.ones(np.size(X,1)),0)
        Y = to_categorical(y)
        #Find transpose of X
        X_transpose=X.transpose()
        #Calculate number of columns in X_transpose
        X_transpose_columns=np.size(X_transpose,1)
        X_transpose_rows=np.size(X_transpose,0)
        #Find transpose of Y
        #Y_transpose=Y.transpose()
        #Calculate number of columns in Y_transpose
        Y_rows=np.size(Y,0)
        Y_columns=np.size(Y,1)
        #Y=Y.transpose()
        #Iterate for num_epochs times
        if learning == "Delta":

            for epoch in range(num_epochs):
                for index in range(0,X_transpose_rows,batch_size):
                    submatrix_X=X_transpose[index:(batch_size+index)]
                    p=submatrix_X.transpose()
                    a=np.matmul(self.weights,p)

                    if self.transfer_function == 'Hard_limit':

                        for x in range(0, np.size(a,0)):
                            for y in range(0,np.size(a,1)):
                                if a[x, y] >= 0:
                                    a[x, y] = 1
                                else:
                                    a[x,y]=0

                    elif self.transfer_function == 'Sigmoid':

                        for x in range(0, np.size(a,0)):
                            for y in range(0,np.size(a,1)):
                                exponent=Math.exp(-a[x,y])
                                res=1/(1+exponent)
                                a[x, y] = res
                    submatrix_Y=Y[index:(batch_size+index)]
                    submatrix_Y=submatrix_Y.transpose()
                    error=np.subtract(submatrix_Y,a)
                    term1=np.matmul(error,submatrix_X)
                    self.weights=self.weights+(alpha*term1)

        elif learning == "Filtered":

            for epoch in range(num_epochs):
                for index in range(0,X_transpose_rows,batch_size):
                    submatrix_X=X_transpose[index:(batch_size+index)]
                    #p=submatrix_X.transpose()
                    #a=np.matmul(self.weights,p)

                    
                    submatrix_Y=Y[index:(batch_size+index)]
                    submatrix_Y=submatrix_Y.transpose()
                    
                    term1=np.matmul(submatrix_Y,submatrix_X)
                    self.weights=((1-gamma)*self.weights)+(alpha*term1)

        elif learning == "Unsupervised_hebb":

            for epoch in range(num_epochs):
                for index in range(0,X_transpose_rows,batch_size):
                    submatrix_X=X_transpose[index:(batch_size+index)]
                    p=submatrix_X.transpose()
                    a=np.matmul(self.weights,p)

                    if self.transfer_function == 'Hard_limit':

                        for x in range(0, np.size(a,0)):
                            for y in range(0,np.size(a,1)):
                                if a[x, y] >= 0:
                                    a[x, y] = 1
                                else:
                                    a[x,y]=0

                    elif self.transfer_function == 'Sigmoid':

                        for x in range(0, np.size(a,0)):
                            for y in range(0,np.size(a,1)):
                                exponent=Math.exp(-a[x,y])
                                res=1/(1+exponent)
                                a[x, y] = res
                    #submatrix_Y=Y[index:(300+index)]
                    #submatrix_Y=submatrix_Y.transpose()
                    #error=np.subtract(submatrix_Y,a)
                    term1=np.matmul(a,submatrix_X)
                    self.weights=self.weights+(alpha*term1)
            
                

    def calculate_percent_error(self,X, y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        target=y
        X=np.insert(X,0,np.ones(np.size(X,1)),0)

        number_of_samples=y.size
        #One hot encoding
        #Y = to_categorical(y)
        #print("y",y)
        #print("Y",Y)
        number_of_errors=0
        #X_transpose=X.transpose()
        #Calculate number of columns in X_transpose
        #X_transpose_columns=np.size(X_transpose,1)
        #Find transpose of Y
        #Y_transpose=Y.transpose()
        #print("Y transpose",Y_transpose)
        #Calculate number of columns in Y_transpose
        #Y_rows=np.size(Y,0)
        #Y_columns=np.size(Y,1)
        #Y=np.reshape(Y,(Y_columns,Y_rows))
        a=np.matmul(self.weights,X)
        if self.transfer_function == 'Hard_limit':

            for x in range(0, np.size(a,0)):
                for y in range(0,np.size(a,1)):
                    if a[x, y] >= 0:
                        a[x, y] = 1
                    else:
                        a[x,y]=0

        elif self.transfer_function == 'Sigmoid':

            for x in range(0, np.size(a,0)):
                for y in range(0,np.size(a,1)):
                    try:
                        exponent=Math.exp(-a[x,y])
                        res=1/(1+exponent)
                        a[x, y] = res
                    except:
                        a[x, y] = 0    

        
        #print("a",a)
        print('numsamples',number_of_samples)
        print('out shape',a.shape)

        for i in range(0, number_of_samples):
    
            y_predicted=np.argmax(a[:,i])
            if(y_predicted != target[i]):
                number_of_errors+=1
            
        print("nerrors",number_of_errors)
        print("nsamples",number_of_samples)
        return (number_of_errors/number_of_samples)

    def calculate_confusion_matrix(self,X,y):
        """
        Given a desired (true) output as one hot and the predicted output as one-hot,
        this method calculates the confusion matrix.
        If the predicted class output is not the same as the desired output,
        then it is considered one error.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """
        target=y
        X=np.insert(X,0,np.ones(np.size(X,1)),0)
        confusion_matrix=np.zeros([self.number_of_classes,self.number_of_classes])
        number_of_samples=y.size
        #One hot encoding
        #Y = to_categorical(y)
        # print("y",y)
        # print("Y",Y)
        # number_of_errors=0
        # X_transpose=X.transpose()
        # #Calculate number of columns in X_transpose
        # X_transpose_columns=np.size(X_transpose,1)
        # #Find transpose of Y
        # Y_transpose=Y.transpose()
        # print("Y transpose",Y_transpose)
        # #Calculate number of columns in Y_transpose
        # Y_rows=np.size(Y,0)
        # Y_columns=np.size(Y,1)
        # Y=np.reshape(Y,(Y_columns,Y_rows))
        a=np.matmul(self.weights,X)
        if self.transfer_function == 'Hard_limit':

            for x in range(0, np.size(a,0)):
                for y in range(0,np.size(a,1)):
                    if a[x, y] >= 0:
                        a[x, y] = 1
                    else:
                        a[x,y]=0

        elif self.transfer_function == 'Sigmoid':

            for x in range(0, np.size(a,0)):
                for y in range(0,np.size(a,1)):
                    try:
                        exponent=Math.exp(-a[x,y])
                        res=1/(1+exponent)
                        a[x, y] = res
                    except:
                        a[x, y] = 0    

        
        #print("a",a)
       
        for i in range(0, number_of_samples):
    
            y_predicted=np.argmax(a[:,i])
            # print("i: {0}".format(i))
            # print("target[i]: {0}".format(target[i]))
            # print("y_predicted: {0}".format(y_predicted))
            confusion_matrix[target[i]][y_predicted]+=1
            '''if(y_predicted != target[i]):
                number_of_errors += 1'''

        return confusion_matrix


if __name__ == "__main__":

    # Read mnist data
    number_of_classes = 10
    number_of_training_samples_to_use = 700
    number_of_test_samples_to_use = 100
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_vectorized=((X_train.reshape(X_train.shape[0],-1)).T)[:,0:number_of_training_samples_to_use]
    y_train = y_train[0:number_of_training_samples_to_use]
    X_test_vectorized=((X_test.reshape(X_test.shape[0],-1)).T)[:,0:number_of_test_samples_to_use]
    y_test = y_test[0:number_of_test_samples_to_use]
    number_of_images_to_view=16
    test_x=X_train_vectorized[:,0:number_of_images_to_view].T.reshape((number_of_images_to_view,28,28))
    display_images(test_x)
    input_dimensions=X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit",seed=5)
    # model.initialize_all_weights_to_zeros()
    percent_error=[]
    for k in range (10):
        model.train(X_train_vectorized, y_train,batch_size=300, num_epochs=2, alpha=0.1,gamma=0.1,learning="Delta")
        percent_error.append(model.calculate_percent_error(X_test_vectorized,y_test))
    print("******  Percent Error ******\n",percent_error)
    confusion_matrix=model.calculate_confusion_matrix(X_test_vectorized,y_test)
    print(np.array2string(confusion_matrix, separator=","))
