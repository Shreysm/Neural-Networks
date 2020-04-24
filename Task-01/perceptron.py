# Mohan, Shreyas
# 2019-09-23

import numpy as np
import itertools

class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        #self._initialize_weights()
        self._initialize_weights()
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = np.random.randn(self.number_of_classes,self.input_dimensions+1)
        #self.print_weights()
        #raise Warning("You must implement _initialize_weights! This function should initialize (or re-initialize) your model weights. Bias should be included in the weights")

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = np.zeros([self.number_of_classes,self.input_dimensions+1])
        #self.print_weights()
        #raise Warning("You must implement this function! This function should initialize (or re-initialize) your model weights to zeros. Bias should be included in the weights")

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]
        """
        #Insert first row of ones to X Matrix
        X=np.insert(X,0,np.ones(np.size(X,1)),0)
        #print(X)
        #Output matrix 'a' generated
        a=np.matmul(self.weights,X)
        #print(a)
        #Applying the hardlimit to 'a'
        for x in range(0, np.size(a,0)):
            for y in range(0,np.size(a,1)):
                if a[x, y] >= 0:
                    a[x, y] = 1
                else:
                    a[x,y]=0
        
        return a
        #raise Warning("You must implement predict. This function should make a prediction on a matrix of inputs")


    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)
        #raise Warning("You must implement print_weights")

    def train(self, X, Y, num_epochs=10, alpha=0.001):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        #Insert first row of ones to X Matrix
        X=np.insert(X,0,np.ones(np.size(X,1)),0)
        #Find transpose of X
        X_transpose=X.transpose()
        #Calculate number of columns in X_transpose
        X_transpose_columns=np.size(X_transpose,1)
        #Find transpose of Y
        Y_transpose=Y.transpose()
        #Calculate number of columns in Y_transpose
        Y_transpose_columns=np.size(Y_transpose,1)
        #Iterate for num_epochs times
        for epoch in range(num_epochs):

            for rowX,rowY in zip(X_transpose,Y_transpose):
                column=rowX.transpose()
                column=np.reshape(column,(X_transpose_columns,1))
                print("column",column)
                a=np.matmul(self.weights,column)

                
                for x in range(0, np.size(a,0)):
                    for y in range(0,np.size(a,1)):
                        if a[x, y] >= 0:
                            a[x, y] = 1
                        else:
                            a[x,y]=0
                print("a",a)
                #Target output matrix
                Ycolumn=rowY.transpose()
                Ycolumn=np.reshape(Ycolumn,(Y_transpose_columns,1))
                print("Ycolumn",Ycolumn)
                #Compute error matrix
                error=np.subtract(Ycolumn,a)
                print("Error",error)
            
                rowX=np.reshape(rowX,(1,X_transpose_columns))
                #Compute new weight matrix
                self.weights=self.weights+(alpha*(np.matmul(error,rowX)))
                
        #raise Warning("You must implement train")

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the output is not hte same as the desired output, Y,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :return percent_error
        """
        
        number_of_samples=0
        X=np.insert(X,0,np.ones(np.size(X,1)),0)
        X_transpose=X.transpose()
        X_transpose_columns=np.size(X_transpose,1)
        Y_transpose=Y.transpose()
        Y_transpose_columns=np.size(Y_transpose,1)
        number_of_errors=0
        for rowX,rowY in zip(X_transpose,Y_transpose):
                column=rowX.transpose()
                column=np.reshape(column,(X_transpose_columns,1))
                print("column",column)
                a=np.matmul(self.weights,column)


                for x in range(0, np.size(a,0)):
                    for y in range(0,np.size(a,1)):
                        if a[x, y] >= 0:
                            a[x, y] = 1
                        else:
                            a[x,y]=0
                print("a",a)
                Ycolumn=rowY.transpose()
                Ycolumn=np.reshape(Ycolumn,(Y_transpose_columns,1))
                print("Ycolumn",Ycolumn)
                error=np.subtract(Ycolumn,a)
                print("Error",error)
                flag=0
                #Check error 
                for x in range(0, np.size(error,0)):
                    for y in range(0,np.size(error,1)):
                        if error[x, y] != 0:
                            flag=1
                if flag==1:
                    number_of_errors+=1
                number_of_samples+=1
        print("nerrors",number_of_errors)
        print("nsamples",number_of_samples)
        return (number_of_errors/number_of_samples)
                


        #raise Warning("You must implement calculate_percent_error")

if __name__ == "__main__":
    """
    This main program is a sample of how to run your program.
    You may modify this main program as you desire.
    """

    input_dimensions = 2
    number_of_classes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    model.initialize_all_weights_to_zeros()
    print("****** Model weights ******\n",model.weights)
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.weights)