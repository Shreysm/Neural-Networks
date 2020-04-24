# Mohan, Shreyas
# 2019-10-27
# using tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each the input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None

    def add_layer(self, num_nodes, activation_function):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param activation_function: Activation function for the layer
         :return: None
         """
        # This is our weight matrix
        w = tf.Variable(tf.random.uniform(shape=(self.input_dimension, num_nodes)))
        # This is our bias vector
        b = tf.Variable(tf.zeros(shape=(num_nodes,)))
        self.activations.append(activation_function)
        self.weights.append(w)
        self.biases.append(b)
        self.input_dimension=num_nodes

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases). Note that the biases shape should be [1][number_of_nodes]
         """
        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number]=weights

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number]=biases

    def set_loss_function(self, loss_fn):
        """
        This function sets the loss function.
        :param loss_fn: Loss function
        :return: none
        """
        self.loss = loss_fn

    def sigmoid(self, x):

        return tf.nn.sigmoid(x)

    def linear(self, x):
        return x

    def relu(self, x):
        out = tf.nn.relu(x)
        return out

    def cross_entropy_loss(self, y, y_hat):
        """
        This function calculates the cross entropy loss
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual outputs values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        num_of_layers=len(self.weights)
        #print(num_of_layers)
        for layer in range(num_of_layers):
            W=self.weights[layer]
            #print('w shape',W.shape)
            b=self.biases[layer]
            #print('b shape',b.shape)  
            #print('x shape',X.shape)
            net=tf.matmul(X,W) + b
            out = self.activations[layer](net)
            X = out
            #print('out shape',out.shape)
        return out
        # for i in range(len(self.activations)):
        #     net = tf.matmul(X,self.weights[i]) + self.biases[i]
        #     out = self.activations[i](net)
        #     X = out

        # return out

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8, regularization_coeff=1e-6):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :param regularization_coeff: regularization coefficient
         :return: None
         """
        #Find transpose of X
        #print('X_train_shape',X_train.shape)
        X_transpose=X_train.transpose()
        #Calculate number of columns in X_transpose
        X_transpose_columns=np.size(X_transpose,1)
        X_transpose_rows=np.size(X_transpose,0)

        for epoch in range(num_epochs):
            for index in range(0,X_transpose_rows,batch_size):
                
                
                submatrix_X=X_train[index:(batch_size+index)]
                p=submatrix_X
                #print('submatriX_shape',p.shape)

                submatrix_Y=y_train[index:(batch_size+index)]
                submatrix_Y=submatrix_Y.transpose()
                #error=np.subtract(submatrix_Y,a)
                #term1=np.matmul(error,submatrix_X)
                #a=np.matmul(self.weights,p)
                with tf.GradientTape() as tape:
                    predictions = self.predict(p)
                    loss = self.loss(submatrix_Y, predictions)
                    # Note that `tape.gradient` works with a list as well (w, b).
                    w=self.weights
                    b=self.biases
                    dloss_dw, dloss_db = tape.gradient(loss, [w, b])
                    #print('dloss_dw',dloss_dw)
                for layer in range(len(self.weights)):
                    self.weights[layer].assign_sub(alpha * dloss_dw[layer])
                    self.biases[layer].assign_sub(alpha * dloss_db[layer])

        


    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """

        number_of_samples=y.size
        num_of_layers=len(self.weights)
        #print(num_of_layers)
        number_of_errors=0
        target=y
        for layer in range(num_of_layers):
            W=self.weights[layer]
            #print('w shape',W.shape)
            b=self.biases[layer]
            #print('b shape',b.shape)  
            #print('x shape',X.shape)
            net=tf.matmul(X,W) + b
            out = self.activations[layer](net)
            X = out

            #print('numsamples',number_of_samples)
            #print('out shape',out.shape)

        for i in range(0, number_of_samples):

            y_predicted=np.argmax(out[i,:])
            if(y_predicted != target[i]):
                number_of_errors+=1

        return (number_of_errors/number_of_samples)

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """
        target=y
       
        number_of_samples=y.size
        num_of_layers=len(self.weights)

        for layer in range(num_of_layers):
            W=self.weights[layer]
            #print('w shape',W.shape)
            b=self.biases[layer]
            #print('b shape',b.shape)  
            #print('x shape',X.shape)
            net=tf.matmul(X,W) + b
            out = self.activations[layer](net)
            X = out

        number_of_classes=np.size(out,1)
        confusion_matrix=np.zeros([number_of_classes,number_of_classes])
        for i in range(0, number_of_samples):
    
            y_predicted=np.argmax(out[i,:])
            # print("i: {0}".format(i))
            # print("target[i]: {0}".format(target[i]))
            # print("y_predicted: {0}".format(y_predicted))
            confusion_matrix[y_predicted][target[i]]+=1
           

        return confusion_matrix


if __name__ == "__main__":
#    from tensorflow.keras.datasets import mnist
#
#    np.random.seed(seed=1)
#    (X_train, y_train), (X_test, y_test) = mnist.load_data()
#    # Reshape and Normalize data
#    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
#    y_train = y_train.flatten().astype(np.int32)
#    input_dimension = X_train.shape[1]
#    indices = list(range(X_train.shape[0]))
#    # np.random.shuffle(indices)
#    number_of_samples_to_use = 500
#    X_train = X_train[indices[:number_of_samples_to_use]]
#    y_train = y_train[indices[:number_of_samples_to_use]]
#    multi_nn = MultiNN(input_dimension)
#    number_of_classes = 10
#    activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
#    number_of_neurons_list = [50, 20, number_of_classes]
#    for layer_number in range(len(activations_list)):
#        multi_nn.add_layer(number_of_neurons_list[layer_number], activation_function=activations_list[layer_number])
#    for layer_number in range(len(multi_nn.weights)):
#        W = multi_nn.get_weights_without_biases(layer_number)
#        W = tf.Variable((np.random.randn(*W.shape)) * 0.1, trainable=True)
#        multi_nn.set_weights_without_biases(W, layer_number)
#        b = multi_nn.get_biases(layer_number=layer_number)
#        b = tf.Variable(np.zeros(b.shape) * 0, trainable=True)
#        multi_nn.set_biases(b, layer_number)
#    multi_nn.set_loss_function(multi_nn.cross_entropy_loss)
#    percent_error = []
#    for k in range(10):
#        multi_nn.train(X_train, y_train, batch_size=100, num_epochs=20, alpha=0.8)
#        percent_error.append(multi_nn.calculate_percent_error(X_train, y_train))
#    confusion_matrix = multi_nn.calculate_confusion_matrix(X_train, y_train)
#    print("Percent error: ", np.array2string(np.array(percent_error), separator=","))
#    print("************* Confusion Matrix ***************\n", np.array2string(confusion_matrix, separator=","))

    np.random.seed(seed=1)
    input_dimension = 4
    number_of_samples=7
    number_of_layers=3
    number_of_nodes_in_layers_list=list(np.random.randint(3,high=15,size=(number_of_layers,)))
    number_of_nodes_in_layers_list[-1]=5
    multi_nn = MultiNN(input_dimension)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        multi_nn.add_layer(number_of_nodes_in_layers_list[layer_number], activation_function=multi_nn.sigmoid)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        W=multi_nn.get_weights_without_biases(layer_number)
        W=np.random.randn(*W.shape)
        multi_nn.set_weights_without_biases(W,layer_number)
        b = multi_nn.get_biases(layer_number)
        b = np.random.randn(*b.shape)
        multi_nn.set_biases(b, layer_number)
    X=np.random.randn(number_of_samples,input_dimension)
    Y=multi_nn.predict(X)
    assert np.allclose(Y.numpy(),np.array( \
        [[0.8325159, 0.66208966, 0.5367257, 0.78715318, 0.61613198],
         [0.84644084, 0.61953579, 0.66037588, 0.80119257, 0.5786144],
         [0.76684143, 0.66854621, 0.44115054, 0.79724872, 0.78264913],
         [0.79286549, 0.54751305, 0.53022196, 0.79516922, 0.60459471],
         [0.73018733, 0.59293959, 0.24156932, 0.68624113, 0.64357039],
         [0.77498083, 0.64774851, 0.39803303, 0.74039891, 0.69847998],
         [0.79041249, 0.6593798, 0.38533034, 0.80670202, 0.62567229]]),rtol=1e-3, atol=1e-3)
