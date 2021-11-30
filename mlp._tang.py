import numpy as np


class MLP:
    " Multi-layer perceptron "

    def __init__(self, sizes, beta=1, momentum=0.9):

        """
        sizes is a list of length four. The first element is the number of features 
                in each samples. In the MNIST dataset, this is 784 (28*28). The second 
                and the third  elements are the number of neurons in the first 
                and the second hidden layers, respectively. The fourth element is the 
                number of neurons in the output layer which is determined by the number 
                of classes. For example, if the sizes list is [784, 5, 7, 10], this means 
                the first hidden layer has 5 neurons and the second layer has 7 neurons. 

        beta is a scalar used in the sigmoid function
        momentum is a scalar used for the gradient descent with momentum
        """
        self.beta = beta
        self.momentum = momentum

        self.nin = sizes[0]  # number of features in each sample
        self.nhidden1 = sizes[1]  # number of neurons in the first hidden layer
        self.nhidden2 = sizes[2]  # number of neurons in the second hidden layer
        self.nout = sizes[3]  # number of classes / the number of neurons in the output layer

        # Initialise the network of two hidden layers
        self.weights1 = (np.random.rand(self.nin + 1, self.nhidden1) - 0.5) * 2 / np.sqrt(self.nin)  # hidden layer 1 [in+1, h1]
        self.weights2 = (np.random.rand(self.nhidden1 + 1, self.nhidden2) - 0.5) * 2 / np.sqrt(      # hidden 2 [h1+1, h2]
            self.nhidden1)  # hidden layer 2
        self.weights3 = (np.random.rand(self.nhidden2 + 1, self.nout) - 0.5) * 2 / np.sqrt(          # output layer [h2+1, out]
            self.nhidden2)  # output layer

    def train(self, inputs, targets, eta, niterations):
        """
        inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.

        targets is a numpy array of shape (num_train, D) containing the training labels
                    consisting of num_train samples each of dimension D.

        eta is the learning rate for optimization
        niterations is the number of iterations for updating the weights

        """
        ndata = np.shape(inputs)[0]  # number of data samples
        # adding the bias
        inputs = np.concatenate((inputs, -np.ones((ndata, 1))), axis=1)

        # numpy array to store the update weights
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
        updatew3 = np.zeros((np.shape(self.weights3)))

        for n in range(niterations):

            #############################################################################
            # TODO: implement the training phase of one iteration which consists of two phases:
            # the forward phase and the backward phase. you will implement the forward phase in
            # the self.forwardPass method and return the outputs to self.outputs. Then compute
            # the error (hints: similar to what we did in the lab). Next is to implement the
            # backward phase where you will compute the derivative of the layers and update
            # their weights.
            #############################################################################

            # forward phase
            self.outputs = self.forwardPass(inputs)

            # Error using the sum-of-squares error function
            error = 0.5 * np.sum((self.outputs - targets) ** 2)
            # error = -np.sum( targets * np.log(self.outputs+ 1e-5) ) / ndata #题目给的是MSE，我写了CE

            if (np.mod(n, 1) == 0):
                print("Iteration: ", n, " Error: ", error)

            # backward phase
            # Compute the derivative of the output layer. NOTE: you will need to compute the derivative of
            # the softmax function. Hints: equation 4.55 in the book.

            # deltao_a = self.outputs-targets  # [9000,10]  # CE+softmax

            deltao_a = np.zeros((ndata,10)) # initial to be (9000,10)

            v_w1 = 0
            v_w2 = 0
            v_w3 = 0

            for i in range(ndata):
                each_output = self.outputs[i] #shape (10)
                each_Kronecker = np.diag(each_output) # Kronecker delta function δij * y
                each_output_dim = np.expand_dims(each_output,axis=0) #shape (1,10)
                each_jacobian = each_output_dim.T * each_output_dim # jacobian matrix for yi*yj
                part_1 = each_output - targets[i] # shape (10)
                part_2 = each_Kronecker - each_jacobian # shape (10,10)
                each_deltaho_a = np.dot(part_1 ,part_2) # (10) dot (10,10) -> (10)
                deltao_a[i] = each_deltaho_a
            deltao_a /= ndata

            # deltao_a = (self.outputs - targets) * self.outputs * ( targets-xx)
            deltaho = np.dot(self.hidden2_addbias.T, deltao_a)  # (h2+1, batch) dot (batch , out) -> (h2+1, out)
            # deltaho =  (self.outputs - targets)

            # ( (batch, out) dot (out,h2+1) ) * (batch,h2+1) * (batch,h2+1) -> (batch, h2+1)
            deltah2_a = np.dot(deltao_a, self.weights3.T) * self.beta * self.hidden2_addbias * (1 - self.hidden2_addbias)
            deltah2 = np.dot(self.hidden1_addbias.T, deltah2_a[:,:-1]) # ( h1+1,batch) dot ( batch, h2) -> (h1+1,h2)

            # compute the derivative of the first hidden layer
            # ( (batch, h2) dot (h2, h1+1) ) * ( batch, h1+1) * ( batch, h1+1) -> (batch,h1+1)
            deltah1_a = np.dot(deltah2_a[:,:-1], self.weights2.T ) * self.beta * self.hidden1_addbias * (1 - self.hidden1_addbias)
            deltah1 =  np.dot(inputs.T,deltah1_a[:,:-1])  # (in+1, batch) dot (batch, h1) -> (in+1, h1)

            # update the weights of the three layers: self.weights1, self.weights2 and self.weights3
            # here you can update the weights as we did in the week 4 lab (using gradient descent)
            # but you can also add the momentum

            # if v_w1 == 0:
            #     v_w1 = deltah1
            #     v_w2 = deltah2
            #     v_w3 = deltaho

            # v_w1 = self.momentum * v_w1 + (1-self.momentum) * deltah1
            # v_w2 = self.momentum * v_w2 + (1 - self.momentum) * deltah2
            # v_w3 = self.momentum * v_w3 + (1 - self.momentum) * deltaho

            updatew1 = eta*deltah1 + self.momentum * updatew1
            updatew2 = eta*deltah2 + self.momentum * updatew2
            updatew3 = eta*deltaho + self.momentum * updatew3
            # updatew1 = eta*v_w1
            # updatew2 = eta*v_w2
            # updatew3 = eta*v_w3

            self.weights1 -= updatew1
            self.weights2 -= updatew2
            self.weights3 -= updatew3
            #############################################################################
            # END of YOUR CODE
            #############################################################################

    def forwardPass(self, inputs):
        """
            inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.
        """
        #############################################################################
        # TODO: Implement the forward phase of the model. It has two hidden layers
        # and the output layer. The activation function of the two hidden layers is
        # sigmoid function. The output layer activation function is the softmax function
        # because we are working with multi-class classification.
        #############################################################################

        def Softmax(y):
            y = y - np.max(y, axis=1, keepdims=True)
            y_exp = np.exp(y)
            y_sum = np.sum(y_exp, axis=1, keepdims=True)
            return y_exp / y_sum

        # def Sigmoid(x,b = self.beta):
        #     return 1.0 / (1.0 + np.exp(-self.beta * x))

        # layer 1
        # compute the forward pass on the first hidden layer with the sigmoid function
        ndata = np.shape(inputs)[0]
        self.hidden1 = np.dot(inputs,self.weights1)  # (batch, in+1) dot (in+1 , h1) -> (batch,h1)
        self.hidden1_sig = 1.0 / (1.0 + np.exp(-self.beta * self.hidden1))
        self.hidden1_addbias = np.concatenate((self.hidden1_sig, -np.ones((ndata, 1))), axis=-1)  # add bias  (batch, h1+1)

        # layer 2
        # compute the forward pass on the second hidden layer with the sigmoid function
        self.hidden2 = np.dot(self.hidden1_addbias,self.weights2) # (batch, h1+1) dot (h1+1, h2) -> (batch, h2)
        self.hidden2_sig = 1.0 / (1.0 + np.exp(-self.beta * self.hidden2))
        self.hidden2_addbias = np.concatenate((self.hidden2_sig, -np.ones((ndata, 1))), axis=-1)  # add bias (batch, h2+1)

        # output layer
        # compute the forward pass on the output layer with softmax function
        outputs = np.dot(self.hidden2_addbias, self.weights3) # (batch, h2+1) dot (h2+1 ,out) -> (batch, out)
        outputs = Softmax(outputs)

        #############################################################################
        # END of YOUR CODE
        #############################################################################
        return outputs

    def evaluate(self, X, y):
        """
            this method is to evaluate our model on unseen samples
            it computes the confusion matrix and the accuracy

            X is a numpy array of shape (num_train, D) containing the testing images
                    consisting of num_train samples each of dimension D.
            y is  a numpy array of shape (num_train, D) containing the testing labels
                    consisting of num_train samples each of dimension D.
        """

        inputs = np.concatenate((X, -np.ones((np.shape(X)[0], 1))), axis=1)
        outputs = self.forwardPass(inputs)
        nclasses = np.shape(y)[1]

        # 1-of-N encoding
        outputs = np.argmax(outputs, 1)
        targets = np.argmax(y, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        # print("The confusion matrix is:")
        # print(cm)
        print("The accuracy is ", np.trace(cm) / np.sum(cm) * 100)

        return cm

