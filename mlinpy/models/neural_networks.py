import numpy as np
from mlinpy.utils.functions import sigmoid, relu, leaky_relu

class LRAsANeuralNetwork():
    def __init__(self):
        pass

    def __parameters_initializer(self, input_size):
        # initial parameters with zeros
        w = np.zeros((input_size, 1), dtype=float)
        b = 0.0
        return w, b

    def __forward_propagation(self, X):
        m = X.shape[1]
        A = sigmoid(np.dot(self.w.T, X) + self.b)
        return A

    def __compute_cost(self, A, Y):
        m = A.shape[1]
        cost = -np.sum(Y*np.log(A) + (1-Y)*(np.log(1-A))) / m
        return cost

    def cost_function(self, X, Y):
        # use the result from forward propagation and the label Y to compute cost
        A = self.__forward_propagation(X)
        cost = self.__compute_cost(A, Y)
        return cost

    def __backward_propagation(self, A, X, Y):
        m = X.shape[1]
        # backward propagation computes gradients
        dw = np.dot(X, (A-Y).T) / m
        db = np.sum(A-Y) / m
        grads = {"dw": dw, "db": db}
        return grads

    def __update_parameters(self, grads, learning_rate):
        self.w -= learning_rate * grads['dw']
        self.b -= learning_rate * grads['db']

    def fit(self, X, Y, num_iterations, learning_rate, print_cost=False, print_num=100):
        self.w, self.b = self.__parameters_initializer(X.shape[0])
        for i in range(num_iterations):
            # forward_propagation
            A = self.__forward_propagation(X)
            # compute cost
            cost = self.__compute_cost(A, Y)
            # backward_propagation
            grads = self.__backward_propagation(A, X, Y)
            # update parameters
            self.__update_parameters(grads, learning_rate)
            # print cost
            if i % print_num == 0 and print_cost:
                print ("Cost after iteration {}: {:.6f}".format(i, cost))
        return self

    def predict_prob(self, X):
        # result of forward_propagation is the probability
        A = self.__forward_propagation(X)
        return A

    def predict(self, X, threshold=0.5):
        pred_prob = self.predict_prob(X)
        threshold_func = np.vectorize(lambda x: 1 if x > threshold else 0)
        Y_prediction = threshold_func(pred_prob)
        return Y_prediction

    def accuracy_score(self, X, Y):
        pred = self.predict(X)
        return len(Y[pred == Y]) / Y.shape[1]

class SimpleNeuralNetwork():
    # simple neural network with one hidden layer
    def __init__(self, input_size, hidden_layer_size):
        self.paramters = self.__parameter_initailizer(input_size, hidden_layer_size)

    def __parameter_initailizer(self, n_x, n_h):
        # W cannot be initialized with zeros
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(1, n_h) * 0.01
        b2 = np.zeros((1, 1))
        return {'W1': W1,'b1': b1,'W2': W2,'b2': b2}

    def __forward_propagation(self, X):
        W1 = self.paramters['W1']
        b1 = self.paramters['b1']
        W2 = self.paramters['W2']
        b2 = self.paramters['b2']
        # forward propagation
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        cache = {'X': X, 'Z1': Z1,'A1': A1,'Z2': Z2,'A2': A2}
        return A2, cache

    def __compute_cost(self, A2, Y):
        m = A2.shape[1]
        cost = -np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2)) / m
        return cost

    def cost_function(self, X, Y):
        # use the result from forward propagation and the label Y to compute cost
        A2, cache = self.__forward_propagation(X)
        cost = self.__compute_cost(A2, Y)
        return cost

    def __backward_propagation(self, cache, Y):
        A1, A2 = cache['A1'], cache['A2']
        W2 = self.paramters['W2']
        X = cache['X']
        m = X.shape[1]
        # backward propagation computes gradients
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        grads = {'dW1': dW1,'db1': db1,'dW2': dW2,'db2': db2}
        return grads

    def __update_parameters(self, grads, learning_rate):
        self.paramters['W1'] -= learning_rate * grads['dW1']
        self.paramters['b1'] -= learning_rate * grads['db1']
        self.paramters['W2'] -= learning_rate * grads['dW2']
        self.paramters['b2'] -= learning_rate * grads['db2']

    def fit(self, X, Y, num_iterations, learning_rate, print_cost=False, print_num=100):
        for i in range(num_iterations):
            # forward propagation
            A2, cache = self.__forward_propagation(X)
            # compute cost
            cost = self.cost_function(X, Y)
            # backward propagation
            grads = self.__backward_propagation(cache, Y)
            # update parameters
            self.__update_parameters(grads, learning_rate)
            # print cost
            if i % print_num == 0 and print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
        return self

    def predict_prob(self, X):
        # result of forward_propagation is the probability
        A2, _ = self.__forward_propagation(X)
        return A2

    def predict(self, X, threshold=0.5):
        pred_prob = self.predict_prob(X)
        threshold_func = np.vectorize(lambda x: 1 if x > threshold else 0)
        Y_prediction = threshold_func(pred_prob)
        return Y_prediction

    def accuracy_score(self, X, Y):
        pred = self.predict(X)
        return len(Y[pred == Y]) / Y.shape[1]

class BinaryNeuralNetwork():
    def __init__(self, layers_dim, activations, init='he'):
        np.random.seed(3)
        self.layers_dim = layers_dim
        self.__num_layers = len(layers_dim)
        self.activations = activations
        self.input_size = layers_dim[0]
        self.parameters = self.__parameters_initializer(layers_dim, init)
        self.output_size = layers_dim[-1]

    def __parameters_initializer(self, layers_dim, init):
        # special initialzer with np.sqrt(layers_dims[l-1])
        assert (init in {'zero', 'large', 'he', 'other'})
        L = len(layers_dim)
        parameters = {}
        if init == 'zero':
            for l in range(1, L):
                parameters['W'+str(l)] = np.zeros((layers_dim[l], layers_dim[l-1]))
                parameters['b'+str(l)] = np.zeros((layers_dim[l], 1))
        elif init == 'large':
            for l in range(1, L):
                parameters['W'+str(l)] = np.random.randn(layers_dim[l], layers_dim[l-1]) * 10
                parameters['b'+str(l)] = np.zeros((layers_dim[l], 1))
        elif init == 'other':
            for l in range(1, L):
                parameters['W'+str(l)] = np.random.randn(layers_dim[l], layers_dim[l-1]) * np.sqrt((1 / layers_dim[l-1]))
                parameters['b'+str(l)] = np.zeros((layers_dim[l], 1))
        else:
            for l in range(1, L):
                parameters['W'+str(l)] = np.random.randn(layers_dim[l], layers_dim[l-1]) * np.sqrt((2 / layers_dim[l-1]))
                parameters['b'+str(l)] = np.zeros((layers_dim[l], 1))
        return parameters

    def __one_layer_forward(self, A_prev, W, b, activation, keep_prob):
        Z = np.dot(W, A_prev) + b
        if activation == 'sigmoid':
            A = sigmoid(Z)
        if activation == 'relu':
            A = relu(Z)
        if activation == 'leaky_relu':
            A = leaky_relu(Z)
        if activation == 'tanh':
            A = np.tanh(Z)
        if keep_prob == 1:
            D = np.ones((A.shape[0], A.shape[1]))
        else:
            D = np.random.rand(A.shape[0], A.shape[1])
            D = D < keep_prob
            A = A * D
            A = A / keep_prob
        cache = {'Z': Z, 'A': A, 'D': D}
        return A, cache

    def __forward_propagation(self, X, keep_prob, parameters):
        np.random.seed(1)
        caches = []
        A_prev = X
        if keep_prob[0] == 1:
            D = np.ones((X.shape[0], X.shape[1]))
        else:
            D = np.random.rand(X.shape[0], X.shape[1])
            D = D < keep_prob[0]
        caches.append({'A': A_prev, 'D': D})
        # forward propagation by laryer
        for l in range(1, len(self.layers_dim)):
            W, b = parameters['W'+str(l)], parameters['b'+str(l)]
            A_prev, cache = self.__one_layer_forward(A_prev, W, b, self.activations[l-1], keep_prob[l])
            caches.append(cache)
        AL = caches[-1]['A']
        return AL, caches

    def __compute_cost(self, AL, Y, l2, parameters):
        m = Y.shape[1]
        cross_entropy_cost = -np.nansum(Y*np.log(AL) + (1-Y)*np.log(1-AL)) / m
        l2_cost = 0
        for l in range(1, len(self.layers_dim)):
             W = parameters['W'+str(l)]
             l2_cost += np.sum(np.square(W))
        cost = cross_entropy_cost + (l2 / (2*m)) * l2_cost
        return cost

    def sigmoid_backward(self, dA, Z):
        s = sigmoid(Z)
        dZ = dA * s*(1-s)
        return dZ

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def leaky_relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0.01
        return dZ

    def tanh_backward(self, dA, Z):
        s = np.tanh(Z)
        dZ = 1 - s*s
        return dZ

    def __linear_backward(self, dZ, A_prev, W, l2, D, keep_prob):
        m = A_prev.shape[1]
        dW = (np.dot(dZ, A_prev.T) + l2 * W) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        dA_prev = dA_prev * D
        dA_prev = dA_prev / keep_prob
        return dA_prev, dW, db

    def __activation_backward(self, dA, Z, activation):
        assert (dA.shape == Z.shape)
        if activation == 'sigmoid':
            dZ = self.sigmoid_backward(dA, Z)
        if activation == 'relu':
            dZ = self.relu_backward(dA, Z)
        if activation == 'leaky_relu':
            dZ = self.leaky_relu_backward(dA, Z)
        if activation == 'tanh':
            dZ = self.tanh_backward(dA, Z)
        return dZ

    def __backward_propagation(self, caches, Y, l2, keep_prob):
        m = Y.shape[1]
        L = self.__num_layers
        grads = {}
        # backward propagate last layer
        AL, A_prev, D_prev = caches[L-1]['A'], caches[L-2]['A'], caches[L-2]['D']
        grads['dZ'+str(L-1)] = AL - Y
        grads['dA'+str(L-2)], \
        grads['dW'+str(L-1)], \
        grads['db'+str(L-1)] = self.__linear_backward(grads['dZ'+str(L-1)],
                                                      A_prev,
                                                      self.parameters['W'+str(L-1)],
                                                      l2,
                                                      D_prev, keep_prob[L-1])
        # backward propagate by layer
        for l in reversed(range(1, L-1)):
            grads['dZ'+str(l)] = self.__activation_backward(grads['dA'+str(l)],
                                                            caches[l]['Z'],
                                                            self.activations[l-1])
            A_prev = caches[l-1]['A']
            D_prev = caches[l-1]['D']
            grads['dA'+str(l-1)], \
            grads['dW'+str(l)], \
            grads['db'+str(l)] = self.__linear_backward(grads['dZ'+str(l)],
                                                        A_prev,
                                                        self.parameters['W'+str(l)],
                                                        l2,
                                                        D_prev, keep_prob[l])
        return grads

    def __update_parameters(self, grads, learning_rate):
        for l in range(1, self.__num_layers):
            # assert (self.parameters['W'+str(l)].shape == grads['dW'+str(l)].shape)
            # assert (self.parameters['b'+str(l)].shape == grads['db'+str(l)].shape)
            self.parameters['W'+str(l)] -= learning_rate * grads['dW'+str(l)]
            self.parameters['b'+str(l)] -= learning_rate * grads['db'+str(l)]

    def __vector_to_parameters(self, theta):
        parameters = {}
        L = len(self.layers_dim)
        index = 0
        for l in range(1, L):
            w_shape = self.parameters['W'+str(l)].shape
            b_shape = self.parameters['b'+str(l)].shape
            parameters['W'+str(l)] = theta[index:index+(w_shape[0]*w_shape[1])].reshape(w_shape)
            index += w_shape[0]*w_shape[1]
            parameters['b'+str(l)] = theta[index:index+b_shape[0]].reshape(b_shape)
            index += b_shape[0]
        return parameters

    def __gradients_to_vector(self, gradients):
        L = len(self.layers_dim)
        count = 0
        for l in range(1, L):
            w_vector = np.reshape(gradients['dW'+str(l)], (-1,1))
            b_vector = np.reshape(gradients['db'+str(l)], (-1,1))
            if count == 0:
                theta = np.concatenate((w_vector, b_vector), axis=0)
            else:
                theta = np.concatenate((theta, w_vector, b_vector), axis=0)
            count += 1
        return theta

    def __parameters_to_vector(self, parameters):
        L = len(self.layers_dim)
        count = 0
        for l in range(1, L):
            w_vector = np.reshape(parameters['W'+str(l)], (-1,1))
            b_vector = np.reshape(parameters['b'+str(l)], (-1,1))
            if count == 0:
                theta = np.concatenate((w_vector, b_vector), axis=0)
            else:
                theta = np.concatenate((theta, w_vector, b_vector), axis=0)
            count += 1
        return theta

    def __gradient_checking(self, parameters, gradients, X, Y, l2, epsilon=1e-7):
        parameters_vector  = self.__parameters_to_vector(parameters)
        grad = self.__gradients_to_vector(gradients)
        num_parameters = parameters_vector.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))
        keep_prob = [1 for i in range(len(self.layers_dim))]
        # Compute gradapprox
        for i in range(num_parameters):
            thetaplus = np.copy(parameters_vector)
            thetaplus[i][0] = thetaplus[i][0] + epsilon
            parameters_plus = self.__vector_to_parameters(thetaplus)
            AL, _ = self.__forward_propagation(X, keep_prob, parameters_plus)
            J_plus[i] = self.__compute_cost(AL, Y, l2, parameters_plus)

            thetaminus = np.copy(parameters_vector)
            thetaminus[i][0] = thetaminus[i][0] - epsilon
            parameters_minus = self.__vector_to_parameters(thetaminus)
            AL, _ = self.__forward_propagation(X, keep_prob, parameters_minus)
            J_minus[i] = self.__compute_cost(AL, Y, l2, parameters_minus)

            gradapprox[i] = (J_plus[i]-J_minus[i]) / (2 * epsilon)
        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
        difference = numerator / denominator
        if difference > 1e-6:
            print ('There is a mistake in the backward propagation! difference = ' + str(difference))
        else:
            print ('Your backward propagation works perfectly fine! difference = ' + str(difference))
        return difference

    def fit(self, X, Y, num_iterations=300, learning_rate=0.03, l2=0, keep_prob=None, grad_check=False, epsilon=1e-7, print_cost=False, print_num=100):
        if not keep_prob:
            keep_prob = [1 for i in range(len(self.layers_dim))]
        for i in range(num_iterations):
            # forward propagation
            AL, caches = self.__forward_propagation(X, keep_prob, self.parameters)
            # compute cost
            cost = self.__compute_cost(AL, Y, l2, self.parameters)
            # backward propagation
            grads = self.__backward_propagation(caches, Y, l2, keep_prob)
            # gradient checking
            if grad_check:
                self.__gradient_checking(self.parameters, grads, X, Y, l2, epsilon=1e-7)
            # update parameters
            self.__update_parameters(grads, learning_rate)
            # print cost
            if i % print_num == 0 and print_cost:
                    print ("Cost after iteration %i: %f" %(i, cost))
        return self

    def predict_prob(self, X):
        keep_prob = [1 for i in range(len(self.layers_dim))]
        A, _ = self.__forward_propagation(X, keep_prob, self.parameters)
        return A

    def predict(self, X, threshold=0.5):
        pred_prob = self.predict_prob(X)
        threshold_func = np.vectorize(lambda x: 1 if x > threshold else 0)
        Y_prediction = threshold_func(pred_prob)
        return Y_prediction

    def accuracy_score(self, X, Y):
        pred = self.predict(X)
        return len(Y[pred == Y]) / Y.shape[1]
