# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os


class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        OUTPUT_SIZE = 1

        self.input_dim = input_dim

        # Random generator for generating initial values
        rng = np.random.default_rng(123456)

        """
        The next lines sets the variables that will be used to keep track of the state of the network.
        All representations and calculations are done with numpy matrices and vectors to improve runtime.

        W: List contaning matrices with the weights between layers.
           The element W[l]_ij represents the weight from node i in layer l to node j in layer l+1
           Initialized with random values between -0.5 and 0.5

        D: List containing lists of deltas for the output layer and optionally the hidden layer.

        B: List of biases for each layer. Initialized with random values between -0.5 and 0.5

        A: List of output values for nodes in each layer.

        In: List of summed weighted input values for nodes in each layer.

        No_Layers: Total number of layers, including input layer.

        """

        if (hidden_layer):
            self.W = [rng.random((self.input_dim, self.hidden_units), np.float32)-0.5,
                      rng.random((self.hidden_units, OUTPUT_SIZE), np.float32)-0.5]
            self.D = [np.zeros(self.hidden_units, np.float32),
                      np.zeros(OUTPUT_SIZE, np.float32)]
            self.B = [rng.random(self.hidden_units, np.float32)-0.5,
                      rng.random(OUTPUT_SIZE, np.float32)-0.5]
            self.A = [np.zeros(self.input_dim, np.float32), np.zeros(
                self.hidden_units, np.float32), np.zeros(1, np.float32)]
            self.In = [np.zeros(self.input_dim, np.float32), np.zeros(
                self.hidden_units, np.float32), np.zeros(1, np.float32)]
            self.No_Layers = 3
        else:
            self.W = [rng.random(self.input_dim, np.float32)-0.5]
            self.D = [np.zeros(OUTPUT_SIZE, np.float32)]
            self.B = [rng.random(OUTPUT_SIZE, np.float32)-0.5]
            self.A = [np.zeros(self.input_dim, np.float32),
                      np.zeros(OUTPUT_SIZE, np.float32)]
            self.In = [np.zeros(self.input_dim, np.float32),
                       np.zeros(OUTPUT_SIZE, np.float32)]
            self.No_Layers = 2

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def g(self, t):  # Sigmoid activation function
        return 1 / (1 + np.exp(-t))

    def g_prime(self, t):  # Derivative of sigmoid function
        return self.g(t)*(1-self.g(t))

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        # Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer.

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.
        o = self.No_Layers - 1  # index of output layer

        for _ in range(self.epochs):
            for x, y in zip(self.x_train, self.y_train):

                # Execute forward pass
                self.forward_pass(x)

                # Update Delta in output layer
                self.D[o-1] = self.g_prime(self.In[o]) * (y - self.A[o])

                # Update Deltas in hidden layer
                for l in range(self.No_Layers-2, 0, -1):
                    self.D[l-1] = self.g_prime(self.In[l]) * \
                        (self.W[l] @ self.D[l])

                for l in range(1, self.No_Layers):
                    # Update weights using learning rate, output values and deltas
                    self.W[l-1] = self.W[l-1] + \
                        self.lr * self.A[l-1][:, np.newaxis] * self.D[l - 1]
                    # Could also use self.lr * np.einsum('i,j->ij',self.A[l-1],  self.D[l-1]), but it seems to run a bit slower.

                    # Update biases using learning rate and deltas
                    self.B[l-1] = self.B[l-1] + \
                        self.lr * self.D[l-1]

        # Line 27 in Figure 18.24 says "return network". Here you do not need to return anything as we are coding
        # the neural network as a class

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        # Run forward pass
        self.forward_pass(x)

        # Return the output value of the output node
        return self.A[-1][0]

    def forward_pass(self, x: np.ndarray):

        for i, x_i in enumerate(x):
            # Set output values in the input layer
            self.A[0][i] = x_i

        for j in range(1, self.No_Layers):
            # Calculate weighed sums and add bias
            self.In[j] = self.B[j-1] + self.A[j-1] @ (self.W[j-1])
            # Use activation function to set output values
            self.A[j] = self.g(self.In[j])


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly"""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()
