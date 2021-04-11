# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
import random
import math


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

        # TODO: Make necessary changes here. For example, assigning the arguments "input_dim" and "hidden_layer" to
        # variables and so forth.

        self.input_dim = input_dim
        self.hidden_layer = hidden_layer

        self.layers = []

        if(hidden_layer):
            return
        else:
            input_nodes = [Neuron() for i in range(input_dim)]
            output_node = [Neuron(
                (self.input_nodes, [random.uniform(0, 0.1) for i in range(input_dim)]))]

            self.layers.append(self.input_nodes)
            self.layers.append(self.output_node)

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

    def g(t) -> float:
        return 1 / (1 + math.exp(-t))

    def g_prime(t) -> float:
        return g(t)*(1-g(t))

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        # TODO: Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer.

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.

        for i in range(self.epochs):

            for x, y in zip(self.x_train, self.y_train):
                for node, value in zip(self.layers[0], x):
                    node.value = value

                for j in range(1, len(self.layers)):
                    layer = self.layers[j]
                    for n in layer:
                        in_weights = np.array(n.weights)
                        in_values = np.array(map(lambda x: x.value, n.inputs))
                        n.in_j = in_weights.dot(in_values)
                        n.value = g(n.in_j)

                for node in self.layers[-1]:

                    node.delta = g_prime(node.in_j) * (y - node.value)

                for l in range(len(self.layers)-2, 0, -1):
                    for node in self.layers[i]:
                        # TODO: Find weight to node i +1
                        node.delta = g_prime(node.in_j)

        # Line 27 in Figure 18.24 says "return network". Here you do not need to return anything as we are coding
        # the neural network as a class
        pass

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        # TODO: Implement the forward pass.
        return 1  # Placeholder, remove when implementing


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
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


class Neuron:
    # TODO: Decide on class structure
    """
    Class for representing a neuron. The "inputs" array keeps track if incoming links,
    and the "weights" array has the input weights on the corresponding index.
    """

    def __init__(self, incoming_nodes=[]) -> None:

        self.incoming_nodes = incoming_nodes


class Layer:

    def __init__(self, size):
        self.nodes = [Neuron() for _ in range(size)]

    def connect(self, layer: Layer):
        self.next_layer = layer
        layer.previous_layer = self
        for node in self.nodes:
            for n in Layer.nodes:
                n.incoming_nodes.append([node, 0])


if __name__ == '__main__':
    unittest.main()
