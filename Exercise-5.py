# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
import random
import math
import time


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

        OUTPUT_SIZE = 1

        self.input_dim = input_dim
        self.hidden_layer = hidden_layer

        self.input_layer = Layer(input_dim)

        self.output_layer = Layer(OUTPUT_SIZE)
        if (hidden_layer):
            h_L = Layer(self.hidden_units)
            self.input_layer.connect(h_L)
            h_L.connect(self.output_layer)
        else:
            self.input_layer.connect(self.output_layer)

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

    def g(self, t) -> float:
        return 1 / (1 + math.exp(-t))

    def g_prime(self, t) -> float:
        return self.g(t)*(1-self.g(t))

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        # TODO: Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer.

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.

        print(len(self.x_train))
        print(len(self.y_train))
        print(len(self.x_train[0]))
        for _ in range(self.epochs):
            tic = time.perf_counter()
            print(1)
            for x, y in zip(self.x_train, self.y_train):
                print(2)
                print(tic - time.perf_counter())
                for node, value in zip(self.input_layer.nodes, x):
                    node.value = value
                    node.in_v = value

                current_layer = self.input_layer.next_layer
                while (current_layer):

                    for j in current_layer.nodes:
                        in_j = 0.0
                        for i in j.incoming_nodes:
                            in_j += i[0].value * i[1]
                        j.in_v = in_j
                        j.value = self.g(in_j)
                    current_layer = current_layer.next_layer
                print(3)
                print(tic - time.perf_counter())
                for j in self.output_layer.nodes:  # Can support multiple output nodes, but in this case we only have one
                    j.delta = self.g_prime(j.in_v) * (y - j.value)

                current_layer = self.output_layer.previous_layer
                print(4)
                print(tic - time.perf_counter())
                while(current_layer):
                    next_layer = current_layer.next_layer

                    for i in range(len(current_layer.nodes)):
                        d_i = 0.0
                        node_i = current_layer.nodes[i]

                        for j in next_layer.nodes:
                            d_i += self.g_prime(node_i.in_v) * \
                                j.incoming_nodes[i][1] * j.delta

                        node_i.delta = d_i

                    current_layer = current_layer.previous_layer
                print(5)
                print(tic - time.perf_counter())
                current_layer = self.input_layer.next_layer
                while(current_layer):

                    for node in current_layer.nodes:

                        for in_node in node.incoming_nodes:
                            n = in_node[0]
                            w = in_node[1]
                            in_node[1] = w + self.lr * n.value * node.delta
                    current_layer = current_layer.next_layer
                print(6)
                print(tic - time.perf_counter())
        # Line 27 in Figure 18.24 says "return network". Here you do not need to return anything as we are coding
        # the neural network as a class

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        # TODO: Implement the forward pass.

        current_layer = self.input_layer.next_layer
        while (current_layer):

            for j in current_layer.nodes:
                in_j = 0.0
                for i in j.incoming_nodes:
                    in_j += i[0].value * i[1]
                j.in_v = in_j
                j.value = self.g(in_j)
            current_layer = current_layer.next_layer
        return self.output_layer.nodes[0].value


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
        self.value = 0.0

    def __repr__(self):
        return f'Neuron: {self.value}, {self.in_v}'


class Layer:

    def __init__(self, size: int):
        self.next_layer = None
        self.previous_layer = None
        self.nodes = [Neuron() for _ in range(size)]

    def connect(self, layer):
        self.next_layer = layer
        layer.previous_layer = self
        for node in self.nodes:
            for n in layer.nodes:
                # Add links and init weights to a small random number
                n.incoming_nodes.append([node, random.uniform(-0.1, 0.1)])


if __name__ == '__main__':
    unittest.main()
