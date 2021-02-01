from collections import defaultdict

import numpy as np


class Variable:
    def __init__(self, name, no_states, table, parents=[], no_parent_states=[]):
        """
        name (string): Name of the variable
        no_states (int): Number of states this variable can take
        table (list or Array of reals): Conditional probability table (see below)
        parents (list of strings): Name for each parent variable.
        no_parent_states (list of ints): Number of states that each parent variable can take.

        The table is a 2d array of size #events * #number_of_conditions.
        # number_of_conditions is the number of possible conditions (prod(no_parent_states))
        If the distribution is unconditional #number_of_conditions is 1.
        Each column represents a conditional distribution and sum to 1.

        Here is an example of a variable with 3 states and two parents cond0 and cond1,
        with 3 and 2 possible states respectively.
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond0   | cond0(0) | cond0(1) | cond0(2) | cond0(0) | cond0(1) | cond0(2) |
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond1   | cond1(0) | cond1(0) | cond1(0) | cond1(1) | cond1(1) | cond1(1) |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(0) |  0.2000  |  0.2000  |  0.7000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(1) |  0.3000  |  0.8000  |  0.2000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(2) |  0.5000  |  0.0000  |  0.1000  |  1.0000  |  0.6000  |  0.2000  |
        +----------+----------+----------+----------+----------+----------+----------+

        To create this table you would use the following parameters:

        Variable('event', 3, [[0.2, 0.2, 0.7, 0.0, 0.2, 0.4],
                              [0.3, 0.8, 0.2, 0.0, 0.2, 0.4],
                              [0.5, 0.0, 0.1, 1.0, 0.6, 0.2]],
                 parents=['cond0', 'cond1'],
                 no_parent_states=[3, 2])
        """
        self.name = name
        self.no_states = no_states
        self.table = np.array(table)
        self.parents = parents
        self.no_parent_states = no_parent_states

        if self.table.shape[0] != self.no_states:
            raise ValueError(f"Number of states and number of rows in table must be equal. "
                             f"Recieved {self.no_states} number of states, but table has "
                             f"{self.table.shape[0]} number of rows.")

        if self.table.shape[1] != np.prod(no_parent_states):
            raise ValueError(
                "Number of table columns does not match number of parent states combinations.")

        if not np.allclose(self.table.sum(axis=0), 1):
            raise ValueError("All columns in table must sum to 1.")

        if len(parents) != len(no_parent_states):
            raise ValueError(
                "Number of parents must match number of length of list no_parent_states.")

    def __str__(self):
        """
        Pretty string for the table distribution
        For printing to display properly, don't use variable names with more than 7 characters
        """
        width = int(np.prod(self.no_parent_states))
        grid = np.meshgrid(*[range(i) for i in self.no_parent_states])
        s = ""
        for (i, e) in enumerate(self.parents):
            s += '+----------+' + '----------+' * width + '\n'
            gi = grid[i].reshape(-1)
            s += f'|{e:^10}|' + \
                '|'.join([f'{e + "("+str(j)+")":^10}' for j in gi])
            s += '|\n'

        for i in range(self.no_states):
            s += '+----------+' + '----------+' * width + '\n'
            state_name = self.name + f'({i})'
            s += f'|{state_name:^10}|' + \
                '|'.join([f'{p:^10.4f}' for p in self.table[i]])
            s += '|\n'

        s += '+----------+' + '----------+' * width + '\n'

        return s

    def probability(self, state, parentstates):
        """
        Returns probability of variable taking on a "state" given "parentstates"
        This method is a simple lookup in the conditional probability table, it does not calculate anything.

        Input:
            state: integer between 0 and no_states
            parentstates: dictionary of {'parent': state}
        Output:
            float with value between 0 and 1
        """
        if not isinstance(state, int):
            raise TypeError(
                f"Expected state to be of type int; got type {type(state)}.")
        if not isinstance(parentstates, dict):
            raise TypeError(
                f"Expected parentstates to be of type dict; got type {type(parentstates)}.")
        if state >= self.no_states:
            raise ValueError(
                f"Recieved state={state}; this variable's last state is {self.no_states - 1}.")
        if state < 0:
            raise ValueError(
                f"Recieved state={state}; state cannot be negative.")

        table_index = 0
        for variable in self.parents:
            if variable not in parentstates:
                raise ValueError(
                    f"Variable {variable.name} does not have a defined value in parentstates.")

            var_index = self.parents.index(variable)
            table_index += parentstates[variable] * \
                np.prod(self.no_parent_states[:var_index])

        return self.table[state, int(table_index)]


class BayesianNetwork:
    """
    Class representing a Bayesian network.
    Nodes can be accessed through self.variables['variable_name'].
    Each node is a Variable.

    Edges are stored in a dictionary. A node's children can be accessed by
    self.edges[variable]. Both the key and value in this dictionary is a Variable.
    """

    def __init__(self):
        # All nodes start out with 0 edges
        self.edges = defaultdict(lambda: [])
        self.variables = {}                   # Dictionary of "name":TabularDistribution

    def add_variable(self, variable):
        """
        Adds a variable to the network.
        """
        if not isinstance(variable, Variable):
            raise TypeError(f"Expected {Variable}; got {type(variable)}.")
        self.variables[variable.name] = variable

    def add_edge(self, from_variable, to_variable):
        """
        Adds an edge from one variable to another in the network. Both variables must have
        been added to the network before calling this method.
        """
        if from_variable not in self.variables.values():
            raise ValueError(
                "Parent variable is not added to list of variables.")
        if to_variable not in self.variables.values():
            raise ValueError(
                "Child variable is not added to list of variables.")
        self.edges[from_variable].append(to_variable)

    def sorted_nodes(self):
        """
        An implementation of Kahn's algorithm for topological sorting of nodes in a DAG.
        I also use a dictionary for keeping track of how many incoming edges a node has, called inbound_edges.
        Decreasing the number of inbound edges for a node is essentially the same as removing an edge for our usecase. 
        We don't actaully need to care about exactly which edge we are removing, keeping track of the count is enough.

        Returns: List of sorted variables.
        """

        L = []  # Will hold the final sorted list of nodes
        S = []  # Contains the set of nodes without inbound edges

        edgesCopy = self.edges.copy()

        # Create a dictionary to keep track of incoming edges for the nodes. Key: a Variable object, Value: the number of incoming edges for the given node/variable.
        inbound_edges = {node: 0 for node in self.variables.values()}

        for i in edgesCopy:
            for node in edgesCopy[i]:
                inbound_edges[node] += 1

        # Populate S with all nodes that has no incoming edges
        for node, edges in inbound_edges.items():
            if edges == 0:
                S.append(node)

        while S:
            # Remove first element from S, and insert it to L
            n = S.pop(0)
            L.append(n)

            # Iterate through all neighbouring nodes of n
            for m in edgesCopy[n]:
                inbound_edges[m] -= 1
                # If inbound edges becomes zero, add m to S
                if inbound_edges[m] == 0:
                    S.append(m)

        # If there are any nodes that still has an inbound edge, we have a cycle in the graph
        if sum(edges for edges in inbound_edges.values()):
            raise Exception("There is a cycle in the graph! :(")
        else:
            # Return the sorted nodes
            return L


class InferenceByEnumeration:
    # This class implements the enumeration-ask algorithm for inference in a Bayesian Network

    def __init__(self, bayesian_network):
        self.bayesian_network = bayesian_network
        # Sort the nodes using a variant of Kahn's algorithm
        self.topo_order = bayesian_network.sorted_nodes()

    # The implementation is heavily based in the pseudocode in Russel & Norvig
    def _enumeration_ask(self, X, evidence):

        StatesCount = self.bayesian_network.variables[X].no_states
        e = evidence.copy()

        # initialize a vector for holding the distribution over X
        Q = [0] * StatesCount

        for i in range(StatesCount):
            e[X] = i
            Q[i] = self._enumerate_all(self.topo_order, e)
        alpha = 1/sum(Q)  # normalization factor

        # Use a numpy array to enable easy normalization. Multiplying native lists in Python does not work the same way as what we want here.
        return alpha * np.array(Q)

    # Called by _enumeration_ask
    # Uses recursive depth-first approach to calcuate the probabilities
    def _enumerate_all(self, vars, evidence):

        # Return 1 if there is no variables to evaluate
        if not vars:
            return 1
        varsCopy = vars.copy()
        # y is the first element of the variables (which are top-sorted). The pop() method also removes the value from the varsCopy list
        y = varsCopy.pop(0)

        e = evidence.copy()

        if y.name in e.keys():
            state = e[y.name]
            return y.probability(state, e) * self._enumerate_all(varsCopy, e)

        val = 0
        states = y.no_states
        # Sum over the states, using the extended evidence set including y.name
        for i in range(states):
            e[y.name] = i
            element = y.probability(i, e) * \
                self._enumerate_all(varsCopy, e)
            val = val + element
        return val

    def query(self, var, evidence={}):
        """
        Wrapper around "_enumeration_ask" that returns a
        Tabular variable instead of a vector
        """
        q = self._enumeration_ask(var, evidence).reshape(-1, 1)
        return Variable(var, self.bayesian_network.variables[var].no_states, q)


def problem3c():
    d1 = Variable('A', 2, [[0.8], [0.2]])
    d2 = Variable('B', 2, [[0.5, 0.2],
                           [0.5, 0.8]],
                  parents=['A'],
                  no_parent_states=[2])
    d3 = Variable('C', 2, [[0.1, 0.3],
                           [0.9, 0.7]],
                  parents=['B'],
                  no_parent_states=[2])
    d4 = Variable('D', 2, [[0.6, 0.8],
                           [0.4, 0.2]],
                  parents=['B'],
                  no_parent_states=[2])

    # print(f"Probability distribution, P({d1.name})")
    # print(d1)
#
    # print(f"Probability distribution, P({d2.name} | {d1.name})")
    # print(d2)
#
    # print(f"Probability distribution, P({d3.name} | {d2.name})")
    # print(d3)
#
    # print(f"Probability distribution, P({d4.name} | {d2.name})")
    # print(d4)

    bn = BayesianNetwork()

    bn.add_variable(d1)
    bn.add_variable(d2)
    bn.add_variable(d3)
    bn.add_variable(d4)
    bn.add_edge(d1, d2)
    bn.add_edge(d2, d3)
    bn.add_edge(d2, d4)

    inference = InferenceByEnumeration(bn)
    posterior = inference.query('A', {'C': 1, 'D': 0})

    # print(f"Probability distribution, P({d3.name} | !{d4.name})")
    print(posterior)


def monty_hall():
    # The monty hall problem as described in Problem 4c)

    # Probability distribution for Prize
    prize = Variable('P', 3, [[1/3], [1/3], [1/3]])

    # Probability distribution for ChosenByGuest
    guest = Variable('G', 3, [[1/3], [1/3], [1/3]])

    # Probability distribution for OpenedByHost given Prize and ChosenByGuest
    host = Variable('H', 3, [[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5],
                             [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5],
                             [0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]],
                    parents=['G', 'P'],
                    no_parent_states=[3, 3])

    print(f"Probability distribution, P({prize.name}) \n{prize}")
    print(f"Probability distribution, P({guest.name}) \n{guest}")
    print(
        f"Probability distribution, P({host.name} | {guest.name}, {prize.name}) \n{host}")

    # Create the network
    bn = BayesianNetwork()

    # Add variables
    bn.add_variable(prize)
    bn.add_variable(guest)
    bn.add_variable(host)

    # Add edges
    bn.add_edge(prize, host)
    bn.add_edge(guest, host)

    # Execute inference by enumeration
    inference = InferenceByEnumeration(bn)
    posterior = inference.query('P', {'G': 0, 'H': 2})

    # Results for Monty Hall
    print(
        f"Probability distribution, P({prize.name} | {guest.name}=0, {host.name}=2) \n{posterior}")


if __name__ == '__main__':
    # problem3c()
    monty_hall()
