import numpy as np
import matplotlib.pyplot as plt

# Transition Matrix
T = np.array([[0.8, 0.2],
              [0.3, 0.7]])


O_1 = O_2 = O_4 = O_6 = np.array([[0.75, 0.0],
                                  [0.0, 0.2]])

O_3 = O_5 = np.array([[0.25, 0.0],
                      [0.0, 0.8]])

initial_probabilities = np.array([[0.5], [0.5]])

# Observations
Evidence = [O_1, O_2, O_3, O_4, O_5, O_6]

# The forward equation


def forward(t):
    if t == 0:
        return initial_probabilities

    x = Evidence[t-1].dot(T.transpose()).dot(forward(t-1))
    return x / np.sum(x)

# The backward equation


def backward(k, t):
    if k == t + 1:
        return np.array([[1],
                         [1]])
    return T.dot(Evidence[k-1]).dot(backward(k+1, t))


# Prediction
def Prediction(t, k):
    if k == t:
        return forward(t)
    x = T.transpose().dot(Prediction(t, k-1))
    return x / np.sum(x)


# Smoothing
def Smoothing(k, t):
    # Equation 15.8
    x = forward(k)*backward(k+1, t)
    return x / np.sum(x)

# Viterbi algorithm


def Viterbi(t):

    path = []  # list for storing the path to the most likely sequence
    # Initialize a table for storing the "m" messages
    m = np.array([[[0.0], [0.0]] for i in range(t)])
    m[0] = forward(1)  # The first probabilities are just P(X_1 | e_1)

    path.append(np.argmax(m[0]))  # Save the state for the maximum probability
    for i in range(1, t):

        k = np.argmax(m[i-1])  # Find maximizing argument of predeceding state
        v = m[i-1][k]  # Max value

        x = Evidence[i].dot(T.transpose())*v  # Calculate probabilities

        # Select the correct values from the calculation
        y = np.array([[x[0][k]], [x[1][k]]])
        m[i] = y  # Add probabilities to m
        path.append(np.argmax(y))  # Add state to the path list
    return m, path


def taskB():
    fig, ax = plt.subplots()  # Make a Matplotlib figure that can display the results
    x = []  # List for indexes , t
    y = []  # List for probabilities of "True"
    y2 = []  # List for probabilities of "False"

    for i in range(1, 7):  # Calculate the forward messages and add values to the lists
        x.append(i)
        y.append(forward(i)[0][0])
        y2.append(forward(i)[1][0])

    # Print values for "True" to the console
    print(f"Filtering, t from 1 to 6 \n{y}\n")

    # Plot the points
    ax.plot(x, y, '-o', label='P(xt | e1:t)')
    ax.plot(x, y2, '-o', label='P(¬xt | e1:t)')

    # Add labels with values to make the figure more informative
    for a, b in zip(x, y):
        ax.annotate(s="{:.4f}".format(b),
                    xy=(a, b),
                    textcoords='offset points',
                    xytext=(0, 10),
                    ha=('center'))

    for a, b in zip(x, y2):
        ax.annotate(s="{:.4f}".format(b),
                    xy=(a, b),
                    textcoords='offset points',
                    xytext=(0, 10),
                    ha=('center'))

    ax.set_xlabel('t')
    ax.set_ylabel('Probability')
    ax.set_title("Task B, Filtering")
    ax.legend()
    plt.show()


def taskC():
    fig, ax = plt.subplots()  # Make a Matplotlib figure that can display the results
    x = []  # List for indexes , t
    y = []  # List for probabilities of "True
    y2 = []  # List for probabilities of "False
    for k in range(7, 31):
        x.append(k)
        y.append(Prediction(6, k)[0][0])
        y2.append(Prediction(6, k)[1][0])

    print(f"Prediction from 7 to 30: \n{y}\n")
    ax.plot(x, y, label='P(xt | e1:6)')
    ax.plot(x, y2, label='P(¬xt | e1:6)')

    ax.set_xlabel('t')
    ax.set_ylabel('Probability')
    ax.set_title("Task C, Prediction")
    ax.legend()
    plt.show()


def taskD():
    fig, ax = plt.subplots()  # Make a Matplotlib figure that can display the results
    x = []  # List for indexes ,
    y = []  # List for probabilities of "True
    y2 = []  # List for probabilities of "False
    for k in range(6):
        s = Smoothing(k, 6)
        x.append(k)
        y.append(s[0][0])
        y2.append(s[1][0])

    print(f"Smoothing:\n{y}\n")

    ax.plot(x, y, '-o', label='P(xt | e1:6)')
    ax.plot(x, y2, '-o', label='P(¬xt | e1:6)')
    for a, b in zip(x, y):
        ax.annotate(s="{:.4f}".format(b),
                    xy=(a, b),
                    textcoords='offset points',
                    xytext=(0, 10),
                    ha=('center'))
    for a, b in zip(x, y2):
        ax.annotate(s="{:.4f}".format(b),
                    xy=(a, b),
                    textcoords='offset points',
                    xytext=(0, 10),
                    ha=('center'))
    ax.set_xlabel('t')
    ax.set_ylabel('Probability')
    ax.set_title("Task D, Smoothing")
    ax.legend()
    plt.show()


def taskE():
    print("Viterbi:")
    for i in range(1, 7):
        probs, path = Viterbi(i)

        # Make a nice display of the results in the console
        print(f"t = {i} \nProbabilities:")
        print("-"+"---------"*i)
        x = "|"
        y = "|"
        for m in probs:
            x = x + str(round(m[0][0], 6)).ljust(8) + "|"
            y = y + str(round(m[1][0], 6)).ljust(8) + "|"

        print(x)
        print(y)
        print("-"+"---------"*i)

        print(f"Most Likely Path:\n{path}\n")


# The first 3 tasks will create a Matplotlib plot each. To see the next one and continue the program, close the current plot.
taskB()
taskC()
taskD()
taskE()
