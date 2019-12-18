import numpy as np
import tensorflow as tf

def eigen(A_np, inputs, num_hidden_neurons, num_iter=1000, learning_rate=0.01, tolerance=0.0001, max_value=True):
    """
    Finds largest or smallest eigenvalue of symmetric matrix using neural network methods
    Inputs:     A_np                np array: Quadratic symmetric np matrix
                inputs              int: Number of neurons in input layer
                num_hidden_neurons  list: Numbers of neurons in hidden layers
                num_iter            int: Number of iterations for training
                learning_rate       float: Learning rate for training
                tolerance           float: Training stops if difference in norm between two iterations falls below this
                max_value           boolean: True if max eigenvalue is wanted, False if min eigenvalue is wanted

    Output:     eigenvector corresponding to min or max eigenvalue

    """
    np.random.seed(21)
    input_tensor = tf.reshape(tf.convert_to_tensor(np.random.randn(inputs)), shape=(-1, 1))
    A = tf.convert_to_tensor(A_np)
    dim = A.shape[0]

    with tf.variable_scope('dnn'):
        num_hidden_layers = np.size(num_hidden_neurons)
        previous_layer = input_tensor
        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], activation=tf.nn.tanh)
            previous_layer = current_layer
        dnn_output = tf.layers.dense(previous_layer, dim)

    with tf.name_scope('loss'):
        output = tf.reshape(dnn_output[-1], shape=(-1, 1))
        raleigh_quotient = (tf.matmul(tf.transpose(output), tf.matmul(A, output)))/(tf.matmul(tf.transpose(output), output))

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        if max_value:
            training_op = optimizer.minimize(-raleigh_quotient)
        else:
            training_op = optimizer.minimize(raleigh_quotient)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        previous = output.eval()
        for i in range(num_iter):
            if i % 100 == 0:
                print('Iteration {}'.format(i))
            sess.run(training_op)
            current = output.eval()
            if np.linalg.norm(current - previous) < tolerance:
                break
            previous = current

    # Normalize output
    return current/np.linalg.norm(current)

if __name__ == '__main__':
    np.random.seed(43)
    Q = np.random.randn(3, 3)
    A = 0.5*(Q.transpose() + Q)
    print(eigen(A, 10, [10], max_value=False))
