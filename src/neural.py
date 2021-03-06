import numpy as np
import tensorflow as tf

def neural(t_end, Nx, Nt, num_hidden_neurons = [90], num_iter=1000, learning_rate=0.01, tolerance=0.001):
    """
    Computes solution of heat equation with predefined initial/boundary
    conditions, using a neural network
    Inputs:         t_end       float, max value of t
                    Nx          int, number of grid lines in x direction
                    Nt          int, number of grid lines in t direction
                    num_hidden_neurons  list: Numbers of neurons in hidden layers
                    num_iter            int: Number of iterations for training
                    learning_rate       float: Learning rate for training
                    tolerance           float: Training stops current value of loss function drops below this


    Returns:        solution    np array, function values on grid
    """
    print('Neural')
    x_np = np.linspace(0, 1, Nx)
    t_np = np.linspace(0, t_end, Nt)

    X, T = np.meshgrid(x_np, t_np)

    x, t = X.ravel(), T.ravel()

    zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)), shape=(-1, 1))
    x = tf.reshape(tf.convert_to_tensor(x), shape=(-1, 1))
    t = tf.reshape(tf.convert_to_tensor(t), shape=(-1, 1))

    points = tf.concat([x, t], 1)

    X = tf.convert_to_tensor(X)
    T = tf.convert_to_tensor(T)

    with tf.variable_scope('dnn'):
        num_hidden_layers = np.size(num_hidden_neurons)
        previous_layer = points
        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], activation=tf.nn.tanh)
            previous_layer = current_layer
        dnn_output = tf.layers.dense(previous_layer, 1)

    with tf.name_scope('loss'):
        g_trial = (1 - t)*tf.sin(np.pi*x) + x*(1 - x)*t*dnn_output
        g_trial_dt = tf.gradients(g_trial, t)
        g_trial_d2x = tf.gradients(tf.gradients(g_trial, x), x)
        print(g_trial_dt)
        print(g_trial_d2x)

        loss = tf.losses.mean_squared_error(zeros, g_trial_dt[0] - g_trial_d2x[0])

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    g_dnn = None
    with tf.Session() as sess:
        init.run()
        for i in range(num_iter):
            if i % 1000 == 0:
                current_loss = loss.eval()
                print('Iteration {}. Loss: {}'.format(i, current_loss))
                # Stop early if loss is low:
                if current_loss < tolerance:
                    break
            sess.run(training_op)
        g_dnn = g_trial.eval()
        print(g_dnn)
        print(g_dnn.shape)
        return g_dnn.reshape((Nt, Nx))
if __name__ == '__main__':
    print(neural(1, 10, 10))
