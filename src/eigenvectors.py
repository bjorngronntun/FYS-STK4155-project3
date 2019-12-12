import numpy as np
import tensorflow as tf
'''
def f(x, A):
    x = x.reshape((-1, 1))
    return_value = np.dot(np.dot(x.transpose(), x)*A + (1 - np.dot(x.transpose(), np.dot(A, x)))*np.eye(A.shape[0]), x)
    return return_value
'''
def eigenvectors(A_np, t_end, Nt, num_hidden_neurons = [90], num_iter=1000, learning_rate=0.01):
    t_np = np.linspace(0, t_end, Nt)
    u_initial_np = np.random.randn(6)
    t = tf.reshape(tf.convert_to_tensor(t_np), shape=(-1, 1))
    u_initial = tf.convert_to_tensor(u_initial_np)
    A = tf.convert_to_tensor(A_np)

    with tf.variable_scope('dnn'):
        num_hidden_layers = np.size(num_hidden_neurons)
        previous_layer = t  # input is just t
        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], activation=tf.nn.tanh)
            previous_layer = current_layer
        dnn_output = tf.layers.dense(previous_layer, 6)

    with tf.name_scope('loss'):
        print('*** t ***', t)
        print('*** x initial ***', u_initial)
        print('*** dnn output ***', dnn_output)
        u_initials = tf.reshape(tf.tile(u_initial, [11]), shape=(11, 6))
        print('u_initials', u_initials)
        g_trial =  (1 - t)*u_initials + t*dnn_output
        print('*** g_trial ***', g_trial)
        g_trial_dt = tf.reshape(tf.stack([tf.gradients(g_trial[:, i], t) for i in range(g_trial.shape[1])]), shape=(11,6))
        print('*** g trial dt ***', g_trial_dt)
        f = lambda x:tf.reshape(tf.matmul(tf.reduce_sum(tf.multiply(x, x))*A + (tf.constant(1.0, dtype='float64') - tf.matmul(tf.transpose(tf.reshape(x, shape=(-1, 1))), tf.matmul(A, tf.reshape(x, shape=(-1, 1)))))*tf.eye(A.shape[0]), tf.reshape(x, shape=(-1, 1))), shape=(-1))
        test = tf.map_fn(f, g_trial)





        # loss = tf.losses.mean_squared_error()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        g_dnn = g_trial.eval()
        print(g_dnn)
if __name__ == '__main__':
    Q = np.random.randn(6, 6)
    A = 0.5*(Q.transpose() + Q)
    eigenvectors(A, 1, 11)
