import numpy as np
import tensorflow as tf
'''
def f(x, A):
    x = x.reshape((-1, 1))
    return_value = np.dot(np.dot(x.transpose(), x)*A + (1 - np.dot(x.transpose(), np.dot(A, x)))*np.eye(A.shape[0]), x)
    return return_value
'''
def eigenvectors(A_np, t_end, Nt, num_hidden_neurons = [15, 15], num_iter=1000, learning_rate=0.01):
    m, n = Nt, A_np.shape[0]
    t_np = np.linspace(0, t_end, Nt)
    u_initial_np = np.random.randn(n)
    t = tf.reshape(tf.convert_to_tensor(t_np), shape=(-1, 1))
    u_initial = tf.convert_to_tensor(u_initial_np)
    A = tf.convert_to_tensor(A_np)

    with tf.variable_scope('dnn'):
        num_hidden_layers = np.size(num_hidden_neurons)
        previous_layer = t  # input is just t
        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], activation=tf.nn.relu)
            previous_layer = current_layer
        dnn_output = tf.layers.dense(previous_layer, n)

    with tf.name_scope('loss'):
        print('*** t ***', t)
        print('*** x initial ***', u_initial)
        print('*** dnn output ***', dnn_output)
        u_initials = tf.reshape(tf.tile(u_initial, [m]), shape=(m, n))
        print('u_initials', u_initials)
        g_trial =  (1 - t)*u_initials + t*dnn_output
        print('*** g_trial ***', g_trial)
        g_trial_dt = tf.reshape(tf.stack([tf.gradients(g_trial[:, i], t) for i in range(g_trial.shape[1])]), shape=(m, n))
        print('*** g trial dt ***', g_trial_dt)
        # f = lambda x:tf.reshape(tf.matmul(tf.reduce_sum(tf.multiply(x, x))*A + (tf.constant(1.0, dtype='float64') - tf.matmul(tf.transpose(tf.reshape(x, shape=(-1, 1))), tf.matmul(A, tf.reshape(x, shape=(-1, 1)))))*tf.eye(A.shape[0]), tf.reshape(x, shape=(-1, 1))), shape=(-1))
        # test = tf.map_fn(f, g_trial)

        f = lambda x : tf.reshape(tf.matmul((tf.reduce_sum(tf.multiply(x, x))*A    ), tf.reshape(x, shape=(-1, 1))), shape=(1, n))
        print('g trial [0] shape', g_trial[0].shape)
        print('f of g_trial[0]', f(g_trial[0]))

        f_u = f(g_trial[0])
        for i in range(1, g_trial.shape[0]):
            f_u = tf.concat([f_u, f(g_trial[i])], axis=0)
        print('f(u)', f_u)

        zeros = tf.convert_to_tensor(np.zeros((m, n)))

        loss = tf.losses.mean_squared_error(zeros, g_trial_dt + g_trial - f_u)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)






        # loss = tf.losses.mean_squared_error()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        for i in range(num_iter):
            if i % 10 == 0:
                current_loss = loss.eval()
                print('Iteration {}. Loss: {}'.format(i, current_loss))
                #print()
                #print('SOLUTION')
                #print(g_trial.eval())
                if current_loss < 0.01:
                    break

            sess.run(training_op)
        computed_eig = g_trial[-1].eval()
        print('Computed eigenvector', computed_eig/np.linalg.norm(computed_eig))
        print('Computed eigenvalue', ((np.dot(computed_eig, np.dot(A_np, computed_eig)))/(np.dot(computed_eig, computed_eig))))


if __name__ == '__main__':
    Q = np.random.randn(3, 3)
    A = 0.5*(Q.transpose() + Q)
    eigenvectors(A, 1, 401)

    print('np eigenvectors:', np.linalg.eig(A)[1])
    print('np eigenvalues:', np.linalg.eig(A)[0])
