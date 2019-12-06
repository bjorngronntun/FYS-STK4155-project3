import numpy as np
import tensorflow as tf

def build_network(num_hidden_neurons):
    print('build')
    with tf.variable_scope('dnn'):
        num_hidden_layers = len(num_hidden_neurons)
        

def solve(t_end, Nx, Nt):
    print('solve')
