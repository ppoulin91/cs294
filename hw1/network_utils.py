import tensorflow as tf

ACTIVATION_FUNCTIONS = {"relu": tf.nn.relu,
                        "sigmoid": tf.nn.sigmoid,
                        "tanh": tf.nn.tanh,
                        "identity": tf.identity}
OPTIMIZERS = {"SGD": tf.train.GradientDescentOptimizer,
              "AdaDelta": tf.train.AdadeltaOptimizer,
              "AdaGrad": tf.train.AdagradOptimizer,
              "Adam": tf.train.AdamOptimizer}


def build_dense_layer(x, in_size, out_size, name, weight_init=None, activation="identity", reg=0.):
    w = tf.get_variable(name + "/w", (in_size, out_size), initializer=weight_init, regularizer=tf.contrib.layers.l2_regularizer(scale=reg))
    b = tf.get_variable(name + "/b", (out_size,), initializer=tf.zeros_initializer)
    a = tf.add(tf.matmul(x, w), b)
    activation_fn = ACTIVATION_FUNCTIONS[activation]
    return activation_fn(a)


def build_model(x, input_size, hidden_sizes, output_size, activation, reg=0.):
    last_layer_outputs = x
    last_layer_size = input_size
    for i, layer_size in enumerate(hidden_sizes):
        with tf.variable_scope("layer_{}".format(i)):
            layer_outputs = build_dense_layer(last_layer_outputs, last_layer_size, layer_size, name="dense_{}".format(activation),
                                              weight_init=tf.orthogonal_initializer, activation=activation, reg=reg)
            last_layer_outputs = layer_outputs
            last_layer_size = layer_size
    with tf.variable_scope("layer_{}".format(len(hidden_sizes))):
        output = build_dense_layer(last_layer_outputs, last_layer_size, output_size, name="dense", weight_init=tf.orthogonal_initializer, reg=reg)
    return output