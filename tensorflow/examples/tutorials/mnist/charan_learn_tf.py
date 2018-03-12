import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# getting input data from mnist as 'datasets' into the memory
# Here, one_hot represents that only one of the variable is active (eg: 0001000 or 010000)
mnist = input_data.read_data_sets("/Users/ganeshsr/Desktop/mnist", one_hot=True)

# Number of nodes in the hidden layers in the dnn we're about to form
hidden1 = 100
hidden2 = 100
hidden3 = 100

# num of classes are the number of labels that the dnn is going to predict
n_classes = 10

# The batch size is the batch of train data used for each weight update
batch_size = 100

# Data placeholders for input data
x = tf.placeholder('float') # input data (784 pixels wide  -- this is a 28x28 image flattened into 784 values)
y = tf.placeholder('float') # output labels (len=10)

"""
    Y = Weights*X + biases 
    eg: 
            W         *    X       +     b
    Y = [[1,.....784] * [1....784] +    [1,
         [2,.....784]                    2 
         ....                            ..
         [100....784]                    100]
                 
"""


def neural_network_model(data):
    """
    This method builds a computational model of the dnn
    :param data
    :return: output
    """
    # First hidden layer
    h1_layer = {
        "weights": tf.Variable(tf.random_normal([784, hidden1])),
        "biases": tf.Variable(tf.random_normal([hidden1]))
    }

    # Second hidden layer
    h2_layer = {
        "weights": tf.Variable(tf.random_normal([hidden1, hidden2])),
        "biases": tf.Variable(tf.random_normal([hidden2]))
    }

    # Third hidden layer
    h3_layer = {
        "weights": tf.Variable(tf.random_normal([hidden2, hidden3])),
        "biases": tf.Variable(tf.random_normal([hidden3]))
    }

    # output layer
    output_layer = {
        "weights": tf.Variable(tf.random_normal([hidden3, n_classes])),
        "biases": tf.Variable(tf.random_normal([n_classes]))
    }

    l1 = tf.add(tf.matmul(data, h1_layer["weights"]), h1_layer["biases"])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, h2_layer["weights"]), h2_layer["biases"])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, h3_layer["weights"]), h3_layer["biases"])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer["weights"]), output_layer["biases"])

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)

    # cost = 0.5*(prediction - y)^2 # this is also called loss
    # why is prediction logits here?
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # An optimizer is Stochastic Gradient Descent or Ada... other optimizers
    # The learning rate for this optimizer, which by default is 0.01
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.055).minimize(cost)

    # Number of epochs.
    # 1 epoch is one forward + one back prop
    hm_epochs = 10

    # Create a session and run the optimizer to minimize cost
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)


        # evaluate the results how correct the predictions are
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))




train_neural_network(x)













