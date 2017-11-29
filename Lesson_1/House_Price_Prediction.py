import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generation some house sizes between 1000 and 3500 (typical sq ft of house)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# Generate house price from size with a random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# Plot generated house nad size
plt.plot(house_size, house_price, 'bx')
plt.ylabel('Price')
plt.xlabel('Size')
plt.show()


# You need to normalize values to prevent under/overflows.
def normalize(array):
    return (array - array.mean()) / array.std()


# Define number of training samples, 0.7 = 70%. We can take the first 70% since the values are randomized
num_train_samples = math.floor(num_house * 0.7)

# Define training data
train_house_size = np.array(house_size[:num_train_samples])
train_house_price = np.array(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

# Define test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# Set up the TensorFlow placeholders that get updated as we descend down the gradient
tf_house_size = tf.placeholder('float', name='house_size')
tf_house_price = tf.placeholder('float', name='house_price')

# Define the variables holding the size_factors and price we set during training.
# We initialize them to some random values based on the normal distribution.
tf_size_factor = tf.Variable(np.random.randn(), name='size_factor')
tf_price_offset = tf.Variable(np.random.randn(), name='price_offset')

# 2. Define the operations for the predicting values - predicted price = (size_factor * house_size) + price_offset
# Notice, the use of the tensorflow add and multiply functions. These add the operations to the computation graph,
# and the tensorflow methods understand how to deal with Tensors. Therefore do not try to use numpy or other library
# methods.
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_price), tf_price_offset)

# 3. Define the Loss function (MSE)
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_house_price, 2) / (2 * num_train_samples))

# Optimizer learning rate. The size of the steps down the gradient
learning_rate = 0.1

# 4. Define a Gradinet descent optimizer that will minimize the loss defined in the operation 'cost'
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Initializing the graph on the sesion
init = tf.global_variables_initializer()

# Launch the graph in the session
with tf.Session() as sess:
    sess.run(init)

    # set how often to diplay training progress and number of training iterations
    display_every = 2
    num_training_iter = 50

    # keep interating the training data
    for iteration in range(num_training_iter):

        # Fit all training data
        for (x, y) in zip(train_house_size_norm, train_house_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_house_price: y})

        # Display current status
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost,
                         feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})
            print('itearation #:', '%04d' % (iteration + 1), 'cost=', '{:.9f}'.format(c), 'size_factor=',
                  sess.run(tf_size_factor), 'price_offset=', sess.run(tf_price_offset))

    print('Optimization Finished!')
    training_cost = sess.run(tf_cost,
                             feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})
    print('Trained cost=', training_cost, 'size_factor=', sess.run(tf_size_factor), 'price_offset=',
          sess.run(tf_price_offset), '\n')

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()
    train_price_mean = train_house_price.mean()
    train_price_std = train_house_price.std()

    # Plot the graph
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    plt.ylabel('Price')
    plt.xlabel('Size (sq.ft)')
    plt.plot(train_house_size, train_house_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(
                 tf_price_offset)) * train_price_std + train_price_mean,
             label='Learned Regression')
    plt.legend(loc='upper left')
    plt.show()
