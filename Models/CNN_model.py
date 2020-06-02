import tensorflow as tf
from tensorflow.keras import Model, layers


# Create TF Model.
class Embedding(Model):
    """
    Class to define the embedding model as a convolutional neural network.
    It extends the Model class, so that it has the properties of such class.

    Args:
      -

    Attributes:
      n_input (int): Length of the input to the CNN
      conv1 (layer): First convolutional layer with 32 filters, kernel size of 10,
                    1 stride and relu activation function
      maxpool1 (layer): Max pooling layer with size 10 and 4 strides
      conv2 (layer): Secocd convolutional layer with 32 filters, kernel size of 10,
                    1 stride and relu activation function
      maxpool2 (layer): Max pooling layer with size 10 and 4 strides
      conv3 (layer): Third convolutional layer with 32 filters, kernel size of 10,
                    1 stride and relu activation function
      maxpool3 (layer): Max pooling layer with size 10 and 4 strides
      conv4 (layer): Fourth convolutional layer with 64 filters, kernel size of 10,
                    1 stride and relu activation function
      flatten (layer): Layer that flattens the output of the previous convolutional layers
      fc1 (layer): First fully connected layer with 32 weights
      dropout (layer): Dropout layer with dropout parameter of 0.1
      fc2 (layer): Second fully connected layer with 20 weights
      dropout2 (layer): Dropout layer with dropout parameter of 0.1
      embeddings (layer): Output layer, the normalized embeddings
    """

    # Set layers.
    def __init__(self, n_input=1000):
        self.n_input = n_input
        super(Embedding, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 10.
        self.conv1 = layers.Conv1D(filters=32, kernel_size=10, strides=1,
                                   activation=tf.nn.relu, padding='same')
        # Max Pooling (down-sampling) with kernel size of 10 and strides of 4. 
        self.maxpool1 = layers.MaxPool1D(10, strides=4, padding='same')

        # Convolution Layer with 32 filters and a kernel size of 10.
        self.conv2 = layers.Conv1D(filters=32, kernel_size=10, strides=1,
                                   activation=tf.nn.relu, padding='same')
        # Max Pooling (down-sampling) with kernel size of 10 and strides of 4. 
        self.maxpool2 = layers.MaxPool1D(10, strides=4)

        self.conv3 = layers.Conv1D(filters=32, kernel_size=10, strides=1,
                                   activation=tf.nn.relu, padding='same')

        # Max Pooling (down-sampling) with kernel size of 10 and strides of 4. 
        self.maxpool3 = layers.MaxPool1D(10, strides=4, padding='same')

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv4 = layers.Conv1D(filters=64, kernel_size=10, strides=1,
                                   activation=tf.nn.relu, padding='same')

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(32)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout = layers.Dropout(rate=0.1)

        # Fully connected layer.
        self.fc2 = layers.Dense(20)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout2 = layers.Dropout(rate=0.1)

        # Output layer, class prediction.
        self.embeddings = layers.BatchNormalization()

    # Set forward pass.
    @tf.function
    def call(self, x, is_training=False):
        """
        Forward pass of the embedding model

        Args:
          x (tensor): X data to pass through the network
          is_training (bool): If training, True, otherwise, False

        Returns:
          out (tensor): Output tensor containing the embeddings
        """
        x = tf.reshape(x, [-1, self.n_input, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=is_training)
        x = self.fc2(x)
        x = self.dropout2(x, training=is_training)
        out = self.embeddings(x)
        return out


class Classification(Model):
    """
    Classification model for bearing failures.

    Args:
      embedding(Model): Embedding model
      n_input (int): Length of the input to the CNN
      num_classes (int): Number of failure modes

    Attributes:
      n_input (int): Length of the input to the CNN
      embedding (Model): Embedding model for feature extraction
      fc1 (layer): Fully connected layer with 64 weights
      fc2 (layer): Fully connected layer with 64 weights
      fc3 (layer): Fully connected layer with 32 weights
      out (layer): Output fully connected layer with 4 weights (the number of failures)
    """

    # Set layers.
    def __init__(self, embedding, n_input=1000, num_classes=4):
        self.n_input = 1000
        super(Classification, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 10.
        self.embedding = embedding
        # Fully-connected layer
        self.fc1 = layers.Dense(64)
        # Fully-connected layer
        self.fc2 = layers.Dense(64)
        # Fully-connected layer
        self.fc3 = layers.Dense(32)
        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass.
    @tf.function
    def call(self, x, is_training=False):
        """
        Forward pass of the classification model

        Args:
          x (tensor): X data to pass through the network
          is_training (bool): If training, True, otherwise, False

        Returns:
          out (tensor): Output tensor containing the prediction of the failures
        """
        x = tf.reshape(x, [-1, self.n_input, 1])
        x = self.embedding(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            out = tf.nn.softmax(out)
        return out

    def freeze(self):
        """
        Freezes the embedding model of the classification model

        Args:
        -

        Returns:
        -
        """
        self.embedding.trainable = False

    def unfreeze(self):
        """
        Unfreezes the embedding model of the classification model

        Args:
        -

        Returns:
        -
        """
        self.embedding.trainable = True
