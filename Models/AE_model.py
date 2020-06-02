import tensorflow as tf
from tensorflow.keras import Model, layers


class _Encoder(tf.keras.layers.Layer):
    """
    Subclassed keras tf.keras.layers.Layer API. Presenting the Encoder layer of the Autoencoder
    Args:
      input_size (int): Input size for the CNN
    Attributes:
      input_size (int): Input size for the CNN
      conv1 (layer): First convolutional layer with 128 filters, kernel size of 20,
                    1 stride and relu activation function
      maxpool1 (layer): Max pooling layer with size 2
      dropout1 (layer): Dropout layer with dropout parameter of 0.3
      conv2 (layer): Second convolutional layer with 128 filters, kernel size of 20,
                    1 stride and relu activation function
      maxpool2 (layer): Max pooling layer with size 2
      dropout2 (layer): Dropout layer with dropout parameter of 0.3
      conv3 (layer): Third convolutional layer with 512 filters, kernel size of 10,
                    1 stride and relu activation function
      conv4 (layer): Fourth convolutional layer with 9 filters, kernel size of 5,
                    1 stride and relu activation function
      maxpool3 (layer): Max pooling layer with size 2
    """
    def __init__(self,
                 name='encoder', input_size=4000,
                 **kwargs):
        self.input_size = input_size
        super(_Encoder, self).__init__(name=name, **kwargs)

        self.conv1 = tf.keras.layers.Convolution1D(filters=64, kernel_size=20, strides=1, activation=tf.nn.relu,
                                                    padding='same')
        self.maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid')
        self.dropout1= tf.keras.layers.Dropout(0.3)

        self.conv2 = tf.keras.layers.Convolution1D(filters=128, kernel_size=20, strides=1, activation=tf.nn.relu,
                                                    padding='same')
        self.maxpool2 = tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid')
        self.dropout2 = tf.keras.layers.Dropout(0.3)

        self.conv3 = tf.keras.layers.Convolution1D(filters=256, kernel_size=10, strides=1, activation=tf.nn.relu,
                                                    padding='same')
        self.conv4 = tf.keras.layers.Convolution1D(filters=256, kernel_size=10, strides=1, activation=tf.nn.relu,
                                                    padding='same')
        self.maxpool3 = tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid')

    @tf.function
    def call(self, inputs, is_training=False):
        """
        Forward pass of the encoder layer

        Args:
          inputs (tensor): X data to pass through the network
          is_training (bool): If training, True, otherwise, False

        Returns:
          out (tensor): Output tensor containing the embeddings
        """
        x = tf.reshape(inputs, tf.constant([-1, self.input_size, 1]))
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x, training=is_training)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x, training=is_training)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.maxpool3(x)
        return out


class _Decoder(tf.keras.layers.Layer):
    """
    Subclassed keras tf.keras.layers.Layer API. Presenting the Decoder layer of the Autoencoder
    Args:
      -
    Attributes:
      up1 (layer): 1D upsampling layer with size 2
      conv5 (layer): First convolutional layer with 512 filters, kernel size of 10,
                    1 stride and relu activation function

      dropout5 (layer): Dropout layer with dropout parameter of 0.3
      conv2 (layer): Second convolutional layer with 128 filters, kernel size of 20,
                    1 stride and relu activation function
      up2 (layer): 1D upsampling layer with size 2
      conv6 (layer): Third convolutional layer with 128 filters, kernel size of 20,
                    1 stride and relu activation function
      dropout4 (layer): Dropout layer with dropout parameter of 0.3
      up3 (layer): 1D upsampling layer with size 2
      conv7 (layer): Fourth convolutional layer with 1 filters, kernel size of 20,
                    1 stride and relu activation function
    """

    def __init__(self,
                 name='decoder',
                 **kwargs):
        super(_Decoder, self).__init__(name=name, **kwargs)

        self.up1 = tf.keras.layers.UpSampling1D(2)
        self.conv5 = tf.keras.layers.Convolution1D(filters=256, kernel_size=10, activation=tf.nn.relu, strides=1,
                                                    padding='same')
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        self.conv6 = tf.keras.layers.Convolution1D(filters=256, kernel_size=10, activation=tf.nn.relu, strides=1,
                                                    padding='same')
        
        self.up2 = tf.keras.layers.UpSampling1D(2)
        self.conv6 = tf.keras.layers.Convolution1D(filters=128, kernel_size=20, activation=tf.nn.relu, strides=1,
                                                    padding='same')
        self.dropout4 = tf.keras.layers.Dropout(0.3)

        self.up3 = tf.keras.layers.UpSampling1D(2)
        self.conv7 = tf.keras.layers.Convolution1D(filters=1, kernel_size=20, activation='sigmoid', strides=1,
                                                    padding='same')

    @tf.function
    def call(self, inputs, is_training=False):
        """
        Forward pass of the decoder model

        Args:
          inputs (tensor): X data to pass through the network
          is_training (bool): If training, True, otherwise, False

        Returns:
          out (tensor): Output tensor containing the embeddings
        """
        x = self.up1(inputs)
        x = self.conv5(x)
        x = self.dropout3(x, training=is_training)
        x = self.up2(x)
        x = self.conv6(x)
        x = self.dropout4(x, training=is_training)
        x = self.up3(x)
        out = self.conv7(x)
        return out


class Autoencoder(tf.keras.Model):
    """
    Subclassed keras tf.keras.Model API. Joining the _Decoder and _Encoder.
    Args:
      -
    Attributes:
      encoder (Layer): Encoder layer
      decoder (Layer): Decoder layer
    """
    def __init__(self,
                 name='CNN_AE',
                 **kwargs):
        super(Autoencoder, self).__init__(name=name, **kwargs)
        self.encoder = _Encoder()
        self.decoder = _Decoder()

    @tf.function
    def call(self, inputs, is_training=False):
        """
        Forward pass of the autoencoder model

        Args:
          inputs (tensor): X data to pass through the network
          is_training (bool): If training, True, otherwise, False

        Returns:
          out (tensor): Output tensor containing the embeddings
        """
        x = self.encoder(inputs, is_training)
        out = self.decoder(x, is_training)
        return out
