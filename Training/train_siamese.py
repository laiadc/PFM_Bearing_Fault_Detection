import tensorflow as tf
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Activation, Flatten, Input, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import random


class Bearings_Network:
    """
    Defines the CNN models (both the pretraining and the Siamese),
    and countains fit functions for both models.
    Args:
        type_net (str): Either "Siamese" or "Pretrain"
        base_network (Model): Embedding model
    Attributes:
        num_classes (int): Number of bearing failure modes
        type_net (str): Either "Siamese" or "Pretrain"
        base_network (Model): Embedding model
        tr_pairs (numpy array): Training pairs for the siamese network
        tr_pairs_y (numpy array): Training labels for the siamese network
        te_pairs (numpy array): Test pairs for the siamese network
        te_pairs_y (numpy array): Test labels for the siamese network
        model (Model): Siamese or Pretraining model
    """
    def __init__(self, type_net="Siamese", base_network=None):
        self.num_classes = 4
        self.type_net = type_net
        self.base_network = base_network
        self.tr_pairs = None
        self.tr_pairs_y = None
        self.te_pairs = None
        self.te_pairs_y = None

        if type_net == "Siamese":
            print("Creating ", type_net, " Network")
            self.model = self.siamese_model()
        elif type_net == "Pretrain":
            print("Creating ", type_net, " Network")
            self.model = self.classification_model()
        else:
            print("Type of network ", type_net, " not supported")

    def euclidean_distance(self, vects):
        """
        Calculates the euclidean distance of two sets of vectors
        Args:
            vects (tuple of numpy arrays): Tuple containing the x and y vectors
        Returns:
            (tensor): Mean euclidean distance between x and y
        """
        x, y = vects
        sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

    def eucl_dist_output_shape(self, shapes):
        """
        Returns the desired output shape of the distance layer
        Args:
            shapes(tuple): shapes of the euclidean distance
        Returns:
            (tuple): desired output shape
        """
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def custom_loss(self, y_true, y_pred, margin=1.):
        """
        Contrastive loss function as a keras custom loss.
        Args:
            y_true (tensor): True labels
            y_pred (tensor): Predicted labels
            margin (int): Margin for the contrastive loss
        Returns:
            (tensor): Contrastive loss function for y_true and y_pred
        """
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        y_true = tf.cast(y_true, tf.float32)
        return tf.reduce_mean(y_true * square_pred + (1. - y_true) * margin_square)

    def create_pairs(self, x, y, load):
        """
        Positive and negative pair creation.
        Alternates between positive and negative pairs.

        Args:
            x (numpy array): x data
            y (numpy array): labels of x
            load (numpy array): loads of x
        Returns:
            pairs (numpy array): pairs of x data
            labels (numpy array): 1 if the label is the same, 0 otherwise
            pairs_y (numpy array): y labels of the pairs
            pairs_load (numpy array): loads of the pairs
        """
        digit_indices = [np.where(y == i)[0] for i in range(self.num_classes)]
        pairs = []
        pairs_y = []
        pairs_load = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(self.num_classes)]) - 1
        for d in range(self.num_classes):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                pairs_y += [[y[z1], y[z2]]]
                pairs_load += [[load[z1], load[z2]]]
                inc = random.randrange(1, self.num_classes)
                dn = (d + inc) % self.num_classes
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                pairs_y += [[y[z1], y[z2]]]
                pairs_load += [[load[z1], load[z2]]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels), np.array(pairs_y), np.array(pairs_load)

    def create_base_network(self, input_shape):
        """
        Base network to be shared (eq. to feature extraction).
        Args:
            input_shape (tuple): input shape of the neural network
        Returns:
            (Model): Embeddings model
        """
        inputs = Input(shape=input_shape)
        C1 = Conv1D(filters=32,
                    kernel_size=20,
                    strides=1,
                    padding='same',
                    input_shape=input_shape, activation='relu')(inputs)
        S2 = MaxPooling1D(pool_size=10, strides=4, padding='same')(C1)
        C3 = Conv1D(filters=32,
                    kernel_size=20,
                    strides=1,
                    padding='same',
                    kernel_initializer="uniform", trainable=True, activation='relu')(S2)
        S4 = MaxPooling1D(pool_size=10, strides=4, padding='same')(C3)
        C5 = Conv1D(filters=32,
                    kernel_size=20,
                    strides=1,
                    padding='same',
                    kernel_initializer="uniform", trainable=True, activation='relu')(S4)
        S6 = MaxPooling1D(pool_size=10, strides=4, padding='same')(C5)
        C7 = Conv1D(filters=64,
                    kernel_size=20,
                    strides=1,
                    padding='same',
                    kernel_initializer="uniform", trainable=True, activation='relu')(S6)
        C8 = Flatten()(C7)
        C9 = Dense(32, activation="relu", kernel_initializer="uniform")(C8)
        C9 = Dropout(0.1)(C9)
        C10 = Dense(20, activation="relu", kernel_initializer="uniform")(C9)
        C10 = Dropout(0.1)(C10)

        return Model(inputs, C10)

    def siamese_model(self, length=1000, w_emb=10, w_class=1):
        """
        Creates Siamese model
        Args:
            length (int): length of the input samples
            w_emb (int): weight of the contrastive loss
            w_class (int): weight of the cross entropy loss
        Returns
            model (Model): Compiled Siamese model
        """
        input_shape = (length, 1)
        # base network (embedding) definition
        if not self.base_network:
            print('initializing network')
            self.base_network = self.create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)

        # Layer to compute the euclidean distance between the two outputs of the siamese model
        distance = Lambda(self.euclidean_distance,
                          output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        # Classification output
        output_a = Dense(self.num_classes, activation="softmax", kernel_initializer="uniform")(processed_a)

        output_b = Dense(self.num_classes, activation="softmax", kernel_initializer="uniform")(processed_b)

        # Define inputs and outputs of the model
        model = Model(inputs=[input_a, input_b], outputs=[distance, output_a, output_b])

        # Compile the model using both the cross entropy loss function and the contrastive loss function
        model.compile(
            loss=[self.custom_loss, tf.keras.losses.categorical_crossentropy, tf.keras.losses.categorical_crossentropy],
            optimizer='Adam', metrics=['categorical_accuracy'], loss_weights=[w_emb, w_class, w_class])

        return model

    def classification_model(self, length=1000):
        """
        Creates the classification (pretrain) model.
        Args:
            length (int): length of the input samples
        Returns:
            model (Model): classification model
        """
        # Define input shape
        input_shape = (length, 1)
        # Create embedding network
        if not self.base_network:
            self.base_network = self.create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        processed_a = self.base_network(input_a)

        # Classification layer
        output_a = Dense(self.num_classes, activation="softmax", kernel_initializer="uniform")(processed_a)

        # Define inputs and outputs of the model
        model = Model(inputs=input_a, outputs=output_a)

        # Compile model using cross entropy
        opt = tf.optimizers.Adam(lr=0.0001)
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=opt, metrics=['categorical_accuracy'])
        return model

    def load_weights(self, filepath, filepath_base, type_net="Siamese"):
        """
        Function to load the weight of the model
        Args:
            filepath (str): path to the file containing the weights of the model
            filepath_base (str): path to the file containing the weights of the base model (embedding network)
            type_net (str): Type of network (Pretraining or Siamese)
        Returns:
            -
        """
        self.base_network = tf.keras.models.load_model(filepath_base)
        if type_net == "Siamese":
            self.model = tf.keras.models.load_model(filepath,
                                                    custom_objects={'euclidean_distance': self.euclidean_distance,
                                                                    'eucl_dist_output_shape': self.eucl_dist_output_shape,
                                                                    'custom_loss': self.custom_loss})
        else:
            self.model = tf.keras.models.load_model(filepath)

    def get_training_pairs(self, X_train, y_train, load_train, X_test,
                           y_test, load_test, prop=1):
        """
        Function that creates the training and test pairs to train the siamese network.
        Args:
            X_train (numpy array): training data
            y_train (numpy array): training labels
            load_train (numpy array): training loads
            X_test (numpy array): test data
            y_test (numpy array): test labels
            load_test (numpy array): test loads
            prop (int): Proportion of data to be left out
        Returns:
            tr_pair0 (numpy array): training data to give to the first (copy of the) network
            tr_pair1 (numpy array): training data to give to the second (copy of the) network
            tr_pairs_y0 (numpy array): training data labels to give to the first (copy of the) network
            tr_pairs_y1 (numpy array): training data labels to give to the second (copy of the) network
            tr_y (numpy array): wether the trianing pair components have the same label or not
            te_pair0 (numpy array): test data to give to the first (copy of the) network
            te_pair1 (numpy array): test data to give to the second (copy of the) network
            te_pairs_y0 (numpy array): test data labels to give to the first (copy of the) network
            te_pairs_y1 (numpy array): test data labels to give to the second (copy of the) network
            te_y (numpy array): wether the test pair components have the same label or not
        """
        # create training+test positive and negative pairs
        tr_pairs, tr_y, tr_pairs_y, tr_pairs_load = self.create_pairs(X_train, y_train, load_train)
        te_pairs, te_y, te_pairs_y, te_pairs_load = self.create_pairs(X_test, y_test, load_test)

        self.tr_pairs = tr_pairs
        self.tr_pairs_y = tr_pairs_y
        self.te_pairs = te_pairs
        self.te_pairs_y = te_pairs_y

        # Select a potion of the training set
        if prop < 1:
            size = tr_pairs.shape[0]
            idx, _, _, _ = train_test_split(np.arange(size), tr_pairs_y, test_size=1.0 - prop, random_state=42)
            tr_pairs = tr_pairs[idx, :, :]
            tr_pairs_y = tr_pairs_y[idx, :]
            tr_y = tr_y[idx]
            tr_pairs_load = tr_pairs_load[idx, :]

        tr_pair0 = np.expand_dims(tr_pairs[:, 0], axis=2)
        tr_pair1 = np.expand_dims(tr_pairs[:, 1], axis=2)
        te_pair0 = np.expand_dims(te_pairs[:, 0], axis=2)
        te_pair1 = np.expand_dims(te_pairs[:, 1], axis=2)

        # Reshape labels
        tr_pairs_y0 = np_utils.to_categorical(tr_pairs_y[:, 0], self.num_classes)
        tr_pairs_y1 = np_utils.to_categorical(tr_pairs_y[:, 1], self.num_classes)
        te_pairs_y0 = np_utils.to_categorical(te_pairs_y[:, 0], self.num_classes)
        te_pairs_y1 = np_utils.to_categorical(te_pairs_y[:, 1], self.num_classes)

        return tr_pair0, tr_pair1, tr_pairs_y0, tr_pairs_y1, tr_y, te_pair0, te_pair1, te_pairs_y0, te_pairs_y1, te_y

    def fit_classification(self, X_train, y_train, load_train, X_test, y_test, load_test,
                           epochs=50, batch_size=64, prop=1, patience=45, min_lr=0.00001, save=True,
                           filepath=None, filepath_base=None):
        """
        Fits the classification (pretraining) model.
        Args:
            X_train (numpy array): Training data
            y_train (numpy array): Labels for training data
            load_train (numpy array): Loads for training data
            X_test (numpy array): Test data
            y_test (numpy array): Labels for test data
            load_test (numpy array): Loads for test data
            epochs (int): Training epochs
            batch_size (int): batch size
            prop (int): Proportion of data to be left out
            patience (int): Patience for early stopping
            min_lr (float): minimum learning rate for reduce LR on Plateu
            save (bool): Whether to save the model or not
            filepath (str): path to the file containing the weights of the model
            filepath_base (str): path to the file containing the weights of the base model (embedding network)
        Returns:
            hist (keras history): Fitting history
        """

        # Get training and test data
        tr_pair1, _, tr_pairs_y0, _, _, te_pair1, _, te_pairs_y0, _, _ = self.get_training_pairs(X_train, y_train,
                                                                                                 load_train,
                                                                                                 X_test, y_test,
                                                                                                 load_test, prop)
        reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5,
                                      patience=5, min_lr=min_lr)
        es = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.000001, patience=patience, verbose=1,
                           restore_best_weights=True)

        # Training model without constastive loss

        hist = self.model.fit(tr_pair1,
                              tr_pairs_y0,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(te_pair1, te_pairs_y0), callbacks=[reduce_lr, es])

        # Save model
        if save:
            tf.keras.models.save_model(self.model, filepath)
            tf.keras.models.save_model(self.base_network, filepath_base)
            print("Saved model to disk ")
        return hist

    def fit_siamese(self, X_train, y_train, load_train, X_test, y_test, load_test,
                    epochs=150, batch_size=64, prop=1, patience=5, min_lr=0.00001, save=True,
                    filepath=None, filepath_base=None):

        '''
        Fits the siamese model.
        Args:
            X_train (numpy array): Training data
            y_train (numpy array): Labels for training data
            load_train (numpy array): Loads for training data
            X_test (numpy array): Test data
            y_test (numpy array): Labels for test data
            load_test (numpy array): Loads for test data
            epochs (int): Training epochs
            batch_size (int): batch size
            prop (int): Proportion of data to be left out
            patience (int): Patience for early stopping
            min_lr (float): minimum learning rate for reduce LR on Plateu
            save (bool): Whether to save the model or not
            filepath (str): path to the file containing the weights of the model
            filepath_base (str): path to the file containing the weights of the base model (embedding network)
        Returns:
            hist (keras history): Fitting history
        '''
        # Get training and test data
        tr_pair1, tr_pair2, tr_pairs_y0, tr_pairs_y1, tr_y, te_pair1, te_pair2, te_pairs_y0, te_pairs_y1, te_y = self.get_training_pairs(
            X_train, y_train, load_train,
            X_test, y_test, load_test, prop)

        # Early stopping
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=20, min_lr=min_lr)
        es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=180, verbose=1, restore_best_weights=True)

        # Train model
        hist = self.model.fit([tr_pair1, tr_pair2],
                              [tr_y, tr_pairs_y0, tr_pairs_y1],
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=([te_pair1, te_pair2],
                                               [te_y, te_pairs_y0, te_pairs_y1]), callbacks=[reduce_lr, es])
        # Save model
        if save:
            tf.keras.models.save_model(self.model, filepath)
            tf.keras.models.save_model(self.base_network, filepath_base)
            print("Saved model to disk ")

        return hist

    def fit(self, X_train, y_train, load_train, X_test, y_test, load_test,
            epochs, batch_size=64, prop=1, patience=15, min_lr=0.00001, save=True,
            filepath=None, filepath_base=None):
        """
        Fit method
        Args:
            X_train (numpy array): Training data
            y_train (numpy array): Labels for training data
            load_train (numpy array): Loads for training data
            X_test (numpy array): Test data
            y_test (numpy array): Labels for test data
            load_test (numpy array): Loads for test data
            epochs (int): Training epochs
            batch_size (int): batch size
            prop (int): Proportion of data to be left out
            patience (int): Patience for early stopping
            min_lr (float): minimum learning rate for reduce LR on Plateu
            save (bool): Whether to save the model or not
            filepath (str): path to the file containing the weights of the model
            filepath_base (str): path to the file containing the weights of the base model (embedding network)
        Returns:
            hist (keras history): Fitting history
        """
        if self.type_net == "Siamese":
            hist = self.fit_siamese(X_train, y_train, load_train, X_test, y_test, load_test,
                                    epochs, batch_size, prop, patience, min_lr, save, filepath, filepath_base)

        else:
            hist = self.fit_classification(X_train, y_train, load_train, X_test, y_test, load_test,
                                           epochs, batch_size, prop, patience, min_lr, save, filepath, filepath_base)

        return hist
