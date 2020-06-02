import numpy as np
from sklearn.metrics import mean_absolute_error
import tensorflow as tf


def cal_error(data, model):
    """
    Calculate error for test data
    Args:
      data (numpy araray): Preprocessed data
      model (Model): Trained Autoencoder
    Returns:
     error: Mean Absolute Error for each sample waveform calculated based on model
    """
    error = np.zeros((data.shape[0], 1))
    for i, row in enumerate(data):
        pred = model(row.reshape(1, -1, 1)).numpy()
        error[i, :] = mean_absolute_error(row.flatten(), pred.flatten())
    error = error.reshape(-1, )
    return error


class Training_AE:
    """
    Performs the training of the autoencoder model using mean absolute error loss
    Args:
        net (Model): Model to train
        learning_rate (float): Learning Rate for Adam optimizer
        training_iters (int): Numer of training iterations
        batch_size (int): Batch size
        display_step (int): Number of iterations to wait to print the current performance of the model
        early_stopping (int): Number of epochs to wait for the validation loss to increase before performing early stopping
        filepath (str): File path to store and recover the model weights
        restore (bool): If true, it looks for existing weights to reestore them

    Attributes:
        net (Model): Model to train
        learning_rate (float): Learning Rate for Adam optimizer
        training_iters (int): Numer of training iterations
        batch_size (int): Batch size
        display_step (int): Number of iterations to wait to print the current performance of the model
        stopping_step (int): How many epochs we have waited so far without the validation loss decreasing
        early_stopping (int): Number of epochs to wait for the validation loss to increase before performing early stopping
        filepath (str): File path to store and recover the model weights
        restore (bool): If true, it looks for existing weights to reestore them
        loss (function): Loss function to optimize. In this case, mean absolute error
        optimizer (tf.Optimizer): Adam optimizer for the learning steps
        ckpt (tf.Checkpoint): Checkpoint that stores weights and optimizer state
        manager (tf.CheckpointManager): Controls that not too many checkpoint files are stored
    """

    def __init__(self, net, learning_rate, training_iters, batch_size, display_step, early_stopping=10, filepath=None,
                 restore=True):
        self.net = net
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.stopping_step = 0
        self.early_stopping = early_stopping
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.loss = tf.keras.losses.MeanAbsoluteError()
        self.filepath = filepath
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=filepath, max_to_keep=3)
        if restore:
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

    def loss_val(self, val_batch):
        """
        Computes the validation loss
        Args:
            val_batch (tensor): batch of validation sample
        Returns:
            val_loss(tensor): validation loss
        """
        reconstructed = self.net(val_batch, False)
        val_loss = self.loss(reconstructed, val_batch)
        return val_loss

    def early_stop(self, epoch, val_loss, stop):
        """
        Assesses if we have to stop training
        Args:
            epoch (int): current epoch
            val_loss (tensor): current validation loss
            stop (bool): early stop parameter
        Returns:
            stop(bool): True if the models stops training, false if it continues training
        """
        # Store best validation loss
        if epoch == 0:
            self.best_loss = val_loss
        else:
            if val_loss < self.best_loss:
                self.stopping_step = 0
                self.best_loss = val_loss
            else:
                # If the validation loss does not decrease, we increase the number of stopping steps
                self.stopping_step += 1
        # If such number reaches the maximum, we stop training
        if self.stopping_step == self.early_stopping:
            stop = True
            print('Early stopping was triggered ')
        return stop

    @tf.function()
    def run_optimization(self, x):
        """
        Performs one step of the learning process. It calculates the loss function and
        applies backpropagation algorithm to update the weights.
        Args:
            x (tensor): Samples of training data used to train the model
        Returns:
            loss (tensor): Loss
        """
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            # Forward pass.
            pred = self.net(x)
            # Compute loss.
            loss = self.loss(pred, x)

        # Variables to update, i.e. trainable variables.
        trainable_variables = self.net.trainable_variables

        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables)

        # Update W and b following gradients.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    def fit(self, X_train, X_test, save=True):
        """
        Main fit function

        Args:
            X_train (numpy array): Processed training data
            X_test (numpy array): Processed test data
            save (bool): If true, we save the weights at the end of the training
        Returns:
            -
        """
        # Create train and test datasets
        train_data = tf.data.Dataset.from_tensor_slices((X_train))
        train_data = train_data.shuffle(buffer_size=1024).batch(self.batch_size)

        test_data = tf.data.Dataset.from_tensor_slices((X_test))
        test_data = test_data.shuffle(buffer_size=1024).batch(self.batch_size)

        loss_batch = []
        val_loss_batch = []

        stop = False
        epoch = 0

        # Run training for the given number of steps (and while not early stopping).
        while epoch < self.training_iters and stop == False:
            for step, x_batch_train in enumerate(train_data):
                # Apply backpropagation algorithm
                loss = self.run_optimization(x_batch_train)

            for test_x in test_data:
                # Compute validation loss
                val_loss = self.loss_val(test_x)

            val_loss_batch.append(val_loss.numpy())
            loss_batch.append(loss.numpy())
            stop = self.early_stop(epoch, val_loss, stop)
            epoch += 1

            # Display the result
            if epoch % self.display_step == 0:
                print('Epoch: ', epoch, "Validation loss: ", val_loss.numpy(), "Loss: ", loss.numpy())

        # Save the weights
        if save:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}".format(save_path))
