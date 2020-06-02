import tensorflow as tf


class Training_classifier:
    """
    Performs training of the classifier (and not the embeddings) using cross entropy loss
    Args:
        net (Model): Model to train
        learning_rate (float): Learning Rate for Adam optimizer
        training_iters (int): Numer of training iterations
        batch_size (int): Batch size
        display_step (int): Number of iterations to wait to print the current performance of the model
        filepath (str): Filepath from saved models
        restore (bool): Whether to restore the saved models or not
    Attributes:
        net (Model): Model to train
        learning_rate (float): Learning Rate for Adam optimizer
        training_iters (int): Numer of training iterations
        batch_size (int): Batch size
        display_step (int): Number of iterations to wait to print the current performance of the model
        optimizer (tf.Optimizer): Adam optimizer for the learning steps
        ckpt (tf.Checkpoint): Checkpoint that stores weights and optimizer state
        manager (tf.CheckpointManager): Controls that not too many checkpoint files are stored
    """

    def __init__(self, net, learning_rate, training_iters, batch_size, display_step, filepath, restore=True):
        self.net = net
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=filepath, max_to_keep=3)
        if restore:
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

    def model(self):
        """
        Returns the current model
        Args:
            -
        Returns:
            net (Model): Model with current weights
        """
        return self.net

    @tf.function()
    def cross_entropy_loss(self, x, y):
        """
        Calculates the cross entropy loss to the actual labels (y) and the predictions (x)
        Args:
            x (tensor): Predictions of the model for some samples
            y (tensor): True labels for the predicted samples
        Returns:
            (tensor): Average cross entropy loss
        """
        # Convert labels to int 64 for tf cross-entropy function.
        y = tf.cast(y, tf.int64)
        # Apply softmax to logits and compute cross-entropy.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
        # Average loss across the batch.
        return tf.reduce_mean(loss)

    @tf.function()
    def accuracy(self, y_pred, y_true):
        """
        Calculates the accuracy of the predictions
        Args:
            y_pred (tensor): Predicted labels
            y_true (tensor): True labels
        Returns:
            (tensor): Accuracy
        """
        # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    @tf.function()
    def run_optimization(self, x, y):
        """
        Performs one step of the learning process. It calculates the loss function and
        appies backpropagation algorithm to update the weights.
        Args:
            x (tensor): Samples of training data used to train the model
            y (tensor): True labels for such samples
        Returns:
            -
        """
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            # Forward pass.
            pred = self.net(x, is_training=True)
            # Compute loss.
            loss = self.cross_entropy_loss(pred, y)

        # Variables to update, i.e. trainable variables.
        trainable_variables = self.net.trainable_variables

        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables)

        # Update weights.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def fit(self, X_train, y_train, save=True):
        """
        Main fit function
        Args:
            X_train (data frame): Processed training data
            y_train (numpy array): Labels for training data
            save(bool): If true, the model is saved at the end of the training
        Returns:
        -
        """
        # Freeze embedding
        self.net.freeze()
        # Use tf.data API to shuffle and batch data.
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_data = train_data.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        # Run training for the given number of steps.
        for step, (batch_x, batch_y) in enumerate(train_data.take(self.training_iters), 1):
            # Run the optimization to update W and b values.
            self.run_optimization(batch_x, batch_y)

            if step % self.display_step == 0:
                pred = self.net(batch_x)
                loss = self.cross_entropy_loss(pred, batch_y)
                acc = self.accuracy(pred, batch_y)
                print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

        if save:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}".format(save_path))
