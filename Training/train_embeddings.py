import tensorflow as tf
from Triplet_loss import Triplet_Loss


class Train_Embeddings:
    """
    Performs training of the embeddings using triplet loss
    Args:
        net (Model): Model to train
        learning_rate (float): Learning Rate for Adam optimizer
        training_iters (int): Numer of training iterations
        batch_size (int): Batch size
        display_step (int): Number of iterations to wait to print the current performance of the model
        triplet_strategy (str): Either 'batch_all' or 'batch_hard'
        margin (int): Margin for the triplet loss function
        squared (bool): Squared parameter for the triplet loss function
                        (wether to use squared euclidean distances or euclidean distances)
        filepath (str): File path to store and recover the model weights
        restore (bool): If true, it looks for existing weights to reestore them
    Attributes:
        net (Model): Model to train
        learning_rate (float): Learning Rate for Adam optimizer
        training_iters (int): Numer of training iterations
        batch_size (int): Batch size
        display_step (int): Number of iterations to wait to print the current performance of the model
        triplet_strategy (str): Either 'batch_all' or 'batch_hard'
        margin (int): Margin for the triplet loss function
        squared (bool): Squared parameter for the triplet loss function
                        (wether to use squared euclidean distances or euclidean distances)
        optimizer (tf.Optimizer): Adam optimizer for the learning steps
        ckpt (tf.Checkpoint): Checkpoint that stores weights and optimizer state
        manager (tf.CheckpointManager): Controls that not too many checkpoint files are stored
    """

    def __init__(self, net, learning_rate, training_iters, batch_size, display_step, triplet_strategy,
                 margin, squared, filepath, restore=True):
        self.net = net
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.triplet_strategy = triplet_strategy
        self.margin = margin
        self.squared = squared
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

    def get_triplet_loss(self, triplet_loss):
        """
        Calculates the loss function (and the fraction of positive triplets for batch_all strategy)

        Args:
            triplet_loss (Triplet_loss): Triplet loss object

        Returns:
            loss (tensor): Average loss function for the batch
            fraction (tensor): Fraction of positive triplets. It only makes sense for the batch all strategy.
                         For batch hard strategy it returns zero.
        """
        # Define triplet loss
        if self.triplet_strategy == "batch_all":
            loss, fraction, _ = triplet_loss.batch_all_triplet_loss()
            return loss, fraction

        elif self.triplet_strategy == "batch_hard":
            loss, _ = triplet_loss.batch_hard_triplet_loss()
            return loss, 0
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(self.triplet_strategy))
            return 0,0

    @tf.function()
    def run_optimization(self, x, y):
        """
        Performs one step of the learning process. It calculates the loss function and
        applies backpropagation algorithm to update the weights.
        Args:
            x (tensor): Samples of training data used to train the model
            y (tensor): True labels for such samples
        Returns:
            loss (tensor): Triplet loss
            fraction (tensor): Fraction of positive triplets
        """
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            # Forward pass.
            pred = self.net(x, is_training=True)
            # Compute loss.
            triplet_loss = Triplet_Loss(tf.cast(y, tf.float32), pred,
                                        margin=tf.constant(self.margin), squared=tf.constant(self.squared))
            loss, fraction = self.get_triplet_loss(triplet_loss)

        # Variables to update, i.e. trainable variables.
        trainable_variables = self.net.trainable_variables

        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables)

        # Update weights.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss, fraction

    def fit(self, X_train, y_train, save=True):
        """
        Main fit function

        Args:
            X_train (data frame): Training data
            y_train (numpy array): Training labels
            save(bool): If true, the model is saved at the end of the training
        Returns:
            -
        """
        # Use tf.data API to shuffle and batch data.
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_data = train_data.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)

        # Run training for the given number of steps.
        for step, (batch_x, batch_y) in enumerate(train_data.take(self.training_iters), 1):

            # Run the optimization to update W and b values.
            loss, fraction = self.run_optimization(tf.constant(batch_x), tf.constant(batch_y))
            # Display the results
            if step % self.display_step == 0:
                print("step: ", step, "Loss", loss.numpy(), "Fraction positive triplets", fraction.numpy())

        if save:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}".format(save_path))
