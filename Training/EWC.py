import tensorflow as tf
from Triplet_loss import Triplet_Loss
import numpy as np


class Fisher_matrix:
    """
    Class to compute and store Fisher information matrix.
    Args:
        X_train (numpy array): Training data
        y_train (numpy array): Labels for training data
        net (Model): Neural network Model
        task (str): Either 'triplet loss' or 'cross entropy'
        batch_size (int): Batch size
        triplet_strategy (str): batch_all or batch_hard for triplet loss, None otherwise
        margin (int): Margin size for triplet loss
        squared (bool): Squared for triplet loss
        num_batches (int): Number of batches
    Attributes:
        X_train (numpy array): Training data
        y_train (numpy array): Labels for training data
        net (Model): Neural network Model
        task (str): Either 'triplet loss' or 'cross entropy'
        batch_size (int): Batch size
        triplet_strategy (str): batch_all or batch_hard for triplet loss, None otherwise
        margin (int): Margin size for triplet loss
        squared (bool): Squared for triplet loss
        weightsA (list of tensors): Weights for task A (change dynamically)
        weights_A_numpy (list of numpy arrays): Weights for task A (static)
        fisher_matrix (tensor): Fisher information matrix
        num_batches (int): Number of batches
    """

    def __init__(self, X_train, y_train, net, task='triplet loss', batch_size=32,
                 triplet_strategy=None, margin=None, squared=None, num_batches=16):
        self.X_train = X_train
        self.y_train = y_train
        self.net = net
        self.task = task
        self.batch_size = batch_size
        self.triplet_strategy = triplet_strategy
        self.margin = margin
        self.squared = squared
        self.num_batches = num_batches
        self.get_weights_A()

        if self.task == 'triplet loss':
            self.fisher_matrix = self.fisher_triplet_loss()
        else:
            self.fisher_matrix = self.fisher_cross_entropy()

    def update_fisher(self, X_train, y_train, net, task='triplet_loss'):
        """
        Update fisher matrix for multiple taks

        Args:
          X_train (numpy array): Training data
          y_train (numpy array): Labels for training data
          net (Model): Current model
          task (str):  batch_all or batch_hard for triplet loss, None otherwise
        Returns:
          -
        """
        self.X_train = X_train
        self.y_train = y_train
        self.task = task
        self.net = net
        self.get_weights_A()

        if self.task == 'triplet loss':
            self.fisher_matrix += self.fisher_triplet_loss()
        else:
            self.fisher_matrix += self.fisher_cross_entropy()

    def get_weights_A(self):
        """
        Store the weights of the model trained for task A

        Args:
          -
        Returns:
          -
        """
        # Get the weights from task A (in a numpy array, so that they are static)
        self.weightsA = []
        # Convert trainable weights to tensors
        for w in self.net.trainable_variables:
            self.weightsA.append(tf.convert_to_tensor(w.numpy()))

    def get_triplet_loss(self, triplet_loss):
        """
        Function to get the triplet loss
        Args:
            triplet_loss (Triplet loss): Triplet loss object
        Returns:
            loss (tensor): Average Loss function
            fraction (tensor): Fraction of positive triplets
            loss_matrix (tensor): Matrix of loss functions for each triplet
        """
        # Define triplet loss
        if self.triplet_strategy == "batch_all":
            loss, fraction, loss_matrix = triplet_loss.batch_all_triplet_loss()
            return loss, fraction, loss_matrix

        elif self.triplet_strategy == "batch_hard":
            loss, loss_matrix = triplet_loss.batch_hard_triplet_loss()
            return loss, 0, loss_matrix

        else:
            raise ValueError("Triplet strategy not recognized: {}".format(triplet_strategy))
            return 0, 0, 0

    def fisher_triplet_loss(self):
        """
        Fisher information matrix trained with triplet loss
        Args:
          -
        Returns:
          gradient_list (tensors): Fisher information matrix calculated as: 1/N sum_i grad(loss_i)^2
        """
        # Create tensorflow datasets
        train_data = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_data = train_data.repeat().shuffle(5000).batch(int(self.batch_size)).prefetch(1)
        num_triplets = 0
        gradient_list = []

        # Calculate the loss function for every triplet
        for step, (batch_xA, batch_yA) in enumerate(train_data.take(int(self.num_batches)), 1):
            # We want to see which variables are more important to perform task A
            # Therefore, we calculate the fisher information matrix with the weights of task A
            # Forward pass.
            pred = self.net(batch_xA, is_training=True)
            # Compute loss.
            triplet_loss = Triplet_Loss(tf.cast(batch_yA, tf.float32), pred,
                                        margin=tf.constant(self.margin), squared=tf.constant(self.squared))
            loss, _, loss_matrix = self.get_triplet_loss(triplet_loss)

            num_triplets += loss_matrix.shape[0]
            # We calculate the triplet loss for a batch, then for each hard triplet in a batch
            # we calculate grad(loss)^2 and add it up
            for i in range(loss_matrix.shape[0]):
                with tf.GradientTape() as g:
                    pred = self.net(batch_xA, is_training=True)
                    triplet_loss = Triplet_Loss(tf.cast(batch_yA, tf.float32), pred,
                                                margin=tf.constant(self.margin), squared=tf.constant(self.squared))
                    loss, _, loss_matrix = self.get_triplet_loss(triplet_loss)
                    # We take the ith element of the batch and compute its gradient
                    loss_valid_slice = tf.slice(tf.reshape(loss_matrix, shape=(loss_matrix.shape[0],)), [i], [1])

                # Variables to update, i.e. trainable variables.
                trainable_variables = self.net.trainable_variables
                # Compute gradients.
                gradients = g.gradient(loss_valid_slice, trainable_variables)

                # We compute the gradients per batches, and them average them
                if len(gradient_list) == 0:
                    for j in range(len(gradients)):
                        gradient_list.append(tf.square(gradients[j]))
                else:
                    for j in range(len(gradients)):
                        gradient_list[j] += tf.square(gradients[j])
        # Divide the fisher information matrix by the number of samples
        for i in range(len(gradients)):
            gradient_list[i] = gradient_list[i] / num_triplets

        return gradient_list

    @tf.function()
    def cross_entropy_loss(self, x, y):
        """
        Calculates the cross entropy loss to the actual labels (y) and the predictions (x)

        Args:
        x (tensor): Predictions of the model for some samples
        y (tensor): True labels for the predicted samples

        Returns:
        loss (tensor): Average cross entropy loss
        """
        # Convert labels to int 64 for tf cross-entropy function.
        y = tf.cast(y, tf.int64)
        # Apply softmax to logits and compute cross-entropy.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
        # Average loss across the batch.
        return tf.reduce_mean(loss)

    def fisher_cross_entropy(self):
        """
        This function returns the gradients of the cross entropy loss function for task A
        Args:
          -
        Returns:
          gradient_list (tensors): Fisher information matrix calculated as: 1/N sum_i grad(loss_i)^2
        """
        # Convert data to tensorflow dataset
        train_data = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        gradient_list = []

        step = 0
        for (batch_xA, batch_yA) in train_data:
            step += 1
            # We want to see which variables are more important to perform task A
            # Therefore, we calculate the fisher information matrix with the weights of task A
            # Try to compute EWC loss
            batch_yA = tf.reshape(batch_yA, shape=(1,))
            with tf.GradientTape() as g:
                # Forward pass.
                pred = self.net(batch_xA, is_training=True)
                # Compute loss.
                loss = self.cross_entropy_loss(pred, batch_yA)

            # Variables to update, i.e. trainable variables.
            trainable_variables = self.net.trainable_variables
            # Compute gradients.
            gradients = g.gradient(loss, trainable_variables)
            # We average the square gradients
            if (len(gradient_list) == 0):
                for grad in gradients:
                    gradient_list.append(tf.square(grad))
            else:
                for i in range(len(gradients)):
                    gradient_list[i] += tf.square(gradients[i])
        # Divide the fisher matrix by the number of samples
        for i in range(len(gradients)):
            gradient_list[i] = gradient_list[i] / (self.X_train.shape[0])

        return gradient_list


class Train_Embeddings_EWC:
    """
    Performs training of the embeddings using triplet loss and EWC

    Args:
        net (Model): Model to train
        fisher_matrix (Fisher_matrix): Object containing fisher information matrix
        learning_rate (float): Learning Rate for Adam optimizer
        training_iters (int): Numer of training iterations
        batch_size (int): Batch size
        display_step (int): Number of iterations to wait to print the current performance of the model
        triplet_strategy (str): Either 'batch_all' or 'batch_hard'
        margin (int): Margin for the triplet loss function
        squared (bool): Squared parameter for the triplet loss function
                        (wether to use squared euclidean distances or euclidean distances)
        lamb (int): Lambda for EWC
        filepath (str): File path to store the model
        restore (bool): Whether to restore the model from checkpoint or not
    Attributes:
        net (Model): Model to train
        fisher_matrix (tensor): Fisher information matrix
        learning_rate (float): Learning Rate for Adam optimizer
        training_iters (int): Numer of training iterations
        batch_size (int): Batch size
        display_step (int): Number of iterations to wait to print the current performance of the model
        triplet_strategy (str): Either 'batch_all' or 'batch_hard'
        margin (int): Margin for the triplet loss function
        squared (bool): Squared parameter for the triplet loss function
                        (wether to use squared euclidean distances or euclidean distances)
        stop (bool): Whether to stop training or not
        lamb (int): Lambda for EWC
        optimizer (tf.Optimizer): Adam optimizer for the learning steps
        weightsA (tensor): Weights for task A
        gradients_A (tensor): Fisher information matrix for task A
        stop (tensor): Whether we have to early stop or not
        ckpt (tf.Checkpoint): Checkpoint that stores weights and optimizer state
        manager (tf.CheckpointManager): Controls that not too many checkpoint files are stored
    """

    def __init__(self, net, fisher_matrix, learning_rate, training_iters, batch_size,
                 display_step, triplet_strategy, margin, squared, lamb, filepath, restore=True):
        self.net = net
        self.fisher_matrix = fisher_matrix
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.triplet_strategy = triplet_strategy
        self.margin = margin
        self.squared = squared
        self.stop = False
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.weightsA = self.fisher_matrix.weightsA

        self.lamb = lamb
        self.gradients_A = self.fisher_matrix.fisher_matrix

        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=filepath, max_to_keep=3)
        if restore:
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

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
            loss, fraction, loss_matrix = triplet_loss.batch_all_triplet_loss()
            return loss, fraction

        elif self.triplet_strategy == "batch_hard":
            loss, loss_matrix = triplet_loss.batch_hard_triplet_loss()
            return loss, 0
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(triplet_strategy))
            return 0,0

    def run_optimization(self, x, y):
        """
        Performs one step of the learning process. It calculates the loss function and
        applies backpropagation algorithm to update the weights.
        Args:
            x (tensor): Samples of training data used to train the model
            y (tensor): True labels for such samples
        Returns:
            final_loss (tensor): Final loss: Triplet loss + Fisher loss
            fisher_loss (tensor): Fisher loss
            loss (tensor): Triplet loss
            fraction (tensor): Fraction of positive triplets
            l2_loss (tensor): L2 loss
        """
        # Wrap computation inside a GradientTape for automatic differentiation.
        weights_A = self.weightsA
        weights_B = self.net.trainable_variables

        with tf.GradientTape() as g:
            # TRIPLET LOSS
            # Forward pass.
            pred = self.net(x, is_training=True)

            # Compute loss.
            triplet_loss = Triplet_Loss(tf.cast(y, tf.float32), pred,
                                        margin=tf.constant(self.margin), squared=tf.constant(self.squared))
            loss, fraction = self.get_triplet_loss(triplet_loss)

            # REGULARIZATION
            g.watch(weights_A)
            g.watch(weights_B)
            fisher_loss = 0
            l2_loss = 0
            i = 0
            # Each grad in gradients is the gradient of the loss with respect to the weights.
            # It is a tensor with the layer weights shape. Therefore,

            for grad in self.gradients_A:
                fisher_diag = grad  # F_i = (\partial L_A/\partial \theta_i)^2
                fisher_loss += tf.reduce_sum(tf.multiply(fisher_diag, tf.square(weights_B[i] - weights_A[i])))
                l2_loss += tf.reduce_sum(tf.square(weights_A[i] - weights_B[i]))
                # fisher loss = \sum_i F_i (theta_i - \theta_i*)^2
                i += 1

            final_loss = loss + self.lamb * fisher_loss

        # Variables to update, i.e. trainable variables.
        trainable_variables = self.net.trainable_variables
        # Compute gradients.
        gradients = g.gradient(final_loss, trainable_variables)
        # Update W and b following gradients.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return final_loss, fisher_loss, loss, fraction, l2_loss

    def early_stop(self, accA, accB):
        """
        Early stopping.
        Args:
            accA (float): Accuracy from task A
            accB (float): Accuracy from task B
        Returns:
            -
        """
        if accB <= accA:
            self.stop = True

    def fit(self, X_trainB, y_trainB, X_testA, y_testA, save=True):
        """
        Main fit function
        Args:
          X_trainB (data frame): Training data (task B)
          y_trainB (numpy array): Training labels (task B)
          X_testA (data frame): Test data (task A)
          y_testA (numpy array): Test labels (task A)
          save (bool): If True, the model is saved at the end of training
        Returns:
          -
        """
        # Use tf.data API to shuffle and batch data.
        train_dataB = tf.data.Dataset.from_tensor_slices((X_trainB, y_trainB))
        train_dataB = train_dataB.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)

        # Run training for the given number of steps.
        for stepB, (batch_xB, batch_yB) in enumerate(train_dataB.take(self.training_iters), 1):
            # Run the optimization to update W and b values.
            loss, fisher_loss, triplet_loss, fraction, l2_loss = self.run_optimization(batch_xB, batch_yB)
            # Obtain predictions
            pred = self.net(X_testA, is_training=False)
            triplet_loss_obj = Triplet_Loss(tf.cast(y_testA, tf.float32), pred,
                                            margin=tf.constant(self.margin), squared=tf.constant(self.squared))
            # Obtain triplet loss
            lossA, fractionA = self.get_triplet_loss(triplet_loss_obj)
            # Check if we have to perform early stopping
            self.early_stop(fractionA, fraction)

            if stepB % self.display_step == 0:
                tf.print('step', stepB, 'loss', loss, 'fisher_loss', self.lamb * fisher_loss,
                         'triplet loss', triplet_loss, 'l2_loss', l2_loss, 'fraction B', fraction,
                         'lossA', lossA, 'fraction A', fractionA)
            # Early stopping
            if self.stop:
                break
        # Save model
        if save:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}".format(save_path))


class Train_Classifier_EWC:
    """
    Performs training of the classifier (and not the embeddings) using cross entropy loss and EWC

    Args:
        net (Model): Model to train
        fisher_matrix (Fisher_Matrix): Fisher matrix object
        learning_rate (float): Learning Rate for Adam optimizer
        training_iters (int): Numer of training iterations
        batch_size (int): Batch size
        display_step (int): Number of iterations to wait to print the current performance of the model
        lamb (int): Lambda for EWC
        filepath (str): File path to store the model
        num_batches (int): Number of training batches
        restore (bool): Whether to restore the model from checkpoint or not

    Attributes:
        net (Model): Model to train
        gradients_A (tensor):Fisher information matrix for task A
        fisher_matrix (Fisher_matrix): Fisher matrix object
        weightsA (tensor): Weights for task A
        acc (tensor): Accuracy
        best_weights (tensor): Current best weights
        stop (bool): whether we have to do early stopping now or not
        stopping_steps (int): Patience for early stopping
        learning_rate (float): Learning Rate for Adam optimizer
        training_iters (int): Numer of training iterations
        batch_size (int): Batch size
        display_step (int): Number of iterations to wait to print the current performance of the model
        optimizer (tf.Optimizer): Adam optimizer for the learning steps
        ckpt (tf.Checkpoint): Checkpoint that stores weights and optimizer state
        manager (tf.CheckpointManager): Controls that not too many checkpoint files are stored
        lamb (int): Lambda for EWC
    """

    def __init__(self, net, fisher_matrix, learning_rate, training_iters, batch_size,
                 display_step, lamb, filepath, num_batches=16, restore=True):
        self.net = net
        self.fisher_matrix = fisher_matrix
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.num_batches = num_batches
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.weightsA = self.fisher_matrix.weightsA
        self.lamb = lamb
        self.stop = False
        self.gradients_A = self.fisher_matrix.fisher_matrix
        self.acc = 0
        self.stopping_steps = 0
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=filepath, max_to_keep=3)
        if restore:
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

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

    def run_optimization(self, x, y):
        """
        Performs one step of the learning process. It calculates the loss function and
        appies backpropagation algorithm to update the weights.
        Args:
            x (tensor): Samples of training data used to train the model
            y (tensor): True labels for such samples
        Returns:
            final_loss (tensor): Final loss: Triplet loss + Fisher loss
            fisher_loss (tensor): Fisher loss
            L2_loss (tensor): L2 loss
        """
        # Wrap computation inside a GradientTape for automatic differentiation.
        weights_A = self.weightsA
        weights_B = self.net.trainable_variables

        with tf.GradientTape() as g:
            g.watch(weights_A)
            g.watch(weights_B)
            g.watch(self.gradients_A)
            # CROSS ENTROPY LOSS
            pred = self.net(x, is_training=True)

            # Compute loss.
            loss = self.cross_entropy_loss(pred, y)

            # REGULARIZATION
            fisher_loss = 0
            l2_loss = 0
            i = 0
            # Each grad in gradients is the gradient of the loss with respect to the weights.
            # It is a tensor with the layer weights shape. Therefore,
            for grad in self.gradients_A:
                fisher_diag = grad  # F_i = (\partial L_A/\partial \theta_i)^2
                fisher_loss += tf.reduce_sum(tf.multiply(fisher_diag, tf.square(weights_B[i] - weights_A[i])))
                l2_loss += tf.reduce_sum(tf.square(weights_A[i] - weights_B[i]))
                # fisher loss = \sum_i F_i (theta_i - \theta_i*)^2
                i += 1

            final_loss = loss + self.lamb * fisher_loss

        # Variables to update, i.e. trainable variables.
        trainable_variables = self.net.trainable_variables
        # Compute gradients.
        gradients = g.gradient(final_loss, trainable_variables)
        # Update W and b following gradients.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return final_loss, fisher_loss, l2_loss

    def early_stop(self, accA, accB):
        """
        Early stopping
        Args:
            accA (float): Accuracy from task A
            accB (float): Accuracy from task B
        """
        if accA <= accB:
            self.stopping_steps += 1
        else:
            self.stopping_steps = max(0, self.stopping_steps - 1)
        if self.stopping_steps == 5:
            self.stop = True

    def fit(self, X_trainB, y_trainB, X_testA, y_testA, save=True):
        """
        Main fit function
        Args:
          X_trainB (data frame): Processed training data (tak B)
          y_trainB (numpy array): Labels for training data (task B)
          X_testA (data frame): Processed test data (task A)
          y_testA (numpy array): Labels for test data (task A)
          save (bool): If True, the model is saved at the end of training
        Returns:
          -
        """
        # Use tf.data API to shuffle and batch data.
        train_dataB = tf.data.Dataset.from_tensor_slices((X_trainB, y_trainB))
        train_dataB = train_dataB.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)

        # Run training for the given number of steps.
        for stepB, (batch_xB, batch_yB) in enumerate(train_dataB.take(self.training_iters), 1):
            # Run the optimization to update W and b values.

            loss, fisher_loss, l2_loss = self.run_optimization(batch_xB, batch_yB)

            if stepB % self.num_batches == 0:
                # Calculate accuracy
                pred = self.net(X_trainB)
                self.acc = self.accuracy(pred, y_trainB)

                predA = self.net(X_testA)
                accA = self.accuracy(predA, y_testA)
                self.early_stop(accA, self.acc)

                if (stepB) % self.display_step == 0:
                    tf.print('step', stepB, 'loss', loss, 'fisher_loss', self.lamb * fisher_loss,
                             'l2_loss', l2_loss, 'accuracy B', self.acc, 'accuracy A', accA)
                # Early stopping
                if self.stop:
                    break
        # Save model
        if save:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}".format(save_path))
