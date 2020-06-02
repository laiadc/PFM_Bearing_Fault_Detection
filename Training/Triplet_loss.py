import tensorflow as tf


class Triplet_Loss:
    """
    Contains the functions needed to create the triplet loss with online triplet mining.

    Args:
        labels (tensor): Tensor of labels (containing the failure mode of the training data)
        embeddigns (tensor): Tensor containing the features in the embedding space
        margin (float): Margin value to calculate the triplet loss
        squared (bool): If true, output is the pairwise squared euclidean distance matrix.
                        If false, output is the pairwise euclidean distance matrix.

    Attributes:
        labels (tensor): Tensor of labels (containing the failure mode of the training data)
        embeddigns (tensor): Tensor containing the features in the embedding space
        margin (float): Margin value to calculate the triplet loss
        squared (bool): If true, output is the pairwise squared euclidean distance matrix.
                        If false, output is the pairwise euclidean distance matrix.
    """

    def __init__(self, labels, embeddings, margin, squared=False):
        self.labels = labels
        self.embeddings = embeddings
        self.margin = margin
        self.squared = squared

    @tf.function()
    def _pairwise_distances(self):
        """Compute the 2D matrix of distances between all the embeddings.
      Args:
          -
      Returns:
          distances (tensor): tensor of shape (batch_size, batch_size)
      """
        # Get the dot product between all embeddings
        # shape (batch_size, batch_size)
        dot_product = tf.matmul(self.embeddings, tf.transpose(self.embeddings))

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = tf.linalg.diag_part(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = tf.maximum(distances, 0.0)

        if not self.squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)

        return distances

    @tf.function()
    def _get_anchor_positive_triplet_mask(self):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        Args:
            -
        Returns:
            mask (boolean): tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(self.labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(self.labels, 0), tf.expand_dims(self.labels, 1))

        # Combine the two masks
        mask = tf.logical_and(indices_not_equal, labels_equal)

        return mask

    @tf.function()
    def _get_anchor_negative_triplet_mask(self):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
      Args:
          -
      Returns:
          mask (boolean): tf.bool `Tensor` with shape [batch_size, batch_size]
      """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(self.labels, 0), tf.expand_dims(self.labels, 1))

        mask = tf.logical_not(labels_equal)

        return mask

    @tf.function()
    def _get_triplet_mask(self):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
      A triplet (i, j, k) is valid if:
          - i, j, k are distinct
          - labels[i] == labels[j] and labels[i] != labels[k]
      Args:
          -
      Returns:
          mask (boolean): tf.bool `Tensor` with shape [batch_size, batch_size]
      """
        # Check that i, j and k are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(self.labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = tf.equal(tf.expand_dims(self.labels, 0), tf.expand_dims(self.labels, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        # Combine the two masks
        mask = tf.logical_and(distinct_indices, valid_labels)

        return mask

    @tf.function()
    def batch_all_triplet_loss(self):
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            -
        Returns:
          triplet_loss (tensor): scalar tensor containing the mean triplet loss
          fraction_positive_triplets (tensor): Fraction positive triplets
          triplet_loss_matrix (tensor): Triplet loss of all triplets
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances()

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = self._get_triplet_mask()
        mask = tf.cast(mask, tf.float32)

        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        triplet_loss_matrix = triplet_loss
        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets, triplet_loss_matrix

    @tf.function()
    def batch_hard_triplet_loss(self):
        """Build the triplet loss over a batch of embeddings.
      For each anchor, we get the hardest positive and hardest negative to form a triplet.
      Args:
          -
      Returns:
          triplet_loss (tensor): scalar tensor containing the triplet loss
          triplet_loss_matrix (tensor): Triplet loss of all triplets
      """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances()

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask()
        mask_anchor_positive = tf.cast(mask_anchor_positive, tf.float32)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask()
        mask_anchor_negative = tf.cast(mask_anchor_negative, tf.float32)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + self.margin, 0.0)

        triplet_loss_matrix = triplet_loss
        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss, triplet_loss_matrix
