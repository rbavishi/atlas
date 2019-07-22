import tensorflow as tf
from typing import List, Optional


def SegmentBasedAttention(data, segment_ids, num_segments):
    """
    A function to compute attention when a dynamic number of elements are involved. This creates when doing
    normalization such as softmax as you cannot use the traditional tf.softmax like functions. So need to
    implement the logexpsum trick manually

    Notation :
    N - number of elements
    G - number of groups

    Args:
        data: The un-normalized scores of the elements (Shape : [N x 1])
        segment_ids: The group of elements which this score is for (Shape : [N x 1])
        num_segments: The total number of groups (G)

    Returns:
        A score for each element normalized over the sum of scores of all the elements in its group (Shape : [N x 1])

    """
    #  Step (1) : Obtain shift constant as max of messages going into a segment

    max_per_segment = tf.unsorted_segment_max(data=data,
                                              segment_ids=segment_ids,
                                              num_segments=num_segments)

    #  Step (2) : Distribute max out to the corresponding segments and shift scores
    scores = data - tf.gather(params=max_per_segment, indices=segment_ids)

    #  Step (3) : Take the exponent, sum up per segment, and compute exp(score) / exp(sum) as the attention
    exp_scores = tf.exp(scores)
    exp_score_sum_per_segment = tf.unsorted_segment_sum(data=exp_scores,
                                                        segment_ids=segment_ids,
                                                        num_segments=num_segments)

    #  Distribute to each segment item
    sum_per_segment_item = tf.gather(params=exp_score_sum_per_segment, indices=segment_ids)
    attention = exp_scores / (sum_per_segment_item + 1e-7)

    return attention


class MLP:
    """ A factory class for a multi-layer perceptron """
    def __init__(self, in_size, out_size, hid_sizes: List[int] = None,
                 activations: List[Optional[str]] = None):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes or []
        self.activations = activations or ['relu'] * (len(self.hid_sizes) + 1)
        self.layers = []
        self.build()

    def build(self):
        layer_dims = self.hid_sizes + [self.out_size]
        self.layers.append(tf.keras.layers.Dense(layer_dims[0], activation=self.activations[0],
                                                 input_dim=self.in_size, kernel_initializer='glorot_uniform'))
        for idx, layer_dim in enumerate(layer_dims[1:], 1):
            self.layers.append(tf.keras.layers.Dense(layer_dim, activation=self.activations[idx],
                                                     kernel_initializer='glorot_uniform'))

    def __call__(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer(result)

        return result
