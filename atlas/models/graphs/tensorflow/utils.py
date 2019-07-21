import tensorflow as tf


def SegmentBasedAttention(data, segment_ids, num_segments):
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
