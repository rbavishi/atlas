import tensorflow as tf

from atlas.models.tensorflow.graphs.gnn import GNNComponent


class GGNNOptimizer(GNNComponent):
    def __init__(self,
                 learning_rate: float = 0.001,
                 clamp_gradient_norm: float = 1.0,
                 **kwargs):

        super().__init__()
        self.learning_rate = learning_rate
        self.clamp_gradient_norm = clamp_gradient_norm

    def build(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        clipped_grads = [
            (tf.clip_by_norm(grad, self.clamp_gradient_norm), var) if grad is not None else (grad, var)
            for grad, var in grads_and_vars]

        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
