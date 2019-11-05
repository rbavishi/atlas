import os
import tempfile
from abc import ABC, abstractmethod
from typing import Collection, Any, Optional, Tuple

import tensorflow as tf

from atlas.models.core import TrainableModel, SerializableModel


class KerasModel(TrainableModel, SerializableModel, ABC):
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def preprocess(self, data: Collection, mode: str = 'training') -> Tuple[Collection, Collection]:
        pass

    def train(self, train_data: Collection[Any], val_data: Collection[Any] = None,
              num_epochs: int = 10, batch_size: int = 128, **kwargs):

        if self.model is None:
            self.build()

        train_inputs, train_targets = self.preprocess(train_data)
        val_inputs, val_targets = (None, None)
        if val_data is not None:
            val_inputs, val_targets = self.preprocess(val_data)

        ckpt_path = f"{tempfile.mkdtemp()}/model.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                        monitor='val_acc' if val_data is not None else 'loss',
                                                        verbose=1, save_best_only=True, mode='max')

        self.model.fit(train_inputs, train_targets, epochs=num_epochs,
                       batch_size=batch_size,
                       validation_data=(None if val_data is None else (val_inputs, val_targets)),
                       callbacks=[checkpoint])

        if os.path.exists(ckpt_path):
            self.model = tf.keras.models.load_model(ckpt_path)

    def infer(self, data: Collection, **kwargs):
        return self.model.predict(self.preprocess(data, mode='inference'))

    def serialize(self, path_dir: str):
        tf.keras.models.save_model(self.model, f"{path_dir}/model.h5")

    def deserialize(self, path_dir: str):
        self.model = tf.keras.models.load_model(f"{path_dir}/model.h5")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('model')

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None
