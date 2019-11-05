from abc import ABC

from atlas import models
import atlas.models.tensorflow
import atlas.models.keras
import atlas.models.imitation


class Models:
    class Tensorflow(models.tensorflow.TensorflowModel, ABC):
        pass

    class Keras(models.keras.KerasModel, ABC):
        pass

    class Generators:
        class Imitation:
            class IndependentOperators(models.imitation.IndependentOperatorsModel, ABC):
                pass
