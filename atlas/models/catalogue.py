from abc import ABC

import atlas.models.keras
import atlas.models.tensorflow
from atlas import models


class Models:
    class Tensorflow(models.tensorflow.TensorflowModel, ABC):
        pass

    class Keras(models.keras.KerasModel, ABC):
        pass

    class Generators:
        class Imitation:
            class IndependentOperators(models.imitation.IndependentOperatorsModel, ABC):
                pass
