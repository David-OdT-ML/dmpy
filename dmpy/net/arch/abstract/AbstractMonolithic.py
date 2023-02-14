from abc import ABC, abstractmethod
from dgbpy.keras_classes import DataPredType

class Monolithic(ABC):

    def __init__(self, ):
        self._model_shape = None
        self._nroutputs = None
        self._predtype: DataPredType = None
        self._modelsize = None

    @abstractmethod
    def build(model_shape, nroutputs, predtype, modelsize=None): pass
