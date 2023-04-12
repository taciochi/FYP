from abc import ABCMeta

from torch import Tensor
from numpy import ndarray


class ImagePreprocessor(metaclass=ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__get_edges') and hasattr(subclass, 'preprocess_image') and
                callable(subclass.__get_edges) and callable(subclass.preprocess_image) or NotImplemented)

    @staticmethod
    def __get_edges(image: ndarray) -> ndarray:
        raise NotImplementedError

    @staticmethod
    def preprocess_image(image: ndarray, requires_grad: bool) -> Tensor:
        raise NotImplementedError
