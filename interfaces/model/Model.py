from abc import ABCMeta, abstractmethod

from torch import Tensor
from torch.nn import Module


class Model(Module, metaclass=ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'forward') and hasattr(subclass, 'get_conv_output_size') and
                callable(subclass.forward) and callable(subclass.get_conv_output_size) or NotImplemented)

    @abstractmethod
    def forward(self, x: Tensor) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_conv_output_size(self) -> int:
        raise NotImplementedError
