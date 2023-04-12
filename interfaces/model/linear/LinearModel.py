from abc import ABCMeta, abstractmethod

from torch import Tensor
from torch.nn import Module


class LinearModel(Module, metaclass=ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'forward') and callable(subclass.forward) and
                hasattr(subclass, 'get_action') and callable(subclass.get_action) or NotImplemented)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_action(self, x: Tensor, epsilon: float) -> int:
        raise NotImplementedError
