from abc import ABCMeta, abstractmethod

from interfaces.model.linear.LinearModel import LinearModel


class ConvolutionalModel(LinearModel, metaclass=ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'forward') and hasattr(subclass, 'get_conv_output_size') and
                callable(subclass.forward) and callable(subclass.get_conv_output_size) and
                hasattr(subclass, 'get_action') and callable(subclass.get_action) or NotImplemented)

    @abstractmethod
    def get_conv_output_size(self, width: int, height: int) -> int:
        raise NotImplementedError
