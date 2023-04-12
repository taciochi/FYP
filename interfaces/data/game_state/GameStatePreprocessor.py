from abc import ABCMeta

from numpy import ndarray


class GameStatePreprocessor(metaclass=ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, 'extract_state_info') and callable(subclass.extract_state_info) or NotImplemented

    @staticmethod
    def extract_state_info(game_state: dict) -> ndarray:
        raise NotImplementedError
