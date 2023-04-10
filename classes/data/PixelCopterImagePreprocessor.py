from torch import tensor, Tensor
from torch import float32 as t_float
from numpy import vectorize, ndarray, expand_dims

from classes.Globals import Globals
from interfaces.data.Preprocessor import Preprocessor


class PixelCopterImagePreprocessor(Preprocessor):

    @staticmethod
    def __get_edges(image: ndarray) -> ndarray:
        return vectorize(lambda x: 255 if x != 0 else x)(image)

    @staticmethod
    def preprocess_image(image: ndarray, requires_grad: bool) -> Tensor:
        image = PixelCopterImagePreprocessor.__get_edges(image=image)
        image = vectorize(lambda pixel: pixel / 255)(image)  # normalize image
        image = expand_dims(image, axis=0)  # add channels input
        return tensor(data=image, requires_grad=requires_grad, dtype=t_float)
