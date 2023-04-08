import cv2
from torch import tensor, Tensor
from torch import float32 as t_float
from torchvision.transforms import Resize
from numpy import vectorize, ndarray, expand_dims

from classes.Globals import Globals
from interfaces.data.Preprocessor import Preprocessor


class FlappyBirdImagePreprocessor(Preprocessor):

    @staticmethod
    def __get_edges(image: ndarray) -> ndarray:
        return cv2.threshold(image, 75, 255, cv2.THRESH_BINARY_INV)[1]

    @staticmethod
    def preprocess_image(image: ndarray, requires_grad: bool = False) -> Tensor:
        image = FlappyBirdImagePreprocessor.__get_edges(image=image)
        image = vectorize(lambda pixel: pixel / 255)(image)  # normalize image
        image = expand_dims(image, axis=0)  # add channels input
        return tensor(data=image, requires_grad=requires_grad, device=Globals.DEVICE_TYPE, dtype=t_float)
