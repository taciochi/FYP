from torch import float32 as t_float
from torchvision.transforms import Resize
from torch import tensor, Tensor, from_numpy
from numpy import vectorize, ndarray, expand_dims

from classes.utils.Globals import Globals
from interfaces.data.image.ImagePreprocessor import ImagePreprocessor


class PixelCopterImagePreprocessor(ImagePreprocessor):

    @staticmethod
    def __get_edges(image: ndarray) -> ndarray:
        return vectorize(lambda x: 255 if x != 0 else x)(image)

    @staticmethod
    def preprocess_image(image: ndarray, requires_grad: bool = False) -> Tensor:
        # image = PixelCopterImagePreprocessor.__get_edges(image=image)
        tmp: Tensor = Resize((Globals.IMG_SIZE, Globals.IMG_SIZE))(from_numpy(expand_dims(image, axis=0)))
        image = tmp.squeeze(0).detach().numpy()
        del tmp
        image = vectorize(lambda pixel: pixel / 255)(image)  # normalize image
        image = expand_dims(image, axis=0)  # add channels input
        return tensor(data=image, requires_grad=requires_grad, dtype=t_float)
