from typing import Union

from torch import Tensor
from numpy import ndarray
from matplotlib.pyplot import imshow, show


class Artist:

    @staticmethod
    def show_image(img: Union[ndarray, Tensor]) -> None:
        if not isinstance(img, ndarray):
            img = img.detach().numpy() if not img.is_cuda else img.cpu().detach().numpy()
        imshow(img, cmap='gray')
        show()
