from os import mkdir
from os.path import exists
from typing import Union, List

from torch import Tensor
from numpy import ndarray
from matplotlib.pyplot import imshow, show, plot, title, savefig, clf


class Artist:
    __DPI: int = 1000
    __IMAGES_DIR_PATH: str = 'images/'

    @staticmethod
    def __save_to_disk(file_name: str) -> None:
        if not exists(Artist.__IMAGES_DIR_PATH):
            mkdir(Artist.__IMAGES_DIR_PATH)
        savefig(fname=f'{Artist.__IMAGES_DIR_PATH}{file_name}', format='png', dpi=Artist.__DPI)

    @staticmethod
    def line_graph(plot_data: List[float], plot_title: str, save_plot: bool = True, show_plot: bool = False) -> None:
        plot(plot_data)
        title(plot_title)
        if save_plot:
            Artist.__save_to_disk(file_name=f'{plot_title}.png')
        if show_plot:
            show()
        clf()

    @staticmethod
    def show_image(img: Union[ndarray, Tensor], file_name: str,
                   save_image: bool = True, show_image: bool = False) -> None:
        if not isinstance(img, ndarray):
            img = img.detach().numpy() if not img.is_cuda else img.cpu().detach().numpy()
        if save_image:
            Artist.__save_to_disk(file_name=f'{file_name}.png')
        if show_image:
            imshow(img, cmap='gray')
        clf()
