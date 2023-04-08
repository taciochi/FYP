from ple import PLE
from ple.games.flappybird import FlappyBird

from classes.Globals import Globals
from classes.data.Artist import Artist
from classes.data.FlappyBirdImagePreprocessor import FlappyBirdImagePreprocessor


def train() -> None:
    ENV: PLE = PLE(FlappyBird(width=Globals.GAME_WIDTH, height=Globals.GAME_HEIGHT), fps=30,
                   reward_values=Globals.TRAINING_REWARD_VALUES, display_screen=True)
    ENV.init()
    ENV.getScreenGrayscale()
    ENV.act(0)  # act to pass the black screen

    while not ENV.game_over():
        state = ENV.getScreenGrayscale().T
        Artist.show_image(FlappyBirdImagePreprocessor.preprocess_image(state, requires_grad=False))


if __name__ == '__main__':
    train()
