from math import inf
from json import dump
from os import listdir
from typing import Dict, Union, List

from ple import PLE
from torch import Tensor
from torch import load as t_load
from ple.games.flappybird import FlappyBird

from classes.utils.Globals import Globals
from classes.model.linear.LinearDQN import LinearDQN
from classes.model.convolutional.ConvDQN import ConvolutionalDQN
from classes.model.linear.LinearDuelingDQN import LinearDuelingDQN
from classes.model.convolutional.ConvDuelingDQN import ConvolutionalDuelingDQN
from classes.data.preprocessors.image.FlappyBirdImagePreprocessor import FlappyBirdImagePreprocessor
from classes.data.preprocessors.game_state.FlappyBirdGameStatePreprocessor import FlappyBirdGameStatePreprocessor


def get_brain_types() -> List[str]:
    desired: List[str] = []
    brain_types: List[str] = listdir(f'{Globals.BRAIN_FLAPPY_DIR_PATH}')

    for brain_type in brain_types:
        if 'solo' in brain_type:
            desired = [*desired, brain_type]
            continue
        if 'final' in brain_type:
            desired = [*desired, brain_type]

    return desired


def play_flappy(number_of_episodes: int = 10) -> None:
    outcomes: Dict[str, Dict[str, float]] = dict()
    ENV: PLE = PLE(FlappyBird(), display_screen=True, reward_values=Globals.PLAYING_REWARD_VALUES)
    ENV.init()
    ACTION_SET = {0: ENV.getActionSet()[0], 1: ENV.getActionSet()[1]}
    for brain_type in get_brain_types():
        outcomes[brain_type] = dict()
        brain: Union[ConvolutionalDQN, ConvolutionalDuelingDQN,
                     LinearDQN, LinearDuelingDQN] = Globals.get_brain(brain_type=brain_type,
                                                                      number_of_actions=len(ENV.getActionSet()))
        brain.to(Globals.DEVICE_TYPE)
        state_dict: dict = t_load(f'{Globals.BRAIN_FLAPPY_DIR_PATH}{brain_type}')
        brain.load_state_dict(state_dict=state_dict)
        get_state: callable = ENV.getGameState if 'linear' in brain_type else ENV.getScreenGrayscale
        preprocess: callable = FlappyBirdGameStatePreprocessor.extract_state_info if 'linear' in brain_type else \
            FlappyBirdImagePreprocessor.preprocess_image

        rewards: List[float] = []
        for episode in range(number_of_episodes):
            ENV.reset_game()
            ENV.act(0)  # pass initial black screen
            episode_reward: float = 0.0

            while not ENV.game_over():
                state: Tensor = preprocess(get_state().T if 'conv' in brain_type else get_state())
                episode_reward += ENV.act(ACTION_SET[brain.get_action(x=state.unsqueeze(0).to(Globals.DEVICE_TYPE),
                                                                      epsilon=-inf)])

            rewards = [*rewards, episode_reward]

        outcomes[brain_type]['max_reward'] = max(rewards)
        outcomes[brain_type]['min_reward'] = min(rewards)
        outcomes[brain_type]['mean_reward'] = sum(rewards) / len(rewards)

    with open(f'{Globals.OUTCOMES_FILE_NAME_START}playing.json', 'w') as outcomes_file:
        dump(outcomes, outcomes_file, indent=2)


if __name__ == '__main__':
    play_flappy()
