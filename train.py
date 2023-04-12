from os import mkdir
from math import exp
from time import time
from copy import deepcopy
from os.path import exists
from typing import List, Union, Tuple

from ple import PLE
from torch.optim import Adam
from torch import save as t_save
from torch import load as t_load
from torch import Tensor, no_grad
from ple.games.flappybird import FlappyBird
from ple.games.pixelcopter import Pixelcopter

from classes.utils.Globals import Globals
from classes.utils.Artist import Artist
from classes.model.linear.LinearDQN import LinearDQN
from classes.data.memory.ReplayBuffer import ReplayBuffer
from classes.model.convolutional.ConvDQN import ConvolutionalDQN
from classes.model.linear.LinearDuelingDQN import LinearDuelingDQN
from classes.model.convolutional.ConvDuelingDQN import ConvolutionalDuelingDQN
from classes.data.preprocessors.image.FlappyBirdImagePreprocessor import FlappyBirdImagePreprocessor
from classes.data.preprocessors.image.PixelCopterImagePreprocessor import PixelCopterImagePreprocessor
from classes.data.preprocessors.game_state.FlappyBirdGameStatePreprocessor import FlappyBirdGameStatePreprocessor
from classes.data.preprocessors.game_state.PixelCopterGameStatePreprocessor import PixelCopterGameStatePreprocessor


def update_target(current: Union[ConvolutionalDQN, ConvolutionalDuelingDQN, LinearDQN, LinearDuelingDQN],
                  target: Union[ConvolutionalDQN, ConvolutionalDuelingDQN, LinearDQN, LinearDuelingDQN]) -> None:
    target.load_state_dict(current.state_dict())


def get_epsilon(frame_number: int) -> float:
    INIT_EPSILON: float = 0.4
    MIN_EPSILON: float = 0.01
    EPSILON_DECAY: float = 500
    return MIN_EPSILON + (INIT_EPSILON - MIN_EPSILON) * exp(-1.0 * frame_number / EPSILON_DECAY)


def get_beta(frame_number: int) -> float:
    INIT_BETA: float = 0.4
    FRAMES_BETA: float = 1000
    return min(1.0, INIT_BETA + frame_number * (1.0 - INIT_BETA) / FRAMES_BETA)


def save_model(model: Union[ConvolutionalDQN, ConvolutionalDuelingDQN, LinearDQN, LinearDuelingDQN],
               file_name: str) -> None:
    if not exists(Globals.BRAIN_DIR_PATH):
        mkdir(Globals.BRAIN_DIR_PATH)
    if not exists(Globals.BRAIN_COPTER_DIR_PATH):
        mkdir(Globals.BRAIN_COPTER_DIR_PATH)
    if not exists(Globals.BRAIN_FLAPPY_DIR_PATH):
        mkdir(Globals.BRAIN_FLAPPY_DIR_PATH)
    state_dict: dict = model.state_dict()
    t_save(state_dict, file_name)


def get_environment(is_flappy: bool) -> PLE:
    GAME: Union[Pixelcopter,
                FlappyBird] = FlappyBird(width=Globals.GAME_WIDTH, height=Globals.GAME_HEIGHT) if is_flappy else \
        Pixelcopter(width=Globals.GAME_WIDTH, height=Globals.GAME_HEIGHT)
    return PLE(GAME, reward_values=Globals.TRAINING_REWARD_VALUES, display_screen=True)


def get_brain(brain_type: str, env: PLE) -> Union[ConvolutionalDQN, ConvolutionalDuelingDQN,
                                                  LinearDQN, LinearDuelingDQN]:
    number_of_actions: int = len(env.getActionSet())
    if 'linear' in brain_type:
        return LinearDQN(number_of_observations=7, number_of_actions=number_of_actions) if 'dqn' in brain_type else \
            LinearDuelingDQN(number_of_observations=7, number_of_actions=number_of_actions)
    return ConvolutionalDQN(in_channels=1, number_of_actions=number_of_actions, game_width=Globals.GAME_WIDTH,
                            game_height=Globals.GAME_HEIGHT) if 'dqn' in brain_type else \
        ConvolutionalDuelingDQN(in_channels=1, number_of_actions=number_of_actions, game_height=Globals.GAME_HEIGHT,
                                game_width=Globals.GAME_WIDTH)


def get_preprocessing_function(is_flappy: bool, brain_type: str) -> callable:
    if is_flappy:
        return FlappyBirdGameStatePreprocessor.extract_state_info if 'linear' in brain_type else \
            FlappyBirdImagePreprocessor.preprocess_image
    return PixelCopterGameStatePreprocessor.extract_state_info if 'linear' in brain_type else \
        PixelCopterImagePreprocessor.preprocess_image


def learn(replay_buffer: ReplayBuffer, replay_amount: int, beta: float,
          current: Union[ConvolutionalDQN, ConvolutionalDuelingDQN],
          target: Union[ConvolutionalDQN, ConvolutionalDuelingDQN], optimizer: Adam) -> float:
    GAMMA: float = 0.99
    states, actions, rewards, next_states, terminals, indices, weights = replay_buffer.get_sample(
        amount_of_memories=replay_amount, beta=beta)

    q_values: Tensor = current(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with no_grad():
        best_actions: Tensor = current(next_states).max(1)[1].unsqueeze(1)
        next_q_values: Tensor = target(next_states).gather(1, best_actions).squeeze(1)

    expected_q_values: Tensor = rewards + GAMMA * next_q_values * (1 - terminals)
    temporal_difference: Tensor = q_values - expected_q_values

    loss: Tensor = (temporal_difference.pow(2) * weights)
    priorities: Tensor = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    replay_buffer.update_priorities(indices=indices, priorities=priorities.cpu().detach().numpy())
    return loss.cpu().detach().item()


def run_training(env: PLE, replay_buffer: ReplayBuffer, number_of_frames: int, replay_amount: int, get_state: callable,
                 current_brain: Union[ConvolutionalDQN, ConvolutionalDuelingDQN, LinearDQN, LinearDuelingDQN],
                 target_brain: Union[ConvolutionalDQN, ConvolutionalDuelingDQN, LinearDQN, LinearDuelingDQN],
                 file_name: str, preprocess: callable, update_threshold: int, action_set: dict) -> Tuple[List[float],
                                                                                                         List[float]]:
    losses: List[float] = []
    rewards: List[float] = []
    env.reset_game()
    env.act(0)
    state: Tensor = preprocess(get_state())
    episode_reward: float = 0.0
    OPTIMIZER: Adam = Adam(params=current_brain.parameters(), lr=0.0075)

    for frame_number in range(1, number_of_frames + 1):
        epsilon: float = get_epsilon(frame_number)
        action: int = current_brain.get_action(x=state.unsqueeze(0).to(Globals.DEVICE_TYPE), epsilon=epsilon)
        reward: float = env.act(action=action_set[action])
        is_done: bool = env.game_over()
        next_state: Tensor = preprocess(get_state())
        replay_buffer.store_memory(state=state.unsqueeze(0), action=action, reward=reward,
                                   next_state=next_state.unsqueeze(0), is_done=is_done)
        episode_reward += reward
        state = next_state

        if is_done:
            env.reset_game()
            rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) >= replay_amount:
            beta = get_beta(frame_number=frame_number)
            loss = learn(replay_buffer=replay_buffer, replay_amount=replay_amount, beta=beta, current=current_brain,
                         target=target_brain, optimizer=OPTIMIZER)
            losses.append(loss)

        if frame_number % update_threshold == 0 and frame_number >= replay_amount:
            update_target(current=current_brain, target=target_brain)
            save_model(model=target_brain, file_name=file_name)

        if frame_number % 1_000 == 0:
            print(f'{(frame_number / number_of_frames) * 100}% done training')

    return losses, rewards


def train_agents(number_of_frames: int, replay_amount: int, update_threshold: int, capacity: int, alpha: float) -> None:
    REPLAY_BUFFER: ReplayBuffer = ReplayBuffer(capacity=capacity, alpha=alpha)
    kwa: dict = {
        'env': None,
        'action_set': None,
        'replay_buffer': REPLAY_BUFFER,
        'number_of_frames': number_of_frames,
        'current_brain': None,
        'target_brain': None,
        'get_state': None,
        'file_name': None,
        'preprocess': None,
        'replay_amount': replay_amount,
        'update_threshold': update_threshold
    }

    for is_flappy in [True, False]:
        brain_dir: str = Globals.BRAIN_FLAPPY_DIR_PATH if is_flappy else Globals.BRAIN_COPTER_DIR_PATH
        game_name: str = 'flappy' if is_flappy else 'copter'
        env: PLE = get_environment(is_flappy=is_flappy)
        env.init()
        kwa['env'] = env
        action_set: dict = env.getActionSet()
        kwa['action_set'] = {0: action_set[0], 1: action_set[1]}

        total_training_time: float = 0.0
        for brain_type in Globals.BRAIN_TYPES:
            kwa['get_state'] = env.getGameState if 'linear' in brain_type else env.getScreenGrayscale
            FILE_NAME: str = f'{brain_dir}{brain_type}.pth'
            current_brain: Union[ConvolutionalDQN, ConvolutionalDuelingDQN,
                                 LinearDQN, LinearDuelingDQN] = get_brain(brain_type=brain_type, env=env)

            # if is_flappy:
            #     state_dict: dict = t_load(f=f'{Globals.BRAIN_COPTER_DIR_PATH}{brain_type}.pth')
            #     current_brain.load_state_dict(state_dict=state_dict)

            kwa['file_name'] = FILE_NAME
            kwa['current_brain'] = current_brain.to(Globals.DEVICE_TYPE)
            kwa['target_brain'] = deepcopy(current_brain).to(Globals.DEVICE_TYPE)
            kwa['preprocess'] = get_preprocessing_function(is_flappy=is_flappy, brain_type=brain_type)

            losses: List[float]
            rewards: List[float]
            training_start: float = time()
            losses, rewards = run_training(**kwa)
            training_end: float = time() - training_start
            REPLAY_BUFFER.clear_memory()
            total_training_time += training_end
            print(f'{game_name} {brain_type} took {training_end} seconds to train')

            Artist.save_line_graph(plot_data=losses, plot_title=f'{game_name}_{brain_type}_losses.png')
            Artist.save_line_graph(plot_data=losses, plot_title=f'{game_name}_{brain_type}_losses.png')


if __name__ == '__main__':
    KWA: dict = {
        'alpha': 0.6,
        'capacity': 2_500,
        'replay_amount': 200,
        'number_of_frames': 100_000,
        'update_threshold': 400
    }

    train_agents(**KWA)
