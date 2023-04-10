from os import mkdir
from math import exp
from time import time
from copy import deepcopy
from os.path import exists
from typing import List, Dict, Union

from ple import PLE
from torch import Tensor
from torch.optim import Adam
from torch import save as t_save
from torch import load as t_load
from ple.games.flappybird import FlappyBird
from ple.games.pixelcopter import Pixelcopter

from classes.Globals import Globals
from classes.data.Artist import Artist
from interfaces.model.Model import Model
from classes.model.architectures.DQN import DQN
from classes.data.ReplayBuffer import ReplayBuffer
from classes.model.architectures.DuelingDQN import DuelingDQN
from classes.data.FlappyBirdImagePreprocessor import FlappyBirdImagePreprocessor
from classes.data.PixelCopterImagePreprocessor import PixelCopterImagePreprocessor


def update_target(current: Model, target: Model) -> None:
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


def save_model(model: Model, file_name: str) -> None:
    if not exists(Globals.BRAIN_DIR_PATH):
        mkdir(Globals.BRAIN_DIR_PATH)
    FULL_PATH: str = Globals.BRAIN_DIR_PATH + file_name
    state_dict: dict = model.state_dict()
    t_save(state_dict, FULL_PATH)


def get_architectures(game_name: str, number_of_actions: int, game_width: int, game_height: int) -> List[Model]:
    architectures: List[Model] = [
        DQN(in_channels=1, number_of_actions=number_of_actions, game_width=game_width, game_height=game_height),
        DuelingDQN(in_channels=1, number_of_actions=number_of_actions, game_width=game_width, game_height=game_height)
    ]

    if game_name == 'flappy':
        architectures[0].load_state_dict(t_load('brains/brain_copter_DQN_state_dict.pth'))
        architectures[1].load_state_dict(t_load('brains/brain_copter_DuelingDQN_state_dict.pth'))

    return architectures


def learn(replay_buffer: ReplayBuffer, replay_amount: int, beta: float, current: Model, target: Model,
          optimizer: Adam) -> float:
    GAMMA: float = 0.99
    states, actions, rewards, next_states, terminals, indices, weights = replay_buffer.get_sample(
        amount_of_memories=replay_amount, beta=beta)

    q_values = current(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    next_q_values_current = current(next_states)
    next_q_values_target = target(next_states)
    next_q_values = next_q_values_target.gather(1, next_q_values_current.max(1)[1].unsqueeze(1)).squeeze(1)

    expected_q_values = rewards + GAMMA * next_q_values * (1 - terminals)

    loss = (q_values - expected_q_values).pow(2) * weights
    priorities = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    replay_buffer.update_priorities(indices=indices, priorities=priorities.cpu().detach().numpy())
    return loss.cpu().detach().item()


def train(env: PLE, action_set: Dict[int, Union[int, None]], current: Model, target: Model, replay_buffer: ReplayBuffer,
          preprocess_image: callable, number_of_frames: int, replay_amount: int,
          frame_update_threshold: int) -> None:
    losses: List[float] = []
    rewards: List[float] = []
    episode_reward: float = 0.0
    OPTIMIZER: Adam = Adam(params=current.parameters(), lr=0.05)
    env.reset_game()  # ensure game is ready to play
    env.act(0)  # act to pass the initial black screen
    state: Tensor = preprocess_image(env.getScreenGrayscale().T, requires_grad=True)
    FILE_NAME: str = 'brain' + ('_copter' if isinstance(env.game, Pixelcopter) else '_flappy') + \
                     ('_DQN' if type(target) == DQN else '_DuelingDQN')

    for frame_number in range(1, number_of_frames + 1):
        epsilon: float = get_epsilon(frame_number)
        action: int = current.get_action(x=state.to(Globals.DEVICE_TYPE), epsilon=epsilon)
        reward: float = env.act(action=action_set[action])
        is_done: bool = env.game_over()
        next_state: Tensor = PixelCopterImagePreprocessor.preprocess_image(env.getScreenGrayscale().T,
                                                                           requires_grad=True)
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
            loss = learn(replay_buffer=replay_buffer, replay_amount=replay_amount, beta=beta, current=current,
                         target=target, optimizer=OPTIMIZER)
            losses.append(loss)

        if frame_number % frame_update_threshold == 0 and frame_number >= replay_amount:
            update_target(current=current, target=target)
            save_model(model=target, file_name=f'{FILE_NAME}_state_dict.pth')

    Artist.line_graph(plot_data=losses, plot_title=f'{FILE_NAME}_losses.png', save_plot=True, show_plot=False)
    Artist.line_graph(plot_data=rewards, plot_title=f'{FILE_NAME}_rewards.png', save_plot=True, show_plot=False)


def train_agents(game_width: int, game_height: int, capacity: int, alpha: float, number_of_frames: int,
                 frame_update_threshold: int, replay_amount: int) -> None:
    start_training_time: float = time()
    for game_name in ['copter', 'flappy']:
        GAME: Union[Pixelcopter, FlappyBird] = Pixelcopter(width=game_width,
                                                           height=game_height) if game_name == 'copter' else \
            FlappyBird(width=game_width, height=game_height)
        ENV: PLE = PLE(GAME, fps=30,
                       reward_values=Globals.REWARD_VALUES, display_screen=True)
        ENV.init()
        ACTIONS: List[Union[int, None]] = ENV.getActionSet()
        ACTION_SET: Dict[int, Union[int, None]] = {0: ACTIONS[0], 1: ACTIONS[1]}
        ARCHITECTURES: List[Model] = get_architectures(game_name=game_name, number_of_actions=len(ACTIONS),
                                                       game_width=game_width, game_height=game_height)
        preprocess_image: callable = PixelCopterImagePreprocessor.preprocess_image if game_name == 'copter' else \
            FlappyBirdImagePreprocessor.preprocess_image

        total_game_training_time: float = time()
        for architecture in ARCHITECTURES:
            start_time: float = time()
            train(env=ENV, action_set=ACTION_SET, current=architecture.to(Globals.DEVICE_TYPE),
                  target=deepcopy(architecture.to(Globals.DEVICE_TYPE)),
                  replay_buffer=ReplayBuffer(capacity=capacity, alpha=alpha),
                  number_of_frames=number_of_frames,
                  preprocess_image=preprocess_image, replay_amount=replay_amount,
                  frame_update_threshold=frame_update_threshold)
            architecture_time_required: float = time() - start_time
            NAME: str = ('copter' if isinstance(GAME, Pixelcopter) else 'flappy') + \
                        ('_DQN' if type(architecture) == DQN else '_DuelingDQN')
            print(f'time to train {NAME}: {architecture_time_required}')
        print(f'total training time for {game_name}: {time() - total_game_training_time}')
    print(f'total training time: {time() - start_training_time}')


if __name__ == '__main__':
    ALPHA: float = 0.6
    GAME_WIDTH: int = 256
    GAME_HEIGHT: int = 512
    CAPACITY: int = 5_000
    REPLAY_AMOUNT: int = 300
    NUMBER_OF_FRAMES: int = 100_000
    FRAME_UPDATE_THRESHOLD: int = 100
    train_agents(game_width=GAME_WIDTH, game_height=GAME_HEIGHT, number_of_frames=NUMBER_OF_FRAMES, capacity=CAPACITY,
                 alpha=ALPHA, frame_update_threshold=FRAME_UPDATE_THRESHOLD, replay_amount=REPLAY_AMOUNT)
