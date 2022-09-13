import gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def get_output_shape(model, image_dim):
    return model(th.rand(*(image_dim))).data.shape[1:]


def weights_init_(m):
    if hasattr(m, 'weight') and m.weight is not None and len(m.weight.shape) > 2:
        th.nn.init.xavier_uniform_(m.weight, gain=1)
    if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, th.Tensor):
        th.nn.init.constant_(m.bias, 0)


class GridEncoder(nn.Module):
    def __init__(self, grid_shape, depth=16, act=nn.ReLU):
        super(GridEncoder, self).__init__()
        self.kernels = (4, 4, 4, 4) if max(grid_shape) > 64 else (3, 3, 3, 3)
        self.strides = (2, 2, 2, 2) if max(grid_shape) > 64 else (1, 1, 1, 1)
        self.padding = (1, 1)
        layers = []
        current_channels = grid_shape[0]
        for i, (k, s) in enumerate(zip(self.kernels, self.strides)):
            layers.append(nn.Conv2d(current_channels, depth, k, stride=s, padding=self.padding))
            layers.append(nn.GroupNorm(1, depth))
            layers.append(act())
            current_channels = depth
            depth *= 2

        layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*layers)

    def forward(self, obs):
        return self.encoder(obs.float())


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2, act=nn.ReLU):
        super(MLP, self).__init__()
        if num_layers > 1:
            layers = [nn.Linear(input_dim, hidden_dim), act()]
        else:
            layers = [nn.Linear(input_dim, output_dim), act()]
        for i in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        if num_layers > 1:
            layers += [nn.Linear(hidden_dim, output_dim), act()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs):
        return self.mlp(obs)


class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2, act=nn.ReLU):
        super(LSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)#, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)
        # fancy learnable initial states taken from https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/7
        self.h0 = th.zeros((1, hidden_dim))
        self.c0 = th.zeros((1, hidden_dim))
        # self.h0 = nn.Parameter(h0, requires_grad=True)  # Parameter() to update weights
        # self.c0 = nn.Parameter(c0, requires_grad=True)

    def reset(self, batch_size=None, traj_change_mask=None):
        if traj_change_mask is None:
            # print('RESETTING ALL')
            self.hi = self.h0.repeat(batch_size, 1).detach()
            self.ci = self.c0.repeat(batch_size, 1).detach()
        else:
            # print('RESETTING SOME')
            # print(self.h0.repeat(batch_size, 1).shape, traj_change_mask.shape, self.hi.shape)
            self.hi = self.h0.repeat(batch_size, 1).detach() * traj_change_mask + self.hi.detach() * (1 - traj_change_mask)
            self.ci = self.c0.repeat(batch_size, 1).detach() * traj_change_mask + self.ci.detach() * (1 - traj_change_mask)
            # self.hi[traj_change_mask.bool(), :] = self.h0.repeat(th.sum(traj_change_mask), 1).detach()
            # self.ci[traj_change_mask.bool(), :] = self.c0.repeat(th.sum(traj_change_mask), 1).detach()
            # self.hi[(1 - traj_change_mask).bool()] = self.hi[(1 - traj_change_mask).bool()]

    def forward(self, new_obs):
        self.hi, self.ci = self.lstm_cell(new_obs, (self.hi, self.ci))
        return self.linear(self.hi)


# LSTM TASK
class OAISinglePlayerLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
        :param observation_space: (gym.Space)
        :param features_dim: (int) Number of features extracted.
            This corresponds to the number of unit for the last layer.
        """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(OAISinglePlayerLSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        self.use_visual_obs = np.prod(observation_space['visual_obs'].shape) > 0
        self.use_vector_obs = np.prod(observation_space['agent_obs'].shape) > 0
        self.use_subtask_obs = 'subtask' in observation_space.keys()
        input_dim = 0
        if self.use_visual_obs:
            self.vis_encoder = GridEncoder(observation_space['visual_obs'].shape)
            test_shape = [1, *observation_space['visual_obs'].shape]
            input_dim += get_output_shape(self.vis_encoder, test_shape)[0]
        if self.use_vector_obs:
            input_dim += np.prod(observation_space['agent_obs'].shape)
        if self.use_subtask_obs:
            input_dim += np.prod(observation_space['subtask'].shape)

        # Define MLP for vector/feature based observations
        self.lstm_encoder = LSTM(input_dim=input_dim, output_dim=features_dim)
        self.apply(weights_init_)
        self.prev_traj_ids = None

    def reset(self, traj_ids):
        # print(self.prev_traj_ids == traj_ids)
        if self.prev_traj_ids is None or self.prev_traj_ids.shape != traj_ids.shape or all(self.prev_traj_ids != traj_ids):
            # print('1', th.tensor(self.prev_traj_ids != traj_ids).int().squeeze())
            self.lstm_encoder.reset(batch_size=traj_ids.shape[0])
        elif any(self.prev_traj_ids != traj_ids):
            # print('2', th.tensor(self.prev_traj_ids != traj_ids).int().squeeze())
            traj_change_mask = (self.prev_traj_ids != traj_ids).int()
            self.lstm_encoder.reset(batch_size=traj_ids.shape[0], traj_change_mask=traj_change_mask)
        self.prev_traj_ids = traj_ids

    def forward(self, observations: th.Tensor) -> th.Tensor:
        self.reset(observations['traj_id'])
        latent_state = []
        # Concatenate all input features before passing them to MLP
        if self.use_visual_obs:
            # Convert all grid-like observations to features using CNN
            latent_state.append(self.vis_encoder.forward(observations['visual_obs']))
        if self.use_vector_obs:
            latent_state.append(th.flatten(observations['agent_obs'], start_dim=1))
        if self.use_subtask_obs:
            latent_state.append(th.flatten(observations['subtask'], start_dim=1))

        return self.lstm_encoder.forward(th.cat(latent_state, dim=-1))


class OAISinglePlayerFeatureExtractor(BaseFeaturesExtractor):
    """
        :param observation_space: (gym.Space)
        :param features_dim: (int) Number of features extracted.
            This corresponds to the number of unit for the last layer.
        """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(OAISinglePlayerFeatureExtractor, self).__init__(observation_space, features_dim)
        self.use_visual_obs = np.prod(observation_space['visual_obs'].shape) > 0
        self.use_vector_obs = np.prod(observation_space['agent_obs'].shape) > 0
        self.use_subtask_obs = 'subtask' in observation_space.keys()
        input_dim = 0
        if self.use_visual_obs:
            self.vis_encoder = GridEncoder(observation_space['visual_obs'].shape)
            test_shape = [1, *observation_space['visual_obs'].shape]
            input_dim += get_output_shape(self.vis_encoder, test_shape)[0]
        if self.use_vector_obs:
            input_dim += np.prod(observation_space['agent_obs'].shape)
        if self.use_subtask_obs:
            input_dim += np.prod(observation_space['subtask'].shape)

        # Define MLP for vector/feature based observations
        self.vector_encoder = MLP(input_dim=input_dim, output_dim=features_dim)
        self.apply(weights_init_)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        latent_state = []
        # Concatenate all input features before passing them to MLP
        if self.use_visual_obs:
            # Convert all grid-like observations to features using CNN
            latent_state.append(self.vis_encoder.forward(observations['visual_obs']))
        if self.use_vector_obs:
            latent_state.append(th.flatten(observations['agent_obs'], start_dim=1))
        if self.use_subtask_obs:
            latent_state.append(th.flatten(observations['subtask'], start_dim=1))

        return self.vector_encoder.forward(th.cat(latent_state, dim=-1))


class OAIDoublePlayerFeatureExtractor(BaseFeaturesExtractor):
    """
        :param observation_space: (gym.Space)
        :param features_dim: (int) Number of features extracted.
            This corresponds to the number of unit for the last layer.
        """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(OAIDoublePlayerFeatureExtractor, self).__init__(observation_space, features_dim)
        self.use_visual_obs = np.prod(observation_space['visual_obs'].shape) > 0
        self.use_vector_obs = np.prod(observation_space['agent_obs'].shape) > 0
        if self.use_visual_obs:
            self.vis_encoders = [GridEncoder(observation_space['visual_obs'].shape[1:]),
                                 GridEncoder(observation_space['visual_obs'].shape[1:])]
            test_shape = [1, *observation_space['visual_obs'].shape[1:]]
            self.encoder_output_shape = get_output_shape(self.vis_encoders[0], test_shape)[0] * 2
        else:
            self.encoder_output_shape = 0

        # Define MLP for vector/feature based observations
        self.vector_encoder = MLP(input_dim=self.encoder_output_shape + np.prod(observation_space['agent_obs'].shape),
                                  output_dim=features_dim)
        self.apply(weights_init_)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        visual_obs, agent_obs = observations['visual_obs'], observations['agent_obs']
        latent_state = []
        # Concatenate all input features before passing them to MLP
        if self.use_visual_obs:
            # Convert all grid-like observations to features using CNN
            latent_state += [self.vis_encoders[i].forward(visual_obs[:, i]) for i in range(2)]
        if self.use_vector_obs:
            latent_state += [th.flatten(agent_obs[:, i], start_dim=1) for i in range(2)]
        return self.vector_encoder.forward(th.cat(latent_state, dim=-1))
