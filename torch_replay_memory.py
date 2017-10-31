import numpy as np
import torch

class Memory(object):

    def __init__(self,
                 max_size,
                 observation_shape,
                 action_shape,
                 observation_dtype=torch.FloatTensor, #np.float32,
                 action_dtype=torch.FloatTensor): #np.float32):
        self._max_size = max_size
        self._observation_shape = observation_shape
        self._action_shape = action_shape

        self._observations = observation_dtype(*((max_size,)+observation_shape)).zero_() #, dtype=observation_dtype)
        self._actions = action_dtype(*((max_size,)+(action_shape,))).zero_() #, dtype=action_dtype)
        self._rewards = torch.zeros(max_size, 1) # dtype=np.float32)
        self._bonuses = torch.zeros(max_size, 1) # dtype=np.float32)
        self._terminals = torch.ByteTensor(max_size, 1).zero_() #np.zeros(max_size, dtype=np.uint8)

        # self._obs_mean = observation_dtype(observation_shape).zero_() #, dtype=observation_dtype)
        # self._obs_stddev = observation_dtype(observation_shape).zero_() #, dtype=observation_dtype)
        # self._act_mean = action_dtype(action_shape).zero_() #, dtype=action_dtype)
        # self._act_stddev = action_dtype(action_shape).zero_() #, dtype=action_dtype)

        self._bottom = torch.LongTensor([0]) #0
        self._top = torch.LongTensor([0])
        self._size = torch.LongTensor([0])

    def add_sample(self, obs, act, rew, term, bonus=0.):
        obs = obs.reshape(self._observation_shape)
        act = act.reshape(self._action_shape)
        obs = torch.from_numpy(obs)
        act = torch.from_numpy(act)
        rew = torch.from_numpy(rew)
        bonus = torch.FloatTensor([bonus])
        term = torch.from_numpy(term)

        top = self._top[0]

        self._observations[self._top[0]] = obs #obs.reshape(self._observation_shape)
        self._actions[self._top[0]] = act #act.reshape(self._action_shape)
        self._rewards[self._top[0]] = rew
        self._bonuses[self._top[0]] = bonus
        self._terminals[self._top[0]] = term
        self._top[0] = (self._top[0] + 1) % self._max_size

        # TODO: keep track of moving average and stddev
        # # add new samples to means
        # self._obs_mean += obs.reshape(self._observation_shape) / (self._size + 1)
        # self._act_mean += act.reshape(self._action_shape) / (self._size + 1)

        if self._size[0] >= self._max_size:
            # remove old samples from means
            # self._obs_mean -= self._observations[self._bottom] / self._size
            # self._act_mean -= self._actions[self._bottom] / self._size
            self._bottom[0] = (self._bottom[0] + 1) % self._max_size
        else:
            self._size[0] = self._size[0] + 1

    def share_memory(self):
        self._observations.share_memory_()
        self._actions.share_memory_()
        self._rewards.share_memory_()
        self._bonuses.share_memory_()
        self._terminals.share_memory_()
        self._bottom.share_memory_()
        self._top.share_memory_()
        self._size.share_memory_()

    def sample(self, batch_size):
        assert self._size[0] > batch_size
        indices = np.zeros(batch_size, dtype='int64')
        transition_indices = np.zeros(batch_size, dtype='int64')
        count = 0
        while count < batch_size:
            index = np.random.randint(
                self._bottom[0], self._bottom[0] + self._size[0]) % self._max_size
            # make sure that the transition is valid: if we are at the end of the pool,
            # we need to discard this sample
            if index == self._size[0] - 1 and self._size[0] <= self._max_size:
                continue
            transition_index = (index + 1) % self._max_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        indices = torch.from_numpy(indices).long()
        transition_indices = torch.from_numpy(transition_indices).long()
        return dict(
            observations=self._observations[indices].numpy(),
            actions=self._actions[indices].numpy(),
            rewards=self._rewards[indices].numpy(),
            bonuses=self._bonuses[indices].numpy(),
            terminals=self._terminals[indices].numpy(),
            next_observations=self._observations[transition_indices].numpy())

    # def sample_last(self, batch_size):
    #     """Get a sample of the final batch_size samples"""
    #     assert self._size > batch_size
    #     indices = np.arange(self._bottom, self._bottom + batch_size)
    #     transition_indices = indices + 1
    #     indices = indices % self._size
    #     transition_indices = transition_indices % self._size
    #
    #     return dict(
    #         observations=self._observations[indices],
    #         actions=self._actions[indices],
    #         rewards=self._rewards[indices],
    #         bonuses=self._bonuses[indices],
    #         terminals=self._terminals[indices],
    #         next_observations=self._observations[transition_indices])

    def mean_obs_act(self):
        if self._size[0] >= self._max_size:
            obs = self._observations
            act = self._actions
        else:
            obs = self._observations[:self._top + 1]
            act = self._actions[:self._top + 1]
        obs_mean = np.mean(obs, axis=0)
        obs_std = np.std(obs, axis=0)
        act_mean = np.mean(act, axis=0)
        act_std = np.std(act, axis=0)
        return obs_mean, obs_std, act_mean, act_std

    @property
    def size(self):
        return self._size[0]

    def __len__(self):
        return self._size[0]
