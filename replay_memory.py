import numpy as np

class Memory(object):

    def __init__(self,
                 max_size,
                 observation_shape,
                 action_shape,
                 observation_dtype=np.float32,
                 action_dtype=np.float32):
        self._max_size = max_size
        self._observation_shape = observation_shape
        self._action_shape = action_shape

        self._observations = np.zeros((max_size,)+observation_shape, dtype=observation_dtype)
        self._actions = np.zeros((max_size,)+(action_shape,), dtype=action_dtype)
        self._rewards = np.zeros(max_size, dtype=np.float32)
        self._bonuses = np.zeros(max_size, dtype=np.float32)
        self._terminals = np.zeros(max_size, dtype=np.uint8)

        self._obs_mean = np.zeros(observation_shape, dtype=observation_dtype)
        self._obs_stddev = np.zeros(observation_shape, dtype=observation_dtype)
        self._act_mean = np.zeros(action_shape, dtype=action_dtype)
        self._act_stddev = np.zeros(action_shape, dtype=action_dtype)

        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, obs, act, rew, term, bonus=0.):
        self._observations[self._top] = obs.reshape(self._observation_shape)
        self._actions[self._top] = act.reshape(self._action_shape)
        self._rewards[self._top] = rew
        self._bonuses[self._top] = bonus
        self._terminals[self._top] = term
        self._top = (self._top + 1) % self._max_size

        # add new samples to means
        self._obs_mean += obs.reshape(self._observation_shape) / (self._size + 1)
        self._act_mean += act.reshape(self._action_shape) / (self._size + 1)

        if self._size >= self._max_size:
            # remove old samples from means
            self._obs_mean -= self._observations[self._bottom] / self._size
            self._act_mean -= self._actions[self._bottom] / self._size
            self._bottom = (self._bottom + 1) % self._max_size
        else:
            self._size = self._size + 1

        # TODO: keep track of moving average and stddev

    def sample(self, batch_size):
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(
                self._bottom, self._bottom + self._size) % self._max_size
            # make sure that the transition is valid: if we are at the end of the pool,
            # we need to discard this sample
            if index == self._size - 1 and self._size <= self._max_size:
                continue
            transition_index = (index + 1) % self._max_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            bonuses=self._bonuses[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices])

    def sample_last(self, batch_size):
        """Get a sample of the final batch_size samples"""
        assert self._size > batch_size
        indices = np.arange(self._bottom, self._bottom + batch_size)
        transition_indices = indices + 1
        indices = indices % self._size
        transition_indices = transition_indices % self._size

        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            bonuses=self._bonuses[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices])

    def mean_obs_act(self):
        if self._size >= self._max_size:
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
        return self._size

    def __len__(self):
        return self._size
