import numpy as np

class Memory(object):

    def __init__(self,
                 max_size,
                 observation_shape1,
                 observation_shape2,
                 action_shape,
                 observation_dtype1=np.float32,
                 observation_dtype2=np.float32,
                 action_dtype=np.float32):
        self._max_size = max_size
        self._observation_shape1 = observation_shape1
        self._observation_shape2 = observation_shape2
        self._action_shape = action_shape

        self._observations1 = np.zeros((max_size,)+observation_shape1, dtype=observation_dtype1)
        self._observations2 = np.zeros((max_size,)+observation_shape2, dtype=observation_dtype2)
        self._actions = np.zeros((max_size,)+(action_shape,), dtype=action_dtype)
        self._rewards = np.zeros(max_size, dtype=np.float32)
        self._bonuses = np.zeros(max_size, dtype=np.float32)
        self._terminals = np.zeros(max_size, dtype=np.uint8)

        self._obs_mean1 = np.zeros(observation_shape1, dtype=observation_dtype1)
        self._obs_stddev1 = np.zeros(observation_shape1, dtype=observation_dtype1)
        self._obs_mean2 = np.zeros(observation_shape2, dtype=observation_dtype2)
        self._obs_stddev2 = np.zeros(observation_shape2, dtype=observation_dtype2)
        self._act_mean = np.zeros(action_shape, dtype=action_dtype)
        self._act_stddev = np.zeros(action_shape, dtype=action_dtype)

        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, obs1, obs2, act, rew, term, bonus=0.):
        self._observations1[self._top] = obs1.reshape(self._observation_shape1)
        self._observations2[self._top] = obs2.reshape(self._observation_shape2)
        self._actions[self._top] = act.reshape(self._action_shape)
        self._rewards[self._top] = rew
        self._bonuses[self._top] = bonus
        self._terminals[self._top] = term
        self._top = (self._top + 1) % self._max_size

        if self._size < self._max_size:
            self._size = self._size + 1

        # # add new samples to means
        # self._obs_mean += obs.reshape(self._observation_shape) / (self._size + 1)
        # self._act_mean += act.reshape(self._action_shape) / (self._size + 1)
        #
        # if self._size >= self._max_size:
        #     # remove old samples from means
        #     self._obs_mean -= self._observations[self._bottom] / self._size
        #     self._act_mean -= self._actions[self._bottom] / self._size
        #     self._bottom = (self._bottom + 1) % self._max_size
        # else:
        #     self._size = self._size + 1
        #
        # # TODO: keep track of moving average and stddev

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
            observations1=self._observations1[indices],
            observations2=self._observations2[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            bonuses=self._bonuses[indices],
            terminals=self._terminals[indices],
            next_observations1=self._observations1[transition_indices],
            next_observations2=self._observations2[transition_indices])

    def sample_last(self, batch_size):
        """Get a sample of the final batch_size samples"""
        assert self._size > batch_size
        indices = np.arange(self._bottom, self._bottom + batch_size)
        transition_indices = indices + 1
        indices = indices % self._size
        transition_indices = transition_indices % self._size

        return dict(
            observations1=self._observations1[indices],
            observations2=self._observations2[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            bonuses=self._bonuses[indices],
            terminals=self._terminals[indices],
            next_observations1=self._observations1[transition_indices],
            next_observations2=self._observations2[transition_indices])


    def mean_obs_act(self):
        if self._size >= self._max_size:
            obs1 = self._observations1
            obs2 = self._observations2
            act = self._actions
        else:
            obs1 = self._observations1[:self._top + 1]
            obs2 = self._observations2[:self._top + 1]
            act = self._actions[:self._top + 1]
        obs_mean1 = np.mean(obs1, axis=0)
        obs_std1 = np.std(obs1, axis=0)
        obs_mean2 = np.mean(obs2, axis=0)
        obs_std2 = np.std(obs2, axis=0)
        act_mean = np.mean(act, axis=0)
        act_std = np.std(act, axis=0)
        return obs_mean1, obs_std1, obs_mean2, obs_std2, act_mean, act_std

    @property
    def size(self):
        return self._size

    def __len__(self):
        return self._size
