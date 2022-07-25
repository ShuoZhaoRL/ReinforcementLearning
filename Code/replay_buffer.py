import numpy as np

import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_capacity, device):
        """
        store data sampling from environment
        :parameter
            store_dim: [state, action, reward, state_new, done]
        """
        self.max_capacity = max_capacity
        self.max_capacity_done = max_capacity//100
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.store_dim = state_dim*2 + action_dim + 1 + 1
        self.container = np.zeros([max_capacity, self.store_dim])
        self.container_done = np.zeros([self.max_capacity_done, self.store_dim])

        self.pointer = 0
        self.final = 0
        self.pointer_done = 0
        self.final_done = 0

        self.done_flag = False
        self.device = device

    def add(self, sample_data: np.array):
        """
        data: [the number of data, data]
        """
        sample_data = np.array(sample_data).reshape(-1, self.store_dim)
        data_shape = np.shape(sample_data)
        if self.pointer+data_shape[0] > self.max_capacity:
            margin = self.pointer+data_shape[0] - self.max_capacity
            difference = self.max_capacity - self.pointer
            self.container[self.pointer:self.max_capacity, :] = sample_data[0:difference, :]
            self.container[0:margin] = sample_data[difference:, :]
            self.pointer = margin
            self.final = self.max_capacity
        else:
            self.container[self.pointer:self.pointer + data_shape[0], :] = sample_data
            self.pointer += data_shape[0]
        pass

    def add_done(self, sample_data: np.array):
        """Add a new experience to memory."""
        sample_data = np.array(sample_data).reshape(-1, self.store_dim)
        data_shape = np.shape(sample_data)
        if self.pointer_done + data_shape[0] > self.max_capacity_done:
            margin = self.pointer_done + data_shape[0] - self.max_capacity_done
            difference = self.max_capacity_done - self.pointer_done
            self.container_done[self.pointer_done:self.max_capacity_done, :] = sample_data[0:difference, :]
            self.container_done[0:margin] = sample_data[difference:, :]
            self.pointer_done = margin
            self.final_done = self.max_capacity_done
        else:
            self.container_done[self.pointer_done:self.pointer_done + data_shape[0], :] = sample_data
            self.pointer_done += data_shape[0]
        pass

    def sample(self, batch_size) -> torch.Tensor:
        if batch_size > np.max((self.pointer, self.final)):
            print('the number of data stored in container is too few')

        if self.done_flag:
            lis = np.random.randint(0, np.max((self.pointer, self.final)), batch_size-batch_size//4)
            arr = self.container[lis, :]
            lis_done = np.random.randint(0, np.max((self.pointer_done, self.final_done)), batch_size//4)
            arr = np.vstack([arr, self.container_done[lis_done, :]])
            sample_data = torch.from_numpy(arr).float().to(self.device)
        else:
            lis = np.random.randint(0, np.max((self.pointer, self.final)), batch_size)
            sample_data = torch.from_numpy(self.container[lis, :]).float().to(self.device)
        return torch.split(sample_data, [self.state_dim, self.action_dim, 1, self.state_dim, 1], dim=-1)

    def get_container(self):
        return self.container

    def clear(self):
        self.container = np.zeros([self.max_capacity, self.store_dim])
        self.pointer = 0
        self.final = 0

    def __len__(self):
        """Return the current size of internal memory."""
        return np.max((self.pointer, self.final))
