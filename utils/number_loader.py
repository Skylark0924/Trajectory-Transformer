from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset
import numpy as np


class NumberLoader(Dataset):
    def __init__(self, x, y, inp_len=3, out_len=3, D3=False, pose_dim=7):
        if len(x) != len(y):
            raise ValueError("len(x) != len(y)")
        self.x = [[x[i + j] for j in range(inp_len)] for i in range(len(x) - inp_len + 1)]
        self.y = [[y[i + j] for j in range(out_len)] for i in range(len(y) - out_len + 1)]
        self.D3 = D3
        self.pose_dim = pose_dim

    def __getitem__(self, index):
        # Dimension amend
        if self.D3:
            # print('num', np.array(self.y[index]).shape)
            return FloatTensor(self.x[index]), FloatTensor(np.vstack(([0] * self.pose_dim, self.x[index])))
        else:
            return LongTensor(self.x[index]), LongTensor([0] + self.y[index])

    def __len__(self):
        return len(self.x)
