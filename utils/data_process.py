import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from torch import randperm
from torch.utils.data import Subset
from torch._utils import _accumulate
import os


def process_test(data_path):
    df_left = pd.read_csv(os.path.join(data_path, '_slash_xsens_slash_left_tcp.csv'))
    df_right = pd.read_csv(os.path.join(data_path, '_slash_xsens_slash_right_tcp.csv'))

    print(df_left.head(3))

    ax = plt.axes(projection='3d')

    # 三维散点的数据
    zdata = df_left['z'][:200]
    xdata = df_left['x'][:200]
    ydata = df_left['y'][:200]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

    zdata = df_right['z'][:200]
    xdata = df_right['x'][:200]
    ydata = df_right['y'][:200]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Oranges')

    plt.show()


def get_data(data_path, trunc_at=None):
    df_left = pd.read_csv(os.path.join(data_path, '_slash_xsens_slash_left_tcp.csv'))
    df_right = pd.read_csv(os.path.join(data_path, '_slash_xsens_slash_right_tcp.csv'))
    if trunc_at is not None:
        df_left = df_left[:trunc_at]
        df_right = df_right[:trunc_at]
        
    zdata_l = np.array(df_left['z'])
    xdata_l = np.array(df_left['x'])
    ydata_l = np.array(df_left['y'])
    qxdata_l = np.array(df_left['x.1'])
    qydata_l = np.array(df_left['y.1'])
    qzdata_l = np.array(df_left['z.1'])
    qwdata_l = np.array(df_left['w'])

    zdata_r = np.array(df_right['z'])
    xdata_r = np.array(df_right['x'])
    ydata_r = np.array(df_right['y'])
    qxdata_r = np.array(df_right['x.1'])
    qydata_r = np.array(df_right['y.1'])
    qzdata_r = np.array(df_right['z.1'])
    qwdata_r = np.array(df_right['w'])

    return xdata_l, ydata_l, zdata_l, qxdata_l, qydata_l, qzdata_l, qwdata_l, xdata_r, ydata_r, zdata_r, qxdata_r, \
           qydata_r, qzdata_r, qwdata_r


def data_process_delta(data_path=None, data=None, aug=False):
    # For getting the delta of data
    if data_path is not None:
        dual_data = get_data(data_path)
    elif data is not None:
        dual_data = data
    else:
        raise print('Something wrong')

    xdata_l, ydata_l, zdata_l, qxdata_l, qydata_l, qzdata_l, qwdata_l, xdata_r, ydata_r, zdata_r, qxdata_r, \
    qydata_r, qzdata_r, qwdata_r = dual_data

    def get_delta(arr, arr_name):
        arr_shift = np.zeros_like(arr)
        arr_shift[:-1] = arr[1:]
        delta_arr = arr - arr_shift
        delta_arr += 0.5
        delta_arr *= 10000
        print('{}: {}, {}'.format(arr_name, np.max(delta_arr), np.min(delta_arr)))
        return delta_arr

    delta_zdata_l = get_delta(zdata_l, 'zdata_l')
    delta_xdata_l = get_delta(xdata_l, 'xdata_l')
    delta_ydata_l = get_delta(ydata_l, 'ydata_l')

    delta_zdata_r = get_delta(zdata_r, 'zdata_r')
    delta_xdata_r = get_delta(xdata_r, 'xdata_r')
    delta_ydata_r = get_delta(ydata_r, 'ydata_r')

    delta_pos_l = delta_xdata_l, delta_ydata_l, delta_zdata_l
    delta_pos_r = delta_xdata_r, delta_ydata_r, delta_zdata_r
    rot_l = qxdata_l, qydata_l, qzdata_l, qwdata_l
    rot_r = qxdata_r, qydata_r, qzdata_r, qwdata_r

    data = [np.array(delta_pos_l), np.array(delta_pos_r), np.array(rot_l), np.array(rot_r)]

    if aug:
        all_data = None
        for x_rate in range(5, 12):
            aug_data = data_process_trans(data=data, x_rate=x_rate / 10.)
            if all_data is not None:
                for item in range(len(aug_data)):
                    all_data[item] = np.hstack((all_data[item], aug_data[item]))
            else:
                all_data = aug_data
        data = all_data

    return data


def data_process_trans(data_path=None, data=None, x_rate=None, y_rate=None, z_rate=None, all_rate=None):
    if data_path is not None:
        delta_pos_l, delta_pos_r, rot_l, rot_r = data_process_delta(data_path)
    elif data is not None:
        if len(data) == 14:
            xdata_l, ydata_l, zdata_l, qxdata_l, qydata_l, qzdata_l, qwdata_l, xdata_r, ydata_r, zdata_r, qxdata_r, \
            qydata_r, qzdata_r, qwdata_r = data
            delta_pos_l = xdata_l, ydata_l, zdata_l
            delta_pos_r = xdata_r, ydata_r, zdata_r
            rot_l = qxdata_l, qydata_l, qzdata_l, qwdata_l
            rot_r = qxdata_r, qydata_r, qzdata_r, qwdata_r
        elif len(data) == 4:
            delta_pos_l, rot_l, delta_pos_r, rot_r = data
        else:
            raise print('Something wrong!')
    else:
        raise print('Something wrong')
    delta_pos_l = list(delta_pos_l)
    delta_pos_r = list(delta_pos_r)

    dual_data = np.vstack((delta_pos_l, rot_l, delta_pos_r, rot_r))

    if x_rate is not None:
        delta_pos_l[0] *= x_rate
        delta_pos_r[0] *= x_rate
    if y_rate is not None:
        delta_pos_l[1] *= y_rate
        delta_pos_r[1] *= y_rate
    if z_rate is not None:
        delta_pos_l[2] *= z_rate
        delta_pos_r[2] *= z_rate
    if all_rate is not None:
        dual_data = dual_data * all_rate

    return dual_data


def data_process_mean_std(data_path=None, data=None, aug=False):
    if data_path is not None:
        dual_data = get_data(data_path)
    elif data is not None:
        dual_data = data
    else:
        raise print('Something wrong')

    left_data = np.array(dual_data)[:7]
    right_data = np.array(dual_data)[7:]
    concat_data = np.concatenate((left_data, right_data), axis=1)
    data_mean, data_std = np.mean(concat_data, axis=1), np.std(concat_data, axis=1)  # (7,)

    # dual_data_mean = np.mean(np.array(dual_data), axis=1)  # (14,)
    # dual_data_std = np.std(np.array(dual_data), axis=1)
    # data_mean, data_std = dual_data_mean, dual_data_std

    return data_mean, data_std


def manually_random_split(dataset, lengths, sequence_len, generator):
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths),
                       generator=generator).tolist()  # Randomly shuffle a sequence of numbers less than sum(lengths)
    # train_indices, val_indices = [indices[offset - length: offset] for offset, length in zip(_accumulate(lengths), lengths)]

    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]


if __name__ == '__main__':
    # data_process('./data/002-chen-04-dualarmstirfry', aug=True)
    data_process_mean_std('./data/002-chen-04-dualarmstirfry', aug=False)
    # process_test('./data/002-chen-04-dualarmstirfry')
    # process_test('./data/002-chen-02-dualarmstirfry')
