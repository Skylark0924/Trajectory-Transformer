import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

all_epoch_loss = []
all_epoch_loss_val = []


def plt_process_vis(process_vis_log_file, epoch_loss, epoch_loss_val):
    all_epoch_loss.append(epoch_loss)
    all_epoch_loss_val.append(epoch_loss_val)

    plt.figure(0)

    # 绘制训练损失曲线
    plt.plot(all_epoch_loss, label="Train Loss")
    plt.plot(all_epoch_loss_val, color="red", label="Val Loss")

    # 保存图片
    plt.savefig("{}/loss_7.png".format(process_vis_log_file))
    plt.close()


def plt_img_log(output, tgt, epoch_index, step_i, img_log_file, val_mode=False, val_img_log=None):
    m = nn.Softmax(dim=-1)
    output_softmax = m(output)
    output_3d = torch.max(output_softmax, dim=-1)[1]
    output_np = output_3d[:, 0, :].cpu().numpy()
    tgt_np = tgt[1:, 0, :].cpu().numpy()
    fig = plt.figure('3D Comparison')
    # plt.plot(output_np, 'b--')
    # plt.plot(tgt_np, 'r--')

    # cm = plt.cm.get_cmap('RdYlBu')

    x = output_np[:, 0]
    y = output_np[:, 1]
    z = output_np[:, 2]

    xt = tgt_np[:, 0]
    yt = tgt_np[:, 1]
    zt = tgt_np[:, 2]

    ax = fig.gca(projection='3d')
    t = np.array([i for i in range(len(output_np[:, 0]))])
    N_points = len(output_np[:, 0])
    t = t / (max(t) * 1.5) + 0.3
    for i in range(1, N_points):
        ax.plot(x[i - 1:i + 1], y[i - 1:i + 1], z[i - 1:i + 1], c=(t[i - 1], 0, 0), linestyle='dashed')
        ax.plot(xt[i - 1:i + 1], yt[i - 1:i + 1], zt[i - 1:i + 1], c=(0, 0, t[i - 1]), linestyle='dashed')
    # plt.show()

    # ax.plot3D(output_np[:, 0], output_np[:, 1], output_np[:, 2], co)
    # ax.plot3D(tgt_np[:, 0], tgt_np[:, 1], tgt_np[:, 2], 'r--')
    if val_mode:
        if not os.path.exists('./%s/epoch_%04d/' % (val_img_log, epoch_index)):
            os.makedirs('./%s/epoch_%04d/' % (val_img_log, epoch_index))
        plt.savefig('./%s/epoch_%04d/%04d.jpg' % (val_img_log, epoch_index, step_i))
    else:
        if not os.path.exists('./%s/epoch_%04d/' % (img_log_file, epoch_index)):
            os.makedirs('./%s/epoch_%04d/' % (img_log_file, epoch_index))
        plt.savefig('./%s/epoch_%04d/%04d.jpg' % (img_log_file, epoch_index, step_i))
    plt.close()


def plt_img_log_new(output, tgt, epoch_index, step_i, img_log_file):
    if type(output) == list:
        output_np = np.array(output)[1:, :]
        tgt_np = tgt[1:, 0, :].detach().numpy()
    elif output.device.type == 'cpu':
        output_np = output[:, 0, :].detach().numpy()
        tgt_np = tgt[1:, 0, :].detach().numpy()
    else:  # cuda
        output_np = output[:, 0, :].detach().cpu().numpy()
        tgt_np = tgt[1:, 0, :].detach().cpu().numpy()

    fig = plt.figure('3D Comparison')

    x = output_np[:, 0]
    y = output_np[:, 1]
    z = output_np[:, 2]

    xt = tgt_np[:, 0]
    yt = tgt_np[:, 1]
    zt = tgt_np[:, 2]

    ax = fig.gca(projection='3d')
    t = np.array([i for i in range(len(output_np[:, 0]))])
    N_points = len(output_np[:, 0])
    t = t / (max(t) * 1.5) + 0.3
    for i in range(1, N_points):
        ax.plot(x[i - 1:i + 1], y[i - 1:i + 1], z[i - 1:i + 1], c=(t[i - 1], 0, 0), linestyle='dashed')
        ax.plot(xt[i - 1:i + 1], yt[i - 1:i + 1], zt[i - 1:i + 1], c=(0, 0, t[i - 1]), linestyle='dashed')
    # plt.show()

    # ax.plot3D(output_np[:, 0], output_np[:, 1], output_np[:, 2], co)
    # ax.plot3D(tgt_np[:, 0], tgt_np[:, 1], tgt_np[:, 2], 'r--')
    if not os.path.exists('%s/epoch_%04d/' % (img_log_file, epoch_index)):
        os.makedirs('%s/epoch_%04d/' % (img_log_file, epoch_index))
    plt.savefig('%s/epoch_%04d/%04d.jpg' % (img_log_file, epoch_index, step_i))
    plt.close()
