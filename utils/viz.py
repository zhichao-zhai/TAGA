import torch
import matplotlib.pyplot as plt
import numpy as np


def viz_render(gt_rgb, gt_mask, pred_pkg, save_path=None):
    pred_rgb = pred_pkg["rgb"].permute(1, 2, 0)
    pred_mask = pred_pkg["alpha"].squeeze(0)
    pred_depth = pred_pkg["dep"].squeeze(0)
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(torch.clamp(gt_rgb, 0.0, 1.0).detach().cpu().numpy())
    plt.title("GT"), plt.axis("off")
    plt.subplot(1, 5, 2)
    plt.imshow(torch.clamp(pred_rgb, 0.0, 1.0).detach().cpu().numpy())
    plt.title("Pred view"), plt.axis("off")
    plt.subplot(1, 5, 3)
    error = (
        torch.clamp(abs(pred_rgb - gt_rgb), 0.0, 1.0)
        .detach()
        .cpu()
        .numpy()
        .max(axis=-1)
    )
    cmap = plt.imshow(error)
    plt.title("Render Error (max in rgb)"), plt.axis("off")
    plt.colorbar(cmap, shrink=0.8)

    plt.subplot(1, 5, 4)
    error = torch.clamp(pred_mask - gt_mask, -1.0, 1.0).detach().cpu().numpy()
    cmap = plt.imshow(error)
    plt.title("(Pr - GT) Mask Error"), plt.axis("off")
    plt.colorbar(cmap, shrink=0.8)

    plt.subplot(1, 5, 5)
    depth = pred_depth.detach().cpu().numpy()
    cmap = plt.imshow(depth)
    plt.title("Pred Depth"), plt.axis("off")
    plt.colorbar(cmap, shrink=0.8)

    plt.tight_layout()
    fig.canvas.draw()
    fig_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_np = fig_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)
    return fig_np


def viz_img(pred_pkg, save_path=None):
    pred_rgb = pred_pkg["rgb"].permute(1, 2, 0)

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(torch.clamp(pred_rgb, 0.0, 1.0).detach().cpu().numpy())
    plt.title("Pred view")
    plt.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    fig.canvas.draw()
    fig_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_np = fig_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return fig_np


BONE_NAMES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
]


def showPose(template, pose, path):

    joints = template.get_posed_joints(pose)[0].detach().cpu().numpy()

    joints[:, [1, 2]] = joints[:, [2, 1]]

    joints[:, 1] = -joints[:, 1]  # Flip the Y axis.
    joints[:, 2] = -joints[:, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for joint in joints:
        ax.scatter(*joint, c="red")

    for bone in BONE_NAMES:
        start_idx, end_idx = bone
        joint_start = joints[start_idx]
        joint_end = joints[end_idx]
        ax.plot(
            [joint_start[0], joint_end[0]],
            [joint_start[1], joint_end[1]],
            [joint_start[2], joint_end[2]],
            "k",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set the equal aspect ratio for all axes
    max_range = (
        np.array(
            [
                joints[:, 0].max() - joints[:, 0].min(),
                joints[:, 1].max() - joints[:, 1].min(),
                joints[:, 2].max() - joints[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (joints[:, 0].max() + joints[:, 0].min()) * 0.5
    mid_y = (joints[:, 1].max() + joints[:, 1].min()) * 0.5
    mid_z = (joints[:, 2].max() + joints[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    fig.savefig(path)
