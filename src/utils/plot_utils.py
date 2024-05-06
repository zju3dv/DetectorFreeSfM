import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
import matplotlib.cm as cm

matplotlib.use("Agg")

jet = cm.get_cmap("jet")  # "Reds"
jet_colors = jet(np.arange(256))[:, :3]  # color list: normalized to [0,1]


def plot_image_pair(imgs, dpi=100, size=6, pad=0.5):
    n = len(imgs)
    assert n == 2, "number of images must be two"
    figsize = (size * n, size) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap("gray"), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)

def plot_images(imgs, match_locations=None, plot_center=True, dpi=80, size=6, pad=0.5):
    n, m = imgs.shape[:2]
    window_size = imgs.shape[-2]
    if match_locations is not None:
        n_, m_ = match_locations.shape[:2]
        assert n==n_ and m == m_
    figsize = (size * m, size * n) if size is not None else None # W * H

    _, ax = plt.subplots(n, m, figsize=figsize, dpi=dpi)
    for i in range(n):
        for j in range(m):
            # ax[i, j].imshow(imgs[i, j], cmap=plt.get_cmap("gray"), vmin=0, vmax=255)
            ax[i, j].imshow(imgs[i, j])
            ax[i, j].get_yaxis().set_ticks([])
            ax[i, j].get_xaxis().set_ticks([])
            for spine in ax[i, j].spines.values():  # remove frame
                spine.set_visible(False)
            
            if match_locations is not None:
                ax[i, j].scatter(match_locations[i,j,0], match_locations[i,j,1], c='b', s=240, marker='x')
            
            if plot_center:
                ax[i, j].scatter(window_size//2, window_size//2, c='r', s=160, marker='o')

    plt.tight_layout(pad=pad)

def plot_keypoints(kpts0, kpts1, color="w", ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps, marker="x")
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps, marker="x")


def plot_keypoints_for_img0(kpts, color="w", ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts[:, 0], kpts[:, 1], c=color, s=ps)


def plot_keypoints_for_img1(kpts, color="w", ps=2):
    ax = plt.gcf().axes
    ax[1].scatter(kpts[:, 0], kpts[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=0.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [
        matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]),
            (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1,
            transform=fig.transFigure,
            c=color[i],
            linewidth=lw,
        )
        for i in range(len(kpts0))
    ]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

def make_matching_plot(
    image0,
    image1,
    kpts0,
    kpts1,
    mkpts0,
    mkpts1,
    color,
    text,
    path=None,
    show_keypoints=False,
    fast_viz=False,
    opencv_display=False,
    opencv_title="matches",
    small_text=[],
):

    if fast_viz:
        make_matching_plot_fast(
            image0,
            image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            path,
            show_keypoints,
            10,
            opencv_display,
            opencv_title,
            small_text,
        )
        return

    plot_image_pair([image0, image1])  # will create a new figure
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color="k", ps=4)
        plot_keypoints(kpts0, kpts1, color="w", ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = "k" if image0[:100, :150].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )

    txt_color = "k" if image0[-100:, :150].mean() > 200 else "w"
    fig.text(
        0.01,
        0.01,
        "\n".join(small_text),
        transform=fig.axes[0].transAxes,
        fontsize=5,
        va="bottom",
        ha="left",
        color=txt_color,
    )
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        # TODO: Would it leads to any issue without current figure opened?
        return fig

def make_matching_plot_fast(
    image0,
    image1,
    kpts0,
    kpts1,
    mkpts0,
    mkpts1,
    color,
    text,
    path=None,
    show_keypoints=False,
    margin=10,
    opencv_display=False,
    opencv_title="",
    small_text=[],
):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin :] = image1
    out = np.stack([out] * 3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1, lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(
            out,
            (x0, y0),
            (x1 + margin + W0, y1),
            color=c,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640.0, 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(
            out,
            t,
            (int(8 * sc), Ht * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 * sc,
            txt_color_bg,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (int(8 * sc), Ht * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 * sc,
            txt_color_fg,
            1,
            cv2.LINE_AA,
        )

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(
            out,
            t,
            (int(8 * sc), int(H - Ht * (i + 0.6))),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5 * sc,
            txt_color_bg,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (int(8 * sc), int(H - Ht * (i + 0.6))),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5 * sc,
            txt_color_fg,
            1,
            cv2.LINE_AA,
        )

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def reproj(K, pose, pts_3d):
    """ 
    Reproj 3d points to 2d points 
    @param K: [3, 3] or [3, 4]
    @param pose: [3, 4] or [4, 4]
    @param pts_3d: [n, 3]
    """
    assert K.shape == (3, 3) or K.shape == (3, 4)
    assert pose.shape == (3, 4) or pose.shape == (4, 4)

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K

    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose

    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_3d_homo = pts_3d_homo.T

    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points  # [n, 2]