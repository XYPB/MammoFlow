# eval AP axis distribution
import numpy as np
from models.preprocess import read_image
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt


def emd_from_samples(x, y):
    normalized_x = x / np.max(x) if np.max(x) > 0 else x
    normalized_y = y / np.max(y) if np.max(y) > 0 else y
    return wasserstein_distance(normalized_x, normalized_y)


def overlap_ratio(x, y):
    normalized_x = x / np.max(x) if np.max(x) > 0 else x
    normalized_y = y / np.max(y) if np.max(y) > 0 else y
    intersection = np.minimum(normalized_x, normalized_y).sum()
    union = np.maximum(normalized_x, normalized_y).sum() + 1e-12
    return intersection / union


def make_hist_probs(x, nbins=50, range_=None):
    hist, edges = np.histogram(x, bins=nbins, range=range_, density=False)
    p = hist.astype(float)
    p = p / p.sum() if p.sum() > 0 else p
    return p, edges


def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    q = np.asarray(q, dtype=float)
    q = q / q.sum()
    p = np.clip(p, eps, 1)  # avoid log(0)
    q = np.clip(q, eps, 1)
    return np.sum(p * (np.log(p) - np.log(q)))


def kl_from_samples(x, y, nbins=50, range_=None, eps=1e-12):
    # shared range for fair comparison
    if range_ is None:
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        range_ = (lo, hi)
    p, edges = make_hist_probs(x, nbins, range_)
    q, _ = make_hist_probs(y, nbins, range_)
    return kl_divergence(p, q, eps)


def mutual_information_from_samples(x, y, nbins=50, range_=None, base=2):
    x = np.asarray(x)
    y = np.asarray(y)

    # Shared range for fair binning
    if range_ is None:
        lo_x, hi_x = np.min(x), np.max(x)
        lo_y, hi_y = np.min(y), np.max(y)
        range_ = ((lo_x, hi_x), (lo_y, hi_y))

    H, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=range_)
    Pxy = H / H.sum()

    Px = Pxy.sum(axis=1)
    Py = Pxy.sum(axis=0)

    # Safe entropy (skip zeros)
    def H_discrete(P):
        nz = P > 0
        return -np.sum(P[nz] * np.log(P[nz]))

    Hx = H_discrete(Px)
    Hy = H_discrete(Py)
    Hxy = H_discrete(Pxy)

    mi = Hx + Hy - Hxy  # in nats
    if base == 2:  # set base=2 for bits
        mi /= np.log(2)
    return mi

def mutual_invariance(p, q, base=2):
    if not np.allclose(np.sum(p), 1.0):
        p = p / np.sum(p)
    if not np.allclose(np.sum(q), 1.0):
        q = q / np.sum(q)

    def H_discrete(P):
        nz = P > 0
        return -np.sum(P[nz] * np.log(P[nz]))
    Hx = H_discrete(p)
    Hy = H_discrete(q)
    Hxy = H_discrete(0.5 * (p + q))
    mi = Hx + Hy - Hxy  # in nats
    if base == 2:
        mi /= np.log(2)
    return mi


def get_ap_axis_dist(image, smooth_sigma=10, axis=0):
    y_axis_dist = np.sum(np.array(image), axis=axis)
    if smooth_sigma > 0:
        y_axis_dist = gaussian_filter1d(y_axis_dist, sigma=smooth_sigma)
    return y_axis_dist


def plot_1d_histogram(
    array,
    bins=50,
    xlabel="",
    ylabel="",
):
    plt.figure(figsize=(8, 4))
    plt.hist(array, bins=bins, color="tab:blue", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_1d_distribution(
    distribution, xlabel="AP axis", ylabel="Pixel count", smooth_sigma=10
):
    plt.figure(figsize=(8, 4))
    plt.plot(distribution / np.max(distribution), color="b")
    plt.fill_between(
        range(len(distribution)),
        distribution / np.max(distribution),
        alpha=0.3,
        color="b",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_paired_1d_distribution(
    mlo_distribution, cc_distribution, xlabel="AP axis", ylabel="Pixel count"
):
    plt.figure(figsize=(8, 4))
    plt.plot(mlo_distribution / np.max(mlo_distribution), label="MLO", color="b")
    plt.plot(cc_distribution / np.max(cc_distribution), label="CC", color="r")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def plot_2d_distribution(
    distribution, xlabel="x-axis", ylabel="y-axis", x_tick=None, y_tick=None
):
    # plot 3D surface of a 2D distribution
    X, Y = np.meshgrid(
        np.arange(distribution.shape[1]), np.arange(distribution.shape[0])
    )
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, distribution, cmap="viridis", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_tick is not None:
        ax.set_xticks(list(range(len(x_tick))), x_tick, fontsize=8)
    if y_tick is not None:
        ax.set_yticks(list(range(len(y_tick))), y_tick, fontsize=8)
    plt.show()


def display_distribution_res(
    mlo_image, cc_image, smooth_sigma=10, save_path=None, show_img=True
):
    mlo_ap_dist = get_ap_axis_dist(mlo_image, smooth_sigma)
    cc_ap_dist = get_ap_axis_dist(cc_image, smooth_sigma)

    emd = emd_from_samples(mlo_ap_dist, cc_ap_dist)
    # use JSD divergence
    kld1 = kl_divergence(mlo_ap_dist, cc_ap_dist)
    kld2 = kl_divergence(cc_ap_dist, mlo_ap_dist)
    jsd = 0.5 * (kld1 + kld2)
    mi = mutual_invariance(mlo_ap_dist, cc_ap_dist, base=2)
    overlap = overlap_ratio(mlo_ap_dist, cc_ap_dist)

    if show_img:
        plt.figure(figsize=(12, 4))
        plt.title(
            f"AP Axis Distribution Comparison (EMD={emd:.2f}, JSD={jsd:.2f}, MI={mi:.2f}, Overlap={overlap:.2f})"
        )
        plt.axis("off")
        plt.subplot(1, 3, 1)
        plt.imshow(mlo_image, cmap="gray")
        plt.title("MLO Image")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(cc_image, cmap="gray")
        plt.title("CC Image")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.plot(mlo_ap_dist / np.sum(mlo_ap_dist), label="MLO", color="tab:blue")
        plt.fill_between(
            range(len(mlo_ap_dist)),
            mlo_ap_dist / np.sum(mlo_ap_dist),
            alpha=0.3,
            color="tab:blue",
        )
        plt.plot(cc_ap_dist / np.sum(cc_ap_dist), label="CC", color="tab:red")
        plt.fill_between(
            range(len(cc_ap_dist)),
            cc_ap_dist / np.sum(cc_ap_dist),
            alpha=0.3,
            color="tab:red",
        )
        plt.xlim(0, max(len(mlo_ap_dist), len(cc_ap_dist)))
        plt.xlabel("AP axis")
        plt.ylabel("Pixel count")
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    return {"emd": emd, "jsd": jsd, "mi": mi, "overlap": overlap}
