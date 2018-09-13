import numpy as np
from skimage.transform import downscale_local_mean
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import threading


# Create new thread for plotting smoothly
class PlotThread(threading.Thread):

    def __init__(self, threadID, fig, ax):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.fig = fig
        self.ax = ax
        self.h = None
        self.h_raw = None
        self.hr_est = None

    def update_data(self, h, h_raw, hr_est):
        self.h = h
        self.h_raw = h_raw
        self.hr_est = hr_est

    def run(self):
        self.ax[0].clear()
        self.ax[0].plot(self.h)
        self.ax[0].set_title("Filtered signal")
        self.ax[0].text(150, .002, '%s bpm' % int(self.hr_est), fontsize=18)
        self.ax[0].set_ylim((-0.003, 0.003))

        self.ax[1].clear()
        self.ax[1].plot(self.h_raw)
        self.ax[1].set_title("Raw signal")

        plt.pause(0.0000000000001)

def run_camera(camera, frame, patch_size, K):
    pass


# --------------------------------------------- FVP ------------------------------------------------------------------
def fvp(frame, patch_size, K):
    """
    Full Video Pulse Extraction:
    A frame comes in. It is downsampled, a similarity matrix is created and weight masks for the image, clustering the
    regions with similar color feature.

    :param frame: frame to process
    :param patch_size: subregion size: patch_size * patch_size pixels
    :param K: number of largest eigen vectors
    :return: Jt -> 2*K number of weighted subregion statistic value (mean, var)
    """

    # Downsample the image
    Id = downscale_local_mean(frame, (patch_size, patch_size, 1))

    # Reshape Id
    Id = np.reshape(Id, (Id.shape[0] * Id.shape[1], 3))

    # Color channel norm
    In = np.zeros(Id.shape)
    norm_fact = np.sum(Id, axis=1)

    for idx, n in enumerate(norm_fact):
        In[idx, :] = Id[idx, :] / norm_fact[idx]  # divide each row by its sum

    # Create Affinity matrix
    A = cdist(In, In,  metric='euclidean')

    # Compute the eigenvectors
    u, _, _ = np.linalg.svd(A)

    # Create weight vector
    w = np.zeros((u.shape[0], K*2), dtype=np.double)

    w[:, 0:K] = u[:, 0:K]
    w[:, K:2*K] = -1*u[:, 0:K]

    # Weights cannot be negative numbers (shift with the minimum)
    w = w-np.min(w[:])

    # Normalize
    norm_fact = np.sum(w, 0)

    for i in range(len(norm_fact)):
        w[:, i] = w[:, i]/norm_fact[i]

    # Weight subregions with the attained masks
    J = np.zeros((w.shape[1], Id.shape[0], Id.shape[1]))
    for weight_idx in range(w.shape[1]):
        for c in range(Id.shape[1]):
            J[weight_idx, :, c] = np.multiply(w[:, weight_idx], Id[:, c])

    # Create mean and variance statistics of the subregions regarding the specific masks
    Jt_current = np.zeros((4*K, 3))
    Jt_current[0:2*K, :] = np.mean(J, axis=1)
    Jt_current[2*K:4*K, :] = np.var(J, axis=1)

    return Jt_current


# -------------------------------------------- POS -------------------------------------------------------------------
def pos(C):
    """
    Blood volume pulse vector algorithm

    :param C: signal attained after averaging a patch of pixels for each frame.
                       its shape is [frame, weight_mask, color_channel]
    :returns: P: The pulse signal for each weight. shape -> [time, weight_mask]
    :returns: Z: The intensity (energy) of the signal
    """

    # Normalize each color channel with its mean
    Cn = np.divide(C, np.mean(C, axis=0)) - 1

    # Implementing POS algorithm (BGR channel ordering...) to attain pulse
    X = Cn[:, :, 1] - Cn[:, :, 0]
    Y = Cn[:, :, 1] + Cn[:, :, 0] - 2 * Cn[:, :, 2]

    # Calculating pulse signal for each weight
    P = X + np.multiply(Y, np.divide(np.std(X, axis=0), np.std(Y, axis=0)))

    # Calculate the intensity signal to supress noise
    Z = Cn[:, :, 0] + Cn[:, :, 1] + Cn[:, :, 2]

    return np.divide(np.subtract(P, np.mean(P, axis=0)), np.std(P, axis=0)), np.divide(np.subtract(Z, np.mean(Z, axis=0)), np.std(Z, axis=0))


# ---------------------------------------------------- PBV ------------------------------------------------------------
def pbv(raw_signal):
    """
    Plain orthogonal to skin algorithm

    :param raw_signal: signal attained after averaging a patch of pixels for each frame
                       its shape is [frame, subregion, color_channel]
    :return: The pulse signal
    """
    pass


# --------------------------------------------- Signal Comb & Plot ----------------------------------------------------
def signal_combination(Ptn, Ztn, L2, B, f, plt_thread):
    """
    Combine independent pulse signal to construct the finel pulse signal.
    And plot results.

    :param Ptn: Pulse signals for the different weight masks: [time, weight_mask]
    :param Ztn: Intensity signal for the different weight masks: [time, weight_mask]
    :param L2: Window length for Fourier analysis
    :param B: Pulse Band in Hz -> [min, max]
    :param f: frequency vector
    :param plt_thread: A thread for plotting
    :return: Plots the results
    """

    Fp = np.fft.fft(np.array(np.transpose(Ptn))) / L2
    Fz = np.fft.fft(np.array(np.transpose(Ztn))) / L2

    W = np.divide(np.abs(np.multiply(Fp, np.conj(Fp))), 1 + np.abs(np.multiply(Fz, np.conj(Fz))))

    W[:, 0:B[0]] = 0
    W[:, B[1]:] = 0

    hfq = np.sum(np.multiply(W, Fp), axis=0)
    hr_idx = np.argmax(np.abs(hfq.real))
    hr_est = f[hr_idx] * 60

    h_raw = np.fft.ifft(np.sum(Fp, axis=0))
    h_raw = h_raw.real
    h = np.fft.ifft(hfq)
    h = h.real

    return h, h_raw, hr_est
