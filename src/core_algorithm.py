import numpy as np
from skimage.transform import downscale_local_mean
from scipy.spatial.distance import cdist


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
def pos(C, order):
    """
    General Blood volume pulse vector algorithm

    :param C: signal attained after averaging a patch of pixels for each frame.
              its shape is [frame, weight_mask_stat, color_channel]
    :param order: list of channel order based on pulsatile amplitude strength (decreasing order)
    :returns: P: The pulse signal for each weight. shape -> [time, weight_mask]
    :returns: Z: The intensity (energy) of the signal
    """

    # Normalize each color channel with its mean
    Cn = np.divide(C, np.mean(C, axis=0)) - 1

    # Implementing POS algorithm (BGR channel ordering...) to attain pulse
    X = Cn[:, :, order[0]] - Cn[:, :, order[1]]
    Y = Cn[:, :, order[0]] + Cn[:, :, order[1]] - 2 * Cn[:, :, order[2]]

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


def cdf_sb_pos(C, order, B):
    """
    CDF: Color-distortion filtering
    SB: Sub-band optimization (for POS)
    POS: Plain orthogonol to skin (core rPPG method

    :param C: signal attained after averaging a patch of pixels for each frame.
              its shape is [frame, weight_mask_stat, color_channel]
    :param order: list of channel order based on pulsatile amplitude strength (decreasing order)
    :param B: pulse band indexes!!
    :returns: P: The pulse signal for each weight. shape -> [time, weight_mask]
    :returns: Z: The intensity (energy) of the signal
    """

    wnum = C.shape[1]

    # Normalize each color channel with its mean
    Cn = np.divide(C, np.mean(C, axis=0)) - 1

    # Calculate the intensity signal to supress noise later
    I = Cn[:, :, 0] + Cn[:, :, 1] + Cn[:, :, 2]

    # Compute FFT along time axis -> 0
    F = np.fft.fft(Cn, axis=0)

    # Do it for each weight-mask statistics
    P = np.ndarray((C.shape[0], C.shape[1]), dtype=np.double)
    for i in range(wnum):
        Ft = F[:, i, :].transpose()

        # pre-processing step: CDF filter -------------------------------------------------------------------------
        # S = np.dot(np.array([-1, 2, -1])/np.sqrt(6), Ft)  # characteristic transformation
        S = 1/np.sqrt(6)*(2*Ft[order[0], :] - Ft[order[1], :] - Ft[order[2], :])
        W = np.divide(np.multiply(S, S.conjugate()), np.sum(np.multiply(Ft, Ft.conjugate()), axis=0))  # energy measurement

        # weighting
        Fw = np.ndarray(shape=(Ft.shape[0], Ft.shape[1]), dtype=np.complex64)
        Fw[0, :] = np.multiply(W, Ft[0, :])
        Fw[1, :] = np.multiply(W, Ft[1, :])
        Fw[2, :] = np.multiply(W, Ft[2, :])

        # Sub-band POS ----------------------------------------------------------------------------------------
        # Implementing POS algorithm (BGR channel ordering...) to attain pulse
        X = Fw[order[0], :] - Fw[order[1], :]
        Y = Fw[order[0], :] + Fw[order[1], :] - 2 * Fw[order[2], :]

        # Calculating pulse signal for each weight (for ech component)
        Z = X + np.multiply(Y, np.divide(np.abs(X), np.abs(Y)))
        # Z = np.multiply(Z, np.divide(np.abs(Z), np.abs(np.sum(Ft, 0))))
        Z = np.multiply(Z, np.divide(np.abs(Z), np.abs(np.sum(Ft, 0))))

        # band limitation
        Z[0:B[0]] = 0
        Z[B[1]:] = 0

        P[:, i] = np.real(np.fft.ifft(Z))

    return np.divide(np.subtract(P, np.mean(P, axis=0)), np.std(P, axis=0)), np.divide(np.subtract(I, np.mean(I, axis=0)), np.std(I, axis=0))


# --------------------------------------------- Signal Comb & Plot ----------------------------------------------------
def signal_combination(Ptn, Ztn, L2, B, f):
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
    Ptn = np.divide(np.subtract(Ptn, np.mean(Ptn, axis=0)), np.std(Ptn, axis=0))
    Ztn = np.divide(np.subtract(Ztn, np.mean(Ztn, axis=0)), np.std(Ztn, axis=0))

    Fp = np.fft.fft(np.array(np.transpose(Ptn))) / L2
    Fz = np.fft.fft(np.array(np.transpose(Ztn))) / L2

    W = np.divide(np.abs(np.multiply(Fp, np.conj(Fp))), 1 + np.abs(np.multiply(Fz, np.conj(Fz))))

    W[:, 0:B[0]] = 0
    W[:, B[1]:] = 0

    hfq = np.sum(np.multiply(W, Fp), axis=0)
    hr_idx = np.argmax(np.abs(hfq.real))
    hr_est = f[hr_idx] * 60
    hr_est = int(round(hr_est))

    hfq_raw = np.sum(Fp, axis=0)
    max_amp = np.abs(hfq_raw.real)[hr_idx]
    other_average = (np.mean(np.abs(hfq_raw.real[0:hr_idx])) + np.mean(np.abs(hfq_raw.real[hr_idx+1:])))/2

    acc = max_amp/other_average
    print "accuracy: " + str(max_amp/other_average)

    h_raw = np.fft.ifft(hfq_raw)
    h_raw = h_raw.real
    h = np.fft.ifft(hfq)
    h = h.real

    h = (h-np.mean(h))/np.std(h)
    h_raw = (h_raw-np.mean(h_raw))/np.std(h_raw)

    return h, h_raw, hr_est, acc
