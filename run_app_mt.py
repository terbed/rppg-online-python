'''
Real time rPPG system for Basler cameras

'''

from pypylon import pylon
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from skimage.transform import downscale_local_mean
from scipy.spatial.distance import cdist
import threading


# Create new thread for Running the camera
class RunCamera(threading.Thread):
    def __init__(self, threadID, img_width, img_height, exp_val, patch_size, K):
        threading.Thread.__init__(self)
        self.threadID = threadID

        # conecting to the first available camera
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()

        # Set up camera parameters first in PylonViewer!!!
        self.camera.Width.Value = img_width
        self.camera.Height.Value = img_height
        self.camera.OffsetX.Value = 200
        self.camera.OffsetY.Value = 100
        self.camera.ExposureTime.SetValue(exp_val)

        # Grabing Continusely (video) with minimal delay
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()

        # converting to opencv bgr format
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.img_width = img_width
        self.img_height = img_height
        self.patch_size = patch_size
        self.K = K

        self.frame = None

    def run(self):
        global video_lock
        global Jt_lock
        global Jt

        while self.camera.IsGrabbing():
            startTime = time.time()
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            self.frame = np.array((self.img_width, self.img_height, 3), dtype=np.double)
            if grabResult.GrabSucceeded():
                # Access the image data
                image = self.converter.Convert(grabResult)
                with video_lock:
                    self.frame = image.GetArray()

            grabResult.Release()

            # Image processing with FVP algorithm
            with Jt_lock:
                Jt.append(fvp(self.frame, self.patch_size, self.K))

            runningTime = (time.time() - startTime)
            fps = 1.0 / runningTime
            print "%f  FPS" % fps
            print len(Jt)

        self.camera.StopGrabbing()


# Create a new thread for the calculation
class RunCalculation(threading.Thread):
    def __init__(self, threadID, L1, L2, B, f):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.L1 = L1
        self.L2 = L2
        self.B = B
        self.f = f

        self.Pt = []
        self.Zt = []

        self.h = None
        self.h_raw = None
        self.hr_est = None

    def run(self):
        global Jt
        global Jt_lock
        global plot_lock

        while True:
            # Extract the PPG signal
            if len(Jt) == self.L1:
                with Jt_lock:
                    C = np.array(Jt)
                    Jt[:] = []

                # -------------------------------------------------------------------------- Pulse extraction algorithm
                print "Calculating POS for 20 frame"
                P, Z = pos(C)
                self.Pt.append(P)
                self.Zt.append(Z)

            if len(self.Pt) == 13:
                print "Calculation pulse signal for 200 frame"
                Ptn = np.array(self.Pt)
                Ztn = np.array(self.Zt)
                print Ptn.shape

                Ptn = np.reshape(Ptn, (Ptn.shape[0] * Ptn.shape[1], Ptn.shape[2]))
                Ztn = np.reshape(Ztn, (Ztn.shape[0] * Ztn.shape[1], Ztn.shape[2]))

                # --------------------------------------------------------------------------- Create final Pulse signal
                print "Create signal to plot"
                with plot_lock:
                    self.h, self.h_raw, self.hr_est = signal_combination(Ptn, Ztn, self.L2, self.B, self.f)

                del self.Pt[0]
                del self.Zt[0]

        cv2.destroyAllWindows()


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


# ------------------------------------------- PLOT RESULTS ----------------------------------------------------------
def disp_result(frame, ax, h, h_raw, hr_est):
    """
    Semi real-time plot of the result

    :param frame: frame
    :param ax: axes of the figure
    :param h: filtered pulse signal
    :param h_raw: raw pulse signal
    :param hr_est: estimated heart rate
    :return: plots the results
    """

    if type(frame) is np.ndarray:
        cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Video', frame)

    if h is not None:
        ax[0].clear()
        ax[0].plot(h)
        ax[0].set_title("Filtered signal")
        ax[0].text(150, .002, '%s bpm' % int(hr_est), fontsize=18)
        ax[0].set_ylim((-0.003, 0.003))

        ax[1].clear()
        ax[1].plot(h_raw)
        ax[1].set_title("Raw signal")

        plt.pause(0.0000000000001)


################################################################################x MAIN ######################################################################################


frame_rate = 20.
hr_band = [40, 250]
img_width = 500
img_height = 500

# Initialize FVP method
K = 6                   # number of top ranked eigenvectors
patch_size = 25
L1 = frame_rate
u0 = 1
L2 = 260                # window length in frame
l = L2/frame_rate       # window length in seconds
Fb = 1./l                # frequency bin in Hz
f = np.linspace(0, L2*Fb, L2, dtype=np.double)  # frequency vector in Hz
hr_min_idx = np.argmin(np.abs(f*60-hr_band[0]))
hr_max_idx = np.argmin(np.abs(f*60-hr_band[1]))
B = [hr_min_idx, hr_max_idx]             # HR range ~ [50, 220] bpm

Jt = []
Jt_lock = threading.Lock()
plot_lock = threading.Lock()
video_lock = threading.Lock()

Pt = []
Zt = []


fig, ax = plt.subplots(2, 1, figsize=(14, 8))

# Initialize and start camera recording thread
camera_thread = RunCamera(1, img_width, img_height, 10000, patch_size, K)
camera_thread.start()

# Initialize and start calculation thread
calc_thread = RunCalculation(2, L1, L2, B, f)
calc_thread.start()

# Main display thread (because GUI can be only in the main thread...)
while cv2.waitKey(2000) != 27:
    with plot_lock, video_lock:
        disp_result(camera_thread.frame, ax, calc_thread.h, calc_thread.h_raw, calc_thread.hr_est)


plt.show()
# Releasing the resource
cv2.destroyAllWindows()