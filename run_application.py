'''
Real time rPPG system for Basler cameras

'''

from pypylon import pylon
import numpy as np
import cv2
import time
from skimage.transform import downscale_local_mean
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt


frame_rate = 20.
hr_band = [40, 250]
img_width = 500
img_height = 500

# Initialize FVP method
K = 6                   # number of top ranked eigenvectors
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
Pt = []
Zt = []


# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Set up camera parameters first in PylonViewer!!!
camera.Width.Value = img_width
camera.Height.Value = img_height
camera.OffsetX.Value = 200
camera.OffsetY.Value = 100
camera.ExposureTime.SetValue(10000)
camera.AcquisitionFrameRate.SetValue(frame_rate)

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

fig, ax = plt.subplots(3, 1, figsize=(14, 10))
ax[0].set_ylim((0.003, 250))

while camera.IsGrabbing():
    startTime = time.time()
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    frame = np.array((img_width, img_height, 3), dtype=np.double)
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        frame = image.GetArray()
        cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Video', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    grabResult.Release()

    # Downsample the image
    Id = downscale_local_mean(frame, (25, 25, 1))
    downscale_size = Id.shape

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

    # Display a weight mask
    # cv2.namedWindow('Weight mask', cv2.WINDOW_NORMAL)
    # cv2.imshow("Weight mask", np.reshape((w[:, 0]/np.max(w[:, 0])), (downscale_size[0], downscale_size[1])))
    # cv2.resizeWindow("Weight mask", 500, 500)

    # Weight subregions with the attained masks
    J = np.zeros((w.shape[1], Id.shape[0], Id.shape[1]))
    for weight_idx in range(w.shape[1]):
        for c in range(Id.shape[1]):
            J[weight_idx, :, c] = np.multiply(w[:, weight_idx], Id[:, c])

    # Create mean and variance statistics of the subregions regarding the specific masks
    Jt_current = np.zeros((4*K, 3))
    Jt_current[0:2*K, :] = np.mean(J, axis=1)
    Jt_current[2*K:4*K, :] = np.var(J, axis=1)

    Jt.append(Jt_current)

    # Extract the PPG signal
    if len(Jt) == L1:
        C = np.array(Jt)
        Jt[:] = []

        # Normalize each color channel with its mean
        Cn = np.divide(C, np.mean(C, axis=0)) - 1

        # Implementing POS algorithm (BGR channel ordering...) to attain pulse
        X = Cn[:, :, 1] - Cn[:, :, 0]
        Y = Cn[:, :, 1] + Cn[:, :, 0] - 2*Cn[:, :, 2]

        # Calculating pulse signal for each weight
        P = X + np.multiply(Y, np.divide(np.std(X, axis=0), np.std(Y, axis=0)))
        Pt.append(np.divide(np.subtract(P, np.mean(P, axis=0)), np.std(P, axis=0)))

        # Calculate the intensity signal to supress noise
        Z = Cn[:, :, 0] + Cn[:, :, 1] + Cn[:, :, 2]
        Zt.append(np.divide(np.subtract(Z, np.mean(Z, axis=0)), np.std(Z, axis=0)))

    if len(Pt) == 13:

        Ptn = np.array(Pt)

        Ptn = np.reshape(Pt, (Ptn.shape[0]*Ptn.shape[1], Ptn.shape[2]))
        Ztn = np.array(Zt)
        Ztn = np.reshape(Zt, (Ztn.shape[0] * Ztn.shape[1], Ztn.shape[2]))

        Fp = np.fft.fft(np.array(np.transpose(Ptn)))/L2
        Fz = np.fft.fft(np.array(np.transpose(Ztn)))/L2

        W = np.divide(np.abs(np.multiply(Fp, np.conj(Fp))), 1 + np.abs(np.multiply(Fz, np.conj(Fz))))

        W[:, 0:B[0]] = 0
        W[:, B[1]:] = 0

        hfq = np.sum(np.multiply(W, Fp), axis=0)
        hr_idx = np.argmax(np.abs(hfq.real))
        hr_est = f[hr_idx]*60

        h_raw = np.fft.ifft(np.sum(Fp, axis=0))
        h_raw = h_raw.real
        h = np.fft.ifft(hfq)
        h = h.real

        #fig.clf()
        ax[0].clear()
        ax[0].plot(h)
        ax[0].set_title("Filtered signal")
        ax[0].text(150, .002, '%s bpm' % int(hr_est), fontsize=18)
        ax[0].set_ylim((-0.003, 0.003))

        ax[1].clear()
        ax[1].plot(h_raw)
        ax[1].set_title("Raw signal")
        #ax[0].text(150, .002, '%s bpm' % int(hr_est), fontsize=18)
        #ax[1].ylim((-0.003, 0.003))

        # ax[2].clear()
        # ax[2].plot(f[:hr_max_idx]*60, np.abs(hfq.real)[:hr_max_idx])
        # ax[2].axvline(x=hr_est)

        plt.pause(0.0000000000001)

        del Pt[0]
        del Zt[0]

    runningTime = (time.time() - startTime)
    fps = 1.0/runningTime
    print "%f  FPS" % fps


plt.show()
# Releasing the resource    
camera.StopGrabbing()
cv2.destroyAllWindows()