'''
Real time rPPG system for Basler cameras

'''

from pypylon import pylon
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import src.core_algorithm as core

frame_rate = 20.
exp_val = 10000
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

# Channel ordering for BGR
# The largest pulsatile strength is in G then B then R
channel_ordering = [1, 0, 2]

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
camera.ExposureTime.SetValue(exp_val)
camera.AcquisitionFrameRate.SetValue(frame_rate)

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_RGB16packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
bgr_img = frame = np.ndarray(shape=(img_height, img_width, 3), dtype=np.uint16)

fig, ax = plt.subplots(2, 1, figsize=(14, 8))
ax[0].set_title("Filtered signal")
ax[1].set_title("Raw signal")

while camera.IsGrabbing():
    startTime = time.time()
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    frame = np.array((img_width, img_height, 3), dtype=np.double)
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        frame = np.ndarray(buffer=image.GetBuffer(), shape=(image.GetHeight(), image.GetWidth(), 3), dtype=np.uint16)
        bgr_img[:, :, 0] = frame[:, :, 2]
        bgr_img[:, :, 1] = frame[:, :, 1]
        bgr_img[:, :, 2] = frame[:, :, 0]

        cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Video', bgr_img*16)
        k = cv2.waitKey(1)
        if k == 27:
            break
    grabResult.Release()

    # ----------------------------------------------------------------------------- Image processing with FVP algorithm
    Jt.append(core.fvp(bgr_img, patch_size, K))

    # Extract the PPG signal
    if len(Jt) == L1:
        C = np.array(Jt)
        Jt[:] = []

        # -------------------------------------------------------------------------- Pulse extraction algorithm
        P, Z = core.pos(C, channel_ordering)
        Pt.append(P)
        Zt.append(Z)

    if len(Pt) == 13:

        Ptn = np.array(Pt)
        Ztn = np.array(Zt)

        Ptn = np.reshape(Ptn, (Ptn.shape[0]*Ptn.shape[1], Ptn.shape[2]))
        Ztn = np.reshape(Ztn, (Ztn.shape[0] * Ztn.shape[1], Ztn.shape[2]))

        # --------------------------------------------------------------------------- Create final Pulse signal
        h, h_raw, hr_est = core.signal_combination(Ptn, Ztn, L2, B, f)

        ax[0].clear()
        ax[0].plot(h)
        ax[0].set_ylim((-0.01, 0.01))
        ax[0].text(150, .008, '%s bpm' % int(hr_est), fontsize=18)

        ax[1].clear()
        ax[1].plot(h_raw)

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