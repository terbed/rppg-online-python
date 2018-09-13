'''
Real time rPPG system for Basler cameras

'''

from pypylon import pylon
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import src.core_algorithm as core
import thread

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

fig, ax = plt.subplots(2, 1, figsize=(14, 8))
plt_thread = core.PlotThread(1, fig, ax)

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

    # ----------------------------------------------------------------------------- Image processing with FVP algorithm
    Jt.append(core.fvp(frame, patch_size, K))

    # Extract the PPG signal
    if len(Jt) == L1:
        C = np.array(Jt)
        Jt[:] = []

        # -------------------------------------------------------------------------- Pulse extraction algorithm
        P, Z = core.pos(C)
        Pt.append(P)
        Zt.append(Z)

    if len(Pt) == 13:

        Ptn = np.array(Pt)
        Ztn = np.array(Zt)

        Ptn = np.reshape(Ptn, (Ptn.shape[0]*Ptn.shape[1], Ptn.shape[2]))
        Ztn = np.reshape(Ztn, (Ztn.shape[0] * Ztn.shape[1], Ztn.shape[2]))

        # --------------------------------------------------------------------------- Create final Pulse signal
        #thread.start_new_thread(core.signal_combination, (Ptn, Ztn, L2, B, f, plt_thread))
        h, h_raw, hr_est = core.signal_combination(Ptn, Ztn, L2, B, f, plt_thread)
        plt_thread.update_data(h, h_raw, hr_est)
        plt_thread.start()

        del Pt[0]
        del Zt[0]

    runningTime = (time.time() - startTime)
    fps = 1.0/runningTime
    print "%f  FPS" % fps


plt.show()
# Releasing the resource    
camera.StopGrabbing()
cv2.destroyAllWindows()