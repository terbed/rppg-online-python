'''
Real time rPPG system for Basler cameras

This version calculates every frame and overlap-add the results

'''

from pypylon import pylon
import numpy as np
import cv2
import time
import src.core_algorithm as core
from scipy import stats
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import src.disp as disp

frame_rate = 20.
exp_val = 20000
hr_band = [40, 200]
img_width = 500
img_height = 500

# Initialize FVP method
K = 6                   # number of top ranked eigenvectors
patch_size = 25
L1 = 256
u0 = 1
L2 = 512               # window length in frame
l0 = float(L1)/float(frame_rate)       # window length in seconds
Fb0 = 1./l0                # frequency bin in Hz
l = float(L2)/float(frame_rate)       # window length in seconds
Fb = 1./l                # frequency bin in Hz
f0 = np.linspace(0, L1*Fb0, L1, dtype=np.double)  # frequency vector in Hz
f = np.linspace(0, L2*Fb, L2, dtype=np.double)  # frequency vector in Hz
t = np.linspace(0, L2*1./frame_rate, L2)
hr_min_idx0 = np.argmin(np.abs(f0*60-hr_band[0]))
hr_max_idx0 = np.argmin(np.abs(f0*60-hr_band[1]))
hr_min_idx = np.argmin(np.abs(f*60-hr_band[0]))
hr_max_idx = np.argmin(np.abs(f*60-hr_band[1]))
B0 = [hr_min_idx0, hr_max_idx0]
B = [hr_min_idx, hr_max_idx]             # HR range ~ [50, 220] bpm

# Channel ordering for BGR
# The largest pulsatile strength is in G then B then R
channel_ordering = [1, 0, 2]

Jt = []

add_row = np.zeros(shape=(1, K*4), dtype=np.double)
Pt = np.zeros(shape=(L2, K*4), dtype=np.double)
Zt = np.zeros(shape=(L2, K*4), dtype=np.double)

# Container for the overlap-added signal
H = np.zeros(shape=(1, L2), dtype=np.double)
H_RAW = np.zeros(shape=(1, L2), dtype=np.double)

# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Set up camera parameters first in PylonViewer!!!
camera.Width.Value = img_width
camera.Height.Value = img_height
camera.OffsetX.Value = 1200
camera.OffsetY.Value = 0
camera.ExposureTime.SetValue(exp_val)
camera.AcquisitionFrameRate.SetValue(frame_rate)
camera.PixelFormat = "BayerRG12"

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_RGB16packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
bgr_img = frame = np.ndarray(shape=(img_height, img_width, 3), dtype=np.uint16)

# Set up display with PyQt
app = QtWidgets.QApplication([])
pg.setConfigOptions(antialias=False)  # True seems to work as well
qtplt_thread = disp.DispThread(parent=None, t=t)

shift_idx = 0
heart_rates = []
acclist = []
first_run = True

while camera.IsGrabbing():
    startTime = time.time()
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    bgr_img = np.ndarray(shape=(img_height, img_width, 3), dtype=np.uint16)

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
        del Jt[0]       # delete the first element

        # -------------------------------------------------------------------------- Pulse extraction algorithm
        P, Z = core.cdf_sb_pos(C, channel_ordering, B0)
 
        if shift_idx + L1-1 < L2:
            Pt[shift_idx:shift_idx+L1, :] = Pt[shift_idx:shift_idx+L1, :] + P
            Zt[shift_idx:shift_idx+L1, :] = Zt[shift_idx:shift_idx+L1, :] + Z

            # average add, not overlap add
            Pt[shift_idx:shift_idx+L1-1, :] = Pt[shift_idx:shift_idx+L1-1, :]/2
            Zt[shift_idx:shift_idx+L1-1, :] = Zt[shift_idx:shift_idx+L1-1, :]/2

            shift_idx = shift_idx + 1
        else:     # In this case the L2 length is fully loaded, we have to remove the first element and add a new one at the end
            # overlap add resulteros for the new frame point

            Pt = np.delete(Pt, 0, 0)  # delete first row (last frame)
            Pt = np.append(Pt, add_row, 0)    # append z
            Zt = np.delete(Zt, 0, 0)  # delete first row (last frame)
            Zt = np.append(Zt, add_row, 0)    # append zeros for the new frame point

            Pt[shift_idx-1:shift_idx+L1-1, :] = Pt[shift_idx-1:shift_idx+L1-1, :] + P
            Zt[shift_idx-1:shift_idx+L1-1, :] = Zt[shift_idx-1:shift_idx+L1-1, :] + Z

            # overlap average
            Pt[shift_idx-1:shift_idx+L1-2, :] = Pt[shift_idx-1:shift_idx+L1-2, :]/2
            Zt[shift_idx-1:shift_idx+L1-2, :] = Zt[shift_idx-1:shift_idx+L1-2, :]/2

            computing = True

    if shift_idx == L2-L1+1:
        # now we can also calculate fourier and signal combination
        h, h_raw, hr_est, acc = core.signal_combination(Pt, Zt, L2, B, f)
        heart_rates.append(hr_est)
        acclist.append(acc)

        H = np.delete(H, 0, 0)
        H_RAW = np.delete(H_RAW, 0, 0)
        H = np.append(H, 0.)
        H_RAW = np.append(H_RAW, 0.)

        # overlap add
        H = H + h
        H_RAW = H_RAW + h_raw

        # overlap average
        H[0:L2-1] = H[0:L2-1]/2
        H_RAW[0:L2-1] = H_RAW[0:L2-1]/2

        # [DEBUG]
        # qtplt_thread.start_plotting_thread(H, H_RAW, hr_est)

        if first_run:
            qtplt_thread.start_plotting_thread(H, H_RAW, hr_est)
            first_run = False

    if len(heart_rates) == frame_rate*2:
        # Display HR estimate and signals if accuracy is over threshold
        estimated_HR, _ = stats.mode(heart_rates)
        avrg_acc = np.mean(acclist)

        if avrg_acc >= 4:
            qtplt_thread.start_plotting_thread(H, H_RAW, estimated_HR)
        else:
            qtplt_thread.start_plotting_thread(H, H_RAW, 0)
        heart_rates = []
        acclist = []

    runningTime = (time.time() - startTime)
    fps = 1.0/runningTime
    print "%f  FPS" % fps

cv2.destroyAllWindows()
app.exec_()
# Releasing the resource
camera.StopGrabbing()
