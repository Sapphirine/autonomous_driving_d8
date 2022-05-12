import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import time
import random, os , sys
from random import choice
from skimage.feature import hog
from skimage.measure import label, regionprops
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from vehicle_detection_utils import *
from lane_detection import *
import joblib

from tensorflow import keras

lane_fit_frame=[]
lane_fit_average=[]
def detect_lanes(img):
    #print("lol in function"
    img = cv2.imread(img, cv2.COLOR_RGB2BGR)
    print("Image read")
    w = img[0].shape[0]
    h = img[0].shape[1]
    img1 = cv2.resize(img,(h,w))
    img1 = np.array(img1)
    #print(img1.shape)
    img1 = img1[None,:,:,:]
    #print("Image reshaped to shape", img1.shape)
    # cv2.imwrite(os.path.join( "static/tampered", "test.jpg" ), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    # cv2.imwrite("drive/MyDrive/bda_data/img1.jpg",img1)
    # print("cv2 imwrite done")
    model = keras.models.load_model('models/lane_model.h5')
    sample = model.predict(img1)[0]
    cv2.imwrite(os.path.join( "static/tampered", "lane_sample.jpg" ), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
    # cv2.imwrite("drive/MyDrive/bda_data/lane_sample.jpg",sample)
    out_image = sample * 255
    lane_fit_frame.append(out_image)
    lane_fit_average = np.mean(np.array([i for i in lane_fit_frame]),axis = 0)
    pad = np.zeros_like(lane_fit_average).astype(np.uint8)
    lane_lines = np.dstack((pad,lane_fit_average,pad))
    w1 = img.shape[0]
    h1 = img.shape[1]
    lane_image = cv2.resize(lane_lines, (h1,w1))
    result = cv2.addWeighted(img,1,lane_image,1,0, dtype = cv2.CV_32F)
    new = cv2.imwrite(os.path.join( "static/lane", "output.jpg" ), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return result
