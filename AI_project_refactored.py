# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:37:27 2017

@author: Abhishank
"""
import math as m
import numpy as np
import h5py
import rope.base.project
from scipy.fftpack import fft, ifft, fft2, ifft2
from sklearn.cross_validation import LeaveOneOut
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import signal
from collections import Counter
from matplotlib import cm
import glob
from os import listdir
from os.path import isfile, join

plt.close('all')

def get_acc_data (imu_channel, numb_of_samples_total):
  acc_array = np.zeros((np.size(imu_channel, 0), 4))
  acc_array[0:numb_of_samples_total,0] = imu_channel[:,0]
  acc_array[0:numb_of_samples_total,1] = imu_channel[:,1]/(2**12)
  acc_array[0:numb_of_samples_total,2] = imu_channel[:,2]/(2**12)
  acc_array[0:numb_of_samples_total,3] = imu_channel[:,3]/(2**12)
  return acc_array

def highpass (sampling_freq, acc_array):
  b,a = signal.butter(2, 1/(sampling_freq/2), 'high')
  acc_highpassed = signal.lfilter(b, a, acc_array[:,1:4], axis=0)
  acc_array_highpassed = np.zeros((np.size(acc_array,0),4))
  acc_array_highpassed[:,0] = acc_array[:,0]
  acc_array_highpassed[:,1:4] = acc_highpassed[:,0:3]
  return acc_array_highpassed

def signal_processing (sampling_freq, freq_of_interest, numb_of_samples_total, overlap):
  time_of_interest = 1/freq_of_interest
  numb_of_samples_per_window = int (m.floor(sampling_freq*time_of_interest))
  numb_of_windows = m.ceil(numb_of_samples_total/(numb_of_samples_per_window*(1-overlap)))
  sampling_time = 1/sampling_freq
  return sampling_time, time_of_interest, numb_of_windows, numb_of_samples_per_window

def plotting_acc_time_domain(numb_of_samples_total, sampling_freq, acc_array):
  x_axis = np.arange(numb_of_samples_total)/sampling_freq
  direction_array = ['X','Y','Z']
  for i in range (len(direction_array)):
    plt.figure ()
    plt.xlabel('Time')
    plt.ylabel(direction_array[i] + ' - Acceleration - Full')
    plt.title('Acceleration in the ' + direction_array[i] + ' Axis - Full')
    plt.plot(x_axis, acc_array[:,(i+1)])


def classification (numb_of_samples_total, breakpoints_array, acc_array_highpassed):
  class_1 = 1
  class_2 = 2
  class_3 = 3
  labelled_acc_array = np.zeros((numb_of_samples_total, 6))
  i = 0
  j = 0
  #X
  while (i < 6):
      labelled_acc_array[0:numb_of_samples_total,i] = acc_array_highpassed [0:numb_of_samples_total,(j+1)]
      labelled_acc_array [breakpoints_array[0]:breakpoints_array[1],(i+1)] = class_1  #nothing
      labelled_acc_array[breakpoints_array[1]:breakpoints_array[2],(i+1)] = class_2   #walking
      labelled_acc_array[breakpoints_array[2]:breakpoints_array[3],(i+1)] = class_3   #running
      labelled_acc_array[breakpoints_array[3]:breakpoints_array[4],(i+1)] = class_2   #walking
      labelled_acc_array[breakpoints_array[4]:breakpoints_array[5],(i+1)] = class_1   #nothing
      i = i + 2
      j = j + 1
  return labelled_acc_array

def features (labelled_acc_array, numb_of_samples_total, numb_of_windows, numb_of_samples_per_window, overlap):
  start = 0
  end = int (numb_of_samples_per_window)
  increment = m.floor(numb_of_samples_per_window*(1-overlap))
  j = 0
  i = 0
  k = 0
  features_sublist = []
#  features_subarray = np.array()
#  while np.sum(features_list [i,:]!=0):
#    i = i + 1
  while (k<6):
    while (end<numb_of_samples_total):
      features_subrow = []
      features_subrow.extend(Counter(labelled_acc_array[start:end, k+1].astype(int)).most_common(1)[0][0]) #label
      features_subrow.extend(float ((Counter(labelled_acc_array[start:end,k+1].astype(int)).most_common(1)[0][1])/numb_of_samples_per_window))#probablity
      features_subrow.extend(np.mean(labelled_acc_array[start:end,k], axis = 0)) #mean
      features_subrow.extend(np.std(labelled_acc_array[start:end,k],axis = 0)) #standard deviation
      features_sublist.append(features_subrow)
      start = start + increment
      end = end + increment
    k = k + 2

#    features_subarray[i,j] = Counter(labelled_acc_array[start:end, k].astype(int)).most_common(1)[0][0] #label
#    features_subarray[i,(j+1)] = float ((Counter(labelled_acc_array[start:end,(i+1)].astype(int)).most_common(1)[0][1])/numb_of_samples_per_window) #Probability
#    features_subarray[i,(j+2)] = np.mean(labelled_acc_array[start:end,i], axis = 0) #mean
#    features_subarray[i,(j+4)] = np.std(labelled_acc_array[start:end,i],axis = 0) #standard Deviation
#    j = j + 6
#
#    start = start + increment
#    end = end + increment
#    i=i+1

#  n = 3

#  while (n<18):

#    features_subarray[:,n] = (features_subarray[:,(m-1)]-np.amin(features_subarray[:,(n-1)], axis = 0))/(np.amax(features_subarray[:,(n-1)], axis = 0)-np.amin(features_subarray[:,(n-1)], axis = 0)) #normalized mean
#    features_subarray[:,(n+2)] = (features_subarray[:,(n+1)]-np.amin(features_subarray[:,(n+1)], axis = 0))/(np.amax(features_subarray[:,(n+1)],axis = 0)-np.amin(features_subarray[:,(n+1)], axis = 0)) #normalized standard deviation
#    n = n + 6

  return np_features_sublist, k








sampling_freq = 100
freq_of_interest = 0.25
numb_of_samples_total = [41160]
overlap = 0.25
breakpoints_array  = np.array( [[0,11720, 22338,28681,34839,numb_of_samples_total[0]]])
features_list = []




mypath = "./AI_data/"
only_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
files_array  = [len(only_files)]
#for i in range (len(only_files)):
#  f = h5py.File(mypath + only_files[i], "r")
f_1 = h5py.File(mypath + only_files[1], 'r')
imu_channel = np.array(f_1['/signals/imu'])
acc_array = get_acc_data (imu_channel, numb_of_samples_total[0])
acc_array_highpassed = highpass (sampling_freq, acc_array)
plotting_acc_time_domain(numb_of_samples_total[0], sampling_freq, acc_array_highpassed)
labelled_acc_array = classification (numb_of_samples_total[0], breakpoints_array[0,:], acc_array_highpassed)
signal_info = signal_processing (sampling_freq, freq_of_interest, numb_of_samples_total[0], overlap)
features_list = features_list.append(features(labelled_acc_array, numb_of_samples_total[0], signal_info[2], signal_info[3],overlap))

#def signal_processing (sampling_freq, freq_of_interest, numb_of_samples_total, overlap):
#  return sampling_time, time_of_interest, numb_of_windows, numb_of_samples_per_window

#def features (labelled_acc_array, numb_of_samples_total, numb_of_windows, numb_of_samples_per_window, , overlap):