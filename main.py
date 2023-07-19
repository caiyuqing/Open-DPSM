# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:34:10 2023

@author: 7009291
"""
# import packages
import os
import pandas as pd
import numpy as np
from classes.preprocessing import preprocessing
#from classes.video_processing import video_processing
#from classes.image_processing import image_processing
from classes.event_extraction import event_extraction
from classes.pupil_prediction import pupil_prediction
from classes.interactive_plot import interactive_plot

import pickle
import cv2
import threading
from threading import *
from PIL import Image, ImageTk
import sys
import logging

import psutil
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import time
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button,TextBox,CheckButtons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
########################################################################################################################################################
#####################################################Information entered by the user####################################################################
########################################################################################################################################################
## If the movie and the eyetracking data are not the same, what to do?
stretchToMatch = True # True: stretch the eyelinkdata to match the movie data; False: cut whichever is longer
## What is the resolution of eyetracking data
eyetracking_height = 1080
eyetracking_width = 1920
## What is the video (showed on the screen) resolution (respective to eyetracking resolution, can be different from video file).
videoRealHeight = 1080
videoRealWidth = 1920
## maximum luminance of the screen (luminance when white is showed on screen)
maxlum = 212
## if video resolution is not the same as eyetracking resolution, what color is the screen covered with? (enter rgb value, E.g. r,b,g = 0 means black)
screenBgColorR = 0
screenBgColorG = 0
screenBgColorB = 0
# Directories
initialDir = "D:\\Users\\7009291\\Desktop\\Movie pupil perimetry\\codes for both data\\Open-DPSM" # This should be the directory of the Open-DPSM

dataDir = initialDir + '\\Example' # This should be the folder saving the eyetracking data and the video data 
## eyetracking data:
###- should have four columns in the order as: time stamps, gaze position x, gaze position y, pupil size
###- time stamps should be in seconds (not miliseconds). If not, please convert it to seconds
subjectFileName = "csv_example_raw_sec(CB cb1).csv" # name of the subject (data file should be contained by "dataDir") [Comment out this line if no eyetracking data]
## video data
### Format can be used: .mp4,.avi,.mkv,.mwv,.mov,.flv,.webm (other format can also be used as long as it can be read by cv2)
movieName =  "VideoExample_sameRatio.mp4" # name of the movie (data file should be contained by "dataDir")
### Do you want to save:
# - model evaluation & paramters
saveParams = True
# - data used for modeling
saveData = True
########################################################################################################################################################
###########################################################End of information entering.#################################################################
########################################################################################################################################################

##########################################Preprocessing: check the gaze data and the movie data######################
# This is the indicator that the app is not used.
useApp = False
# chdir
os.chdir(initialDir)
# load parameters that should not be changed
exec(open("settings.py").read())
# read video data and check information
prepObj = preprocessing()
filename_movie = dataDir +"\\" + movieName
prepObj.videoFileName = filename_movie
prepObj.preprocessingVideo()
video_nFrame = prepObj.vidInfo['frameN_end']
video_height = prepObj.vidInfo['height']
video_width = prepObj.vidInfo['width']
video_ratio = video_height / video_width 
video_duration = prepObj.vidInfo['duration_end']
video_fps = prepObj.vidInfo['fps_end']
print(f"Video number of frame: {video_nFrame}")
print(f"Video height x width: {video_height}x{video_width}; aspect ratio (width:height): {1/video_ratio}")
print(f"Video duration: {video_duration}")
print(f"Video frame rate: {video_fps}")
movieName = movieName.split(".")[0]

if 'subjectFileName' in globals():
    filename_csv = dataDir + "\\" + subjectFileName
    # read eyetracking data and check information
    df_eyetracking = pd.read_csv(filename_csv, index_col=0)

    eyetracking_duration = df_eyetracking.iloc[-1,0]
    eyetracking_nSample = df_eyetracking.shape[0]
    eyetracking_samplingrate = int(1/(eyetracking_duration/eyetracking_nSample))
    subjectName = subjectFileName.split(".")[0]

    print(f"Eyetracking data duration: {eyetracking_duration} seconds")
    print(f"Eyetracking data sampling rate: {eyetracking_samplingrate} Hz")
else:
    # rename subjectName as NoEyetracking data if there is no eyetracking data

    subjectName = "NoEyetrackingData" # name the events file as NoEyetrackingData
    gazecentered = False # not use gaze data
    print("No eyetracking data. model prediction will not be preformed. Predicted pupil trace will be generated with the response function from our study")
# check if video and the eyetracking data have the same ratio
if videoRealHeight == eyetracking_height and videoRealWidth ==eyetracking_width:
    videoScreenSameRatio = True 
    videoStretched = True
elif videoRealHeight == eyetracking_height or videoRealWidth ==eyetracking_width:
    videoScreenSameRatio = False
    videoStretched = True
else:
    videoScreenSameRatio = False
    videoStretched = False


#%%############################################Feature extraction##############################################
# NOTE: skip this part if feature extraction has already done previously
# create folder to save data
os.chdir(dataDir) 
foldername = "Visual events"
if not os.path.exists(foldername):
   os.makedirs(foldername)
os.chdir(foldername)

# name the feature extracted pickle:
picklename = movieName + "_"+ subjectName + "_VF_" +colorSpace + "_" + imageSector + ".pickle"

# feature extraction class
eeObj = event_extraction()


# load some data and parameters
eeObj.video_duration = video_duration
eeObj.video_fps = video_fps

eeObj.videoRealHeight = videoRealHeight
eeObj.videoRealWidth = videoRealWidth
eeObj.screenBgColorR = screenBgColorR
eeObj.screenBgColorG = screenBgColorG
eeObj.screenBgColorB = screenBgColorB
eeObj.stretchToMatch = stretchToMatch
eeObj.subject = subjectName
eeObj.movieNum = movieName
eeObj.picklename = picklename
eeObj.filename_movie = filename_movie
eeObj.setNBufFrames(nFramesSeqImageDiff + 1)
eeObj.imCompFeatures = True  # creates: imageObj.vectorMagnFrame
eeObj.showVideoFrames = showVideoFrames
eeObj.imColSpaceConv = colorSpace
eeObj.gazecentered = gazecentered
eeObj.nVertMatPartsPerLevel = nVertMatPartsPerLevel  # [4, 8, 16, 32]
eeObj.aspectRatio = aspectRatio 
eeObj.imageSector = imageSector
eeObj.nFramesSeqImageDiff = nFramesSeqImageDiff
eeObj.selectFeatures = featuresOfInterest
eeObj.scrGamFac = scrGamFac
eeObj.A = A
eeObj.maxlum = maxlum
eeObj.useApp = useApp
eeObj.videoScreenSameRatio = videoScreenSameRatio 
eeObj.videoStretched = videoStretched    
eeObj.vidInfo = prepObj.vidInfo # extract vidInfo from preprocessing object

# process eyetracking data
if gazecentered: # if there is eyetracking data, do gaze-contingent visual events extraction
    eeObj.eyetracking_duration = eyetracking_duration
    eeObj.eyetracking_height = eyetracking_height
    eeObj.eyetracking_width = eyetracking_width
    eeObj.eyetracking_samplingrate = eyetracking_samplingrate
    timeStampsSec = np.array(df_eyetracking.iloc[:,0])
    gazexdata = np.array(df_eyetracking.iloc[:,1])
    gazeydata = np.array(df_eyetracking.iloc[:,2])
    pupildata = np.array(df_eyetracking.iloc[:,3])
    # resample the eytracking data to match the video sampling rate
    eeObj.sampledTimeStamps_featureExtraction =eeObj.prepare_sampleData(timeStampsSec, video_nFrame)
    eeObj.sampledgazexData_featureExtraction = eeObj.prepare_sampleData(gazexdata, video_nFrame)
    eeObj.sampledgazeyData_featureExtraction = eeObj.prepare_sampleData(gazeydata, video_nFrame)
    eeObj.sampledpupilData_featureExtraction = eeObj.prepare_sampleData(pupildata, video_nFrame)
# start feature extraction: this can take a while. The extracted features will be saved in folder "Visual events"
eeObj.event_extraction()

#%%############################################Pupil modeling##############################################
# NOTE: this part can only be performed if eyelink data exists
picklename = movieName + "_"+ subjectName + "_VF_" +colorSpace + "_" + imageSector + ".pickle"
#load feature data
os.chdir(dataDir)
os.chdir("Visual events")
with open(picklename, "rb") as handle:
    vidInfo, timeStamps, magnPerImPart,magnPerIm = pickle.load(handle)
    handle.close() 
# create folder to save data

# new folder for modeling results
foldername = "Modeling result"
os.chdir(dataDir) 
if not os.path.exists(foldername):
   os.makedirs(foldername)
os.chdir(foldername)
#Create dictionaries to save results
if os.path.exists("modelDataDict.pickle"):
    with open("modelDataDict.pickle", "rb") as handle:
        modelDataDict = pickle.load(handle)
        handle.close() 
else:
    modelDataDict = {}
        
if os.path.exists("modelResultDict.pickle"):
    with open("modelResultDict.pickle", "rb") as handle:
        modelResultDict = pickle.load(handle)
        handle.close() 
    #subjectProcessed = list(modelResultDict.keys())
    #subjects = [subject for subject in subjects if subject not in subjectProcessed]
else:
    modelResultDict = {}
# To-do: sameWeightFeature may not work
# if subject is already in the dictionary, remove its results
if subjectName in list(modelResultDict.keys()):
    modelResultDict[subjectName] = {}
    modelDataDict[subjectName] = {}
# pupil prediction class
modelObj = pupil_prediction()

modelObj.subject = subjectName
modelObj.movie = movieName
modelObj.sameWeightFeature =sameWeightFeature
modelObj.RF =RF 
modelObj.skipNFirstFrame =skipNFirstFrame 
modelObj.useBH = useBH
modelObj.niter = niter
modelObj.magnPerImPart= magnPerImPart
modelObj.useApp = useApp
modelObj.stretchToMatch = stretchToMatch
modelObj.video_duration = video_duration
modelObj.eyetracking_duration = eyetracking_duration
modelObj.video_fps = video_fps
modelObj.nFramesSeqImageDiff = nFramesSeqImageDiff

# load eyetracking data
modelObj.useEtData = True
timeStampsSec = np.array(df_eyetracking.iloc[:,0])
gazexdata = np.array(df_eyetracking.iloc[:,1])
gazeydata = np.array(df_eyetracking.iloc[:,2])
pupildata = np.array(df_eyetracking.iloc[:,3])

modelObj.sampledTimeStamps  =modelObj.prepare_sampleData(timeStampsSec,video_nFrame)
modelObj.sampledgazexData =modelObj.prepare_sampleData(gazexdata,video_nFrame)
modelObj.sampledgazeyData=modelObj.prepare_sampleData(gazeydata,video_nFrame)
modelObj.sampledpupilData=modelObj.prepare_sampleData(pupildata,video_nFrame)
modelObj.sampledFps = 1/(modelObj.sampledTimeStamps[-1]/(len(modelObj.sampledTimeStamps)))

modelObj.sampledTimeStamps = modelObj.synchronize(modelObj.sampledTimeStamps)
modelObj.sampledgazexData = modelObj.synchronize(modelObj.sampledgazexData)
modelObj.sampledgazeyData = modelObj.synchronize(modelObj.sampledgazeyData)
modelObj.sampledpupilData = modelObj.synchronize(modelObj.sampledpupilData)
modelObj.sampledpupilData= modelObj.zscore(modelObj.sampledpupilData)

modelObj.modelDataDict = modelDataDict
modelObj.modelResultDict = modelResultDict
# 
modelObj.pupil_prediction()

sampledTimeStamps = modelObj.sampledTimeStamps
sampledpupilData = modelObj.sampledpupilData
sampledFps = modelObj.sampledFps
##################################
# save model results
if saveParams:
    foldername = "csv_results"
    os.chdir(dataDir) 
    if not os.path.exists(foldername):
       os.makedirs(foldername)
    os.chdir(foldername)
    params = modelResultDict[subjectName]["modelContrast"]["parameters"]
    if RF == "HL":
            paramNames = ["r",'rmse',"n_luminance", "tmax_luminance", "n_contrast", "tmax_contrast", "weight_contrast", "regional_weight1","regional_weight2","regional_weight3","regional_weight4","regional_weight5","regional_weight6"]
            params = np.insert(params,5,1)
            params = np.insert(params,0,modelObj.r)
            params = np.insert(params,1,modelObj.rmse)
            df = pd.DataFrame(np.vstack([paramNames,params]).T)
            df.columns = ["parameterName", "value"]
            df.to_csv(f"{subjectName}_parameters.csv")    
    elif RF == "KB":
            paramNames = ["r",'rmse',"theta_luminance", "k_luminance", "theta_contrast", "k_contrast", "weight_contrast", "regional_weight1","regional_weight2","regional_weight3","regional_weight4","regional_weight5","regional_weight6"]
            params = np.insert(params,5,1)
            params = np.insert(params,0,modelObj.r)
            params = np.insert(params,1,modelObj.rmse)
            df = pd.DataFrame(np.vstack([paramNames,params]).T)
            df.columns = ["parameterName", "value"]
            df.to_csv(f"{subjectName}_parameters.csv")
# save modeling data
if saveData:
    foldername = "csv_results"
    os.chdir(dataDir) 
    if not os.path.exists(foldername):
       os.makedirs(foldername)
    os.chdir(foldername)
    y_pred = modelResultDict[subjectName]["modelContrast"]["predAll"] 
    lumConv = modelResultDict[subjectName]["modelContrast"]["lumConv"] 
    contrastConv = modelResultDict[subjectName]["modelContrast"]["contrastConv"] 
    sampledpupilData_z = (sampledpupilData -np.nanmean(sampledpupilData)) /np.nanstd(sampledpupilData)
    df = pd.DataFrame(np.vstack([sampledTimeStamps,sampledpupilData, y_pred,lumConv,contrastConv]).T)
    df.columns = ["timeStamps", "Actual pupil (z)", "Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]
    
    df.to_csv(f"{subjectName}_modelPrediction.csv")
####################################################################################

# making plot
# This step have to be done after pupil prediction
plotObj = interactive_plot()
# subject and movie to plot
plotObj.subjectName = subjectName
plotObj.movie = movieName
# other parameters
plotObj.useApp = useApp
plotObj.dataDir = dataDir
plotObj.filename_movie = filename_movie
plotObj.A = A
plotObj.skipNFirstFrame =skipNFirstFrame
plotObj.sampledFps = sampledFps
plotObj.eyetracking_height = eyetracking_height
plotObj.eyetracking_width = eyetracking_width
plotObj.videoRealHeight = videoRealHeight
plotObj.videoRealWidth = videoRealWidth
plotObj.screenBgColorR = screenBgColorR
plotObj.screenBgColorG = screenBgColorG
plotObj.screenBgColorB = screenBgColorB
plotObj.videoScreenSameRatio = videoScreenSameRatio 
plotObj.videoStretched = videoStretched
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plotObj.plot()
#%%######################################Pupil prediction (no Eyetracking data)#######################################
# NOTE: run this part if no eyetracking data available
picklename = movieName + "_"+ subjectName + "_VF_" +colorSpace + "_" + imageSector + ".pickle"
#load feature data
os.chdir(dataDir)
os.chdir("Visual events")
with open(picklename, "rb") as handle:
    vidInfo, timeStamps, magnPerImPart,magnPerIm = pickle.load(handle)
    handle.close() 
# new folder for modeling results
foldername = "Modeling result"
os.chdir(dataDir) 
if not os.path.exists(foldername):
   os.makedirs(foldername)
os.chdir(foldername)
#Create dictionaries to save results
if os.path.exists("modelDataDict.pickle"):
    with open("modelDataDict.pickle", "rb") as handle:
        modelDataDict = pickle.load(handle)
        handle.close() 
else:
    modelDataDict = {}
        
if os.path.exists("modelResultDict.pickle"):
    with open("modelResultDict.pickle", "rb") as handle:
        modelResultDict = pickle.load(handle)
        handle.close() 
    #subjectProcessed = list(modelResultDict.keys())
    #subjects = [subject for subject in subjects if subject not in subjectProcessed]
else:
    modelResultDict = {}
# To-do: sameWeightFeature may not work
# if subject is already in the dictionary, remove its results
if subjectName in list(modelResultDict.keys()):
    modelResultDict[subjectName] = {}
    modelDataDict[subjectName] = {}
# pupil prediction class
modelObj = pupil_prediction()
modelObj.useEtData = False
if RF == 'HL':
    params = [9.67,0.19,0.8,0.52,0.3, 1,1,1,1,1] 
else:
    params = [0.12,4.59,0.14,6.78,0.28,1,1,1,1,1]
modelObj.sampledTimeStamps = timeStamps
modelObj.sampledFps = 1/(modelObj.sampledTimeStamps [-1]/(len(modelObj.sampledTimeStamps)))
modelObj.numRemoveMovFrame = 0
modelObj.modelDataDict = modelDataDict
modelObj.modelResultDict = modelResultDict
modelObj.sameWeightFeature = sameWeightFeature
modelObj.RF = RF
modelObj.magnPerImPart = magnPerImPart
modelObj.subject = subjectName
modelObj.pupil_predictionNoEyetracking(params)
sampledFps = modelObj.sampledFps
sampledTimeStamps = modelObj.sampledTimeStamps
####################################
# save model results
foldername = "csv_results"
os.chdir(dataDir) 
if not os.path.exists(foldername):
   os.makedirs(foldername)
os.chdir(foldername)
# save data used for pupil prediction
if saveData:
    y_pred = modelResultDict[subjectName]["modelContrast"]["predAll"] 
    lumConv = modelResultDict[subjectName]["modelContrast"]["lumConv"] 
    contrastConv = modelResultDict[subjectName]["modelContrast"]["contrastConv"] 
    
    df = pd.DataFrame(np.vstack([sampledTimeStamps,y_pred,lumConv,contrastConv]).T)
    df.columns = ["timeStamps", "Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]
    
    df.to_csv(f"{subjectName}_modelPrediction.csv")
####################################################################################
# making plot
# This step have to be done after pupil prediction
plotObj = interactive_plot()
# subject and movie to plot
plotObj.subjectName = subjectName
plotObj.movie = movieName
# other parameters
plotObj.useApp = useApp
plotObj.dataDir = dataDir
plotObj.filename_movie = filename_movie
plotObj.A = A
plotObj.skipNFirstFrame =skipNFirstFrame
plotObj.sampledFps = sampledFps
plotObj.video_width = video_width
plotObj.video_height = video_height
plotObj.screenBgColorR = screenBgColorR
plotObj.screenBgColorG = screenBgColorG
plotObj.screenBgColorB = screenBgColorB
plotObj.videoScreenSameRatio = videoScreenSameRatio 
plotObj.videoStretched = videoStretched
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plotObj.plot_NoEyetracking()
