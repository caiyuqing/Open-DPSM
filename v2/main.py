# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:34:10 2023

@author: 7009291
"""
# import packages
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
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

from scipy.optimize import minimize
from scipy.optimize import basinhopping
import time
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button,TextBox,CheckButtons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
########################################################################################################################################################
#####################################################Information entered by the user####################################################################
########################################################################################################################################################
# Directories
initialDir = os.getcwd()

dataDir = initialDir + '\\Example_test2' # This should be the folder saving the eyetracking data and the video data 
## eyetracking data:
###- should have four columns in the order as: time stamps, gaze position x, gaze position y, pupil size
###- time stamps should be in seconds (not miliseconds). If not, please convert it to seconds
subjectFileName = "testData.csv" # name of the subject (data file should be contained by "dataDir") [Comment out this line if no eyetracking data]
## video data
### Format can be used: .mp4,.avi,.mkv,.mwv,.mov,.flv,.webm (other format can also be used as long as it can be read by cv2)
movieName =  "01.mp4" # name of the movie (data file should be contained by "dataDir")

## If the movie and the eyetracking data are not the same, what to do?
stretchToMatch = True # True: stretch the eyelinkdata to match the movie data; False: cut whichever is longer
### maximum luminance of the screen (luminance when white is showed on screen)
maxlum = 212

## The following information is only relevant if eyetracking data is available
### What is the resolution of eyetracking data 
eyetracking_height = 1080
eyetracking_width = 1920
### What is the video (showed on the screen) resolution (respective to eyetracking resolution).
# *Note that it is not the resolution in the video file.* For example, if the resolution of the eye-tracking data is 1000x500 and the physical height and width of the video displayed is half of the physical height and width of the screen, then videoRealHeight & videoRealHeight should be 500 and 250
  
videoRealHeight = 1080
videoRealWidth = 1920
# what is the physical width of the screen? (in cm)
screen_width = 145

# what is the distance between the eye and the monitor? (in cm)
eye_to_screen = 75


# What should be the size of the regional weight map? (relative to the size of the video) 
# Default value is twice as the size of the video horizontal visual angle. If the video is very large, consider make it smaller
degVF_param = 2

## if video resolution is not the same as eyetracking resolution, what color is the screen covered with? (enter rgb value, E.g. r,b,g = 0 means black)
screenBgColorR = 0
screenBgColorG = 0
screenBgColorB = 0

# shape of map: can choose between square and circular (default: circular)
mapType = "circular"

# number of weight: (default: 44)
# if mapType is sqaure, can choose among 2 (left or right), 6 (original open-DPSM paper) and 48 (all regions separately)
# if mapType is circluar, can choose between 2 (left or right), and 44 (all regions separately)
nWeight = 44
# Do regularization or not: choose between "" and "ridge"(default: ridge)
regularizationType = 'ridge'

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
    df_eyetracking = pd.read_csv(filename_csv, index_col=None, header = None)
    # change the beginning as 0s
    df_eyetracking.iloc[:,0] = df_eyetracking.iloc[:,0]-df_eyetracking.iloc[0,0]
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


#%%############################################Visual events extraction##############################################
# NOTE: skip this part if feature extraction has already done previously
# create folder to save data
# visual angle of the movie 
videoWidthCM = videoRealWidth / (eyetracking_width/screen_width)
videoWidthDeg =math.degrees(math.atan(videoWidthCM/2/eye_to_screen))*2
# visual angle of the regional weight map 
degVF = videoWidthDeg *degVF_param

os.chdir(dataDir) 
foldername = "Visual events"
if not os.path.exists(foldername):
   os.makedirs(foldername)
os.chdir(foldername)

# name the feature extracted pickle:
if mapType == "square":
    picklename ="square_" + movieName + "_"+ subjectName + "_VF_" +colorSpace + "_" + imageSector + ".pickle"
elif mapType == "circular":
    picklename ="circular_" + movieName + "_"+ subjectName + "_VF_" +colorSpace + "_nWeight_" + str(nWeight) + ".pickle"
# feature extraction class
eeObj = event_extraction()


# load some data and parameters
eeObj.video_duration = video_duration
eeObj.video_fps = video_fps
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

eeObj.maxlum = maxlum
eeObj.useApp = useApp
eeObj.videoScreenSameRatio = videoScreenSameRatio 
eeObj.videoStretched = videoStretched    
eeObj.vidInfo = prepObj.vidInfo # extract vidInfo from preprocessing object
eeObj.mapType = mapType
eeObj.degVF = degVF
eeObj.eye_to_screen = eye_to_screen
eeObj.screen_width= screen_width
# process eyetracking data
if gazecentered: # if there is eyetracking data, do gaze-contingent visual events extraction
    eeObj.eyetracking_duration = eyetracking_duration
    eeObj.eyetracking_height = eyetracking_height
    eeObj.eyetracking_width = eyetracking_width
    eeObj.eyetracking_samplingrate = eyetracking_samplingrate
    eeObj.videoRealHeight = videoRealHeight
    eeObj.videoRealWidth = videoRealWidth
    eeObj.screenBgColorR = screenBgColorR
    eeObj.screenBgColorG = screenBgColorG
    eeObj.screenBgColorB = screenBgColorB
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
if mapType == "square":
    picklename ="square_" + movieName + "_"+ subjectName + "_VF_" +colorSpace + "_" + imageSector + ".pickle"
elif mapType == "circular":
    picklename ="circular_" + movieName + "_"+ subjectName + "_VF_" +colorSpace + "_nWeight_" + str(44) + ".pickle"#load feature data
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

modelObj.nWeight =nWeight
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
modelObj.nWeight = nWeight
modelObj.mapType = mapType
# load eyetracking data
modelObj.useEtData = True
timeStampsSec = np.array(df_eyetracking.iloc[:,0])
gazexdata = np.array(df_eyetracking.iloc[:,1])
gazeydata = np.array(df_eyetracking.iloc[:,2])
pupildata = np.array(df_eyetracking.iloc[:,3])

# downsampling the eyetracking data
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
    foldername = "csv results"
    os.chdir(dataDir) 
    if not os.path.exists(foldername):
       os.makedirs(foldername)
    os.chdir(foldername)
    params = modelResultDict[subjectName]["modelContrast"]["parameters"]
    if RF == "HL":
        paramNames =  ['r', 'rmse'] + modelResultDict[subjectName]["modelContrast"]["parametersNames"]
        params = np.insert(params,0,modelObj.r)
        params = np.insert(params,1,modelObj.rmse)
        df = pd.DataFrame(np.vstack([paramNames,params]).T)
        df.columns = ["parameterName", "value"]
        df.to_csv(f"{subjectName}_parameters.csv")    
    elif RF == "KB":
        paramNames =  ['r', 'rmse'] + modelResultDict[subjectName]["modelContrast"]["parametersNames"]
        params = np.insert(params,0,modelObj.r)
        params = np.insert(params,1,modelObj.rmse)
        df = pd.DataFrame(np.vstack([paramNames,params]).T)
        df.columns = ["parameterName", "value"]
        df.to_csv(f"{subjectName}_parameters.csv")
# save modeling data
if saveData:
    foldername = "csv results"
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
###################################
# Do regularization
if regularizationType == "ridge": # This is the only type of regularization tested
    foldername = "Modeling result"
    os.chdir(dataDir)
    os.chdir(foldername)
    modelObj.regularizationType = regularizationType
    modelObj.regularization()
    # save the modeling results (data do not need to save because they are not different from the model without regularization)
    if saveParams:
        foldername = "csv results"
        os.chdir(dataDir) 
        if not os.path.exists(foldername):
           os.makedirs(foldername)
        os.chdir(foldername)
        params = modelResultDict[subjectName]["modelRegularization"]["parameters"]
        if RF == "HL":
            paramNames =  ['r', 'rmse'] + modelResultDict[subjectName]["modelRegularization"]["parametersNames"]
            params = np.insert(params,0,modelObj.r)
            params = np.insert(params,1,modelObj.rmse)
            df = pd.DataFrame(np.vstack([paramNames,params]).T)
            df.columns = ["parameterName", "value"]
            df.to_csv(f"{subjectName}_parameters_regularization.csv")    
        elif RF == "KB":
            paramNames =  ['r', 'rmse'] + modelResultDict[subjectName]["modelRegularization"]["parametersNames"]
            params = np.insert(params,0,modelObj.r)
            params = np.insert(params,1,modelObj.rmse)
            df = pd.DataFrame(np.vstack([paramNames,params]).T)
            df.columns = ["parameterName", "value"]
            df.to_csv(f"{subjectName}_parameters_regularization.csv")


#%%##################################### interactive plot##############################################
# making plot
# This step have to be done after pupil prediction
eeObj = event_extraction()
eeObj.mapType = mapType
eeObj.degVF = degVF
eeObj.eye_to_screen =eye_to_screen
eeObj.eyetracking_width =eyetracking_width
eeObj.eyetracking_height =eyetracking_height
eeObj.screen_width =screen_width

eeObj.createMapMask()
plotObj = interactive_plot()
# subject and movie to plot
plotObj.subjectName = subjectName
plotObj.movieName = movieName
# other parameters
plotObj.useApp = useApp
plotObj.dataDir = dataDir
plotObj.filename_movie = filename_movie
plotObj.finalImgWidth = eeObj.finalImgWidth
plotObj.finalImgHeight = eeObj.finalImgHeight

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
#%%######################################Pupil prediction (no eye-tracking data)#######################################
# NOTE: run this part if eyetracking data is not available
if mapType == "square":
    picklename ="square_" + movieName + "_"+ subjectName + "_VF_" +colorSpace + "_" + imageSector + ".pickle"
elif mapType == "circular":
    picklename ="circular_" + movieName + "_"+ subjectName + "_VF_" +colorSpace + "_nWeight_" + str(nWeight) + ".pickle"
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
    params = [9.67,0.19,0.8,0.52,0.3, 1,1,1,1,1] # Those are the parameters gained from the our data

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
##########################
# Temporary for revision
fontsize = 22
lumData = modelDataDict[subjectName]["lumData"] 
contrastData = modelDataDict[subjectName]["contrastData"] 
y_pred = modelResultDict[subjectName]["modelContrast"]["predAll"] 
lumConv = modelResultDict[subjectName]["modelContrast"]["lumConv"] 
contrastConv = modelResultDict[subjectName]["modelContrast"]["contrastConv"] 
timeStamps = modelDataDict[subjectName]["timeStamps"]
# tkinter grid
# widget_list = self.all_children()
# for item in widget_list:
#     item.destroy()


frameAxe = timeStamps
   
plt.subplots(3,1, figsize = (20,12), sharex = True)
ax = plt.subplot(3,1,1)
ax.plot(frameAxe,lumData, color = "#4c004c",alpha=1, label = "luminance", linewidth = linesize)

#l_color, = ax[2].plot(frameAxe,labData, color = "green", label = "color")
ax.set_ylim([-20,20])
ax.set_xlim([4,timeStamps[-1]])
ax.set_yticks([-10,0,10])
ax.set_yticklabels([-10,0,10],fontsize=fontsize-2)

#ax.set_xlim([0,timeStamps[-1]])
ax.set_ylabel("Luminance changes",fontsize = fontsize)
#ax.legend(frameon = False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_linewidth(linesize -1)
ax.spines["bottom"].set_linewidth(linesize -1)
ax = plt.subplot(3,1,2)
ax.plot(frameAxe,contrastData,color = "green",alpha=1, label = "contrast", linewidth = linesize)
ax.set_ylim([-20,20])
ax.set_ylabel("Contrast changes",fontsize = fontsize)
#ax.legend(frameon = False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_linewidth(linesize -1)
ax.spines["bottom"].set_linewidth(linesize -1) 
ax.set_xlim([4,timeStamps[-1]])
ax.set_yticks([-10,0,10])
ax.set_yticklabels([-10,0,10],fontsize=fontsize-2)

ax = plt.subplot(3,1,3)
# plot the model (actual and prediction)
#l_actualPupil, = axes[4].plot(frameAxe, sampledpupilData, color = "grey", label = "actual pupil", linewidth = linesize)
ax.plot(frameAxe, y_pred, color = "#744700", label = "predicted pupil", linewidth = linesize)
#ax.plot(frameAxe[5:], lumConv[5:], color = "#4c004c", label = "predicted pupil luminance", linewidth = linesize)
#ax.plot(frameAxe[5:], contrastConv[5:], color = "green", label = "predicted pupil contrast", linewidth = linesize)
   
ax.set_xlim([4,timeStamps[-1]])
ax.legend().set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_linewidth(linesize -1)
ax.spines["bottom"].set_linewidth(linesize -1)
ax.set_ylabel("Pupil size $(z)$",fontsize = fontsize)
#
ax.set_ylim([-4,1])
ax.set_xticks([5,10,15,20, 25])
ax.set_yticks([-2,0])
ax.set_yticklabels([-2,0],fontsize=fontsize-2)

ax.set_xticklabels([5,10,15,20, 25],fontsize=fontsize-2)
ax.set_xlabel("Time (s)",fontsize = fontsize)
#%%
from matplotlib.lines import Line2D
linesize = 3
fontsize = 15
changes = np.where(contrastData !=0)[0]
change_appear = []
for i in range(changes.shape[0]-1):
    if i > 0:
        if changes[i+1]- changes[i]==1 and changes[i]- changes[i-1]>1:
            change_appear.append(changes[i])
y_pred_cut = np.array([y_pred[change_appear[0]:change_appear[2]]])
for j in range(len(change_appear)):
    if j%2 ==0 and j >0:
        if j<6:
            y_pred_cut = np.vstack((y_pred_cut, y_pred[change_appear[j]:change_appear[j+2]]))
        else:
            y_pred_cut = np.vstack((y_pred_cut, y_pred[change_appear[j]:]))

plt.subplots(1,1, figsize = (6,6), sharex = True)
ax = plt.subplot(1,1,1)
for i in range(4):
    y_pred_cut[i,:] = y_pred_cut[i,:]- y_pred_cut[i,0]
t = np.linspace(0,5,150)
ax.plot(t,y_pred_cut[0,:], color = "blue",alpha=1, label = "luminance", linewidth = linesize)
ax.plot(t,y_pred_cut[1,:], color = "blue",alpha=0.5, label = "luminance", linewidth = linesize)
ax.plot(t,y_pred_cut[2,:], color = "red",alpha=0.5, label = "luminance", linewidth = linesize)
ax.plot(t,y_pred_cut[3,:], color = "red",alpha=1, label = "luminance", linewidth = linesize)
legend_elements = [Line2D([0], [0], color='blue',alpha =1, lw=linesize, label='black', linestyle = "solid"),
                   Line2D([0], [0], color='blue',alpha =0.3, lw=linesize, label='dark grey', linestyle = "solid"),
                   Line2D([0], [0], color='red',alpha =0.5, lw=linesize, label='light grey', linestyle = "solid"),
                   Line2D([0], [0], color='red',alpha =1, lw=linesize, label='white', linestyle = "solid"),]
ax.legend(handles=legend_elements,loc = "center right",fontsize = fontsize, frameon = False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_linewidth(linesize -1)
ax.spines["bottom"].set_linewidth(linesize -1)
ax.set_ylabel("Pupil size $(z)$",fontsize = fontsize)
ax.set_xlabel("Time (s)",fontsize = fontsize)
ax.set_yticks([-3,-2,-1,0])
#y_pred_cut = np.array([y_pred[change_appear[2]:change_appear[3]]])
#%%
#########################interactive plot###########################################################
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
# plotObj.videoScreenSameRatio = videoScreenSameRatio 
# plotObj.videoStretched = videoStretched
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plotObj.plot_NoEyetracking()
