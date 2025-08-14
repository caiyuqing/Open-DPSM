# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:34:10 2023

@author: 7009291
"""
# import packages
import os
initialDir = os.path.dirname(os.path.realpath(__file__))
os.chdir(initialDir)
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
## set up Directories
# folder should be organzized as:
# Example (dataset name)
# - Input
# -- Eyetracking (if there is eyetracking data)
# -- Movies
# - Output
# Eyetracking data requirement:
## CSV file, should have four columns in the order as: time stamps (in seconds), gaze position x, gaze position y, pupil size
# Movie data  requiarement:
# Format can be used: .mp4,.avi,.mkv,.mwv,.mov,.flv,.webm (other format can also be used as long as it can be read by cv2)

dataDir = initialDir + '\\Example' # change it to the head directiory of the data folder
inputDir = dataDir + '\\Input' # change it to the head directiory of the data folder
outputDir = dataDir + "\\Output"

movieDir = inputDir + '\\Movies'
eyetrackingDir = inputDir + '\\Eyetracking' # comment out this line if no eyetracking data mode is used
# change to initial directory
os.chdir(initialDir)

#################Set up information for the eyetracking data and movie######################
### maximum luminance of the screen (luminance when white is showed on screen)
maxlum = 212

### What is the resolution of eyetracking data (also is the screen resolution)
# Note: enter screen width and height in pixels when there is no eyetracking data
eyetracking_height = 1080
eyetracking_width = 1920
eyetracking_samplingrate = 500
### What is the video (showed on the screen) resolution (respective to eyetracking resolution).
# *Note that it is not the resolution in the video file.* For example, if the resolution of the eye-tracking data is 1000x500 and the physical height and width of the video displayed is half of the physical height and width of the screen, then videoRealHeight & videoRealHeight should be 500 and 250
  
videoRealHeight = 1080
videoRealWidth = 1920
# what is the physical width of the screen? (in cm)
screen_width = 145

# what is the distance between the eye and the monitor? (in cm)
# Note: use the default 75 when there is no eyetracking data
eye_to_screen = 75

# What should be the size of the regional weight map? (relative to the size of the video) 
# Default value is the same size of the video horizontal visual angle. If the video is very large, consider make it smaller
degVF_param = 1

## if video resolution is not the same as eyetracking resolution, what color is the screen covered with? (enter rgb value, E.g. r,b,g = 0 means black)
screenBgColorR = 0
screenBgColorG = 0
screenBgColorB = 0

# select parameters for event extraction and modeling
# event extraction mode (default: gaze-centered)
gazecentered = True
# map type: can choose between square and circular (default: circular)
mapType = "circular"
# But if not gazecentered, map type can only be square 
if not gazecentered:
    mapType = "square"
# number of weight: (default: 44)
# if mapType is sqaure, can choose among 2 (left or right), 6 (original open-DPSM paper) and 48 (all regions separately)
# if mapType is circluar, can choose between 2 (left or right), and 44 (all regions separately)
nWeight = 44
# load other parameters that should not be changed
exec(open("settings.py").read())
### Do you want to save:
# - model evaluation & paramters
saveParams = True
# - data used for modeling
saveData = True
########################################################################################################################################################
###########################################################End of information entering.#################################################################
########################################################################################################################################################

#%%#########################################Preprocessing and visual even extraction####################################################################
# This is the indicator that the app is not used.
useApp = False
movieFiles = [file for file in os.listdir(movieDir)]
movieListAll = [file.split('.')[0] for file in movieFiles]
print(f"Selected parameters: \n- gazecentered: {gazecentered}\n- map type: {mapType}\n- number of weight: {nWeight}")

# read video data and check information
prepObj = preprocessing()
if "eyetrackingDir" not in globals():
    
    # rename subjectName as NoEyetracking data if there is no eyetracking data
    subjects = ["NoEyetrackingData"] # name the events file as NoEyetrackingData
    # if there is no eyetracking data, only option is screen-cenetered event extraction, square map and 48 weights
    gazecentered = False # not use gaze data
    mapType = "square"
    nWeight = 48
    print("No eyetracking data. Model prediction will not be preformed. Predicted pupil trace will be generated with the default response function and weights.")
else:
    if gazecentered:
        print("Eyetracking folder found.")
        subjects = os.listdir(eyetrackingDir)
    else:
        # rename subjectName as "sc" (screen-centered) data if there is no eyetracking data
        # all the subjects will have the same pickle files
        subjects = ['sc'] 
# iteratively extract events for all the subjects and all the movies
for subjectName in subjects:
    if gazecentered:
        # If gazecentered, check the files from each subject folder
        subjectDir = eyetrackingDir + f"\\{subjectName}"
        ### one folder for one participant, containing all eyetracking data 
        csvFiles = [file for file in os.listdir(subjectDir) if file.endswith('.csv')]
    
        #check whether one eyetracking data is paired with one movie file
        noMovieFiles = [file for file in csvFiles if file.replace(".csv",'') not in movieListAll]
        if len(noMovieFiles)>0:
            print(f"Warning: {len(noMovieFiles)} movies not found")
        movieList = [file.replace('.csv','') for file in csvFiles]
    else:
        # If not gazecentered, check the files from Movies folder (eyetracking data does not matter)
        movieList = [file.split(".")[0] for file in os.listdir(movieDir)]
    for movie in movieList:
        # check video information
        movieName = [file for file in os.listdir(movieDir) if file.startswith(movie)][0]
        filename_movie = movieDir +"\\" + movieName 
        prepObj.videoFileName = filename_movie
        #prepObj.preprocessingVideo()
        prepObj.loadVideo(filename_movie)
        prepObj.getVideoInfo()
        # extract information from video file
        video_nFrame = prepObj.vidInfo['nFrames']
        video_height = prepObj.vidInfo['height']
        video_width = prepObj.vidInfo['width']
        video_ratio = video_height / video_width 
        video_duration = prepObj.vidInfo['duration']
        video_fps = prepObj.vidInfo['fps']
        movieName = movieName.split(".")[0]
        # read eyetracking data if gaze centered
        if gazecentered:
            filename_csv = subjectDir + "\\" + movieName +".csv"
            # read eyetracking data and check information
            df_eyetracking = pd.read_csv(filename_csv, index_col=None, header = None,sep = ",") # Change it to sep = ',' if encounter error
            # change the beginning as 0s
            eyetracking_duration = df_eyetracking.iloc[-1,0]
            eyetracking_nSample = df_eyetracking.shape[0]
            df_eyetracking.iloc[:,0] = df_eyetracking.iloc[:,0]-df_eyetracking.iloc[0,0]
            if eyetracking_samplingrate != int(1/(eyetracking_duration/eyetracking_nSample)):
                df_eyetracking.iloc[:,0] = df_eyetracking.iloc[:,0]/1000
                eyetracking_duration = df_eyetracking.iloc[-1,0]
    
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
        #############################################Visual events extraction##############################################
        # calculate visual angle of the movie displayed 
        videoWidthCM = videoRealWidth / (eyetracking_width/screen_width)
        videoWidthDeg =math.degrees(math.atan(videoWidthCM/2/eye_to_screen))*2
        # visual angle of the regional weight map 
        degVF = videoWidthDeg *degVF_param
        # create folder to save data
        os.chdir(dataDir) 
        foldername = "Output"
        if not os.path.exists(foldername):
           os.makedirs(foldername)
        os.chdir(foldername)
        foldername = "Visual events"
        if not os.path.exists(foldername):
           os.makedirs(foldername)
        os.chdir(foldername)
        # name the feature extracted pickle:
        if mapType == "square":
            picklename ="square_" + movieName + "_"+ subjectName + "_VF_" +colorSpace + "_nWeight_" + str(nWeight)  + ".pickle"
        elif mapType == "circular":
            picklename ="circular_" + movieName + "_"+ subjectName + "_VF_" +colorSpace + "_nWeight_" + str(nWeight) + ".pickle"
        if picklename in os.listdir():
            # skip event extraction if it has already done previously
            print(f"Subject {subjectName} Movie {movieName} already extracted")
        else:
            # start of event extraction for one movie in one subject
            print(f"====Extracting for subject {subjectName} movie {movieName}====")
            print(f"Video number of frame: {video_nFrame}")
            print(f"Video height x width: {video_height}x{video_width}; aspect ratio (width:height): {1/video_ratio}")
            print(f"Video duration: {video_duration}")
            print(f"Video frame rate: {video_fps}")
            if gazecentered:
                print(f"Eyetracking data duration: {eyetracking_duration} seconds")
                print(f"Eyetracking data sampling rate: {eyetracking_samplingrate} Hz")
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
               
            eeObj.vidInfo = prepObj.vidInfo # extract vidInfo from preprocessing object
            eeObj.mapType = mapType
            eeObj.degVF = degVF
            eeObj.eye_to_screen = eye_to_screen
            eeObj.screen_width= screen_width
            eeObj.nWeight = nWeight
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
                eeObj.videoScreenSameRatio = videoScreenSameRatio 
                eeObj.videoStretched = videoStretched 
                timeStampsSec = np.array(df_eyetracking.iloc[:,0])
                gazexdata = np.array(df_eyetracking.iloc[:,1])
                gazeydata = np.array(df_eyetracking.iloc[:,2])
                pupildata = np.array(df_eyetracking.iloc[:,3])
                # resample the eytracking data to match the video sampling rate
                eeObj.sampledTimeStamps_featureExtraction =eeObj.prepare_sampleData(timeStampsSec, video_nFrame)
                eeObj.sampledgazexData_featureExtraction = eeObj.prepare_sampleData(gazexdata, video_nFrame)
                eeObj.sampledgazeyData_featureExtraction = eeObj.prepare_sampleData(gazeydata, video_nFrame)
                eeObj.sampledpupilData_featureExtraction = eeObj.prepare_sampleData(pupildata, video_nFrame)
            # Event extraction function: 
            # This can take a while. The extracted features will be saved in folder "Visual events"
            eeObj.event_extraction()

#%%############################################Pupil modeling##############################################
# NOTE: this part can only be performed if eyelink data exists
os.chdir(outputDir)

# create new folder for modeling results
foldername = "Modeling result"
if not os.path.exists(foldername):
   os.makedirs(foldername)
os.chdir(foldername)
#Create dictionaries to save results
if os.path.exists(f"modelDataDict_nWeight{nWeight}.pickle"):
    with open(f"modelDataDict_nWeight{nWeight}.pickle", "rb") as handle:
        modelDataDict = pickle.load(handle)
        handle.close() 
else:
    modelDataDict = {}
        
if os.path.exists(f"modelResultDict_nWeight{nWeight}.pickle"):
    with open(f"modelResultDict_nWeight{nWeight}.pickle", "rb") as handle:
        modelResultDict = pickle.load(handle)
        handle.close() 
    #subjectProcessed = list(modelResultDict.keys())
    #subjects = [subject for subject in subjects if subject not in subjectProcessed]
else:
    modelResultDict = {}
# To-do: sameWeightFeature not work
# Modeling iteratively for all the subjects (use all the movies together under each subject)
subjects = os.listdir(eyetrackingDir)
for subjectName in subjects:
    # if subject is already in the dictionary, skip
    if subjectName in modelResultDict.keys() and 'modelRegularization' in modelResultDict[subjectName].keys():
        print(f"Modeling already done for subject {subjectName}")
    else:
        # Start of modeling
        subjectDir = eyetrackingDir + f"\\{subjectName}"
        csvFiles = [file for file in os.listdir(subjectDir) if file.endswith('.csv')]
        movieList = [file.replace('.csv','') for file in csvFiles]
    
        # pupil prediction class
        modelObj = pupil_prediction()
        modelObj.eyetrackingDir = eyetrackingDir
        modelObj.subjectDir = subjectDir
        modelObj.outputDir = outputDir
        modelObj.feature_pickle_directory = f"{outputDir}\\visual events"

        modelObj.nWeight =nWeight
        modelObj.subject = subjectName
        modelObj.sameWeightFeature =sameWeightFeature
        modelObj.RF =RF 
        modelObj.skipNFirstFrame =skipNFirstFrame 
        modelObj.useBH = useBH
        modelObj.niter = niter
        
        modelObj.useApp = useApp
        modelObj.stretchToMatch = stretchToMatch
        modelObj.nFramesSeqImageDiff = nFramesSeqImageDiff
        modelObj.mapType = mapType
        modelObj.useEtData = True
        modelObj.modelDataDict = modelDataDict
        modelObj.modelResultDict = modelResultDict
        modelObj.movieList = movieList
        modelObj.gazecentered = gazecentered
        modelObj.pupil_zscore = pupil_zscore
        modelObj.connect_data(movieList)
        # pupil prediction function start:
        # This can also take a while
        modelObj.pupil_prediction()
        # extract the results from modelObj
        modelResultDict = modelObj.modelResultDict
        
        ##################################
        # save model results
        if saveParams:
            foldername = "csv results"
            os.chdir(outputDir) 
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
                df.to_csv(f"{subjectName}_parameters_nWeight{nWeight}.csv")    
            elif RF == "KB":
                paramNames =  ['r', 'rmse'] + modelResultDict[subjectName]["modelContrast"]["parametersNames"]
                params = np.insert(params,0,modelObj.r)
                params = np.insert(params,1,modelObj.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{subjectName}_parameters_nWeight{nWeight}.csv")
        # save modeling data
        if saveData:
            foldername = "csv results"
            os.chdir(outputDir) 
            if not os.path.exists(foldername):
               os.makedirs(foldername)
            os.chdir(foldername)
            y_pred = modelResultDict[subjectName]["modelContrast"]["predAll"] 
            lumConv = modelResultDict[subjectName]["modelContrast"]["lumConvAll"] 
            contrastConv = modelResultDict[subjectName]["modelContrast"]["contrastConvAll"] 
            sampledpupilData_z = modelObj.sampledPupilDataAll 
            df = pd.DataFrame(np.vstack([sampledpupilData_z, y_pred,lumConv,contrastConv]).T)
            df.columns = [ "Actual pupil (z)", "Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]
            
            df.to_csv(f"{subjectName}_modelPrediction_nWeight{nWeight}.csv")
        ###################################
        # Do regularization (This step cannot be skipped)
        if regularizationType == "ridge": # This is the only type of regularization tested
            os.chdir(outputDir)
            os.chdir("Modeling result")
            modelObj.regularizationType = regularizationType
            modelObj.params = modelResultDict[subjectName]["modelContrast"]["parameters"]
            modelObj.regularization()
            modelResultDict = modelObj.modelResultDict
            # save the modeling results (data do not need to save because they are not different from the model without regularization)
            if saveParams:
                foldername = "csv results"
                os.chdir(outputDir) 
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
                    df.to_csv(f"{subjectName}_parameters_regularization_nWeight{nWeight}.csv")    
                elif RF == "KB":
                    paramNames =  ['r', 'rmse'] + modelResultDict[subjectName]["modelRegularization"]["parametersNames"]
                    params = np.insert(params,0,modelObj.r)
                    params = np.insert(params,1,modelObj.rmse)
                    df = pd.DataFrame(np.vstack([paramNames,params]).T)
                    df.columns = ["parameterName", "value"]
                    df.to_csv(f"{subjectName}_parameters_regularization_nWeight{nWeight}.csv")
            if saveData:
                foldername = "csv results"
                os.chdir(outputDir) 
                if not os.path.exists(foldername):
                   os.makedirs(foldername)
                os.chdir(foldername)
                y_pred = modelResultDict[subjectName]["modelRegularization"]["predAll"] 
                lumConv = modelResultDict[subjectName]["modelRegularization"]["lumConvAll"] 
                contrastConv = modelResultDict[subjectName]["modelRegularization"]["contrastConvAll"] 
                sampledpupilData_z = modelObj.sampledPupilDataAll 
                df = pd.DataFrame(np.vstack([sampledpupilData_z, y_pred,lumConv,contrastConv]).T)
                df.columns = [ "Actual pupil (z)", "Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]
                
                df.to_csv(f"{subjectName}_modelPrediction_regularization_nWeight{nWeight}.csv")

#%%##################################### interactive plot##############################################
# making plot (for one movie from one subject)
## select a subject to plot
subjectName = 'p21'
movieName = "01.mp4"
# This step have to be done after pupil prediction
eeObj = event_extraction()
eeObj.mapType = mapType
eeObj.degVF = degVF
eeObj.eye_to_screen =eye_to_screen
eeObj.eyetracking_width =eyetracking_width
eeObj.eyetracking_height =eyetracking_height
eeObj.screen_width =screen_width
eeObj.nWeight = nWeight
eeObj.createMapMask()
plotObj = interactive_plot()

plotObj.subject = subjectName
plotObj.outputDir = outputDir
plotObj.movieName = movieName.split(".")[0]
# other parameters
plotObj.useApp = useApp
plotObj.dataDir = dataDir
plotObj.movieDir = movieDir
plotObj.finalImgWidth = eeObj.finalImgWidth
plotObj.finalImgHeight = eeObj.finalImgHeight

plotObj.skipNFirstFrame =skipNFirstFrame
plotObj.eyetracking_height = eyetracking_height
plotObj.eyetracking_width = eyetracking_width
plotObj.videoRealHeight = videoRealHeight
plotObj.videoRealWidth = videoRealWidth
plotObj.screenBgColorR = screenBgColorR
plotObj.screenBgColorG = screenBgColorG
plotObj.screenBgColorB = screenBgColorB
plotObj.videoScreenSameRatio = videoScreenSameRatio 
plotObj.videoStretched = videoStretched
plotObj.nWeight = nWeight
plotObj.gazecentered = gazecentered
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plotObj.plot()
#%%######################################Pupil prediction (no eye-tracking data)#######################################
# NOTE: run this part if eyetracking data is not available    
# new folder for modeling results
foldername = "Modeling result"
os.chdir(outputDir) 
if not os.path.exists(foldername):
   os.makedirs(foldername)
os.chdir(foldername)
#Create dictionaries to save results
if os.path.exists(f"modelDataDict_noETdata.pickle"):
    with open(f"modelDataDict_noETdata.pickle", "rb") as handle:
        modelDataDict = pickle.load(handle)
        handle.close() 
else:
    modelDataDict = {}
        
if os.path.exists(f"modelResultDict_noETdata.pickle"):
    with open(f"modelResultDict_noETdata.pickle", "rb") as handle:
        modelResultDict = pickle.load(handle)
        handle.close() 
    #subjectProcessed = list(modelResultDict.keys())
    #subjects = [subject for subject in subjects if subject not in subjectProcessed]
else:
    modelResultDict = {}
# To-do: sameWeightFeature may not work
# if subject is already in the dictionary, skip modeling
subjectName= "NoEyetrackingData"
if subjectName in list(modelResultDict.keys()):
    print("Modeling already done.")
else:
    # pupil prediction class
    modelObj = pupil_prediction()
    modelObj.useEtData = False
    if RF == 'HL':
        params = [9.67,0.19,0.8,0.52,0.3] + [1] * nWeight # Those are the parameters gained from the our data
    
    else:
        params = [0.12,4.59,0.14,6.78,0.28]+ [1] * nWeight
    #modelObj.sampledTimeStamps = timeStamps
    #modelObj.sampledFps = 1/(modelObj.sampledTimeStamps [-1]/(len(modelObj.sampledTimeStamps)))
    movieList = [file.split(".")[0] for file in os.listdir(movieDir)]

    modelObj.numRemoveMovFrame = 0
    modelObj.outputDir = outputDir
    modelObj.modelDataDict = modelDataDict
    modelObj.modelResultDict = modelResultDict
    modelObj.sameWeightFeature = sameWeightFeature
    modelObj.RF = RF
    modelObj.subject = subjectName
    modelObj.nWeight= nWeight
    modelObj.movieList = movieList
    modelObj.feature_pickle_directory =  f"{outputDir}\\visual events"
    modelObj.gazecentered = gazecentered
    modelObj.mapType = mapType
    modelObj.pupil_zscore = pupil_zscore
    
    modelObj.connect_data(movieList)
    modelObj.pupil_predictionNoEyetracking(params)
    ####################################
    # save model results
    foldername = "csv results"
    os.chdir(outputDir) 
    if not os.path.exists(foldername):
       os.makedirs(foldername)
    os.chdir(foldername)
    # save data used for pupil prediction
    if saveData:
        y_pred = modelResultDict[subjectName]["modelContrast"]["predAll"] 
        lumConv = modelResultDict[subjectName]["modelContrast"]["lumConvAll"] 
        contrastConv = modelResultDict[subjectName]["modelContrast"]["contrastConvAll"] 
        
        df = pd.DataFrame(np.vstack([y_pred,lumConv,contrastConv]).T)
        df.columns = ["Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]
        
        df.to_csv(f"{subjectName}_modelPrediction.csv")

#%%
#########################interactive plot###########################################################
# making plot 
# select one movie to plot
movieName = "01.mp4" 
# This step have to be done after pupil prediction
plotObj = interactive_plot()
# subject and movie to plot
plotObj.subjectName = subjectName
plotObj.movieName = movieName.split('.')[0]
# other parameters
plotObj.useApp = useApp
plotObj.dataDir = dataDir
plotObj.A = 1
plotObj.skipNFirstFrame =skipNFirstFrame

plotObj.outputDir = outputDir
plotObj.movieDir = movieDir
# plotObj.videoScreenSameRatio = videoScreenSameRatio 
# plotObj.videoStretched = videoStretched
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plotObj.plot_NoEyetracking()
