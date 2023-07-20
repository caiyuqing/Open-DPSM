# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:46:04 2023

@author: 7009291
"""

"""
preprocessing.py
====================================
* This file is for the preprocessing of the eyelink data
* file 'LICENSE.txt', which is part of this source code package.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib import pyplot as plt
import pickle
from moviepy.editor import *
import cv2
class preprocessing:
    def __init__(self):
        self.removeEyeblink = True
        self.plotBlinkRemoval = False
        #self.initialDir = 'D:/Users/7009291/Desktop/Movie pupil perimetry/codes for both data/Toolbox' # This should be the directory of the current script
        #self.dataDir = self.initialDir + '/Example' # This should be the folder for the eyetracking data and the video data 
        #self.videoFileName =  self.dataDir + "/VideoExample.mp4"
        #self.eyelinkDataDir = self.dataDir + "/EyelinkDataExample.pickle"
        self.eyelinkSampleRate = 500
        self.nFramesSeqImageDiff = 2
        self.timeStamps = np.zeros((0))  # append video timestamps
        self.frameCount = 0  # count number of image frames in vode
        self.altTimeStamps = False
        #self.eyelink_width = 1023.0
        #self.eyelink_height = 767.0
    def preprocessingEyelink(self):
        self.preprocessingVideo()
        numVideoFrames = self.vidInfo['frameN_end'] 
        self.load_eyelinkData()
        if self.removeEyeblink:
            self.removeEyelinkBlinks(self.eyelinkData)
        else:
            timeStamps = self.eyelinkData["TIMESTAMPS"]
            pupildata =self. eyelinkData["PUPIL-SIZE"]
            gazex = self.eyelinkData["X-GAZE"]
            gazey = self.eyelinkData["Y-GAZE"]
            timeStampsSec = (np.array(timeStamps) - timeStamps[0])/1000
            self.timeStamps_br = timeStamps
            self.timeStampsSec_br = timeStampsSec
            self.pupildata_br = pupildata
            self.gazexdata_br = gazex
            self.gazeydata_br = gazey
        lengthEyelinkData = self.timeStampsSec_br[-1]
        self.fps_end = round(1/(lengthEyelinkData/self.vidInfo['frameN_end']))
        self.vidInfo['fps_end'] = self.fps_end
        self.synchronize()
        self.sampledgazexData= self.prepare_sampleData_gaze(self.gazexdata_syn, numVideoFrames)
        self.sampledgazeyData= self.prepare_sampleData_gaze(self.gazeydata_syn, numVideoFrames)
        self.sampledpupilData= self.prepare_sampleData_gaze(self.pupildata_syn, numVideoFrames)

        self.sampledTimeStamps= self.prepare_sampleData_gaze(self.timeStamps_syn, numVideoFrames)
        self.sampledgazexData_videoCoor=  self.sampledgazexData/self.eyelink_width * (self.video_x-1)
        self.sampledgazeyData_videoCoor=  self.sampledgazeyData/self.eyelink_height * (self.video_y-1)
    def preprocessingVideo(self):
        self.loadVideo(self.videoFileName)  # load a video into a capture object
        self.getVideoInfo()  # store video info like number of frames 
        # get video coordinate
        self.video_x = self.vidInfo['width']
        self.video_y = self.vidInfo['height']
        # determine the real number of the frames in the movie
        frameCount = 0
        while True:  # Capture frame-by-frame
            self.readFrame()  # reads a frame from video
            if self.ret:
                frameCount = frameCount+1
            else:
                break        
        # get the information of the video from moviepy (seems to be more accurate)
        moviepy_clip = VideoFileClip(self.videoFileName)
        fps = moviepy_clip.fps
        duration = moviepy_clip.duration
        frames = int(moviepy_clip.fps * moviepy_clip.duration)
        if frameCount != frames:
            print("MoviePy and CV2 generate different result, use frame number from CV2")
            
        self.vidInfo['frameN_end'] = frameCount
        self.vidInfo['fps_end'] = fps
        self.vidInfo['duration_end'] = duration

    def load_eyelinkData(self):
        with open(self.eyelinkDataDir, "rb") as handle:
            self.eyelinkData = pickle.load(handle)
            handle.close() 
        #self.eyelinkData = self.eyelinkData[SAMPLES]
            
    def removeEyelinkBlinks(self,
        eyelinkData, 
        blinkDetectionThreshold=[4, 2],# std [below above] mean
        dilutionSize=[20, 40], # number of samples that need to be removed [before, after] a blink period.
        consecutiveRemovalPeriod=20,
        plotBlinkRemoveResult = False
    ):
        """
        eyelinkData: A dictionary of four keys: TIMESTAMPS, PUPIL-SIZE, X-GAZE, y-GAZE
        dilutionSize is list with two integers indicating number of samples that need to be
        removed [before, after] a blink period.
        """
        #eyelinkData = self.eyelinkData["SAMPLES"]
        timeStamps = eyelinkData["TIMESTAMPS"]
        pupildata = eyelinkData["PUPIL-SIZE"]
        gazex = eyelinkData["X-GAZE"]
        gazey = eyelinkData["y-GAZE"]
        timeStamps = np.array(timeStamps, dtype="float32")
        pupildata = np.array(pupildata, dtype="float32")
        gazex = np.array(gazex, dtype="float32")
        gazey = np.array(gazey, dtype="float32")

        # filter out blink periods (blink periods contain value 0; set at 2 just in case)
        pdata = pupildata.copy()
        gazexdata = gazex.copy()
        gazeydata = gazey.copy()
        pdata[pdata < 2] = np.nan
        

        # filter out blinks based on speed changes
        pdiff = np.diff(pdata)  # difference between consecutive pupil sizes

        # remove blink periods
        # for selectIdx in range(len(timeTraceData)):
        # create blinkspeed threshold 4SD below the mean
        blinkSpeedThreshold = np.nanmean(pdiff) - (blinkDetectionThreshold[0] * np.nanstd(pdiff))

        # create blinkspeed threshold 2SD above the mean
        blinkSpeedThreshold2 = np.nanmean(pdiff) + (blinkDetectionThreshold[1] * np.nanstd(pdiff))

        # blink window containing minimum and maximum value
        blinkWindow = [-dilutionSize[0], dilutionSize[1]]
        blinkWindow2 = [-dilutionSize[1], dilutionSize[0]]

        blinkIdx = np.where(
            pdiff < blinkSpeedThreshold
        )  # find where the pdiff is smaller than the lower blinkspeed threshold
        blinkIdx = blinkIdx[0]
        blinkIdx = blinkIdx[np.where(np.diff(blinkIdx) > consecutiveRemovalPeriod)[0]]

        blinkIdx2 = np.where(
            pdiff > blinkSpeedThreshold2
        )  # find where the pdiff is larger than the upper blinkspeed threshold
        blinkIdx2 = blinkIdx2[0]
        blinkIdx2 = blinkIdx2[np.where(np.diff(blinkIdx2) > consecutiveRemovalPeriod)[0]]

        # remove blink segments
        for bl in blinkIdx:
            pdata[np.arange(bl + blinkWindow[0], bl + blinkWindow[1])] = np.nan
            pdiff[np.arange(bl + blinkWindow[0], bl + blinkWindow[1])] = np.nan

        for bl in blinkIdx2:
            pdata[np.arange(bl + blinkWindow2[0], bl + blinkWindow2[1])] = np.nan
            pdiff[np.arange(bl + blinkWindow2[0], bl + blinkWindow2[1])] = np.nan


        # interpolate blink periods
        # find the first element in the array that isn't NaN
        pdataFilt = pdata.copy()
        pdataFilt = np.where(np.isfinite(pdataFilt))[0][0]

        # find the last element in the array that isn't NaN
        pdataFilt2 = pdata.copy()
        pdataFilt2 = np.where(np.isfinite(pdataFilt2))[0][-1]

        missDataIdx = np.where(~np.isfinite(pdata))[0]
        corrDataIdx = np.where(np.isfinite(pdata))[0]
        ## PCHIP interpolation
        pdata_beforeInterpo = pdata.copy()
        # remove gazex and gazey data if pdata were identified as NaN
        gazexdata[~np.isfinite(pdata)] = np.nan
        gazeydata[~np.isfinite(pdata)] = np.nan
        gazexdata_beforeInterpo = gazexdata.copy()
        gazeydata_beforeInterpo = gazeydata.copy()
        #interpolate Pupil and Gaze data
        pdata[missDataIdx] = PchipInterpolator(timeStamps[corrDataIdx], pdata[corrDataIdx])(timeStamps[missDataIdx],extrapolate=False)
        gazexdata[missDataIdx] = PchipInterpolator(timeStamps[corrDataIdx], gazexdata[corrDataIdx])(timeStamps[missDataIdx],extrapolate=False)
        gazeydata[missDataIdx] = PchipInterpolator(timeStamps[corrDataIdx], gazeydata[corrDataIdx])(timeStamps[missDataIdx],extrapolate=False)
        timeStampsSec = (np.array(timeStamps) - timeStamps[0])/1000
        self.timeStamps_br = timeStamps
        self.timeStampsSec_br = timeStampsSec

        self.pupildata_br = pdata
        self.gazexdata_br = gazexdata
        self.gazeydata_br = gazeydata
        self.pupildata_beforeInterpo = pdata_beforeInterpo
        self.gazexdata_beforeInterpo = gazexdata_beforeInterpo
        self.gazeydata_beforeInterpo = gazeydata_beforeInterpo

        #%% plot the blink removal result
        if plotBlinkRemoveResult:
            plt.subplots(5,1,figsize = (25,15),sharex = True)
            plt.tight_layout()
            plt.subplot(5,1,1)
            plt.plot(timeStampsSec,pupildata)
            plt.ylim((0,max(pupildata)))
            plt.title("Raw pupil data")
            plt.subplot(5,1,2)
            plt.plot(timeStampsSec,pdata_beforeInterpo)
            plt.ylim((0,max(pupildata)))
            allNa = np.where(~np.isfinite(pdata_beforeInterpo))[0]
            blink_start = []
            blink_end = []
            for i in range(len(allNa)):
                if i == 0:
                    blink_start.append(allNa[i])
                elif allNa[i]-allNa[i-1] >1:
                    blink_start.append(allNa[i])
                if i != len(allNa)-1 and allNa[i+1]-allNa[i] >1:
                    blink_end.append(allNa[i])
                elif i == len(allNa)-1:
                    blink_end.append(allNa[i])
                
            for i in range(len(blink_start)):
                plt.axvspan(timeStampsSec[blink_start[i]], timeStampsSec[blink_end[i]], alpha=0.1, color='red')
    
            plt.title("Blink removed & before interplotation (pupil)")
            
            plt.subplot(5,1,3)
            plt.plot(timeStampsSec,pdata)
            plt.ylim((0,max(pupildata)))
            #plt.xlim((0, max(timestamps_arr)))
    
            plt.title("After interplotation (pupil)")
            plt.subplot(5,1,4)
            plt.plot(timeStampsSec,gazexdata_beforeInterpo)
            plt.plot(timeStampsSec,gazeydata_beforeInterpo)
            plt.ylim((0,max(gazexdata_beforeInterpo)))
    
            plt.legend(["gazex", "gazey"])
            allNa = np.where(~np.isfinite(gazexdata_beforeInterpo))[0]
            blink_start = []
            blink_end = []
            for i in range(len(allNa)):
                if i == 0:
                    blink_start.append(allNa[i])
                elif allNa[i]-allNa[i-1] >1:
                    blink_start.append(allNa[i])
                if i != len(allNa)-1 and allNa[i+1]-allNa[i] >1:
                    blink_end.append(allNa[i])
                elif i == len(allNa)-1:
                    blink_end.append(allNa[i])
                
            for i in range(len(blink_start)):
                plt.axvspan(timeStampsSec[blink_start[i]], timeStampsSec[blink_end[i]], alpha=0.1, color='red')
            plt.title("Blink removed & before interplotation (gazex & gazey)")
            plt.subplot(5,1,5)
            plt.plot(timeStampsSec,gazexdata)
            plt.plot(timeStampsSec,gazeydata)
            plt.ylim((0,max(gazexdata)))
    
            plt.legend(["gazex", "gazey"])
            plt.title("After interplotation (gazex & gazey)")
            plt.xlabel("TimeStamps")
            plt.tight_layout()
        

    def synchronize(self):
        numRemoveEnd = int(round((self.nFramesSeqImageDiff  / self.fps_end) / (1/self.eyelinkSampleRate)))*(-1)
        # remove the the last data of eyelink according to nFramesSeqImageDiff
        self.timeStamps_syn = self.timeStampsSec_br[:numRemoveEnd]
        self.pupildata_syn=self.pupildata_br[:numRemoveEnd]
        self.gazexdata_syn=self.gazexdata_br[:numRemoveEnd]
        self.gazeydata_syn=self.gazeydata_br[:numRemoveEnd]
    def prepare_sampleData_gaze(self, dataBeforeSample, numVideoFrames):
        sampleTimes = np.linspace(0, len(dataBeforeSample) - 1, numVideoFrames).astype("int")
        sampledData = dataBeforeSample[sampleTimes]
        return sampledData
    ######################From video_processing########################
    def loadVideo(self, videoFileName="example.avi"):
        """

        Creates video object self.cap to read frames out of video file

        Parameters
        ----------
        videoFileName: str with filename with extension (e.g., 'example.avi')


        Return
        ----------
        None; Creates self.cap and self.videoFileName for later processing

        """
        self.cap = cv2.VideoCapture(videoFileName)
        self.videoFileName = videoFileName

        # if self.cap.isOpened() == False:
        #     logger.info("Videocapture could not open video file", "error")
        # else:
        #     logger.info("Reading video file: " + self.videoFileName)

    def getVideoInfo(self):
        """

        Creates dictionary self.vidInfo with video information from self.cap

        Parameters
        ----------
        None


        Return
        ----------
        None; Creates self.vidInfo with variables 'nFrames','fps','height','width','codecnum','codec','duration'
        Note that for some video codecs cv2.cap cannot provide nFrames, fps, and duration.
        Use the self.stop() function to get this video info after all video frames have been processed.

        """
        self.vidInfo = {}
        self.vidInfo["fps"] = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.vidInfo["nFrames"] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vidInfo["codecnum"] = self.cap.get(cv2.CAP_PROP_FOURCC)
        self.vidInfo["codec"] = (
            "".join(
                [chr((int(self.cap.get(cv2.CAP_PROP_FOURCC)) >> 8 * i) & 0xFF) for i in range(4)]
            ),
        )
        self.vidInfo["duration"] = self.vidInfo["nFrames"] / self.vidInfo["fps"]

        self.vidInfo["height"] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vidInfo["width"] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    def readFrame(self):
        """

        Read next frame from self.cap and stores as self.frame
        Also increase self.frameCount and append timestamp to self.timeStamps

        Parameters
        ----------
        None


        Return
        ----------
        None; Creates self.frame; appends self.timeStamps

        """
        self.ret, self.frame = self.cap.read()
        if self.ret:
            self.frame = cv2.cvtColor(
                self.frame, cv2.COLOR_BGR2RGB
            )  # cv2 capture software outputs BGR instead of RGB ... strange but correctable
            self.frameCount += 1
            curTimeStamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if (curTimeStamp == 0) & (self.frameCount > 1):
                pass  # do not update frameTime
            else:
                self.frameTime = curTimeStamp

            self.timeStamps = np.append(self.timeStamps, self.frameTime)
        # else:
        #     logger.info("End of video")