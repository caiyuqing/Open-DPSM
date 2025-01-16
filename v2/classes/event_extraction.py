# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:49:54 2023

@author: 7009291
"""
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import logging
import pickle
import math
logging.basicConfig(level=logging.DEBUG)
handler = logging.FileHandler("gaze_center_pickle_creation.log")        

logger = logging.getLogger('gaze_center')
logger.addHandler(handler)
class event_extraction:
    def __init__(self):
        # for image processing
        self.imShown = False
        self.bufFrames = False  # buffer multiple frames, e.g. for comparison
        self.imCompFeatures = True  # compare subsequent images; bufFrames must be set True
        self.avaiFeatures = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "RGB"]
        #self.selectFeatures = ['Luminance']#, 'Red-Green', 'Blue-Yellow', 'Lum-RG', 'Lum-BY', 'Hue-Sat', 'LAB']
        self.countIm = 0
        # for video processing
        self.frameCount = 0
        self.timeStamps = np.zeros((0))  # append video timestamps

    def event_extraction(self):   
        #imDimRat = self.vidInfo["height"] / self.vidInfo["width"]
        self.createMapMask()
        self.magnPerImPart = {}
        self.magnPerIm ={}
        #gazecenteredPickleDir = 'C:/Users/7009291/OneDrive - Universiteit Utrecht/Movie pupil perimetry project/Movie and movie data/gaze centered pickle files'

        # loop video frames
        pb_value = 0
        if self.useApp:
            self.var = tk.StringVar()
            self.var.set('0% done')
        else:
            print("0% done...")
        self.video_nFrame = self.vidInfo['frameN_end']
        self.cap = cv2.VideoCapture(self.filename_movie)

        while True: #self.frameCount <10:  # Capture frame-by-frame
            self.readFrame()  # reads a frame from video
            
            #self.reportProcess()  # check frame in percentage progress of video; output progress

            if self.ret:
                #self.loadImage(self.frame, self.frameCount,scrGamFac=scrGamFac)
                if self.gazecentered:
                    self.loadImage_gazecentered(self.frame, self.frameCount, self.sampledgazexData_featureExtraction, self.sampledgazeyData_featureExtraction,self.subject,self.movieNum, scrGamFac=self.scrGamFac, maxlum = self.maxlum)
                else:
                    self.loadImage(self.frame, self.frameCount,scrGamFac=self.scrGamFac, maxlum = self.maxlum)
                # track the progress
                if self.frameCount >= np.linspace(0, self.video_nFrame, 11)[pb_value]:
                    
                    percentage_done = pb_value * 10
                    if self.useApp:

                        label_percent = tk.Label(self.window, textvariable = self.var, fg = "green")
                        label_percent.grid(column = 0, row = 14)
                        self.label_info.grid_forget()
                        self.var.set(f"{percentage_done}% done...")
                        self.window.update_idletasks()
                    else:  
                        print(f"{percentage_done}% done")
                    pb_value = pb_value +1 
                    
                    
                # if self.frameCount%1 ==0:
                #     print(self.frameCount) # show proces
                if self.frameCount ==1:
                    print(f"final: {self.img.shape}")
                    print(f"original: {self.frame.shape}")
                    print(f"check if aspect ratio is correct: {self.aspectRatio}")
                # break matrix in parts at several levels and loop matrix parts
                # and calculate mean value in part. Target matrix can be a raw image
                # frame (i.e., self.img), or 1st order frame comparison (i.e., self.vectorMagnFrame)
                # Function creates: self.levelMeanMatPartPerLevel

                # Luminance contrast (1D vector across luminance axis of color space)
                # per image part between two subsequent frames
                if self.mapType == "circular":
                    for vidFeatureName in self.vectorMagnFrame:
                        # this step will get: imageObj.levelMeanMatPart and imageObj.meanImg
                        #imageObj.meanPerMatPartCircle(circleMask=circleMask,circleMask_matrix= circleMask_matrix,num_pixel_matrix = num_pixel_matrix, tarMat = imageObj.vectorMagnFrame[vidFeatureName])
                        self.levelMeanMatPart = self.vectorMagnFrame[vidFeatureName]
                        self.meanImg = np.sum(self.vectorMagnFrame[vidFeatureName])/np.shape(self.levelMeanMatPart)[0]

                        if vidFeatureName in self.magnPerImPart:
                            self.magnPerImPart[vidFeatureName] = np.vstack((self.magnPerImPart[vidFeatureName],self.levelMeanMatPart))
                            self.magnPerIm[vidFeatureName] = np.hstack((self.magnPerIm[vidFeatureName],self.meanImg))
                        else:
                            self.magnPerImPart[vidFeatureName] = self.levelMeanMatPart
                            self.magnPerIm[vidFeatureName] = self.meanImg
                elif self.mapType == "square":
                    for vidFeatureName in self.vectorMagnFrame:
                        # calculate mean feature value per image region (averaged across multiple resolution levels)
                        
                        self.meanPerMatPart(self.nVertMatPartsPerLevel,self.aspectRatio,self.vectorMagnFrame[vidFeatureName],)
                        self.meanImage(self.vectorMagnFrame[vidFeatureName])
                        
                        # store mean feature value per image region
                        if vidFeatureName in self.magnPerImPart:
                            self.magnPerImPart[vidFeatureName] = np.dstack((self.magnPerImPart[vidFeatureName], self.levelMeanMatPart))
                            self.magnPerIm[vidFeatureName] = np.hstack((self.magnPerIm[vidFeatureName],self.meanImg))
                        else:
                            self.magnPerImPart[vidFeatureName] = self.levelMeanMatPart
                            self.magnPerIm[vidFeatureName] = self.meanImg
                # only shows last calculated feature (e.g. B in LAB)
                if (self.showVideoFrames) and (self.countIm > 1):

                    changeIm = self.magnPerImPart[vidFeatureName][:, :, -1]
                    if "maxMagn" not in locals():
                        maxMagn = np.max(changeIm) - np.min(changeIm)
                    elif maxMagn < (np.max(changeIm) - np.min(changeIm)):
                        maxMagn = np.max(changeIm) - np.min(changeIm)

                    changeIm = changeIm - np.min(changeIm)
                    changeIm = changeIm / maxMagn
                    changeIm = np.uint8(255 * changeIm)
                    changeIm = np.dstack((changeIm, changeIm, changeIm))
                    changeIm = cv2.resize(
                        changeIm, (np.shape(self.frame)[1], np.shape(self.frame)[0])
                    )

                    self.frame2show = np.hstack((self.frame, changeIm))
                    textList = [
                        "{:.2f}".format(self.frameTime) + "s",
                        "{:.0f}".format(self.frameCount) + "f",
                    ]
                    self.showImage(text=textList)

                    # Press Q on keyboard to exit
                    # THIS MUST BE INCLUDED, otherwise nothing is shown
                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        break
                self.countIm +=1
            else:
                break
    
        self.stop()
        # print out saving 
        if self.useApp:
            self.var.set(f"Saving the pickle file...")
            self.window.update_idletasks()
        else:
            print("Saving the pickle file")
        # save as a pickle file
        
        with open(
            self.picklename,
            "wb",
        ) as handle:
            pickle.dump(
                [self.vidInfo, self.timeStamps, self.magnPerImPart,self.magnPerIm],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            handle.close()
        if self.useApp:
            self.var.set(f"pickle file has been saved!")
        else:
            print("Pickle file has been saved!")
    ##################################################################################
    # eyetracking data processing
    def prepare_sampleData(self, dataBeforeSample,nsample):
        ## do nothing if choose the stretch to match
        if self.stretchToMatch:
            pass
        ## cut the longer one
        else:
            if self.video_duration > self.eyetracking_duration:
                diff = self.video_duration -self.eyetracking_duration
                nFrameDiff = np.floor(diff/(1/self.video_fps))
                nsample = int(nsample-nFrameDiff)
            else: 
                diff = np.abs(self.video_duration -self.eyetracking_duration)
                nFrameDiff = np.floor(diff/(1/self.eyetracking_samplingrate))
                eyetracking_nsample = int(len(dataBeforeSample)- nFrameDiff)
                dataBeforeSample = dataBeforeSample[:eyetracking_nsample]
        sampleTimes = np.linspace(0, len(dataBeforeSample) - 1, nsample).astype("int")
        sampledData = dataBeforeSample[sampleTimes]
        return sampledData
    ##################################################################################
    # image processing
    def setNBufFrames(self, nBufFrames):
        """
        presets self.refFrames that stores a number of subsequent frames for comparison analyses

        Parameters
        ----------
        nBufFrames: integer

        Return
        ----------
        None

        """
        self.bufFrames = True
        self.nBufFrames = nBufFrames
   
    # store image in object
    #def loadImageGazeCent(self, img, frameNum, scrGamFac=0, rotDeg=0, showImageProcResult=False, gazeY,gazeX):
    def loadImage_gazecentered(self, img, frameNum, sampledgazexData, sampledgazeyData, subject, movieNum, A = 2, scrGamFac=2.2, rotDeg=0, maxlum = 212, showImageProcResult=False):
        """
        Loads an image to self.img for further processing

        Parameters
        ----------
        img: 8-bit RGB image matrix

        rotDeg: integer indication rotation of image (e.g., when video taken by tablet/smartphone)
            90, 180, or 270
        also see rotateImage()

        showImageProcResult: True or False. Set to true if
        image processing steps are displayed per frame, or when a control image/video
        needs to be stored for later inspection.
        If True, image is stored as self.frame2show for further displaying purposes.

        see drawFaceSkinResults() for drawing the image processing result to self.frame2show
        see storeFaceSkinImage() for storing a couple image frames to show which skin areas were selected
        see storeFaceSkinVideo() for storing the video to show which skin areas were selected per frame

        Return
        ----------
        None

        """
        if self.videoScreenSameRatio and self.videoStretched:
            realImg = cv2.resize(img, (int(self.videoRealWidth), int(self.videoRealHeight)))
            img = realImg
        else: 
            screen = np.zeros((int(self.eyetracking_height), int(self.eyetracking_width),np.shape(img)[2]),dtype = 'float32')
            # fill the screen with the color
            colorBgRGB = np.array([self.screenBgColorR,self.screenBgColorG,self.screenBgColorB])
            for i in range(int(self.eyetracking_height)):
                for j in range(int(self.eyetracking_width)):
                    screen[i,j,:] = colorBgRGB
            realImg = cv2.resize(img, (int(self.videoRealWidth), int(self.videoRealHeight))) # resize is (width, height)
            # put the realImg to the screen
            width_diff = screen.shape[1] - realImg.shape[1]
            height_diff = screen.shape[0] - realImg.shape[0]
            screen[int(np.ceil(height_diff/2)):(int(np.ceil(height_diff/2)) + realImg.shape[0]),int(np.ceil(width_diff/2)):(int(np.ceil(width_diff/2)) + realImg.shape[1]),:] = realImg
            img = screen
        bigImg = np.zeros(((np.shape(img)[0]-1)*4+1, (np.shape(img)[1]-1)*4+1,np.shape(img)[2]),dtype = 'float32')
        #bigImg[:] = np.nan
        bigImg_centerx = int((bigImg.shape[1] -1 )/2) # This will be the new center of the data
        bigImg_centery = int((bigImg.shape[0] -1 )/2)
        print(f"frame number is {frameNum}")
        gazex = sampledgazexData[frameNum-1]
        gazey = sampledgazeyData[frameNum-1]
        if self.mapType == "circular":
            finalImgWidth = finalImgHeight = self.circleMask.shape[0]
        elif self.mapType == "square":
            finalImgWidth = self.squareWidth
            finalImgHeight = self.squareHeight
        # remove gaze that is too far off
        if gazex <0 or gazex > self.eyetracking_width:
            if np.abs(gazex) > finalImgWidth/2 or np.abs(gazex) - self.eyetracking_width > finalImgWidth/2:
                gazex = np.nan
        if gazey <0 or gazey > self.eyetracking_height:
            if np.abs(gazey) > finalImgHeight/2 or np.abs(gazey) - self.eyetracking_height > finalImgHeight/2:
                gazey = np.nan
        if np.isfinite(gazex) and np.isfinite(gazey):
            leftTopCornerX = int(round(bigImg_centerx - gazex)) # top left
            leftTopCornerY = int(round(bigImg_centery - gazey))
            if gazex<0 or gazey <0 or gazex > np.shape(img)[1]-1 or  gazey > np.shape(img)[0]-1:
                logger.info(f'{subject} movie {movieNum} frameNum {frameNum} is out of the screen')
                logger.info(f'gazex:{gazex}; gazey:{gazey}')

            
            for i in range(np.shape(img)[2]):
                bigImg[leftTopCornerY:(leftTopCornerY+np.shape(img)[0]), leftTopCornerX:(leftTopCornerX+np.shape(img)[1]), i] = img[:,:,i]
            #finalImg = bigImg[(bigImg_centery-(np.shape(img)[0]-1)):(bigImg_centery+(np.shape(img)[0]-1)+1),(bigImg_centerx-(np.shape(img)[1]-1)):(bigImg_centerx+(np.shape(img)[1]-1)+1),:]
            finalImg = bigImg[(bigImg_centery-int(np.floor(finalImgHeight/2))):(bigImg_centery+int(np.floor(finalImgHeight/2))+1),(bigImg_centerx-int(np.floor(finalImgWidth/2))):(bigImg_centerx+int(np.floor(finalImgWidth/2))+1),:]
            if self.countIm ==1:
                logger.info(f"{subject} movie {movieNum}")
                logger.info(f"original: {img.shape}")
                logger.info(f"final: {finalImg.shape}")
        
        else: 
            leftTopCornerX = 0 #doesn't matter, as the img will become all black
            leftTopCornerY = 0
            img = np.zeros(np.shape(img))
            for i in range(np.shape(img)[2]):
                bigImg[leftTopCornerY:(leftTopCornerY+np.shape(img)[0]), leftTopCornerX:(leftTopCornerX+np.shape(img)[1]), i] = img[:,:,i]
            #finalImg = bigImg[(bigImg_centery-(np.shape(img)[0]-1)):(bigImg_centery+(np.shape(img)[0]-1)+1),(bigImg_centerx-(np.shape(img)[1]-1)):(bigImg_centerx+(np.shape(img)[1]-1)+1),:]
            finalImg = bigImg[(bigImg_centery-int(np.floor(finalImgHeight/2))):(bigImg_centery+int(np.floor(finalImgHeight/2))+1),(bigImg_centerx-int(np.floor(finalImgWidth/2))):(bigImg_centerx+int(np.floor(finalImgWidth/2))+1),:]
            if self.countIm ==1:
                logger.info(f"{subject} movie {movieNum}")
                logger.info(f"original: {img.shape}")
                logger.info(f"final: {finalImg.shape}")
        
        # This step is important!
        finalImg = np.uint8(finalImg)
        self.img = finalImg.copy()
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.imshow(self.img.astype(int))
        #print(self.img[629,487])
        self.convertColorSpace(self.imColSpaceConv)
        
        #print(self.img[629,487])

        self.rotateImage(deg=rotDeg)

        self.frame2show = self.img.copy()

        # convert to luminance values per image pixel
        self.val2lum(scrGamFac=scrGamFac, maxlum = maxlum)
        # apply mask if it is circular
        if self.mapType == "circular":
            # divide the img into circular regions first
            self.meanPerMatPartCircle(self.circleMask,self.circleMask_matrix,self.num_pixel_matrix, self.img)
        # elif self.useSapMask:
        #     self.meanPerMatPartSAP(tarMat=self.img, degPerRegion=6)
            
        if self.bufFrames:
            if hasattr(self, "refFrames"):
                pass
            else:
                # pre-create frame variable
                if self.mapType == "square":
                    self.refFrames = np.zeros((np.shape(self.img)[0],np.shape(self.img)[1],np.shape(self.img)[2],self.nBufFrames,))
                    
                elif self.mapType == "circular":
                    self.refFrames = np.zeros((np.shape(self.img)[0],np.shape(self.img)[1],self.nBufFrames,))
                    
            
            if self.mapType == "square":
                curFrameNum = np.mod(frameNum, self.nBufFrames)
                refFrameNum = np.mod(frameNum - (self.nBufFrames - 1), self.nBufFrames)
                self.refFrames[:, :, :, curFrameNum] = self.img
                curFrame = self.refFrames[:, :, :, curFrameNum]
                refFrame = self.refFrames[:, :, :, refFrameNum]
            elif self.mapType == "circular":
                curFrameNum = np.mod(frameNum, self.nBufFrames)
                refFrameNum = np.mod(frameNum - (self.nBufFrames - 1), self.nBufFrames)
                self.refFrames[:, :,  curFrameNum] = self.img
                curFrame = self.refFrames[:, :,  curFrameNum]
                refFrame = self.refFrames[:, :,  refFrameNum]
            if self.imCompFeatures:
                #if self.countIm == 0:
                    # logger.log("Check for changes across frames for features :")
                    # logger.log(self.avaiFeatures)

                self.diffFrame = curFrame - refFrame
                #print(self.diffFrame[629,487])

                # self.diffFrame = np.array(curFrame - refFrame, dtype="float64")
                self.vectorMagnFrame = {}

                for i in range(3):
                    if self.avaiFeatures[i] in self.selectFeatures:
                        if self.mapType == "square":
                            self.vectorMagnFrame[self.avaiFeatures[i]] = self.diffFrame[:, :, i]
                        elif self.mapType == "circular":
                            self.vectorMagnFrame[self.avaiFeatures[i]] = self.diffFrame[:,  i]
                        
                # If LAB space: 2D vector across Lum and A (red-green) axes
                # If HSV space: 2D vector across hue and sat axes
                if self.avaiFeatures[3] in self.selectFeatures:
                    self.vectorMagnFrame[self.avaiFeatures[3]] = np.sqrt(
                        self.diffFrame[:, :, 0] ** 2 + self.diffFrame[:, :, 1] ** 2
                    )

                if self.avaiFeatures[4] in self.selectFeatures:
                    # If LAB space: 2D vector across Lum and B (blue-yellow) axes
                    # If HSV space: 2D vector across Hue and lum axes
                    self.vectorMagnFrame[self.avaiFeatures[4]] = np.sqrt(
                        self.diffFrame[:, :, 0] ** 2 + self.diffFrame[:, :, 2] ** 2
                    )

                if self.avaiFeatures[5] in self.selectFeatures:
                    # If LAB space: 2D vector across color A&B axes (hue-sat)
                    # If HSV space: 2D vector across Sat and lum axes
                    self.vectorMagnFrame[self.avaiFeatures[5]] = np.sqrt(
                        self.diffFrame[:, :, 1] ** 2 + self.diffFrame[:, :, 2] ** 2
                    )

                if self.avaiFeatures[6] in self.selectFeatures:
                    # 3d vector in color space across all axes
                    self.vectorMagnFrame[self.avaiFeatures[6]] = np.sqrt(
                        self.diffFrame[:, :, 0] ** 2
                        + self.diffFrame[:, :, 1] ** 2
                        + self.diffFrame[:, :, 2] ** 2
                    )
        #self.countIm += 1
    def loadImage(self, img, frameNum,  maxlum, scrGamFac, rotDeg=0,showImageProcResult=False):
        """
        Loads an image to self.img for further processing

        Parameters
        ----------
        img: 8-bit RGB image matrix

        rotDeg: integer indication rotation of image (e.g., when video taken by tablet/smartphone)
            90, 180, or 270
        also see rotateImage()

        showImageProcResult: True or False. Set to true if
        image processing steps are displayed per frame, or when a control image/video
        needs to be stored for later inspection.
        If True, image is stored as self.frame2show for further displaying purposes.

        see drawFaceSkinResults() for drawing the image processing result to self.frame2show
        see storeFaceSkinImage() for storing a couple image frames to show which skin areas were selected
        see storeFaceSkinVideo() for storing the video to show which skin areas were selected per frame

        Return
        ----------
        None

        """
        
        self.img = img
                    
        self.convertColorSpace(self.imColSpaceConv)

        self.rotateImage(deg=rotDeg)
        self.frame2show = self.img.copy()

        # convert to luminance values per image pixel
        self.val2lum(scrGamFac=scrGamFac, maxlum = maxlum)

        if self.bufFrames:
            if hasattr(self, "refFrames"):
                pass
            else:
                # pre-create frame variable
                self.refFrames = np.zeros(
                    (
                        np.shape(self.img)[0],
                        np.shape(self.img)[1],
                        np.shape(self.img)[2],
                        self.nBufFrames,
                    )
                )
                # self.allFrames = np.zeros(
                #     (
                #         np.shape(self.img)[0],
                #         np.shape(self.img)[1],
                #         np.shape(self.img)[2],
                #         10,
                #     )
                # )

            curFrameNum = np.mod(frameNum, self.nBufFrames)
            refFrameNum = np.mod(frameNum - (self.nBufFrames - 1), self.nBufFrames)
            self.refFrames[:, :, :, curFrameNum] = self.img
            curFrame = self.refFrames[:, :, :, curFrameNum]
            refFrame = self.refFrames[:, :, :, refFrameNum]
            #self.allFrames[:,:,:,(frameNum-1)] = self.img
            if self.imCompFeatures:
                #if self.countIm == 0:
                    # logger.log("Check for changes across frames for features :")
                    # logger.log(self.avaiFeatures)

                self.diffFrame = curFrame - refFrame
                # self.diffFrame = np.array(curFrame - refFrame, dtype="float64")
                self.vectorMagnFrame = {}

                for i in range(3):
                    if self.avaiFeatures[i] in self.selectFeatures:
                        self.vectorMagnFrame[self.avaiFeatures[i]] = self.diffFrame[:, :, i]

                # If LAB space: 2D vector across Lum and A (red-green) axes
                # If HSV space: 2D vector across hue and sat axes
                if self.avaiFeatures[3] in self.selectFeatures:
                    self.vectorMagnFrame[self.avaiFeatures[3]] = np.sqrt(
                        self.diffFrame[:, :, 0] ** 2 + self.diffFrame[:, :, 1] ** 2
                    )

                if self.avaiFeatures[4] in self.selectFeatures:
                    # If LAB space: 2D vector across Lum and B (blue-yellow) axes
                    # If HSV space: 2D vector across Hue and lum axes
                    self.vectorMagnFrame[self.avaiFeatures[4]] = np.sqrt(
                        self.diffFrame[:, :, 0] ** 2 + self.diffFrame[:, :, 2] ** 2
                    )

                if self.avaiFeatures[5] in self.selectFeatures:
                    # If LAB space: 2D vector across color A&B axes (hue-sat)
                    # If HSV space: 2D vector across Sat and lum axes
                    self.vectorMagnFrame[self.avaiFeatures[5]] = np.sqrt(
                        self.diffFrame[:, :, 1] ** 2 + self.diffFrame[:, :, 2] ** 2
                    )

                if self.avaiFeatures[6] in self.selectFeatures:
                    # 3d vector in color space across all axes
                    self.vectorMagnFrame[self.avaiFeatures[6]] = np.sqrt(
                        self.diffFrame[:, :, 0] ** 2
                        + self.diffFrame[:, :, 1] ** 2
                        + self.diffFrame[:, :, 2] ** 2
                    )
        #self.countIm += 1
    def convertColorSpace(self, colorSpace):
        if colorSpace == "RGB":
            pass
        elif colorSpace == "LAB":
            self.avaiFeatures = [
                "Luminance",
                "Red-Green",
                "Blue-Yellow",
                "Lum-RG",
                "Lum-BY",
                "Hue-Sat",
                "LAB",
            ]
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)
        elif colorSpace == "HSV":
            self.avaiFeatures = [
                "Hue",
                "Saturation",
                "Luminance",
                "Hue-Sat",
                "Hue-Lum",
                "Sat-Lum",
                "HSV",
            ]
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        elif colorSpace == "HLS":
            self.avaiFeatures = [
                "Hue",
                "Luminance",
                "Saturation",
                "Hue-Lum",
                "Hue-Sat",
                "Lum-Sat",
                "HLS",
            ]
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
        elif colorSpace == "LUV":
            self.avaiFeatures = [
                "Luminance",
                "Cyan-Magenta",
                "Purple-Yellow",
                "Lum-CM",
                "Lum-PY",
                "Hue-Sat",
                "LUV",
            ]
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2LUV)
        else:
            logger.log(
                "Color space "
                + colorSpace
                + " indicated in settings not recognized; sticking to RGB",
                logType="error",
            )

    def val2lum(self, scrGamFac=2.2, maxlum = 212):
        if scrGamFac > 0:
            pixLumCurve = np.divide((np.arange(0, 256) ** scrGamFac), (255 ** scrGamFac)) * maxlum

            if self.imColSpaceConv == "LAB":
                lumChanIdx = [0]
            elif self.imColSpaceConv == "HSV":
                lumChanIdx = [2]
            elif self.imColSpaceConv == "HLS":
                lumChanIdx = [1]
            elif self.imColSpaceConv == "LUV":
                lumChanIdx = [0]
            else:  # RGB
                lumChanIdx = [0, 1, 2]

            for lumChanI in lumChanIdx:
                # NOT SURE WHETHER ALL COLOR CHANNELS ARE BETWEEN 0-255..
                # double check this if using something else than RGB or LAB
                self.img[:, :, lumChanI] = pixLumCurve[np.uint8(self.img[:, :, lumChanI])]

    # rotate image
    def rotateImage(self, deg=90):
        """
        rotates img

        Parameters
        ----------
        img: 8-bit image matrix

        deg: integer indication rotation of image (e.g., when video taken by tablet/smartphone)
            90, 180, or 270

        Return
        ----------
        None; self.img is modified

        """
        if deg == 90:
            self.img = cv2.rotate(self.img, cv2.cv2.ROTATE_90_CLOCKWISE)
        elif deg == 270:
            self.img = cv2.rotate(self.img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif deg == 180:
            self.img = cv2.rotate(self.img, cv2.cv2.ROTATE_180)
    def stimMatrixFunc(self):
        # eye_to_screen is the distance between eye to screen in cm
        # calculate how long it is required for the certain degree
        circleDiameterCM = np.tan((self.degVF/2) * np.pi / 180) * self.eye_to_screen *2
        # calculate number of pixel in eyetracker system in 1cm of the screen
        eyetracker_pixel_1cm = self.eyetracking_width / self.screen_width
        # calculate how many pixel is required for the degree
        circleDiameterPixel = int(round(eyetracker_pixel_1cm * circleDiameterCM))
        # create stimulus matrix
        halfCircleDiameterPixel = circleDiameterPixel/2
        annulusEccentricity_pix = list(np.around(np.array([70, 149, 218, 336, 447]) * (halfCircleDiameterPixel/447)).astype(int))
        nWedgesPerAnnulus = list(np.array([4, 8, 8, 12, 12]))

        stimMatrix = np.zeros(
            (np.max(annulusEccentricity_pix) * 2 + 1, np.max(annulusEccentricity_pix) * 2 + 1)
        )
        [stimMatrixX, stimMatrixY] = np.meshgrid(
            range(-1 * np.max(annulusEccentricity_pix), np.max(annulusEccentricity_pix) + 1),
            range(-1 * np.max(annulusEccentricity_pix), np.max(annulusEccentricity_pix) + 1),
        )

        [stimMatrixR, stimMatrixTH] = self.cart2pol(stimMatrixX, stimMatrixY)

        tempRadii = np.hstack((0, annulusEccentricity_pix))
        countWedge = 0
        for a in range(len(tempRadii) - 1):
            selectRadius = (stimMatrixR > tempRadii[a]) & (stimMatrixR <= tempRadii[a + 1])

            tempWedges = np.linspace(-1 * np.pi, np.pi, nWedgesPerAnnulus[a] + 1)
            for w in range(nWedgesPerAnnulus[a]):
                countWedge = countWedge + 1
                selectWedge = (stimMatrixTH > tempWedges[w]) & (stimMatrixTH <= tempWedges[w + 1])

                stimMatrix[(selectRadius) & (selectWedge)] = countWedge
        #finalImgWidth =int(np.ceil(self.eyelink_width * A) +1)
        #finalImgHeight = int(np.ceil(self.eyelink_height *A)+1)#int(self.eyelink_height * A +1)
        # create a matrix as a mask for later part
        #circleMask = np.empty((int(finalImgHeight), int(finalImgWidth)))
        #circleMask[(int((finalImgHeight-circleDiameterPixel)/2)-1):(int(finalImgHeight - ((finalImgHeight-circleDiameterPixel)/2))), (int((finalImgWidth-circleDiameterPixel)/2)-1):(int(finalImgWidth - ((finalImgWidth-circleDiameterPixel)/2)))] = stimMatrix
        self.finalImgWidth = stimMatrix.shape[0]#finalImgWidth
        self.finalImgHeight= stimMatrix.shape[1]#finalImgHeight
        self.A = stimMatrix.shape[0] / min(self.eyetracking_height, self.eyetracking_width)
        return stimMatrix
    def createMapMask(self):
        if self.mapType == "circular":
            circleMask = self.stimMatrixFunc()
            circleMask_arr = circleMask.reshape(circleMask.shape[0]*circleMask.shape[1])
            circleMask_matrix = np.empty(circleMask_arr.shape[0])
            
            print(f"Generating circular region mask....")
            for i in range(int(max(circleMask_arr))):
                ind = i+1
                    
                circleMask_arr_ind = circleMask_arr.copy()
                circleMask_arr_ind[circleMask_arr_ind!=ind] =0
                circleMask_arr_ind[circleMask_arr_ind==ind] =1
                #print(circleMask_arr_ind.max())
                circleMask_matrix = np.vstack((circleMask_matrix, circleMask_arr_ind))
            circleMask_matrix = circleMask_matrix[1:circleMask_matrix.shape[0], :]
            num_pixel_matrix = np.sum(circleMask_matrix, axis = 1)
            self.circleMask_matrix = circleMask_matrix
            self.num_pixel_matrix = num_pixel_matrix
            self.circleMask = circleMask
            finalImgWidth = finalImgHeight = self.circleMask.shape[0]
            
        elif self.mapType == "square":
            tarMatSizeCM = math.tan(math.radians(self.degVF/2)) * self.eye_to_screen*2
            # calculate number of pixel in eyetracker system in 1cm of the screen
            eyetracker_pixel_1cm = self.eyetracking_width / self.screen_width
            # calculate how many pixel is required for the degree
            self.squareWidth = int(round(eyetracker_pixel_1cm * tarMatSizeCM))
            self.squareHeight = int(self.squareWidth/(self.eyetracking_width/self.eyetracking_height))
            
            finalImgWidth = self.squareWidth
            finalImgHeight = self.squareHeight
        self.finalImgWidth = finalImgWidth
        self.finalImgHeight = finalImgHeight
    def meanPerMatPartCircle(self, circleMask,circleMask_matrix,num_pixel_matrix, tarMat):
        #tarMat = self.vectorMagnFrame['Luminance']
        # levelMeanMatPart = np.zeros((int(np.nanmax(circleMask))))
        # for i in range(levelMeanMatPart.shape[0]):
        #     mask = np.zeros(circleMask.shape)
        #     mask[circleMask == (i+1)] = 1
        #     maskedMat = tarMat * mask
        #     # calculate the number of pixel in the region
        #     unique,count = np.unique(mask,return_counts = True)
        #     num_pixel = count[1]
        #     print(num_pixel)
        #     # calculate the sum of the visual feature change in a region
        #     maskedMatMean = np.sum(maskedMat)/num_pixel
        #     levelMeanMatPart[i] = maskedMatMean
        #    # plt.imshow(maskedMat)
        #     #plt.show()
        
        # tarMat_arr = tarMat.reshape(circleMask.shape[0]*circleMask.shape[1])
        # levelMeanMatPart = np.matmul(tarMat_arr,circleMask_matrix.T)
        # # calculate the mean of every region
        # levelMeanMatPart = levelMeanMatPart / num_pixel_matrix
        # self.levelMeanMatPart = levelMeanMatPart
        # self.meanImg = np.sum(levelMeanMatPart)/int(np.nanmax(circleMask))
        
        levelMeanMatPartMatrix = np.empty((int(circleMask.max()),3))
        
        for i in range(tarMat.shape[2]):
            tarMat_arr = tarMat[:,:,i].reshape(self.circleMask.shape[0]*self.circleMask.shape[1])
            levelMeanMatPart = np.matmul(tarMat_arr,self.circleMask_matrix.T) # circleMask_matrix contains 44 columns, each of them contains 1 or 0, pixels that are in this region is indexed with 1
            # calculate the mean of every region
            levelMeanMatPart = levelMeanMatPart / self.num_pixel_matrix
            levelMeanMatPartMatrix[:,i] = levelMeanMatPart
        self.img = levelMeanMatPartMatrix

    def meanPerMatPart(self, nVertMatPartsPerLevel, aspectRatio, tarMat):

        nBinsYMax = nVertMatPartsPerLevel[-1]
        # imDimRat = np.shape(tarMat)[1] / np.shape(tarMat)[0]
        nBinsXMax = int(np.round(nBinsYMax / aspectRatio))

        meanMatPartPerLevel = np.zeros((nBinsYMax, nBinsXMax, len(nVertMatPartsPerLevel)))
        for binCount in range(len(nVertMatPartsPerLevel)):
            nBinsY = nVertMatPartsPerLevel[binCount]
            yBinEdges = np.array(
                np.floor(np.linspace(0, np.shape(tarMat)[0], nBinsY + 1)), dtype="int"
            )
            nBinsX = int(np.round(nBinsY / aspectRatio))
            xBinEdges = np.array(
                np.floor(np.linspace(0, np.shape(tarMat)[1], nBinsX + 1)), dtype="int"
            )
            refEdgesY = np.array(np.floor(np.linspace(0, nBinsYMax, nBinsY + 1)), dtype="int")
            refEdgesX = np.array(np.floor(np.linspace(0, nBinsXMax, nBinsX + 1)), dtype="int")

            meanMatPart = np.zeros((nBinsYMax, nBinsXMax))
            for y in range(len(yBinEdges) - 1):
                yRange = [refEdgesY[y], refEdgesY[y + 1]]
                for x in range(len(xBinEdges) - 1):
                    xRange = [refEdgesX[x], refEdgesX[x + 1]]

                    meanMatPart[yRange[0] : yRange[1], xRange[0] : xRange[1]] = np.mean(
                        tarMat[
                            yBinEdges[y] : yBinEdges[y + 1],
                            xBinEdges[x] : xBinEdges[x + 1],
                        ]
                    )

            meanMatPartPerLevel[:, :, binCount] = meanMatPart

        # average across levels
        self.tmp = meanMatPartPerLevel
        self.levelMeanMatPart = np.mean(meanMatPartPerLevel, axis=2)
    def meanImage(self,img):
        self.meanImg = np.mean(img.astype(float)[:])

    def showImage(self, text=["a", "b", "c"], textScale=1):
        """
        shows image (self.frame2show) in popup window

        Parameters
        ----------
        None


        Return
        ----------
        Stores image array as self.img_gray

        """

        countTxt = 0
        for curTxt in text:
            countTxt += 1
            cv2.putText(
                self.frame2show,
                curTxt,
                (10, int(textScale * 30 * countTxt)),
                cv2.FONT_HERSHEY_DUPLEX,
                textScale,
                (255, 0, 0),
                int(np.ceil(textScale)),
            )  # , cv2.LINE_AA

        if self.imShown == False:
            # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Frame", cv2.WINDOW_FREERATIO)
            self.imShown = True

        cv2.imshow(
            "Frame", np.uint8(self.frame2show[:, :, [2, 1, 0]])
        )  # this function only works in a loop

    def cart2pol(self, x, y):
        """
        Cartesian coordinates to polar coordinates

        Parameters
        ----------
        x: floats or integers of horizontal cartesian axis
        y: floats or integers of vertical cartesian axis


        Return
        ----------
        rho: radius of coordinates
        phi: angle of coordinates in radii  [0-2*pi]

        """
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    def pol2cart(self, rho, phi):
        """
        Polar coordinates (numpy array) to cartesian coordinates

        Parameters
        ----------
        rho: radius of coordinates
        phi: angle of coordinates in radii [0-2*pi]


        Return
        ----------
        x: floats or integers of horizontal cartesian axis
        y: floats or integers of vertical cartesian axis

        """
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)
    ##################################################################################
    #video processing
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

    def reportProcess(self, interval=10):

        """

        Logs processing status every Nth percentile (if 'nFrames' is available in self.vidInfo)

        Parameters
        ----------
        interval: integer (N) indicating to log every Nth percentile
        If number of frames unknown, then it logs every 10s of video


        Return
        ----------
        None; Creates self.frameMilestoneReached variable.
        This variable can for example be used to decide whether it is time to save a control image for later inspection


        """
        # if self.vidInfo['nFrames']: # if number of frames available (for some video/codec types not available) output every 10% of frames;
        if (
            self.vidInfo["nFrames"] > 0
        ):  # if number of frames available (for some video/codec types not available) output every 10% of frames; sometimes (e.g., with webm videos), this variable is negative
            # print('Framecount" ' + str(self.vidInfo["nFrames"]) + "; Time: " + (self.frameTime))
            if (
                self.vidInfo["fps"] > 200
            ):  # unusually high FPS; probably an encoder meta data mistake; check duration instead of frame number just to be sure
                if sum(self.frameCount == np.arange(1, 90000, 300)):
                    # logger.info(
                    #     "Video processing status: "
                    #     + str(self.frameTime)
                    #     + "s, "
                    #     + str(self.frameCount)
                    #     + " frames"
                    # )
                    self.frameMilestoneReached = True
                else:
                    self.frameMilestoneReached = False
            else:
                if sum(
                    self.frameCount
                    == np.arange(
                        1, self.vidInfo["nFrames"], round(self.vidInfo["nFrames"] / interval)
                    )
                    - 1
                ):
                    # logger.info(
                    #     "Video processing status: "
                    #     + str(round(self.frameCount / self.vidInfo["nFrames"] * 100))
                    #     + "%"
                    # )
                    self.frameMilestoneReached = (
                        True  # for later storage of image with several frames
                    )
                else:
                    self.frameMilestoneReached = False
        # elif self.vidInfo['nFrames'] < 0: # output every 10 seconds
        else:  # output every 10 seconds
            # print("No framecount; Time: " + str(self.frameTime))
            if self.frameTime > self.reportProcessSec:
                # logger.info(
                #     "Time-based video processing status: "
                #     + str(self.frameTime)
                #     + "s, "
                #     + str(self.frameCount)
                #     + " frames"
                # )
                self.reportProcessSec = self.reportProcessSec + interval
                self.frameMilestoneReached = True
            elif self.vidInfo["fps"] <= 200:
                if sum(
                    self.frameCount == np.arange(1, 90000, np.ceil(self.vidInfo["fps"] * interval))
                ): 
                    pass
                    # logger.info(
                    #     "FPS-based video processing status: "
                    #     + str(self.frameTime)
                    #     + "s, "
                    #     + str(self.frameCount)
                    #     + " frames"
                    # )
                else:
                    self.frameMilestoneReached = False
            elif sum(self.frameCount == np.arange(1, 90000, 300)):
                # logger.info(
                #     "Framecount-based video processing status: "
                #     + str(self.frameTime)
                #     + "s, "
                #     + str(self.frameCount)
                #     + " frames"
                # )
                self.frameMilestoneReached = True
            else:
                self.frameMilestoneReached = False

    def stop(self):
        """

        Closes video and stores additional video info to dictionary self.vidInfo,
        including number of frames, frame rate, and duration because
        for some video codecs, cv2.cap cannot provide this information
        when loading the video, but can only calculated at the end
        after having processed all frames

        Parameters
        ----------
        None


        Return
        ----------
        None; Extends self.vidInfo with final video info variables 'nFrames_end', 'fps_end', 'duration_end'

        """
        cv2.destroyAllWindows()
        self.cap.release()

        self.vidInfo["nFrames_end"] = self.frameCount
        # Following is not always correct because frame time is sometimes wrong due to incorrect encoders that do not parse info correctly
        self.vidInfo["fps_end"] = self.frameCount / self.frameTime
        if self.vidInfo["fps_end"] > 999:  # this sometimes happens with webm videos
            self.vidInfo["fps_end"] = self.frameCount / self.vidInfo["duration"]
            self.vidInfo["duration_end"] = self.vidInfo["duration"]
            # if self.vidInfo["duration"] < 0:  # this sometimes happens with webm videos
            #     logger.info(
            #         "DURATION IS INCORRECT. THIS MAY HAPPEN WITH WEBM CODECS (VP80-VP90). CHECK FRAME RATE (FPS). IF ALSO INACCURATE, THEN TIME-BASED ANALYSES WILL FAIL.",
            #         "error",
            #     )

        else:
            self.vidInfo["duration_end"] = self.frameTime
            # self.vidInfo["fps"] = self.frameCount / self.vidInfo["duration"]
    