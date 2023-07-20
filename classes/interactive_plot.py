# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:53:54 2023

@author: 7009291
"""
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import pickle
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
class interactive_plot:
    def plot(self):
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

        fontsize = 15
        linesize = 3
        foldername = "Modeling result"
        os.chdir(self.dataDir) 
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
        self.modelResultDict = modelResultDict
        self.modelDataDict= modelDataDict
        
        sampledTimeStamps = self.modelDataDict[self.subjectName]["timeStamps"]
        sampledpupilData= self.modelDataDict[self.subjectName]["pupil"] 
        sampledgazexData = self.modelDataDict[self.subjectName]["gazex"] 
        sampledgazeyData = self.modelDataDict[self.subjectName]["gazey"] 
        lumData = self.modelDataDict[self.subjectName]["lumData"] 
        contrastData = self.modelDataDict[self.subjectName]["contrastData"] 
        y_pred = self.modelResultDict[self.subjectName]["modelContrast"]["predAll"] 
        lumConv = self.modelResultDict[self.subjectName]["modelContrast"]["lumConv"] 
        contrastConv = self.modelResultDict[self.subjectName]["modelContrast"]["contrastConv"] 
        
        # tkinter grid
        # widget_list = self.all_children()
        # for item in widget_list:
        #     item.destroy()
        if self.useApp:
            pass
        else:
            # if not using app, need to create a window first
            self.window = tk.Tk()
        self.top_interactive_figure = tk.Toplevel(self.window)
        self.top_interactive_figure.geometry("2000x1000")
        self.top_interactive_figure.resizable(True, True)
        
    
        init_frame = 3
        # Create the figure and the line that we will manipulate
        os.chdir(self.dataDir)
        video = cv2.VideoCapture(self.filename_movie)
        # plot the new coordinate system
        video.set(1,init_frame); # set up initail frame
        ret, frame = video.read() 
        if self.videoScreenSameRatio and self.videoStretched:
            realImg = cv2.resize(frame, (int(self.videoRealWidth), int(self.videoRealHeight)))
            frame = realImg
        else: 
            screen = np.zeros((int(self.eyetracking_height), int(self.eyetracking_width),np.shape(frame)[2]),dtype = 'float32')
            # fill the screen with the color
            colorBgRGB = np.array([self.screenBgColorR,self.screenBgColorG,self.screenBgColorB])
            for i in range(int(self.eyetracking_height)):
                for j in range(int(self.eyetracking_width)):
                    screen[i,j,:] = colorBgRGB
            realImg = cv2.resize(frame, (int(self.videoRealWidth), int(self.videoRealHeight))) # resize is (width, height)
            # put the realImg to the screen
            width_diff = screen.shape[1] - realImg.shape[1]
            height_diff = screen.shape[0] - realImg.shape[0]
            screen[int(np.ceil(height_diff/2)):(int(np.ceil(height_diff/2)) + realImg.shape[0]),int(np.ceil(width_diff/2)):(int(np.ceil(width_diff/2)) + realImg.shape[1]),:] = realImg
            frame = screen
            
        video_x = self.videoRealWidth
        video_y = self.videoRealHeight
        # bigImg_x = (video_x -1)*4+1
        # bigImg_y = (video_y -1)*4+1
        # bigImg = np.zeros((int(bigImg_y), int(bigImg_x), 3))
        # bigImg_centerx = int((bigImg_x -1 )/2) # This will be the new center of the data
        # bigImg_centery = int((bigImg_y -1 )/2)
        gazex = sampledgazexData[init_frame]
        gazey = sampledgazeyData[init_frame]
        
        f = plt.figure(figsize = (17,8),constrained_layout = True)
        gs = f.add_gridspec(ncols = 16,nrows = 20)
        axes = list()
        axes.append(f.add_subplot(gs[2:7, 0:4]))
        axes.append(f.add_subplot(gs[0:8, 3:16]))
        axes.append(f.add_subplot(gs[9:12,0:16]))
    
        axes.append(f.add_subplot(gs[13:16, 0:16]))
        axes.append(f.add_subplot(gs[17:20, 0:16]))
        #plt.subplots_adjust(right =20)
        # add movie and gaze to the left
        axes[0].set_xlim([min(0,min(sampledgazexData)),max(video_x,max(sampledgazexData))])
        axes[0].set_ylim([max(video_y, max(sampledgazeyData)),min(0,min(sampledgazeyData))])
        axes[0].set_title("Movie and gaze position", fontsize = fontsize, loc = "left")
        axes[0].spines[:].set_visible(False)
        original_movie = axes[0].imshow(realImg[:,:,::-1])
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        gaze, = axes[0].plot([sampledgazexData[init_frame]],[sampledgazeyData[init_frame]], 'ro')
    
       
        bigImg = np.zeros(((np.shape(frame)[0]-1)*4+1, (np.shape(frame)[1]-1)*4+1,np.shape(frame)[2]),dtype = 'float32')
        bigImg_x = (video_x -1)*4+1
        bigImg_y = (video_y -1)*4+1
        #bigImg[:] = np.nan
        bigImg_centerx = int((bigImg.shape[1] -1 )/2) # This will be the new center of the data
        bigImg_centery = int((bigImg.shape[0] -1 )/2)
        if not np.isnan(gazex) and not np.isnan(gazey):
            leftTopCornerX = int(round(bigImg_centerx - gazex)) # top left
            leftTopCornerY = int(round(bigImg_centery - gazey))
            
        else:
            leftTopCornerX =0
            leftTopCornerY =0
            frame = np.zeros((int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])))
            #add border to frame
            ## show image in the gaze-center coordinate system
        finalImgShape = np.zeros(((int(np.ceil(np.shape(frame)[0]-1)*self.A+1)), int(np.ceil((np.shape(frame)[1]-1)*self.A+1)),np.shape(frame)[2]),dtype = 'float32').shape
        
    
        #frame = cv2.copyMakeBorder(frame, 5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255,255])
        for i in range(3):
            bigImg[int(leftTopCornerY):int(leftTopCornerY+video_y), int(leftTopCornerX):int(leftTopCornerX+video_x),i] = frame[:,:,i]
        finalImg = bigImg[(bigImg_centery-int(np.floor(finalImgShape[0]/2))):(bigImg_centery+int(np.floor(finalImgShape[0]/2))+1),(bigImg_centerx-int(np.floor(finalImgShape[1]/2))):(bigImg_centerx+int(np.floor(finalImgShape[1]/2))+1),:]
    
        #finalImgShow = finalImg*255
        finalImgShow = finalImg[:,:,::-1]
        image = axes[1].imshow(finalImgShow.astype(np.uint8))#,origin='upper')#, extent=[leftTopCornerX, leftTopCornerX + video_x,leftTopCornerY + video_y,leftTopCornerY ])
            #ax[0].figure.figimage(frame[:,:,::-1],10,10, alpha=0.5)
    
    
        axes[1].set_xlim([0,finalImg.shape[1]])
        axes[1].set_ylim([finalImg.shape[0],0])
        axes[1].spines[:].set_visible(False)
        center, = axes[1].plot((finalImg.shape[1]-1)/2,(finalImg.shape[0]-1)/2, 'b', marker = 'P', markersize = 15),
        #axes[1].set_title(f"Subject: {self.subjectName}\nMovie: {self.movieName}",fontsize= fontsize, loc = "left")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        # create legend for marker
        gaze_marker = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                  markersize=10, label='Gaze position')
        center_marker = mlines.Line2D([], [], color='blue', marker='P', linestyle='None',
                                  markersize=10, label='Center of the new coordinate system')
        axes[1].legend(handles=[gaze_marker, center_marker],bbox_to_anchor=(1.6, 1), frameon = False)
        #
        #fig.text(s =f"Subject: {self.subjectName}\nMovie: {self.movieName}", x= 0.7,y =0.8,fontsize= fontsize)
    
        # plot gaze trace
        frameAxe = np.arange(self.skipNFirstFrame, len(sampledTimeStamps))*(1/self.sampledFps)
        l_gazex, =axes[2].plot(frameAxe,sampledgazexData, color = "grey",  label ="gazex", linewidth = linesize)
        l_gazey, = axes[2].plot(frameAxe,sampledgazeyData, color = "black", label ="gazey", linewidth = linesize)
        axes[2].set_xlim([0,sampledTimeStamps[-1]])
        axes[2].set_title("Gaze position")
        #axes[2].legend(bbox_to_anchor=(1.01, 0.6), frameon = False)
        axes[2].spines["right"].set_visible(False)
        axes[2].spines["top"].set_visible(False)
        axes[2].spines["left"].set_linewidth(linesize -1)
        axes[2].spines["bottom"].set_linewidth(linesize -1)
        axes[2].legend(loc = "upper center", ncol = 2,frameon = False)
        # luminance and contrast
    
        l_lum, = axes[3].plot(frameAxe,lumData, color = "#4c004c",alpha=1, label = "luminance", linewidth = linesize)
        l_contrast, = axes[3].plot(frameAxe,contrastData,color = "green",alpha=1, label = "contrast", linewidth = linesize)
        #l_color, = ax[2].plot(frameAxe,labData, color = "green", label = "color")
        axes[3].set_ylim([-4,4])
        axes[3].set_xlim([0,sampledTimeStamps[-1]])
        axes[3].set_title("Visual event")
        axes[3].legend().set_visible(False)
        axes[3].spines["right"].set_visible(False)
        axes[3].spines["top"].set_visible(False)
        axes[3].spines["left"].set_linewidth(linesize -1)
        axes[3].spines["bottom"].set_linewidth(linesize -1)
    
        lines_feature = [l_lum, l_contrast]
    
        # plot the model (actual and prediction)
        l_actualPupil, = axes[4].plot(frameAxe, sampledpupilData, color = "grey", label = "actual pupil", linewidth = linesize)
        l_predictedPupil, = axes[4].plot(frameAxe, y_pred, color = "#744700", label = "predicted pupil", linewidth = linesize)
        l_predictedPupilLum, = axes[4].plot(frameAxe, lumConv, color = "#4c004c", label = "predicted pupil luminance", linewidth = linesize)
        l_predictedPupilContrast, = axes[4].plot(frameAxe, contrastConv, color = "green", label = "predicted pupil contrast", linewidth = linesize)
        l_predictedPupilLum.set_visible(False)
        l_predictedPupilContrast.set_visible(False)
    
        axes[4].set_xlim([0,sampledTimeStamps[-1]])
        axes[4].legend().set_visible(False)
        axes[4].spines["right"].set_visible(False)
        axes[4].spines["top"].set_visible(False)
        axes[4].spines["left"].set_linewidth(linesize -1)
        axes[4].spines["bottom"].set_linewidth(linesize -1)
        axes[4].set_title("Model prediction")
        lines_model = [l_actualPupil,l_predictedPupil,l_predictedPupilLum,l_predictedPupilContrast]
        #
        # ax[4].set_ylim([-2,2])
        # ax[4].set_xlim([0,len(timeStamps)])
        # ax[4].set_title("pupil real")
        # plot the magnPerIm
        # l_lumDark_magnPerIm, = ax[4].plot(frameAxe,lumData_ss, color = "orange",alpha=1, label = "lumDark")
    
        # #ax[4].set_ylim([-5,5])
        # ax[4].set_xlim([0,len(timeStamps)])
        # ax[4].set_title("SteadyState feature change (gaze-centered)")
    
        # add marker of frame and legend
        marker1 = axes[2].axvline(x = init_frame* (1/self.sampledFps), linestyle = "dashed", linewidth = linesize, color = "red")
        #ax[1].legend(bbox_to_anchor=(1, 1.05))
        marker2 = axes[3].axvline(x = init_frame* (1/self.sampledFps), linestyle = "dashed", linewidth = linesize, color = "red")
        #ax[2].legend(bbox_to_anchor=(1, 1.05))
        marker3 = axes[4].axvline(x = init_frame* (1/self.sampledFps), linestyle = "dashed", linewidth = linesize, color = "red")
        #f.tight_layout(pad = 0.5)
        plt.margins(x=0)
        chart = FigureCanvasTkAgg(f, self.top_interactive_figure)
        chart.get_tk_widget().grid(row = 0, column = 0,sticky=tk.N+tk.S+tk.E+tk.W, rowspan =9, padx = (0,5))
        #buttunFrame = tk.Frame(self.top_interactive_figure, bg = "red").grid(row = 0, column =5, rowspan = 2)
        # add buttoms
        def buttomFunc_updateFig1():
            var_lum_value = var_lum.get()
            var_contrast_value = var_contrast.get()
    
            if var_lum_value == 1:
                l_lum.set_visible(True)
            else: 
                l_lum.set_visible(False)
            if var_contrast_value == 1:
                l_contrast.set_visible(True)
            else: 
                l_contrast.set_visible(False)   
            plt.draw()
        def buttomFunc_updateFig2():
    
            var_actual_value = var_actual.get()
            var_predict_value = var_predict.get()
            var_lumConv_value = var_lumConv.get()
            var_contrastConv_value = var_contrastConv.get()
    
            if var_actual_value == 1:
                l_actualPupil.set_visible(True)
            else: 
                l_actualPupil.set_visible(False)
            if var_predict_value == 1:
                l_predictedPupil.set_visible(True)
            else: 
                l_predictedPupil.set_visible(False)  
            if var_lumConv_value == 1:
                l_predictedPupilLum.set_visible(True)
            else: 
                l_predictedPupilLum.set_visible(False)
            if var_contrastConv_value == 1:
                l_predictedPupilContrast.set_visible(True)
            else: 
                l_predictedPupilContrast.set_visible(False) 
            plt.draw()
        # luminance and contrast
        print(self.filename_movie)
        print(self.subjectName)
        
    
        var_lum = tk.IntVar()
        var_lum.set(1)
        tk.Checkbutton(self.top_interactive_figure, text="luminance", variable=var_lum, fg = "#4c004c").grid(row=0, column = 1, sticky=tk.W+tk.S)
        var_contrast = tk.IntVar()
        var_contrast.set(1)
    
        tk.Checkbutton(self.top_interactive_figure, text="contrast", variable=var_contrast, fg = "green").grid(row=1, column = 1, sticky=tk.W)
        # model
        var_actual = tk.IntVar()
        var_actual.set(1)
    
        tk.Checkbutton(self.top_interactive_figure, text="Actual pupil", variable=var_actual, fg = "grey").grid(row=3, column = 1, sticky=tk.W, pady = (30,0), columnspan = 2)
        var_predict = tk.IntVar()
        var_predict.set(1)
    
        tk.Checkbutton(self.top_interactive_figure, text="Predicted(luminance+contrast)", variable=var_predict, fg = "#744700").grid(row=4, column = 1, columnspan = 2, sticky=tk.W)
        var_lumConv = tk.IntVar()
        var_lumConv.set(0)
    
        tk.Checkbutton(self.top_interactive_figure, text="Predicted(luminance)", fg = "#4c004c", variable=var_lumConv).grid(row=5, column = 1, sticky=tk.W, columnspan = 2)
        var_contrastConv = tk.IntVar()
        var_contrastConv.set(0)
    
        tk.Checkbutton(self.top_interactive_figure, text="Predicted(contrast)", fg = "green", variable=var_contrastConv).grid(row=6, column = 1, sticky=tk.W, columnspan = 2)
        
        tk.Button(self.top_interactive_figure, text='Update figure', command=buttomFunc_updateFig1).grid(row=2, column =1, sticky=tk.W+tk.E,columnspan = 2)
    
        tk.Button(self.top_interactive_figure, text='Update figure', command=buttomFunc_updateFig2).grid(row=7, column =1, sticky=tk.W+tk.E,columnspan = 2)
        
        def callback_figure():
            choice = choice_figure.get()
            print(choice)
            return choice
        def buttomFunc_saveFig():
            choice = callback_figure()
            foldername = "Figures"
            os.chdir(self.dataDir) 
            if not os.path.exists(foldername):
               os.makedirs(foldername)
            os.chdir(foldername)
            # save figures
            if choice == "Save gaze":
                extent = axes[2].get_window_extent().transformed(f.dpi_scale_trans.inverted())
                f.savefig(f'{self.movieName}_{self.subjectName}_Fig_gaze.png', bbox_inches=extent.expanded(1.1, 1.3))
                showinfo(title='',message=f'{self.subjectName}_Fig_gaze.png is saved')
            elif choice == 'Save visual events':
                extent = axes[3].get_window_extent().transformed(f.dpi_scale_trans.inverted())
                f.savefig(f'{self.movieName}_{self.subjectName}_Fig_feature.png', bbox_inches=extent.expanded(1.1, 1.3))
                showinfo(title='',message=f'{self.movieName}_{self.subjectName}_Fig_feature.png is saved')
    
            elif choice == "Save model prediction":
                extent = axes[4].get_window_extent().transformed(f.dpi_scale_trans.inverted())
                f.savefig(f'{self.movieName}_{self.subjectName}_Fig_model.png', bbox_inches=extent.expanded(1.1, 1.3))
                showinfo(title='',message=f'{self.movieName}_{self.subjectName}_Fig_model.png is saved')
    
            elif choice == "Save all":
                f.savefig(f'{self.movieName}_{self.subjectName}_Fig_all.png')
                showinfo(title='',message=f'{self.movieName}_{self.subjectName}_Fig_all.png is saved')
    
        # save figs
        choices= ("Save all",'Save gaze','Save visual events', "Save model prediction")
        
        choice_figure= tk.StringVar()
        choice_figure.set('Save all')
        choicebox_figure= ttk.Combobox(self.top_interactive_figure, textvariable= choice_figure, width = 6,font = ("Arial", 10))
        choicebox_figure['values']= choices
        choicebox_figure['state']= 'readonly'
        choicebox_figure.grid(column = 1, row = 9,sticky=tk.E+tk.W)
        #choice_figure.trace('w', callback_figure)        
        tk.Button(self.top_interactive_figure, text='Save fig', command = buttomFunc_saveFig).grid(row=9, column =2, sticky=tk.W+tk.E)
        
        tk.Button(self.top_interactive_figure, text='Back', command = self.close_top_figure).grid(row=10, column =1, sticky=tk.W+tk.S+tk.E)
        tk.Button(self.top_interactive_figure, text='Exit', command = self.close).grid(row=10, column =2, sticky=tk.W+tk.N+tk.S+tk.E)
    
        #tk.Button(self.top_interactive_figure, text='Save all fig', command=buttomFunc_saveFigAll).grid(row=9, sticky=tk.W)
        self.top_interactive_figure.rowconfigure(0, weight = 175)
        self.top_interactive_figure.rowconfigure(1, weight = 1)
        self.top_interactive_figure.rowconfigure(2, weight = 1)
        self.top_interactive_figure.rowconfigure(3, weight = 1)
        self.top_interactive_figure.rowconfigure(4, weight = 1)
        self.top_interactive_figure.rowconfigure(5, weight = 1)
        self.top_interactive_figure.rowconfigure(6, weight = 1)
        self.top_interactive_figure.rowconfigure(7, weight = 1)
        self.top_interactive_figure.rowconfigure(8, weight = 1)
        self.top_interactive_figure.rowconfigure(9, weight = 1)
        self.top_interactive_figure.rowconfigure(10, weight = 1)
    
                
        #scale.bind("<ButtonRelease-1>", updateValue)
        def updateValue(event):
            print(scale.get())
            frameNum_new = int(scale.get()/(1/self.sampledFps))
            marker1.set_xdata(scale.get())
            marker2.set_xdata(scale.get())
            marker3.set_xdata(scale.get())
            video.set(1,frameNum_new-1)
            ret, frame = video.read() 
            if self.videoScreenSameRatio and self.videoStretched:
                realImg = cv2.resize(frame, (int(self.videoRealWidth), int(self.videoRealHeight)))
                frame = realImg
            else: 
                screen = np.zeros((int(self.eyetracking_height), int(self.eyetracking_width),np.shape(frame)[2]),dtype = 'float32')
                # fill the screen with the color
                colorBgRGB = np.array([self.screenBgColorR,self.screenBgColorG,self.screenBgColorB])
                for i in range(int(self.eyetracking_height)):
                    for j in range(int(self.eyetracking_width)):
                        screen[i,j,:] = colorBgRGB
                realImg = cv2.resize(frame, (int(self.videoRealWidth), int(self.videoRealHeight))) # resize is (width, height)
                # put the realImg to the screen
                width_diff = screen.shape[1] - realImg.shape[1]
                height_diff = screen.shape[0] - realImg.shape[0]
                screen[int(np.ceil(height_diff/2)):(int(np.ceil(height_diff/2)) + realImg.shape[0]),int(np.ceil(width_diff/2)):(int(np.ceil(width_diff/2)) + realImg.shape[1]),:] = realImg
                frame = screen
            gazex = sampledgazexData[frameNum_new]
            gazey = sampledgazeyData[frameNum_new]
            print(f"gazex:{gazex}; gazey:{gazey}")
            print(f"sum of frame : {np.sum(frame)}")
            # update gaze-contingent coordinate system
            bigImg = np.zeros(((np.shape(frame)[0]-1)*4+1, (np.shape(frame)[1]-1)*4+1,np.shape(frame)[2]),dtype = 'float32')
            bigImg_centerx = int((bigImg.shape[1] -1 )/2) # This will be the new center of the data
            bigImg_centery = int((bigImg.shape[0] -1 )/2)
            if not np.isnan(gazex) and not np.isnan(gazey):
                leftTopCornerX = int(round(bigImg_centerx - gazex)) # top left
                leftTopCornerY = int(round(bigImg_centery - gazey))
                
            else:
                leftTopCornerX =0
                leftTopCornerY =0
                frame = np.zeros((int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])))
            
                #add border to frame
                ## show image in the gaze-center coordinate system
            #frame = cv2.copyMakeBorder(frame, 5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255,255])
            print(f"sum of frame : {np.sum(frame)}")
            if not np.isnan(gazex) and not np.isnan(gazey):
                leftTopCornerX = int(round(bigImg_centerx - gazex)) # top left
                leftTopCornerY = int(round(bigImg_centery - gazey))
                
            else:
                leftTopCornerX =0
                leftTopCornerY =0
                frame = np.zeros((int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])))
                #add border to frame
                ## show image in the gaze-center coordinate system
            finalImgShape = np.zeros(((int(np.ceil(np.shape(frame)[0]-1)*self.A+1)), int(np.ceil((np.shape(frame)[1]-1)*self.A+1)),np.shape(frame)[2]),dtype = 'float32').shape
            bigImg = np.zeros(((np.shape(frame)[0]-1)*4+1, (np.shape(frame)[1]-1)*4+1,np.shape(frame)[2]),dtype = 'float32')
            #bigImg[:] = np.nan
            bigImg_centerx = int((bigImg.shape[1] -1 )/2) # This will be the new center of the data
            bigImg_centery = int((bigImg.shape[0] -1 )/2)
    
            #frame = cv2.copyMakeBorder(frame, 5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255,255])
            for i in range(3):
                bigImg[int(leftTopCornerY):int(leftTopCornerY+video_y), int(leftTopCornerX):int(leftTopCornerX+video_x),i] = frame[:,:,i]
            finalImg = bigImg[(bigImg_centery-int(np.floor(finalImgShape[0]/2))):(bigImg_centery+int(np.floor(finalImgShape[0]/2))+1),(bigImg_centerx-int(np.floor(finalImgShape[1]/2))):(bigImg_centerx+int(np.floor(finalImgShape[1]/2))+1),:]
    
            finalImgShow = finalImg*255
            finalImgShow = finalImg[:,:,::-1]
            image.set_data(finalImgShow.astype(np.uint8))#,origin='upper')#, extent=[leftTopCornerX, leftTopCornerX + video_x,leftTopCornerY + video_y,leftTopCornerY ])
            original_movie.set_data(realImg[:,:,::-1])
            
            if np.isnan(gazex) or np.isnan(gazey):
                gaze.set_color('#FF000000')
            else:
                gaze.set_color('red')
    
                gaze.set_xdata([sampledgazexData[frameNum_new]])
                gaze.set_ydata([sampledgazeyData[frameNum_new]])
            plt.draw()
            f.canvas.draw_idle()
            #f.canvas.flush_events()
    
         # add slider
        scalevar = tk.IntVar()
        scale = tk.Scale(self.top_interactive_figure, from_=0, to=sampledTimeStamps[-1],length = len(sampledTimeStamps)+1, tickinterval=100, variable=scalevar, orient=tk.HORIZONTAL,resolution = 0.01)#, command=self.updateValue)
        scale.grid(row = 9, column = 0, sticky = tk.W + tk.E + tk.N+tk.S, padx=(5,0))
        scale.bind("<ButtonRelease-1>", updateValue)
        tk.Label(self.top_interactive_figure, text = "Time (s)", font = ("Arial", 10), bg = "#d9d9d9").grid(row = 10, column = 0)
    
        self.top_interactive_figure.columnconfigure(0, weight = 60)
        self.top_interactive_figure.columnconfigure(1, weight = 1)
        self.top_interactive_figure.columnconfigure(2, weight = 1)
        self.window.mainloop()

    def plot_NoEyetracking(self):
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

        fontsize = 15
        linesize = 3
        foldername = "Modeling result"
        os.chdir(self.dataDir) 
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
        self.modelResultDict = modelResultDict
        self.modelDataDict= modelDataDict
        
        
        lumData = self.modelDataDict[self.subjectName]["lumData"] 
        contrastData = self.modelDataDict[self.subjectName]["contrastData"] 
        y_pred = self.modelResultDict[self.subjectName]["modelContrast"]["predAll"] 
        lumConv = self.modelResultDict[self.subjectName]["modelContrast"]["lumConv"] 
        contrastConv = self.modelResultDict[self.subjectName]["modelContrast"]["contrastConv"] 
        timeStamps = self.modelDataDict[self.subjectName]["timeStamps"]
        # tkinter grid
        # widget_list = self.all_children()
        # for item in widget_list:
        #     item.destroy()
        if self.useApp:
            pass
        else:
            # if not using app, need to create a window first
            self.window = tk.Tk()        
        self.top_interactive_figure = tk.Toplevel(self.window)
        self.top_interactive_figure.geometry("2000x1000")
        self.top_interactive_figure.resizable(True, True)
        
    
        init_frame = 3
        # Create the figure and the line that we will manipulate
        os.chdir(self.dataDir)
        video = cv2.VideoCapture(self.filename_movie)
        # plot the new coordinate system
        video.set(1,init_frame); # set up initail frame
        ret, frame = video.read() 
        
            
        video_x = self.video_width
        video_y = self.video_height
        
        
        f = plt.figure(figsize = (17,8),constrained_layout = True)
        gs = f.add_gridspec(ncols = 16,nrows = 20)
        axes = list()
        axes.append(f.add_subplot(gs[0:8, 0:16]))
        axes.append(f.add_subplot(gs[0:8, 15:16]))
        axes.append(f.add_subplot(gs[9:12,0:16]))
    
        axes.append(f.add_subplot(gs[13:16, 0:16]))
        axes.append(f.add_subplot(gs[17:20, 0:16]))
        #plt.subplots_adjust(right =20)
        # add movie and gaze to the left
       
        axes[0].set_title("Movie", fontsize = fontsize, loc = "left")
        axes[0].spines[:].set_visible(False)
        original_movie = axes[0].imshow(frame[:,:,::-1])
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["left"].set_visible(False)
        axes[0].spines["bottom"].set_visible(False)
        #fig.text(s =f"Subject: {self.subjectName}\nMovie: {self.movieName}", x= 0.7,y =0.8,fontsize= fontsize)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["left"].set_visible(False)
        axes[1].spines["bottom"].set_visible(False)
        # plot gaze trace
        frameAxe = timeStamps
        axes[2].set_title("Gaze position")
        #axes[2].legend(bbox_to_anchor=(1.01, 0.6), frameon = False)
        axes[2].text(s = "No eyetracking data!", x = 0.47, y = 0.5, fontsize = fontsize+2)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        # luminance and contrast
    
        l_lum, = axes[3].plot(frameAxe,lumData, color = "#4c004c",alpha=1, label = "luminance", linewidth = linesize)
        l_contrast, = axes[3].plot(frameAxe,contrastData,color = "green",alpha=1, label = "contrast", linewidth = linesize)
        #l_color, = ax[2].plot(frameAxe,labData, color = "green", label = "color")
        axes[3].set_ylim([-4,4])
        axes[3].set_xlim([0,timeStamps[-1]])
        axes[3].set_title("Visual event")
        axes[3].legend().set_visible(False)
        axes[3].spines["right"].set_visible(False)
        axes[3].spines["top"].set_visible(False)
        axes[3].spines["left"].set_linewidth(linesize -1)
        axes[3].spines["bottom"].set_linewidth(linesize -1)
    
        lines_feature = [l_lum, l_contrast]
    
        # plot the model (actual and prediction)
        #l_actualPupil, = axes[4].plot(frameAxe, sampledpupilData, color = "grey", label = "actual pupil", linewidth = linesize)
        l_predictedPupil, = axes[4].plot(frameAxe, y_pred, color = "#744700", label = "predicted pupil", linewidth = linesize)
        l_predictedPupilLum, = axes[4].plot(frameAxe, lumConv, color = "#4c004c", label = "predicted pupil luminance", linewidth = linesize)
        l_predictedPupilContrast, = axes[4].plot(frameAxe, contrastConv, color = "green", label = "predicted pupil contrast", linewidth = linesize)
        l_predictedPupilLum.set_visible(False)
        l_predictedPupilContrast.set_visible(False)
    
        axes[4].set_xlim([0,timeStamps[-1]])
        axes[4].legend().set_visible(False)
        axes[4].spines["right"].set_visible(False)
        axes[4].spines["top"].set_visible(False)
        axes[4].spines["left"].set_linewidth(linesize -1)
        axes[4].spines["bottom"].set_linewidth(linesize -1)
        axes[4].set_title("Model prediction")
        lines_model = [l_predictedPupil,l_predictedPupilLum,l_predictedPupilContrast]
        #
        # ax[4].set_ylim([-2,2])
        # ax[4].set_xlim([0,len(timeStamps)])
        # ax[4].set_title("pupil real")
        # plot the magnPerIm
        # l_lumDark_magnPerIm, = ax[4].plot(frameAxe,lumData_ss, color = "orange",alpha=1, label = "lumDark")
    
        # #ax[4].set_ylim([-5,5])
        # ax[4].set_xlim([0,len(timeStamps)])
        # ax[4].set_title("SteadyState feature change (gaze-centered)")
    
        # add marker of frame and legend
        #marker1 = axes[2].axvline(x = init_frame, linestyle = "dashed", linewidth = linesize, color = "red")
        #ax[1].legend(bbox_to_anchor=(1, 1.05))
        marker2 = axes[3].axvline(x = init_frame* (1/self.sampledFps), linestyle = "dashed", linewidth = linesize, color = "red")
        #ax[2].legend(bbox_to_anchor=(1, 1.05))
        marker3 = axes[4].axvline(x = init_frame* (1/self.sampledFps), linestyle = "dashed", linewidth = linesize, color = "red")
        #f.tight_layout(pad = 0.5)
        plt.margins(x=0)
        chart = FigureCanvasTkAgg(f, self.top_interactive_figure)
        chart.get_tk_widget().grid(row = 0, column = 0,sticky=tk.N+tk.S+tk.E+tk.W, rowspan =9, padx = (0,5))
        #buttunFrame = tk.Frame(self.top_interactive_figure, bg = "red").grid(row = 0, column =5, rowspan = 2)
        # add buttoms
        def buttomFunc_updateFig1():
            var_lum_value = var_lum.get()
            var_contrast_value = var_contrast.get()
    
            if var_lum_value == 1:
                l_lum.set_visible(True)
            else: 
                l_lum.set_visible(False)
            if var_contrast_value == 1:
                l_contrast.set_visible(True)
            else: 
                l_contrast.set_visible(False)   
            plt.draw()
        def buttomFunc_updateFig2():
    
            #var_actual_value = var_actual.get()
            var_predict_value = var_predict.get()
            var_lumConv_value = var_lumConv.get()
            var_contrastConv_value = var_contrastConv.get()
    
            
            if var_predict_value == 1:
                l_predictedPupil.set_visible(True)
            else: 
                l_predictedPupil.set_visible(False)  
            if var_lumConv_value == 1:
                l_predictedPupilLum.set_visible(True)
            else: 
                l_predictedPupilLum.set_visible(False)
            if var_contrastConv_value == 1:
                l_predictedPupilContrast.set_visible(True)
            else: 
                l_predictedPupilContrast.set_visible(False) 
            plt.draw()
        # luminance and contrast
        print(self.filename_movie)
        print(self.subjectName)
        
    
        var_lum = tk.IntVar()
        var_lum.set(1)
        tk.Checkbutton(self.top_interactive_figure, text="luminance", variable=var_lum, fg = "#4c004c").grid(row=0, column = 1, sticky=tk.W+tk.S)
        var_contrast = tk.IntVar()
        var_contrast.set(1)
    
        tk.Checkbutton(self.top_interactive_figure, text="contrast", variable=var_contrast, fg = "green").grid(row=1, column = 1, sticky=tk.W)
        # model
        var_predict = tk.IntVar()
        var_predict.set(1)
    
        tk.Checkbutton(self.top_interactive_figure, text="Predicted(luminance+contrast)", variable=var_predict, fg = "#744700").grid(row=4, column = 1, columnspan = 2, sticky=tk.W)
        var_lumConv = tk.IntVar()
        var_lumConv.set(0)
    
        tk.Checkbutton(self.top_interactive_figure, text="Predicted(luminance)", fg = "#4c004c", variable=var_lumConv).grid(row=5, column = 1, sticky=tk.W, columnspan = 2)
        var_contrastConv = tk.IntVar()
        var_contrastConv.set(0)
    
        tk.Checkbutton(self.top_interactive_figure, text="Predicted(contrast)", fg = "green", variable=var_contrastConv).grid(row=6, column = 1, sticky=tk.W, columnspan = 2)
        
        tk.Button(self.top_interactive_figure, text='Update figure', command=buttomFunc_updateFig1).grid(row=2, column =1, sticky=tk.W+tk.E,columnspan = 2)
    
        tk.Button(self.top_interactive_figure, text='Update figure', command=buttomFunc_updateFig2).grid(row=7, column =1, sticky=tk.W+tk.E,columnspan = 2)
        
        def callback_figure():
            choice = choice_figure.get()
            print(choice)
            return choice
        def buttomFunc_saveFig():
            choice = callback_figure()
            foldername = "Figures"
            os.chdir(self.dataDir) 
            if not os.path.exists(foldername):
               os.makedirs(foldername)
            os.chdir(foldername)
            # save figures
            if choice == 'Save visual events':
                extent = axes[3].get_window_extent().transformed(f.dpi_scale_trans.inverted())
                f.savefig(f'{self.movieName}_{self.subjectName}_Fig_feature.png', bbox_inches=extent.expanded(1.1, 1.3))
                showinfo(title='',message=f'{self.movieName}_{self.subjectName}_Fig_feature.png is saved')
    
            elif choice == "Save model prediction":
                extent = axes[4].get_window_extent().transformed(f.dpi_scale_trans.inverted())
                f.savefig(f'{self.movieName}_{self.subjectName}_Fig_model.png', bbox_inches=extent.expanded(1.1, 1.3))
                showinfo(title='',message=f'{self.movieName}_{self.subjectName}_Fig_model.png is saved')
    
            elif choice == "Save all":
                f.savefig(f'{self.movieName}_{self.subjectName}_Fig_all.png')
                showinfo(title='',message=f'{self.movieName}_{self.subjectName}_Fig_all.png is saved')
    
        # save figs
        choices= ("Save all",'Save visual events', "Save model prediction")
        
        choice_figure= tk.StringVar()
        choice_figure.set('Save all')
        choicebox_figure= ttk.Combobox(self.top_interactive_figure, textvariable= choice_figure, width = 6,font = ("Arial", 10))
        choicebox_figure['values']= choices
        choicebox_figure['state']= 'readonly'
        choicebox_figure.grid(column = 1, row = 9,sticky=tk.E+tk.W)
        #choice_figure.trace('w', callback_figure)        
        tk.Button(self.top_interactive_figure, text='Save fig', command = buttomFunc_saveFig).grid(row=9, column =2, sticky=tk.W+tk.E)
        
        tk.Button(self.top_interactive_figure, text='Back', command = self.close_top_figure).grid(row=10, column =1, sticky=tk.W+tk.S+tk.E)
        tk.Button(self.top_interactive_figure, text='Exit', command = self.close).grid(row=10, column =2, sticky=tk.W+tk.N+tk.S+tk.E)
    
        #tk.Button(self.top_interactive_figure, text='Save all fig', command=buttomFunc_saveFigAll).grid(row=9, sticky=tk.W)
        self.top_interactive_figure.rowconfigure(0, weight = 175)
        self.top_interactive_figure.rowconfigure(1, weight = 1)
        self.top_interactive_figure.rowconfigure(2, weight = 1)
        self.top_interactive_figure.rowconfigure(3, weight = 1)
        self.top_interactive_figure.rowconfigure(4, weight = 1)
        self.top_interactive_figure.rowconfigure(5, weight = 1)
        self.top_interactive_figure.rowconfigure(6, weight = 1)
        self.top_interactive_figure.rowconfigure(7, weight = 1)
        self.top_interactive_figure.rowconfigure(8, weight = 1)
        self.top_interactive_figure.rowconfigure(9, weight = 1)
        self.top_interactive_figure.rowconfigure(10, weight = 2)
    
                
        #scale.bind("<ButtonRelease-1>", updateValue)
        def updateValue(event):
            print(scale.get())
            frameNum_new = int(scale.get()/(1/self.sampledFps))
            marker2.set_xdata(scale.get())
            marker3.set_xdata(scale.get())
            video.set(1,frameNum_new-1)
            ret, frame = video.read() 
            
            original_movie.set_data(frame[:,:,::-1])
            
        
            plt.draw()
            f.canvas.draw_idle()
            #f.canvas.flush_events()
    
         # add slider
        scalevar = tk.IntVar()
        scale = tk.Scale(self.top_interactive_figure, from_=0, to=timeStamps[-1], length = len(timeStamps)+1,tickinterval=100, variable=scalevar, resolution = 0.01,orient=tk.HORIZONTAL)#, command=self.updateValue)
        scale.grid(row = 9, column = 0, sticky = tk.W + tk.E + tk.N+tk.S, padx=(5,0))
        scale.bind("<ButtonRelease-1>", updateValue)
        tk.Label(self.top_interactive_figure,text = "Time (s)", font = ("Arial", 10)).grid(row = 10, column = 0, sticky = tk.N)
    
        self.top_interactive_figure.columnconfigure(0, weight = 60)
        self.top_interactive_figure.columnconfigure(1, weight = 1)
        self.top_interactive_figure.columnconfigure(2, weight = 1)
        self.window.mainloop()

    def close(self):
        self.window.destroy()
        self.window.quit()
        # python = sys.executable
        # os.execl(python, python, *sys.argv)
        #self.sys_exit()
        #self.window.mainloop()
    def close_top_figure(self):
        self.top_interactive_figure.destroy()