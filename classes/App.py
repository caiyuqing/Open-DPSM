# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:00:43 2023

@author: 7009291
"""
#import mttkinter as tk
#from mttkinter import mtTkinter
#from mttkinter import mtTkinter
import os

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import pandas as pd
import numpy as np
from classes.preprocessing import preprocessing
#from classes.video_processing import video_processing
#from classes.image_processing import image_processing
from classes.pupil_prediction import pupil_prediction
from classes.event_extraction import event_extraction
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

class tkfunctions:
    def __init__(self):
        #############################################################
        # do not change the following except you are very sure of what you are doing
        # boolean indicating whether or not to show frame-by-frame of each video with analysis results
        self.showVideoFrames = False

        # boolean indicating whether or not to skip feature extraction of already analyzed videos.
        # If skipped, then video information is loaded (from pickle file with same name as video).
        # If not skipped, then existing pickle files (one per video) are overwritten.
        self.skipAlrAnFiles = True

        # array with multiple levels [2, 4, 8], with each number indicating the number of vertical image parts (number of horizontal parts will be in ratio with vertical); the more levels, the longer the analysis
        self.nVertMatPartsPerLevel = [3, 6]  # [4, 8, 16, 32]
        self.aspectRatio = 0.75
        self.imageSector = "6x8" # number of visual field regions (used for naming the new visual feature)
        # integer indicating number of subsequent frames to calculate the change in features at an image part.
        # it is recommended to set this number such that 100ms is between the compared frames.
        # e.g. for a video with 24fps (50ms between subsequent frames), the variable should be set at 2.
        self.nFramesSeqImageDiff = 2

        # string indicating color space in which a 3D vector is calculated between frames
        # Options: 'RGB', 'LAB', 'HSV', 'HLS', 'LUV'
        self.colorSpace = "LAB"

        # list with strings indicating which features to analyze
        # If colorSpace = 'LAB', options: ["Luminance","Red-Green","Blue-Yellow","Lum-RG","Lum-BY","Hue-Sat","LAB"]
        # If colorSpace = 'HSV', options: ["Hue","Saturation","Luminance","Hue-Sat","Hue-Lum","Sat-Lum","HSV"]
        self.featuresOfInterest = [
            "Luminance"]#,
        #     "Red-Green",
        #     "Blue-Yellow",
        #     "Lum-RG",
        #     "Lum-BY",
        #     "Hue-Sat",
        #     "LAB",
        # ]
        
        # monitor gamma factor
        # for conversion from pixel values to luminance values in cd/m2
        # Set to zero for no conversion
        self.scrGamFac = 2.2
        # What is the ratio between gaze-centered coordinate system and the screen
        self.A = 2
        # Number of movie frames skipped at the beginning of the movie
        self.skipNFirstFrame = 0
        # gaze-contingent
        self.gazecentered = True
        ############pupil prediction parameters##############
        # Response function type
        self.RF = "HL"
        # same regional weights for luminance and contrast
        self.sameWeightFeature = True 
        # Basinhopping or minimizing
        self.useBH = True
        # iteration number for basinhopping
        self.niter = 5
        # use Application instead of the code mode
        self.useApp = True
        
    
    def run_tk(self):
        # create the root window
        self.window = tk.Tk()
        self.window.title('Open DPSM')
        self.window.resizable(False, False)
        self.window.geometry('1000x600')
        self.window.configure(bg='#d9d9d9')

        # set up canvas for welcom message
        top_canvas =  tk.Canvas(self.window, width=800, height=200, bg='#d9d9d9',highlightthickness=0)
        top_canvas.grid(row=0, column=0, padx=100, pady=5)

        # print logo and welcome message
        logo = Image.open("App_fig\\DPSM logo.jpg")
        logo = logo.resize((200,100))
        logo_image = ImageTk.PhotoImage(logo, master=self.window)
        image_logo = top_canvas.create_image(300, 10, anchor="nw", image=logo_image)
        text_welcom = top_canvas.create_text(400,140, text="Welcome to Open DPSM!",anchor = "center", fill="black", font=('Helvetica 20 bold'))
        text_welcom2 = top_canvas.create_text(400,180, text="Please start by loading eyetracking and movie data:",anchor = "center", fill="black", font=('Helvetica 15 bold'))
        # set up canvas for welcom message
        middle_canvas =  tk.Canvas(self.window, width=800, height=150, bg='#d9d9d9',highlightthickness=0)
        middle_canvas.grid(row=1, column=0, padx=100, pady=5)

        # text insert
        self.var_label_csv = tk.StringVar()
        self.var_label_movie = tk.StringVar()

        self.label_csv = tk.Label(middle_canvas, textvariable=self.var_label_csv, width = 50).grid(column = 1, row = 1, sticky = "nsew")
        self.label_movie = tk.Label(middle_canvas,textvariable=self.var_label_movie, width = 50).grid(column =1, row = 2, sticky = "nsew")
        #self.text_csv = tk.Text(self.window, height = 1)
        #self.text_movie = tk.Text(self.window, height = 1)
        # open button
        button_openfile_csv = ttk.Button(middle_canvas,text='Open eyetracking data (.csv)*',command=self.select_file_csv)#threading.Thread(target=self.select_file_csv).start)
        button_openfile_movie = ttk.Button(middle_canvas,text='Open movie (.mp4,.avi,.mkv,.mwv,.mov,.flv,.webm)',command=self.select_file_movie)#threading.Thread(target=self.select_file_movie).start)
        # set up canvas for continue and exit
        self.bottom_canvas =  tk.Canvas(self.window, width=800, height=30, bg='#d9d9d9',highlightthickness=0)
        self.bottom_canvas.grid(row=2, column=0, padx=100, pady=5)
        #self.stop_event1 = threading.Event()
        self.t1 = threading.Thread(target=self.buttonFunc_check_data)
        self.t1.daemon = True
        #t1 = self.threadingFunc(function = self.buttonFunc_check_data)
        button_check_data = ttk.Button(self.bottom_canvas, text = "Continue", command=self.t1.start)#threading.Thread(target=self.buttonFunc_check_data).start)
        button_exit = ttk.Button(self.bottom_canvas,text = "Exit",command = self.close)
        
        button_openfile_csv.grid(column = 0, row = 1)
        button_openfile_movie.grid(column = 0, row = 2)
        button_check_data.grid(column = 0,row = 0)
        button_exit.grid(column = 0,row = 1)
        tk.Label(text = "*If no eye tracking data is loaded, model optimazation will not be preformed. Parameters used to generate predicted pupil trace will be the ones found by our study.",bg='#d9d9d9').grid(column = 0, row = 3, pady = 20)
        tk.Label(text = "Please cite: <citation here>").grid(column = 0, row = 5, pady = 170, sticky = tk.S)
        # run the application
        self.window.mainloop()
    def select_file_csv(self):
        filetypes = (
            ('csv files', '*.csv'),
            ('All files', '*.*')
        )
    
        filename_csv = fd.askopenfilename(
            title='Open a file',
            initialdir=self.dataDir,
            filetypes=filetypes)
        
        file = filename_csv.split("/")[-1]
        self.var_label_csv.set(file)
        self.filename_csv = filename_csv
        self.subjectName = file.split(".")[0]
        
    def select_file_movie(self):
        filetypes = (
            ('movie files', '*.mp4 *.avi *.mkv *.mwv *.mov *.flv *.webm'),
            ('All files', '*.*')
        )
    
        filename_movie = fd.askopenfilename(
            title='Open a file',
            initialdir=self.dataDir,
            filetypes=filetypes)
        # showinfo(
        #     title='Selected File',
        #     message=f"{filename_movie} is opened"
        # )
        file = filename_movie.split("/")[-1]
        self.var_label_movie.set(file)
        self.filename_movie = filename_movie
        self.movieName = file.split(".")[0]
    def buttonFunc_check_data(self):
        # preprocessing eyetracking data and movie data
        ## movie
        # stop_event = threading.Event()
        # stop_event.set()
        
        text_checkdata = tk.Label(self.bottom_canvas, text = "Checking data...Please wait",bg= "#d9d9d9").grid(column = 1, row = 0)
        # use class.preprocessing to check the movie and eyetracking data
        prepObj = preprocessing()
        prepObj.videoFileName = self.filename_movie
        prepObj.preprocessingVideo()
        self.video_nFrame = prepObj.vidInfo['frameN_end']
        self.video_height = prepObj.vidInfo['height']
        self.video_width = prepObj.vidInfo['width']
        self.video_ratio = self.video_height / self.video_width 
        self.video_duration = prepObj.vidInfo['duration_end']
        self.video_fps = prepObj.vidInfo['fps_end']
        # remove all the things from the previous window
        widget_list = self.all_children()
        for item in widget_list:
            item.destroy()
        ## eyetracking data
        
            # information about the eyetracking and movie data
        if hasattr(self, "filename_csv"):
            self.gazecentered  = True

            df_eyetracking = pd.read_csv(self.filename_csv, index_col=0, header = 0)
            # change the first timestamp to 0
            df_eyetracking.iloc[:,0] = df_eyetracking.iloc[:,0]-df_eyetracking.iloc[0,0]
            # extract eye-tracking information
            self.eyetracking_duration = df_eyetracking.iloc[-1,0]
            self.eyetracking_nSample = df_eyetracking.shape[0]
            self.eyetracking_samplingrate = int(1/(self.eyetracking_duration/self.eyetracking_nSample))
            self.df_eyetracking = df_eyetracking
            print(f"This video has {self.video_nFrame} frames")
            # check if the first column is increasing (as time should be increasomg)
            if not all(i < j for i, j in zip(np.array(self.df_eyetracking.iloc[:,0]).tolist(), np.array(self.df_eyetracking.iloc[:,0]).tolist()[1:])):
                self.open_error_popup(errormessage= "The first column of the eyetracking data is not the time. Please recheck!")
            # check if millisecond (open pop up window)
            if self.eyetracking_samplingrate < 30 or self.eyetracking_samplingrate >2000:
                self.top_milli = tk.Toplevel(self.window)
                self.top_milli.geometry("800x100")
                self.top_milli.title("Sample rate of eye tracking data")
                self.top_milli.configure(bg = "#d9d9d9")
                tk.Label(self.top_milli, text = "Standard timestamps should be in seconds. Now it seems that it is in milisecond. Please confirm whether it is in milisecond", font = ("Arial", 10), bg = "#d9d9d9").grid(column = 0, row = 0)
                self.top_milli.columnconfigure(0, weight=1)
                self.top_milli.rowconfigure(0, weight=1)
                self.top_milli.wm_transient(self.window)
                top_milli_canvas =  tk.Canvas(self.top_milli, width=800, height=100, bg='#d9d9d9',highlightthickness=0)
                top_milli_canvas.grid(row=1, column=0, padx=100, pady=5)
                # add button
                button_yes = ttk.Button(top_milli_canvas,text = "Yes, it is in milisecond", command = self.change_to_sec).grid(column = 0, row = 1)#threading.Thread(target=self.change_to_sec).start).grid(column = 0, row = 1)
                button_no = ttk.Button(top_milli_canvas,text = "No, it is in second", command = self.close_top_milli).grid(column = 1, row = 1)#threading.Thread(target=self.close_popup).start).grid(column = 1, row = 1)
                button_exit = ttk.Button(top_milli_canvas,text = "Exit", command = self.close).grid(column = 2, row = 1)
                self.top_milli.wait_window()
        

        # Check the information of eyetracking data and movie
        self.top_checkinfo = tk.Toplevel(self.window)
        self.top_checkinfo.geometry("800x300")
        self.top_checkinfo.title( "Check the information")
        self.top_checkinfo.configure(bg = "#d9d9d9")
        if not hasattr(self, "filename_csv"):
            tk.Label(self.top_checkinfo, text =  f"Please double-check:\nDuration of the video in seconds: {self.video_duration}\n Frame rate of the movie in Hz: {self.video_fps}\n Aspect ratio of the video is (x dimension divided by y dimension) {1/self.video_ratio}", bg = "#d9d9d9",font = ("Arial", 12)).grid(column = 0, row = 0)
        else:
            tk.Label(self.top_checkinfo, text =  f"Please double-check:\n \n Duration of eye tracking data in seconds: {self.eyetracking_duration}\n \nSampling rate of eyetracking data: {self.eyetracking_samplingrate}\n \nDuration of the video in seconds: {self.video_duration}\n \n Frame rate of the movie in Hz: {self.video_fps}\n \nAspect ratio of the video is (x dimension divided by y dimension) {1/self.video_ratio}", bg = "#d9d9d9", font = ("Arial", 12), height = 500).grid(column = 0, row = 0)
        self.top_checkinfo.columnconfigure(0, weight=1)
        self.top_checkinfo.rowconfigure(0, weight=1)
        self.top_checkinfo.wm_transient(self.window)

        #self.open_popup(title = "Check the information", message = )
        #information_check = tk.Label(self.window, text = f"Check if everything is correct:\nLength of eyetracking data: {self.eyetracking_duration}\nSampling rate of eyetracking data: {self.eyetracking_samplingrate}\nLength of the video: {self.video_duration}\n frame rate of the movie: {self.video_fps}\n Aspect ratio of the video is {self.video_ratio}")
        top_checkinfo_canvas =  tk.Canvas(self.top_checkinfo, width=800, height=100, bg='#d9d9d9',highlightthickness=0)
        top_checkinfo_canvas.grid(row=1, column=0, padx=100, pady=5)
        button_yes = ttk.Button(top_checkinfo_canvas,text = "Continue", command = self.close_top_checkinfo).grid(column = 0, row = 1)
        button_exit = ttk.Button(top_checkinfo_canvas,text = "Exit", command = self.close).grid(column = 2, row = 1)
        
        self.top_checkinfo.wait_window()
        #######################enter some information##############################
        
        def callback(*arg):
            self.ans_groundTruth = choice_gt.get()
            print(f"Ground truth: {self.ans_groundTruth} has been chosen")
            
        # instruction
        # ground truth of timing 
        if not hasattr(self, "filename_csv"):
            tk.Label(self.window, text = "More information needed:",font=('Arial', 20),bg = "#d9d9d9").grid(column = 0, row = 0,columnspan =2)
            tk.Label(self.window, text = u"What is the maximum luminance of the screen (i.e. measured physical lumiance (cd/m\u00b3) when color white is showed on screen)?",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 2, pady = 10)
            self.entry_maxLum = ttk.Entry(self.window)
            self.entry_maxLum.grid(column = 1, row = 3)
            
            self.gazecentered  = False
            self.nextButton = ttk.Button(self.window,text='Continue', command = self.buttonFunc_show_featureextraction)
            self.nextButton.grid(column = 1, row = 7)
            
            
        else:
            tk.Label(self.window, text = "More information needed:",font=('Arial', 20),bg = "#d9d9d9").grid(column = 0, row = 0,columnspan =2)

            if self.eyetracking_duration != self.video_duration:
                tk.Label(self.window, text = f"Eyetracking data and video data do not have the (exact) same duration (Movie: {self.video_duration}s, eye tracking: {self.eyetracking_duration}s).\nWhat do you want to do?",  font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 1, pady = 10)    
                
                choices= ('Stretch to match','cut last part of the longer file')
    
                choice_gt= tk.StringVar()
                choice_gt.set('Stretch to match')
                self.ans_groundTruth = "Stretch to match"
                choicebox_gt= ttk.Combobox(self.window, textvariable= choice_gt, width = 20,font = ("Arial", 10))
                choicebox_gt['values']= choices
                choicebox_gt['state']= 'readonly'
                choicebox_gt.grid(column = 1, row = 1, pady = 10)
                choice_gt.trace('w', callback)
            else:
                tk.Label(self.window, text = "Eyetracking data and video data have the same length.",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 1, pady = 10)    
            # Video maximum luminance
            tk.Label(self.window, text = u"What is the maximum luminance of the screen (i.e. measured physical lumiance (cd/m\u00b3) when color white is showed on screen)?",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 2, pady = 10)
            self.entry_maxLum = ttk.Entry(self.window)
            self.entry_maxLum.grid(column = 1, row = 3)
            # spatial resolution of eyetracking
            tk.Label(self.window, text = f"What is the resolution for the coordinate system of eye-tracking data (also the resolution of the screen)",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 4, pady = 10)
            
            eyetracking_width = tk.Label(self.window, text = "Width:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
            eyetracking_width.grid(column = 0, row = 5, pady = 0)
            eyetracking_height = tk.Label(self.window, text = "Height:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
            eyetracking_height.grid(column = 0, row = 6, pady = 0)
        
            self.entry_eyetrackingwidth = ttk.Entry(self.window)
            self.entry_eyetrackingwidth.insert(0, "1920") # this is our data
            self.entry_eyetrackingwidth.grid(column = 1, row = 5)
            self.entry_eyetrackingheight = ttk.Entry(self.window)
            self.entry_eyetrackingheight.insert(0, "1080") # this is our data
            self.entry_eyetrackingheight.grid(column = 1, row = 6)
            self.nextButton = ttk.Button(self.window,text='Continue', command = self.buttonFunc_ratioCheck)
            self.nextButton.grid(column = 1, row = 7)
        self.exitButton = ttk.Button(self.window,text='Exit', command = self.close)
        self.exitButton.grid(column = 1, row = 8)

        self.prepObj = prepObj
        
    def buttonFunc_ratioCheck(self):
        self.maxlum = float(self.entry_maxLum.get())
        self.eyetracking_height = float(self.entry_eyetrackingheight.get())
        self.eyetracking_width = float(self.entry_eyetrackingwidth.get())
        self.eyetracking_aspectRatio = self.eyetracking_height/self.eyetracking_width
        if self.video_ratio != self.eyetracking_aspectRatio:
            self.videoScreenSameRatio = False
            self.top_ratioCheck = tk.Toplevel(self.window)
            self.top_ratioCheck.geometry("800x300")
            self.top_ratioCheck.title( "Check the aspect ratio")
            self.top_ratioCheck.configure(bg = "#d9d9d9")
            self.top_ratioCheck.columnconfigure(0, weight=1)
            self.top_ratioCheck.columnconfigure(1, weight=1)

            tk.Label(self.top_ratioCheck, text =  f"The eyelink data and the video do not have the same aspect ratio.\nWhat does the video look like? (Please exit if the video looks like neither)",font = ("Arial", 15),  bg = "#d9d9d9").grid(column = 0, row = 0, columnspan = 2)
            
            #self.top_ratioCheck.rowconfigure(0, weight=1)
            self.top_ratioCheck.wm_transient(self.window)
            if self.video_ratio < self.eyetracking_aspectRatio:
                choiceA = Image.open("App_fig\\Screen_lower.jpg")
                choiceA = choiceA.resize((300,160))
                choiceA_image = ImageTk.PhotoImage(choiceA, master=self.window)
                choiceB = Image.open("App_fig\\Screen_surrounding_lower.jpg")
                choiceB = choiceB.resize((300,160))
                choiceB_image = ImageTk.PhotoImage(choiceB, master=self.window)
                tk.Label(self.top_ratioCheck, image = choiceA_image).grid(column = 0, row = 1,sticky = "we")
                tk.Label(self.top_ratioCheck, image = choiceB_image).grid(column = 1, row = 1,sticky = "we")
                ttk.Button(self.top_ratioCheck,text='Choose A',command = self.stretch).grid(column = 0, row = 2)
                ttk.Button(self.top_ratioCheck,text='Choose B', command = self.notStretch).grid(column = 1, row = 2)
            else:
                choiceA = Image.open("App_fig\\Screen_higher.jpg")
                choiceA = choiceA.resize((300,160))
                choiceA_image = ImageTk.PhotoImage(choiceA, master=self.window)
                choiceB = Image.open("App_fig\\Screen_surrounding_higher.jpg")
                choiceB = choiceB.resize((300,160))
                choiceB_image = ImageTk.PhotoImage(choiceB, master=self.window)
                tk.Label(self.top_ratioCheck, image = choiceA_image).grid(column = 0, row = 1)
                tk.Label(self.top_ratioCheck, image = choiceB_image).grid(column = 1, row = 1)
                ttk.Button(self.top_ratioCheck,text='Choose A', command = self.stretch).grid(column = 0, row = 2)
                ttk.Button(self.top_ratioCheck,text='Choose B', command = self.notStretch).grid(column = 1, row = 2)
            ttk.Button(self.top_ratioCheck,text='Exit', command = self.close).grid(column = 0, row = 3,columnspan = 2)
            self.top_ratioCheck.wait_window()
        else:
            self.videoScreenSameRatio = True
            self.top_ratioCheck = tk.Toplevel(self.window)
            self.top_ratioCheck.geometry("800x100")
            self.top_ratioCheck.title( "Check the aspect ratio")
            self.top_ratioCheck.configure(bg = "#d9d9d9")

            self.top_ratioCheck.columnconfigure(0, weight=1)
            self.top_ratioCheck.columnconfigure(1, weight=1)
            self.top_ratioCheck.wm_transient(self.window)
            tk.Label(self.top_ratioCheck, text =  f"The eyelink data and the video have the same aspect ratio.\nWas the video fullscreen*?",font = ("Arial", 15),  bg = "#d9d9d9").grid(column = 0, row = 0, columnspan = 3)
            tk.Label(self.top_ratioCheck, text = "*Fullscreen means that there was no verticle or horizontal bars surrounding the video on screen", font = ("Arial", 8), bg = "#d9d9d9").grid(column = 0, row = 1, columnspan = 3)
            ttk.Button(self.top_ratioCheck, text = "Yes", command = self.stretch).grid(column = 0, row = 2, sticky = tk.E)
            ttk.Button(self.top_ratioCheck, text = "No", command = self.notStretch).grid(column = 1, row = 2, sticky = tk.W)
            ttk.Button(self.top_ratioCheck, text = "Exit", command = self.close).grid(column = 2, row =2, sticky = tk.W+ tk.E)
    def buttonFunc_show_featureextraction(self):
        self.nextButton.grid_forget()
        self.exitButton.grid_forget()
        self.maxlum = float(self.entry_maxLum.get())
        if hasattr(self, "filename_csv"):
            self.eyetracking_height = float(self.entry_eyetrackingheight.get()) # start with 0
            self.eyetracking_width = float(self.entry_eyetrackingwidth.get()) # start with 0
            # if not stretch, need to extract the videoRealHeight and videoRealWidth from the entry
            
            
            if not self.videoStretched:
                if len(self.entry_videoheight.get()) == 0 or len(self.entry_videowidth.get()) == 0 or len(self.entry_colorR.get()) == 0 or len(self.entry_colorG.get()) == 0 or len(self.entry_colorB.get()) == 0:
                    showinfo(title = "Warning", 
                             message = "Please enter everything!")
                else:
                    self.videoRealHeight = float(self.entry_videoheight.get())
                    self.videoRealWidth = float(self.entry_videowidth.get())
            if self.videoStretched and self.videoScreenSameRatio:
                self.screenBgColorR = np.nan
                self.screenBgColorG = np.nan
                self.screenBgColorB = np.nan
            else:
                self.screenBgColorR = float(self.entry_colorR.get())
                self.screenBgColorG = float(self.entry_colorG.get())
                self.screenBgColorB = float(self.entry_colorB.get())
    
            
            print(f"eyetracking: {self.eyetracking_height} and {self.eyetracking_width}")
            self.aspectRatio_video_final = self.videoRealHeight/self.videoRealWidth
            if self.aspectRatio_video_final != self.video_ratio:
                showinfo(title = "Warning",
                         message = "Aspect ratio of the video is different from the video file. Please make sure you enter the correct number")
            else:
                print(f"video REAL height: {self.videoRealHeight} and width: {self.videoRealWidth}")
                t2 = threading.Thread(target=self.buttonFunc_feature_extraction)
                t2.daemon = True
                #t2 = self.threadingFunc(function = self.buttonFunc_feature_extraction)
                button_featureextraction= ttk.Button(self.window,text='Start event extraction',command=t2.start)
                button_featureextraction.grid(column = 1, row = 14)
                self.exitButton.grid(column = 1, row = 20)
        else:
            t2 = threading.Thread(target=self.buttonFunc_feature_extraction)
            t2.daemon = True
            #t2 = self.threadingFunc(function = self.buttonFunc_feature_extraction)
            button_featureextraction= ttk.Button(self.window,text='Start event extraction',command=t2.start)
            button_featureextraction.grid(column = 1, row = 14)
            self.exitButton.grid(column = 1, row = 20)
        self.nextButton.grid_forget()
            
    def buttonFunc_feature_extraction(self):    
        # prepare folder for event extraction
        foldername = "Visual events"
        if hasattr(self, "subjectName"):
            pass
        else:
            self.subjectName = "NoEyetrackingData"
        picklename = self.movieName + "_"+ self.subjectName + "_VF_" +self.colorSpace + "_" + self.imageSector + ".pickle"
        self.picklename = picklename
        os.chdir(self.dataDir) 
        if not os.path.exists(foldername):
           os.makedirs(foldername)
        os.chdir(foldername)
        # if events are alredady extracted
        if os.path.exists(picklename):
            print("Feature extraction already done. Loading features...")
            label_info = tk.Label(self.window, text = "Feature extraction already done. Loading features...", fg = "green")
            label_info.grid(column = 0, row = 14)
            with open(picklename, "rb") as handle:
                vidInfo, self.timeStamps, self.magnPerImPart,self.magnPerIm = pickle.load(handle)
                handle.close() 
            label_info.grid_forget()
            label_info = tk.Label(self.window, text = "Visual events are already extracted earlier. Pickle file loaded!", fg = "green")
            label_info.grid(column = 0, row = 14)
        # if events are not extracted, extracting events
        else:
            print("Extracting features...")
            # event extraction object
            eeObj = event_extraction()
            
            if hasattr(self, "filename_csv"):
                # calculate eyetracking data for gaze-contingent feature extraction
                # synchronize the eyetracking data with the movie and resample
                timeStampsSec = np.array(self.df_eyetracking.iloc[:,0])
                gazexdata = np.array(self.df_eyetracking.iloc[:,1])
                gazeydata = np.array(self.df_eyetracking.iloc[:,2])
                pupildata = np.array(self.df_eyetracking.iloc[:,3])
                
                eeObj.eyetracking_height = self.eyetracking_height
                eeObj.eyetracking_width = self.eyetracking_width
                eeObj.videoRealHeight = self.videoRealHeight
                eeObj.videoRealWidth = self.videoRealWidth
                eeObj.screenBgColorR = self.screenBgColorR
                eeObj.screenBgColorG = self.screenBgColorG
                eeObj.screenBgColorB = self.screenBgColorB
                eeObj.videoScreenSameRatio = self.videoScreenSameRatio
                eeObj.videoStretched = self.videoStretched
                if self.ans_groundTruth == "Stretch to match":
                    eeObj.stretchToMatch = True
                else:
                    eeObj.stretchToMatch = False
                eeObj.sampledTimeStamps_featureExtraction =eeObj.prepare_sampleData(timeStampsSec, self.video_nFrame)
                eeObj.sampledgazexData_featureExtraction = eeObj.prepare_sampleData(gazexdata, self.video_nFrame)
                eeObj.sampledgazeyData_featureExtraction = eeObj.prepare_sampleData(gazeydata, self.video_nFrame)
                eeObj.sampledpupilData_featureExtraction = eeObj.prepare_sampleData(pupildata, self.video_nFrame)
                
            # import parameters to eeObj
            eeObj.subject = self.subjectName
            eeObj.movieNum = self.movieName
            eeObj.picklename = self.picklename
            eeObj.filename_movie = self.filename_movie
            eeObj.setNBufFrames(self.nFramesSeqImageDiff + 1)
            eeObj.imCompFeatures = True  # creates: imageObj.vectorMagnFrame
            eeObj.showVideoFrames = self.showVideoFrames
            eeObj.imColSpaceConv = self.colorSpace
            eeObj.gazecentered = self.gazecentered
            eeObj.nVertMatPartsPerLevel = self.nVertMatPartsPerLevel  # [4, 8, 16, 32]
            eeObj.aspectRatio = self.aspectRatio 
            eeObj.imageSector = self.imageSector
            eeObj.nFramesSeqImageDiff = self.nFramesSeqImageDiff
            eeObj.selectFeatures = self.featuresOfInterest
            eeObj.scrGamFac = self.scrGamFac
            eeObj.A = self.A
            eeObj.maxlum = self.maxlum
            eeObj.useApp = self.useApp
            # resample eyetracking data
            
            # load movie
    
            #videoObj.loadVideo(self.filename_movie)  # load a video into a capture object videoObj.cap
            eeObj.vidInfo = self.prepObj.vidInfo # extract vidInfo from preprocessing object
            eeObj.window = self.window
            self.eeObj = eeObj
            eeObj.event_extraction()
            #save new pickle dictionary
            
            
        self.window.update_idletasks() 
        self.nextButton.grid_forget()
        self.exitButton.grid_forget()
        t3 = threading.Thread(target=self.buttonFunc_modeling)
        t3.daemon = True
        button_modeling= ttk.Button(self.window,text='Start modeling', command = t3.start)
        button_modeling.grid(column = 1, row = 15)

        self.exitButton.grid(column = 1, row = 20)
        
    def buttonFunc_modeling(self):
        #load feature data
        modeling_progress = tk.Label(self.window, text = "Modeling in progress. Please wait!",fg = "green")
        modeling_progress.grid(column = 0, row = 15)
        os.chdir(self.dataDir)
        os.chdir("Visual events")
        with open(self.picklename, "rb") as handle:
            vidInfo, self.timeStamps, self.magnPerImPart,self.magnPerIm = pickle.load(handle)
            handle.close() 
        # new folder for modeling results
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
            #subjectProcessed = list(modelResultDict.keys())
            #subjects = [subject for subject in subjects if subject not in subjectProcessed]
        else:
            modelResultDict = {}
        # To-do: now only works for gazecentered = True and sameWeightFeature = True
        if self.subjectName in list(modelResultDict.keys()):
            modelResultDict[self.subjectName] = {}
            modelDataDict[self.subjectName] = {}

        modelObj = pupil_prediction()
            
        modelObj.subject = self.subjectName
        modelObj.movie = self.movieName
        #modelObj.gazecentered = self.gazecentered
        #modelObj.colorSpace = self.colorSpace 
        #modelObj.imageSector = self.imageSector 
        modelObj.sameWeightFeature =self.sameWeightFeature
        
        modelObj.RF =self.RF 
        modelObj.skipNFirstFrame =self.skipNFirstFrame 
        modelObj.useBH = self.useBH
        modelObj.niter = self.niter
        modelObj.window = self.window
        modelObj.magnPerImPart= self.magnPerImPart
        modelObj.useApp = self.useApp
        modelObj.nFramesSeqImageDiff = self.nFramesSeqImageDiff
    
        
        if hasattr(self, "filename_csv"):
            if self.ans_groundTruth == "Stretch to match":
                modelObj.stretchToMatch = True
            else:
                modelObj.stretchToMatch = False
        # load eyetracking data
            modelObj.useEtData = True
            timeStampsSec = np.array(self.df_eyetracking.iloc[:,0])
            gazexdata = np.array(self.df_eyetracking.iloc[:,1])
            gazeydata = np.array(self.df_eyetracking.iloc[:,2])
            pupildata = np.array(self.df_eyetracking.iloc[:,3])
            
            modelObj.sampledTimeStamps  =modelObj.prepare_sampleData(timeStampsSec,self.video_nFrame)
            modelObj.sampledgazexData =modelObj.prepare_sampleData(gazexdata,self.video_nFrame)
            modelObj.sampledgazeyData=modelObj.prepare_sampleData(gazeydata,self.video_nFrame)
            modelObj.sampledpupilData=modelObj.prepare_sampleData(pupildata,self.video_nFrame)
            modelObj.sampledFps = 1/(modelObj.sampledTimeStamps[-1]/(len(modelObj.sampledTimeStamps)))
            # remove last several frames of the sampled eyetracking data because the calculation of feature has gap
            modelObj.sampledTimeStamps = modelObj.synchronize(modelObj.sampledTimeStamps)
            modelObj.sampledgazexData = modelObj.synchronize(modelObj.sampledgazexData)
            modelObj.sampledgazeyData = modelObj.synchronize(modelObj.sampledgazeyData)
            modelObj.sampledpupilData = modelObj.synchronize(modelObj.sampledpupilData)

            modelObj.sampledpupilData= modelObj.zscore(modelObj.sampledpupilData)

            print(f"check sample rate: {modelObj.sampledFps}")
            #modelObj.sampledpupilData = modelObj.zscore(modelObj.sampledpupilData)
            
            
            modelObj.modelDataDict = modelDataDict
            modelObj.modelResultDict = modelResultDict
            modelObj.pupil_prediction()
            self.sampledTimeStamps = modelObj.sampledTimeStamps
            self.sampledpupilData = modelObj.sampledpupilData
            self.r = modelObj.r
            self.rmse = modelObj.rmse
        else:
            modelObj.useEtData = False
            if self.RF == 'HL':
                params = [9.67,0.19,0.8,0.52,0.3, 1,1,1,1,1]
            else:
                params = [0.12,4.59,0.14,6.78,0.28,1,1,1,1,1]
            modelObj.sampledTimeStamps = self.timeStamps
            modelObj.sampledFps = 1/(modelObj.sampledTimeStamps [-1]/(len(modelObj.sampledTimeStamps)))
            modelObj.numRemoveMovFrame = 0
            modelObj.modelDataDict = modelDataDict
            modelObj.modelResultDict = modelResultDict
            modelObj.pupil_predictionNoEyetracking(params)
            modeling_progress.grid_forget()
            tk.Label(self.window,text=f'Pupil prediction done',fg = "green").grid(column = 0, row = 15)
            self.sampledTimeStamps = modelObj.sampledTimeStamps

        # move two result dictionaries to this level of class
        self.modelResultDict = modelObj.modelResultDict
        self.modelDataDict = modelObj.modelDataDict
        self.sampledFps = modelObj.sampledFps
       
        
        # print results
        self.exitButton.grid_forget()
        if hasattr(self, "filename_csv"):
            button_saveResultsParams= ttk.Button(self.window,text='Save parameters & model evaluation', command = self.buttonFunc_save_params)
            button_saveResultsParams.grid(column = 0, row = 16,sticky=tk.E)
        button_savePrediction= ttk.Button(self.window,text='Save model prediction', command = self.buttonFunc_save_model_prediction)
        button_savePrediction.grid(column = 1, row = 16)
        if hasattr(self, "filename_csv"):
            button_interactivePlot= ttk.Button(self.window,text='Interactive plot', command = self.plot)
        else:
            button_interactivePlot= ttk.Button(self.window,text='Interactive plot', command = self.plot_NoEyetracking)

        button_interactivePlot.grid(column = 1, row = 18)
        self.exitButton.grid(column = 1, row = 20)
        

    def buttonFunc_save_params(self):
        foldername = "csv results"
        os.chdir(self.dataDir) 
        if not os.path.exists(foldername):
           os.makedirs(foldername)
        os.chdir(foldername)
        params = self.modelResultDict[self.subjectName]["modelContrast"]["parameters"]
        if self.RF == "HL":
            if self.gazecentered:
                paramNames = ["r","rmse","n_luminance", "tmax_luminance", "n_contrast", "tmax_contrast", "weight_contrast", "regional_weight1","regional_weight2","regional_weight3","regional_weight4","regional_weight5","regional_weight6"]
                params = np.insert(params,5,1)
                params = np.insert(params,0,self.r)
                params = np.insert(params,1,self.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.movieName}_{self.subjectName}_parameters.csv")
            else:
                paramNames = ["r","rmse","n_luminance", "tmax_luminance", "n_contrast", "tmax_contrast", "weight_contrast"]
                params = np.insert(params,0,self.r)
                params = np.insert(params,1,self.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.movieName}_{self.subjectName}_parameters.csv")
        elif self.RF == "KB":
            if self.gazecentered:
                paramNames = ["r","rmse","theta_luminance", "k_luminance", "theta_contrast", "k_contrast", "weight_contrast", "regional_weight1","regional_weight2","regional_weight3","regional_weight4","regional_weight5","regional_weight6"]
                params = np.insert(params,5,1)
                params = np.insert(params,0,self.r)
                params = np.insert(params,1,self.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.movieName}_{self.subjectName}_parameters.csv")
            else:
                paramNames = ["r","rmse","theta_luminance", "k_luminance", "theta_contrast", "k_contrast", "weight_contrast"]
                params = np.insert(params,0,self.r)
                params = np.insert(params,1,self.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.movieName}_{self.subjectName}_parameters.csv")
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.movieName}_{self.subjectName}_parameters.csv")
        showinfo(title = "", message = "Parameters saved!")

        #tk.Label(text = "Parameters saved!", fg = "green").grid(column = 0, row = 18)
        
    def buttonFunc_save_model_prediction(self):
        self.nextButton.grid_forget()
        self.exitButton.grid_forget()
        foldername = "csv results"
        os.chdir(self.dataDir) 
        if not os.path.exists(foldername):
           os.makedirs(foldername)
        os.chdir(foldername)
        y_pred = self.modelResultDict[self.subjectName]["modelContrast"]["predAll"] 
        lumConv = self.modelResultDict[self.subjectName]["modelContrast"]["lumConv"] 
        contrastConv = self.modelResultDict[self.subjectName]["modelContrast"]["contrastConv"] 
        if hasattr(self, "filename_csv"):
            self.sampledpupilData_z = (self.sampledpupilData -np.nanmean(self.sampledpupilData)) /np.nanstd(self.sampledpupilData)
            df = pd.DataFrame(np.vstack([self.sampledTimeStamps,self.sampledpupilData, y_pred,lumConv,contrastConv]).T)
            df.columns = ["timeStamps", "Actual pupil (z)", "Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]
        else:
            df = pd.DataFrame(np.vstack([self.sampledTimeStamps,y_pred,lumConv,contrastConv]).T)
            df.columns = ["timeStamps", "Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]

        df.to_csv(f"{self.movieName}_{self.subjectName}_modelPrediction.csv")
        showinfo(title = "", message = "Model prediction result saved!")
        #tk.Label(text = "Model prediction result saved!", fg = "green").grid(column = 1, row = 18,sticky=tk.E)
        self.exitButton.grid(column = 1, row = 20)

        
    def plot(self):
        # This step have to be done after pupil prediction
        plotObj = interactive_plot()
        # subject and movie to plot
        plotObj.subjectName = self.subjectName
        plotObj.movie = self.movieName
        # other parameters
        plotObj.useApp = self.useApp
        plotObj.dataDir = self.dataDir
        plotObj.filename_movie = self.filename_movie
        plotObj.A = self.A
        plotObj.skipNFirstFrame =self.skipNFirstFrame
        plotObj.sampledFps = self.sampledFps
        plotObj.eyetracking_height = self.eyetracking_height
        plotObj.eyetracking_width = self.eyetracking_width
        plotObj.videoRealHeight = self.videoRealHeight
        plotObj.videoRealWidth = self.videoRealWidth
        plotObj.screenBgColorR = self.screenBgColorR
        plotObj.screenBgColorG = self.screenBgColorG
        plotObj.screenBgColorB = self.screenBgColorB
        plotObj.videoScreenSameRatio = self.videoScreenSameRatio 
        plotObj.videoStretched = self.videoStretched
        plotObj.window = self.window
        plotObj.plot()
        
        
        
    def plot_NoEyetracking(self):
        # This step have to be done after pupil prediction
        plotObj = interactive_plot()
        # subject and movie to plot
        plotObj.subjectName = self.subjectName
        plotObj.movieName = self.movieName
        # other parameters
        plotObj.useApp = self.useApp
        plotObj.dataDir = self.dataDir
        plotObj.filename_movie = self.filename_movie
        plotObj.A = self.A
        plotObj.skipNFirstFrame =self.skipNFirstFrame
        plotObj.sampledFps = self.sampledFps
        plotObj.video_height = self.video_height
        plotObj.video_width = self.video_width
        # plotObj.videoScreenSameRatio = self.videoScreenSameRatio 
        # plotObj.videoStretched = self.videoStretched
        plotObj.window = self.window
        plotObj.plot_NoEyetracking()

       

    def all_children(self):
        _list = self.window.winfo_children()
        for item in _list:
            if item.winfo_children():
                _list.extend(item.winfo_children())
        return _list
    def stretch(self):
        self.videoStretched = True
        self.top_ratioCheck.destroy()
        self.nextButton.grid_forget()
        self.exitButton.grid_forget()

        if self.video_ratio < self.eyetracking_aspectRatio:
            self.videoRealWidth = self.eyetracking_width
            self.videoRealHeight = self.videoRealWidth * self.video_ratio
        elif self.video_ratio > self.eyetracking_aspectRatio:
            self.videoRealHeight = self.eyetracking_height
            self.videoRealWidth = self.videoRealHeight / self.video_ratio
        else:
            self.videoRealHeight = self.eyetracking_height
            self.videoRealWidth = self.eyetracking_width
        videoInfo = tk.Label(self.window, text = f"The spatial resolution of the movie in respect to the screen should be:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        videoInfo.grid(column = 0, row = 7, pady = 10, columnspan =2)
        videoInfo_width = tk.Label(self.window, text = f"Width:{self.videoRealWidth}",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        videoInfo_width.grid(column = 0, row = 8, pady = 0,columnspan =2)
        videoInfo_height = tk.Label(self.window, text = f"Height:{self.videoRealHeight}",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        videoInfo_height.grid(column = 0, row = 9, pady = 0, columnspan =2)
        
        if self.videoScreenSameRatio:
            pass
            # colorInfo = tk.Label(self.window, text = f"No screen unoccupied by video. Pass",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
            # colorInfo.grid(column = 0, row = 9, pady = 10)
            # colorInfo_R = tk.Label(self.window, text = f"R:/",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
            # colorInfo_R.grid(column = 0, row = 10, pady = 0)
            # colorInfo_G = tk.Label(self.window, text = f"G:/",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
            # colorInfo_G.grid(column = 0, row = 11, pady = 0)
            # colorInfo_B = tk.Label(self.window, text = f"B:/",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
            # colorInfo_B.grid(column = 0, row = 12, pady = 0)
        else:
            colorInfo = tk.Label(self.window, text = f"Remaining screen background color outside video: (Please enter r,g,b value of the color)",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
            colorInfo.grid(column = 0, row = 10, pady = 10)
            colorInfo_R = tk.Label(self.window, text = f"R:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
            colorInfo_R.grid(column = 0, row = 11, pady = 0)
            colorInfo_G = tk.Label(self.window, text = f"G:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
            colorInfo_G.grid(column = 0, row = 12, pady = 0)
            colorInfo_B = tk.Label(self.window, text = f"B:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
            colorInfo_B.grid(column = 0, row = 13, pady = 0)
            self.entry_colorR = ttk.Entry(self.window)
            self.entry_colorR.grid(column = 1, row =11)
            self.entry_colorG = ttk.Entry(self.window)
            self.entry_colorG.grid(column = 1, row =12)
            self.entry_colorB = ttk.Entry(self.window)
            self.entry_colorB.grid(column = 1, row =13)
        self.nextButton = ttk.Button(self.window,text='Continue', command = self.buttonFunc_show_featureextraction)
        self.nextButton.grid(column = 1, row = 19)
        self.exitButton.grid(column =1, row = 20)

    def notStretch(self):
        self.videoStretched = False
        self.top_ratioCheck.destroy()
        self.nextButton.grid_forget()
        self.exitButton.grid_forget()
        videoInfo = tk.Label(self.window, text = f"Please enter the spatial resolution of the movie in respect to the screen:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        videoInfo.grid(column = 0, row = 7, pady = 10)
        videoInfo_height = tk.Label(self.window, text = "Height:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        videoInfo_height.grid(column = 0, row = 8, pady = 0)
        videoInfo_width = tk.Label(self.window, text = "Width:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        videoInfo_width.grid(column = 0, row = 9, pady = 0)
        self.entry_videoheight = ttk.Entry(self.window)
        self.entry_videoheight.grid(column = 1, row =8)
        self.entry_videowidth = ttk.Entry(self.window)
        self.entry_videowidth.grid(column = 1, row =9)
        # color information (enter)
        colorInfo = tk.Label(self.window, text = f"The color the the screen background is: (Please enter r,g,b value of the color)",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        colorInfo.grid(column = 0, row = 10, pady = 10)
        colorInfo_R = tk.Label(self.window, text = f"R:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        colorInfo_R.grid(column = 0, row = 11, pady = 0)
        colorInfo_G = tk.Label(self.window, text = f"G:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        colorInfo_G.grid(column = 0, row = 12, pady = 0)
        colorInfo_B = tk.Label(self.window, text = f"B:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        colorInfo_B.grid(column = 0, row = 13, pady = 0)
        self.entry_colorR = ttk.Entry(self.window)
        self.entry_colorR.grid(column = 1, row =11)
        self.entry_colorG = ttk.Entry(self.window)
        self.entry_colorG.grid(column = 1, row =12)
        self.entry_colorB = ttk.Entry(self.window)
        self.entry_colorB.grid(column = 1, row =13)
        self.nextButton = ttk.Button(self.window,text='Continue', command = self.buttonFunc_show_featureextraction)
        self.nextButton.grid(column = 1, row = 19)
        self.exitButton.grid(column =1, row = 20)
    def change_to_sec(self):
        
        self.df_eyetracking.iloc[:,0] = self.df_eyetracking.iloc[:,0]/1000
        print(self.df_eyetracking.iloc[:,0])
        self.eyetracking_duration = self.df_eyetracking.iloc[-1,0]
        self.eyetracking_nSample = self.df_eyetracking.shape[0]
        self.eyetracking_samplingrate = int(1/(self.eyetracking_duration/self.eyetracking_nSample))
        print(f"Now the sampling rate is: {self.eyetracking_samplingrate}")
        
        showinfo(
            title='Change to sec',
            message=f"The sampling rate is: {self.eyetracking_samplingrate}Hz. The time unit of the eyetracking data has been changed to second."
        )
        self.top_milli.destroy()
        self.top_milli.update() 
    def warningMessage(self):
        

        showinfo(
            title='WARNING',
            message="Something is not correct. Please check!"
        )
        #self.stop_event.set()

        #self.window.after(250, self.warningMessage)
        #self.close_top_checkinfo()
        self.close()
    
        #self.window.after(250, self.open_popup(title, message))
    def open_error_popup(self, errormessage):
        top = tk.Toplevel(self.window)
        top.geometry("800x200")
        top.title("Error")
        top.configure(bg = "#d9d9d9")

        tk.Label(top, text = errormessage).grid(column = 0, row = 0)
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        button_exit = ttk.Button(top,text = "Exit", command = self.close).grid(column = 0, row = 1)
        top.wm_transient(self.window)
        self.top = top

    def close(self):
        self.window.destroy()
        self.window.quit()
        # python = sys.executable
        # os.execl(python, python, *sys.argv)
        #self.sys_exit()
        #self.window.mainloop()
    def close_top_figure(self):
        self.top_interactive_figure.destroy()
    def close_top_milli(self):
        self.top_milli.destroy()
        #self.top_milli.update()   
    def close_top_checkinfo(self):
        self.top_checkinfo.destroy()
        #self.top_checkinfo.update()   
    def sys_exit(self):
        import warnings
        warnings.filterwarnings("ignore")
        sys.exit('exiting...')
    
    