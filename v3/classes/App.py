# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:00:43 2023

@author: 7009291
"""
#import mttkinter as tk
#from mttkinter import mtTkinter
#from mttkinter import mtTkinter
import os
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import messagebox

import pandas as pd
import numpy as np
from classes.preprocessing import preprocessing
#from classes.video_processing import video_processing
#from classes.image_processing import image_processing
from classes.pupil_prediction import pupil_prediction
from classes.event_extraction import event_extraction
from classes.interactive_plot import interactive_plot
import pathlib
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
class tkfunctions:
    def __init__(self):
        #############################################################
        # load parameters 
        # Pre-determined parameters: Don't change unless absolutely sure
        # boolean indicating whether or not to show frame-by-frame of each video with analysis results
        self.showVideoFrames = False

        # boolean indicating whether or not to skip feature extraction of already analyzed videos.
        # If skipped, then video information is loaded (from pickle file with same name as video).
        # If not skipped, then existing pickle files (one per video) are overwritten.
        self.skipAlrAnFiles = True
        # Do regularization or not: choose between "" and "ridge"(default: ridge)
        self.regularizationType = 'ridge'
        # Do zscore for pupil size change or not (default: True; False not fully tested yet)
        self.pupil_zscore = True

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

        # Number of movie frames skipped at the beginning of the movie
        self.skipNFirstFrame = 0

        ############pupil prediction parameters##############
        # Response function type
        self.RF = "HL"
        # same regional weights for luminance and contrast
        self.sameWeightFeature = True 
        # Basinhopping or minimizing
        self.useBH = False
        # iteration number for basinhopping
        self.niter = 5
        self.useApp = True
        # whether save extra csv results
        self.saveParams = True
        self.saveData = True
        
    
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
        logo_dir = os.path.join(self.initialDir, "App_fig", "DPSM logo.jpg")
        logo = Image.open(self.resource_path(logo_dir))
        logo = logo.resize((200,100))
        logo_image = ImageTk.PhotoImage(logo, master=self.window)
        image_logo = top_canvas.create_image(300, 10, anchor="nw", image=logo_image)
        text_welcom = top_canvas.create_text(400,140, text="Welcome to Open DPSM!",anchor = "center", fill="black", font=('Helvetica 20 bold'))
        text_welcom2 = top_canvas.create_text(400,180, text="Please start by loading eyetracking and movie data:",anchor = "center", fill="black", font=('Helvetica 15 bold'))
        # set up canvas for welcom message
        middle_canvas =  tk.Canvas(self.window, width=800, height=150, bg='#d9d9d9',highlightthickness=0)
        middle_canvas.grid(row=1, column=0, padx=100, pady=5)

        # text insert
        self.var_label_folder = tk.StringVar()
        #self.var_label_movie = tk.StringVar()

        self.label_folder = tk.Label(middle_canvas, textvariable=self.var_label_folder, width = 50).grid(column = 1, row = 1, sticky = "nsew")
        #self.label_movie = tk.Label(middle_canvas,textvariable=self.var_label_movie, width = 50).grid(column =1, row = 2, sticky = "nsew")
        #self.text_csv = tk.Text(self.window, height = 1)
        #self.text_movie = tk.Text(self.window, height = 1)
        # open button
        button_openfile_folder = ttk.Button(middle_canvas,text='Open folder',command=self.select_folder)#threading.Thread(target=self.select_file_csv).start)
        #button_openfile_movie = ttk.Button(middle_canvas,text='Open movie (.mp4,.avi,.mkv,.mwv,.mov,.flv,.webm)',command=self.select_file_movie)#threading.Thread(target=self.select_file_movie).start)
        # set up canvas for continue and exit
        self.bottom_canvas =  tk.Canvas(self.window, width=800, height=30, bg='#d9d9d9',highlightthickness=0)
        self.bottom_canvas.grid(row=2, column=0, padx=100, pady=5)
        #self.stop_event1 = threading.Event()
        
        # select parameters for the model
        label_info = tk.Label(self.bottom_canvas, text = f"Select event extract mode, map type and number of weight",  font = ("Arial", 10),bg = "#d9d9d9")
        label_info.grid(column = 0, row =0,columnspan = 2)    
        # event extraction mode
        modes = ["gaze-centered", "screen-centered"]
        label_info_m = tk.Label(self.bottom_canvas, text = f"Event extraction mode:",  font = ("Arial", 10), bg = "#d9d9d9")
        label_info_m.grid(column = 0, row = 1) 
        choices_m= tuple(modes)
        def callback(*arg):
            self.mode = choice_gt_m.get()
        choice_gt_m= tk.StringVar()
        choice_gt_m.set("gaze-centered")
        self.mode = "gaze-centered"
        choicebox_gt_m= ttk.Combobox(self.bottom_canvas, textvariable= choice_gt_m, width = 20,font = ("Arial", 10))
        choicebox_gt_m['values']= choices_m
        choicebox_gt_m['state']= 'readonly'
        choicebox_gt_m.grid(column = 1, row =1)
        choice_gt_m.trace('w', callback)
        
        # maptype
        maps = ["circular", "square"]
        label_info_map = tk.Label(self.bottom_canvas, text = f"Map type:",  font = ("Arial", 10), bg = "#d9d9d9")
        label_info_map.grid(column = 0, row = 2) 
        choices_map= tuple(maps)
        def callback(*arg):
            self.mapType = choice_gt_map.get()
        choice_gt_map= tk.StringVar()
        choice_gt_map.set("circular")
        self.mapType = "circular"
        choicebox_gt_map= ttk.Combobox(self.bottom_canvas, textvariable= choice_gt_map, width = 20,font = ("Arial", 10))
        choicebox_gt_map['values']= choices_map
        choicebox_gt_map['state']= 'readonly'
        choicebox_gt_map.grid(column = 1, row =2)
        choice_gt_map.trace('w', callback)
        # nWeight
        nums = [2,6,20,44,48]
        label_info_num = tk.Label(self.bottom_canvas, text = f"Number of weights:",  font = ("Arial", 10), bg = "#d9d9d9")
        label_info_num.grid(column = 0, row = 3) 
        choices_num= tuple(nums)
        def callback(*arg):
            self.nWeight = choice_gt_num.get()
        choice_gt_num= tk.StringVar()
        choice_gt_num.set(44)
        self.nWeight = 44
        choicebox_gt_num= ttk.Combobox(self.bottom_canvas, textvariable= choice_gt_num, width = 20,font = ("Arial", 10))
        choicebox_gt_num['values']= choices_num
        choicebox_gt_num['state']= 'readonly'
        choicebox_gt_num.grid(column = 1, row =3)
        choice_gt_num.trace('w', callback)
        
        # self.t1 = threading.Thread(target=self.buttonFunc_check_data)
        # self.t1.daemon = True
        #t1 = self.threadingFunc(function = self.buttonFunc_check_data)
        label_info = tk.Label(self.bottom_canvas, text = f"Constrains:\n- (1)If screen-centered, map type can only be square;\n- (2) If map type is circular, number of weights can only be 2, 20, 44;\n- (3) If map type is square, number of weights can only be 2, 6, 48",  font = ("Arial", 10),bg = "#d9d9d9")
        label_info.grid(column = 0, row =5,columnspan = 4)   
        button_check_data = ttk.Button(self.bottom_canvas, text = "Continue", command=self.check_parameters)#threading.Thread(target=self.buttonFunc_check_data).start)
        button_exit = ttk.Button(self.bottom_canvas,text = "Exit",command = self.close)
        
        button_openfile_folder.grid(column = 0, row = 1)
        #button_openfile_movie.grid(column = 0, row = 2)
        button_check_data.grid(column = 0,row =6,columnspan = 2)
        button_exit.grid(column = 0,row = 7,columnspan = 2)
        #tk.Label(text = "*If no eye tracking data folder (Input --> Eyetracking) is loaded, model optimazation will not be preformed. Parameters used to generate predicted pupil trace will be the ones found by our study.",bg='#d9d9d9').grid(column = 0, row = 3, pady = 20)
        tk.Label(text = "Please cite: Y. Cai., C. Strauch., S. Van der Stigchel., & M. Naber. Open-DPSM: An open-source toolkit for modeling pupil size changes to dynamic visual inputs.").grid(column = 0, row = 5, pady = 170, sticky = tk.S)
        # run the application
        self.window.mainloop()
    def check_parameters(self):
        self.nWeight = int(self.nWeight)
        if self.mode == "gaze-centered":
            self.gazecentered = True
        else:
            self.gazecentered = False
        if not self.gazecentered and self.mapType == "circular":
            messagebox.showwarning("Invalid combination!", "Map type can only be square if screen-centered mode is selected. Please select different parameters.")
            return
        elif self.mapType == "circular" and (self.nWeight == 6 or self.nWeight == 48):
            messagebox.showwarning("Invalid combination!", "Number of weight can only be 2,20,44 if map type is circular. Please select different parameters")
            return
        elif self.mapType == "square" and (self.nWeight == 20 or self.nWeight == 44):
            messagebox.showwarning("Invalid combination!", "Number of weight can only be 2,6,48 if map type is square. Please select different parameters")
            return
        self.buttonFunc_check_data()
    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)
    def select_folder(self):
        folder_path = fd.askdirectory()
        if folder_path:
            print("Selected folder:", folder_path)
        self.dataDir = folder_path
        self.inputDir = self.dataDir + '\\Input' # change it to the head directiory of the data folder
        self.outputDir = self.dataDir + "\\Output"

        self.movieDir = self.inputDir + '\\Movies'
        if "Eyetracking" in os.listdir(self.inputDir):
            self.eyetrackingDir = self.inputDir + '\\Eyetracking' # comment out this line if no eyetracking data mode is used
        self.var_label_folder.set(folder_path)

    def select_file_csv(self):
        filetypes = (
            ('csv files', '*.csv'),
            ('All files', '*.*')
        )
    
        filename_csv = fd.askopenfilename(
            title='Open a file',
            initialdir=self.initialDir,
            filetypes=filetypes)
        pathlib_path = pathlib.Path(filename_csv)
        file = pathlib_path.name
        self.dataDir = str(pathlib_path.parent)
        #file = filename_csv.split("/")[-1]
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
            initialdir=self.initialDir,
            filetypes=filetypes)
        # showinfo(
        #     title='Selected File',
        #     message=f"{filename_movie} is opened"
        # )
        pathlib_path = pathlib.Path(filename_movie)
        file = pathlib_path.name
        self.dataDir = str(pathlib_path.parent)
        #file = filename_movie.split("/")[-1]
        self.var_label_movie.set(file)
        self.filename_movie = filename_movie
        self.movieName = file.split(".")[0]
    def buttonFunc_check_data(self):
        # preprocessing eyetracking data and movie data
        ## movie
        # stop_event = threading.Event()
        # stop_event.set()
        entries = []
        
        text_checkdata = tk.Label(self.bottom_canvas, text = "Checking data...Please wait",bg= "#d9d9d9").grid(column = 1, row = 0)
        # manually change the aspect ratio if the selected parameters is nWeight ==2 and mapType == square
        if self.nWeight == 2 and self.mapType == "square":
            self.nVertMatPartsPerLevel = [1]
            self.aspectRatio = 0.5
            self.imageSector = "1x2"
        # array with multiple levels [2, 4, 8], with each number indicating the number of vertical image parts (number of horizontal parts will be in ratio with vertical); the more levels, the longer the analysis
        else:
            self.nVertMatPartsPerLevel = [3,6]  # [4, 8, 16, 32]
            self.aspectRatio = 0.75
            self.imageSector = "6x8" # number of visual field regions (used for naming the new visual feature)
        print(f"Selected parameters: \n- gazecentered: {self.gazecentered}\n- map type: {self.mapType}\n- number of weight: {self.nWeight}")

        # remove all the things from the previous window
        widget_list = self.all_children()
        for item in widget_list:
            item.destroy()
        if not "Eyetracking" in os.listdir(self.inputDir):
            print("No eyetracking data. Model prediction will not be preformed. Predicted pupil trace will be generated with the default response function and weights.")

            showinfo(title = "", message = "No eyetracking data. Model prediction will not be preformed. Predicted pupil trace will be generated with the default response function and weights.")
            self.mapType = "square"
            self.nWeight = 48
            self.gazecentered = False
        ############################universal information for all the eyetracking and movie data
        tk.Label(self.window, text = "Please enter information:",font=('Arial', 20),bg = "#d9d9d9").grid(column = 0, row = 0,columnspan =2)
        # Video maximum luminance
        tk.Label(self.window, text = u"What is the maximum luminance of the screen (i.e. measured physical lumiance (cd/m\u00b3) when color white is showed on screen)?",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 2, pady = 10)
        self.entry_maxLum = ttk.Entry(self.window)
        self.entry_maxLum.insert(0, "212") # this is our data
        self.entry_maxLum.grid(column = 1, row = 2)
        entries.append(self.entry_maxLum)
                # spatial resolution of eyetracking
        if "Eyetracking" in os.listdir(self.inputDir):
            tk.Label(self.window, text = f"What is the resolution for the coordinate system of eye-tracking data (also the resolution of the screen) (in pixels)",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 4, pady = 10)
        else:
            tk.Label(self.window, text = f"What is the resolution for the coordinate system of the screen (in pixels)",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 4, pady = 10)

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
        entries.append(self.entry_eyetrackingwidth)
        entries.append(self.entry_eyetrackingheight)

        # spatial resolution of movie
        tk.Label(self.window, text = f"What is the resolution for the video (the size of video displayed on the screen; the video is assumed to be screen-centered)", font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 7, pady = 10)
        
        movie_width = tk.Label(self.window, text = "Width:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        movie_width.grid(column = 0, row = 8, pady = 0)
        movie_height = tk.Label(self.window, text = "Height:",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
        movie_height.grid(column = 0, row = 9, pady = 0)

        self.entry_moviewidth = ttk.Entry(self.window)
        self.entry_moviewidth.insert(0, "1920") # this is our data
        self.entry_moviewidth.grid(column = 1, row = 8)
        self.entry_movieheight = ttk.Entry(self.window)
        self.entry_movieheight.insert(0, "1080") # this is our data
        self.entry_movieheight.grid(column = 1, row = 9)
        entries.append(self.entry_moviewidth)
        entries.append(self.entry_movieheight)
        # 
        colorInfo = tk.Label(self.window, text = f"If video was not played full screen, what is the color for the remaining screen outside video?\n(Don't change the value if the videos were full screen)",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9")
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
        self.entry_colorR.insert(0, "0") # this is our data
        self.entry_colorG.insert(0, "0") # this is our data
        self.entry_colorB.insert(0, "0") # this is our data

        # physical width screen
        tk.Label(self.window, text = f"What is the physical width of the screen? (in cm)",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 14, pady = 10)
        self.entry_screenwidthcm = ttk.Entry(self.window)
        self.entry_screenwidthcm.insert(0, "145") # this is our data
        self.entry_screenwidthcm.grid(column = 1, row = 14)
        entries.append(self.entry_screenwidthcm)

        # distance between screen and eye
        if "Eyetracking" in os.listdir(self.inputDir):
            tk.Label(self.window, text = f"What is the distance between the eye and the monitor? (in cm)",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 15, pady = 10)
            self.entry_screeneyecm = ttk.Entry(self.window)
            self.entry_screeneyecm.insert(0, "75") # this is our data
            self.entry_screeneyecm.grid(column = 1, row = 15)
            entries.append(self.entry_screeneyecm)

        # Size of the regional weights map
        tk.Label(self.window, text = f"The visual angle of the regional weight map should be ( ) times of the visual angle of the video",font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 16, pady = 10)
        self.entry_visualanglemap = ttk.Entry(self.window)
        self.entry_visualanglemap.insert(0, "1") # this is our data
        self.entry_visualanglemap.grid(column = 1, row = 16)
        entries.append(self.entry_visualanglemap)
        self.entries = entries
        if "Eyetracking" in os.listdir(self.inputDir):
            # what to do if eyetracking and video data do not have exactly the same duration?
            tk.Label(self.window, text = f"If the lengths of eyetracking data and video are not exactly the same, what do you want to do?",  font = ("Arial", 10), justify = tk.RIGHT,anchor = "e", width = 100,  bg = "#d9d9d9").grid(column = 0, row = 17)    
            choices= ('Stretch to match','cut last part of the longer file')
            def callback(*arg):
                self.ans_groundTruth = choice_gt.get()
            choice_gt= tk.StringVar()
            choice_gt.set('Stretch to match')
            self.ans_groundTruth = "Stretch to match"
            choicebox_gt= ttk.Combobox(self.window, textvariable= choice_gt, width = 20,font = ("Arial", 10))
            choicebox_gt['values']= choices
            choicebox_gt['state']= 'readonly'
            choicebox_gt.grid(column = 1, row =17)
            choice_gt.trace('w', callback)
        
        ## eyetracking data

        self.nextButton = ttk.Button(self.window,text='Continue', command = self.check_entries)
        self.nextButton.grid(column = 1, row = 18)
        self.exitButton = ttk.Button(self.window,text='Exit', command = self.close)
        self.exitButton.grid(column = 1, row = 19)
    
    def check_entries(self):
        for entry in self.entries:
            if not entry.get().strip():  # empty or only spaces
                messagebox.showwarning("Missing value", "Please fill in all fields.")
                return
        self.check_individual_file()     
    def check_entries2(self):
        for entry in self.entries:
            if not entry.get().strip():  # empty or only spaces
                messagebox.showwarning("Missing value", "Please fill in all fields.")
                return
        self.buttonFunc_show_featureextraction()     

        
    def check_individual_file(self):
        self.maxlum = float(self.entry_maxLum.get())
        if "Eyetracking" in os.listdir(self.inputDir):
            self.eye_to_screen = float(self.entry_screeneyecm.get())
        else:
            self.eye_to_screen = 75
        self.eyetracking_height = float(self.entry_eyetrackingheight.get())
        self.eyetracking_width = float(self.entry_eyetrackingwidth.get())
        self.eyetracking_aspectRatio = round(self.eyetracking_height/self.eyetracking_width,3)
        self.screen_width = float(self.entry_screenwidthcm.get())
        self.degVF_param = float(self.entry_visualanglemap.get())
        self.videoRealHeight = float(self.entry_movieheight.get())
        self.videoRealWidth = float(self.entry_moviewidth.get())
        self.videoWidthCM = self.videoRealWidth / (self.eyetracking_width/self.screen_width)
        self.videoWidthDeg =math.degrees(math.atan(self.videoWidthCM/2/self.eye_to_screen))*2
        

        # visual angle of the regional weight map 
        self.degVF = self.videoWidthDeg *self.degVF_param
        if self.videoRealHeight > self.eyetracking_height or self.videoRealWidth > self.eyetracking_width:
            self.open_error_popup(errormessage= f"Video is larger than screen. Please modify video file first.")
        if self.videoRealHeight == self.eyetracking_height and self.videoRealWidth ==self.eyetracking_width:
            self.videoScreenSameRatio = True 
            self.videoStretched = True
        elif self.videoRealHeight == self.eyetracking_height or self.videoRealWidth ==self.eyetracking_width:
            self.videoScreenSameRatio = False
            self.videoStretched = True
        else:
            self.videoScreenSameRatio = False
            self.videoStretched = False
            
        # RGB value for background
        if self.videoRealHeight * self.videoRealHeight == self.eyetracking_height * self.eyetracking_width:
            self.screenBgColorR = np.nan
            self.screenBgColorG = np.nan
            self.screenBgColorB = np.nan
        else:
            self.screenBgColorB = float(self.entry_colorB.get())
            self.screenBgColorR = float(self.entry_colorR.get())
            self.screenBgColorG = float(self.entry_colorG.get())
        # remove all the things from the previous window
        widget_list = self.all_children()
        for item in widget_list:
            item.destroy()
        self.window.geometry("10x10+0+0")
        ################################################### check basic eyetracking data structure for each csv file####################################################################
        # should be skipped if the user followed the instruction on Gitbub
        
        if "Eyetracking" in os.listdir(self.inputDir):
            # save decision information for each file
            for subjectName in os.listdir(self.eyetrackingDir):
                os.chdir(self.eyetrackingDir)
                os.chdir(subjectName)
                self.subjectDir = os.getcwd()
                for csvFile in os.listdir():
                    movie = csvFile.split(".")[0]

                    filename_csv = self.eyetrackingDir + f"\\{subjectName}\\{csvFile}"
                    df_eyetracking = pd.read_csv(filename_csv, index_col=None, header = None)
                    # change the first timestamp to 0
                    df_eyetracking.iloc[:,0] = df_eyetracking.iloc[:,0]-df_eyetracking.iloc[0,0]
                    # extract eye-tracking information
                    self.eyetracking_duration = df_eyetracking.iloc[-1,0]
                    self.eyetracking_nSample = df_eyetracking.shape[0]
                    self.eyetracking_samplingrate = int(1/(self.eyetracking_duration/self.eyetracking_nSample))
                    self.df_eyetracking = df_eyetracking
                    # check if the first column is increasing (as time should be increasomg)
                    if not all(i < j for i, j in zip(np.array(self.df_eyetracking.iloc[:,0]).tolist(), np.array(self.df_eyetracking.iloc[:,0]).tolist()[1:])):
                        self.open_error_popup(errormessage= f"The first column of the eyetracking data of subject {subjectName} movie {movie} is not the time. Please recheck!")
                    # check if millisecond (open pop up window)
                    if self.eyetracking_samplingrate < 30 or self.eyetracking_samplingrate >2000:
                        self.top_milli = tk.Toplevel(self.window)
                        self.top_milli.geometry("800x100")
                        self.top_milli.title(f"Sample rate of subject {subjectName} movie {movie}")
                        self.top_milli.configure(bg = "#d9d9d9")
                        tk.Label(self.top_milli, text = "Standard timestamps should be in seconds. Now it seems that it is in milisecond. Change to seconds? (csv file will be replaced)", font = ("Arial", 10), bg = "#d9d9d9").grid(column = 0, row = 0)
                        self.top_milli.columnconfigure(0, weight=1)
                        self.top_milli.rowconfigure(0, weight=1)
                        self.top_milli.wm_transient(self.window)
                        top_milli_canvas =  tk.Canvas(self.top_milli, width=800, height=100, bg='#d9d9d9',highlightthickness=0)
                        top_milli_canvas.grid(row=1, column=0, padx=100, pady=5)
                        # add button
                        button_yes = ttk.Button(top_milli_canvas,text = "Yes, and replace csv file", command = self.change_to_sec).grid(column = 0, row = 1)#threading.Thread(target=self.change_to_sec).start).grid(column = 0, row = 1)
                        button_no = ttk.Button(top_milli_canvas,text = "No", command = self.close_top_milli).grid(column = 1, row = 1)#threading.Thread(target=self.close_popup).start).grid(column = 1, row = 1)
                        button_exit = ttk.Button(top_milli_canvas,text = "Exit", command = self.close).grid(column = 2, row = 1)
                        self.top_milli.wait_window()
        ################################################### set up parameters for event extraction and do event extraction####################################################################
        if "Eyetracking" in os.listdir(self.inputDir):
            if self.gazecentered:
                subjects = os.listdir(self.eyetrackingDir) 
            else:
                subjects = ["sc"]
        else:
            subjects = ["NoEyetrackingData"] # name the events file as NoEyetrackingData
            self.gazecentered = False # not use gaze data
        for subjectName in subjects:
            if self.gazecentered:
                os.chdir(self.eyetrackingDir)
                os.chdir(subjectName)
                self.subjectDir = os.getcwd()
                movieList = os.listdir()
            else:
                movieList = [file for file in os.listdir(self.movieDir)]
            for movie in movieList:
                self.subjectName = subjectName
                self.movieName = movie.split(".")[0]
                movie = [file for file in os.listdir(self.movieDir) if file.startswith(self.movieName)][0]
                # extract movie info
                self.filename_movie = self.movieDir + f"\\{movie}"
                print(self.filename_movie)
                # check if this movie already extracted
                os.chdir(self.dataDir) 
                foldername = "Output"
                if not os.path.exists(foldername):
                   os.makedirs(foldername)
                os.chdir(foldername)
                foldername = "Visual events"
                if not os.path.exists(foldername):
                   os.makedirs(foldername)
                os.chdir(foldername)
                # name the feature extracted pickle:
                if self.mapType == "square":
                    picklename ="square_" + self.movieName + "_"+ self.subjectName + "_VF_" +self.colorSpace + "_nWeight_" + str(self.nWeight)  + ".pickle"
                elif self.mapType == "circular":
                    picklename ="circular_" + self.movieName + "_"+ self.subjectName + "_VF_" +self.colorSpace + "_nWeight_" + str(self.nWeight) + ".pickle"
                self.picklename = picklename
                if os.path.exists(picklename):
                    print(f"Subject {subjectName}, Movie {self.movieName} event extraction already done.")
                    showinfo(title = "", message = f"Subject {subjectName}, Movie {self.movieName} event extraction already done.")

                    # label_info = tk.Label(top_checkinfo_canvas, text = "Feature extraction already done. Press [Next] to do the next movie.", fg = "green")
                    # label_info.grid(column = 0, row = 3)
                    # nextMovie = ttk.Button(top_checkinfo_canvas,text = "Next movie", command = self.close_top_checkinfo)
                    # nextMovie.grid(column = 1, row = 4)
                    # self.exitButton = ttk.Button(top_checkinfo_canvas,text = "Exit", command = self.close)
                    # self.exitButton.grid(column = 1, row = 5)
                    
                    # self.top_checkinfo.wait_window()
                else:
                    prepObj = preprocessing()
                    prepObj.videoFileName = self.filename_movie
                    prepObj.loadVideo(self.filename_movie)
                    prepObj.getVideoInfo()
                    self.video_nFrame = prepObj.vidInfo['nFrames']
                    self.video_height = prepObj.vidInfo['height']
                    self.video_width = prepObj.vidInfo['width']
                    self.video_ratio = round(self.video_height / self.video_width,3) 
                    self.video_duration = prepObj.vidInfo['duration']
                    self.video_fps = prepObj.vidInfo['fps']
                    self.prepObj = prepObj
                    print(f"Video number of frame: {self.video_nFrame}")
                    print(f"Video height x width: {self.video_height}x{self.video_width}; aspect ratio (width:height): {1/self.video_ratio}")
                    print(f"Video duration: {self.video_duration}")
                    print(f"Video frame rate: {self.video_fps}")
                    
                    # extract eyetracking info
                    if self.gazecentered:
                        csvFile = self.movieName + ".csv"
                        filename_csv = self.eyetrackingDir + f"\\{subjectName}\\{csvFile}"
                        df_eyetracking = pd.read_csv(filename_csv, index_col=None, header = None)
                        # change the first timestamp to 0
                        df_eyetracking.iloc[:,0] = df_eyetracking.iloc[:,0]-df_eyetracking.iloc[0,0]
                        # extract eye-tracking information
                        self.eyetracking_duration = df_eyetracking.iloc[-1,0]
                        self.eyetracking_nSample = df_eyetracking.shape[0]
                        self.eyetracking_samplingrate = int(1/(self.eyetracking_duration/self.eyetracking_nSample))
                        self.df_eyetracking = df_eyetracking
                    # Check the information of eyetracking data and movie
                    self.top_checkinfo = tk.Toplevel(self.window)
                    self.top_checkinfo.geometry("600x300+0+0")
                
                    self.top_checkinfo.title(f"Event extraction {subjectName}")
                    self.top_checkinfo.configure(bg = "#d9d9d9")
                    
                    #self.top_checkinfo.columnconfigure(0, weight=1)
                    #self.top_checkinfo.rowconfigure(0, weight=1)
                    #self.top_checkinfo.wm_transient(self.window)

                    #self.open_popup(title = "Check the information", message = )
                    #information_check = tk.Label(self.window, text = f"Check if everything is correct:\nLength of eyetracking data: {self.eyetracking_duration}\nSampling rate of eyetracking data: {self.eyetracking_samplingrate}\nLength of the video: {self.video_duration}\n frame rate of the movie: {self.video_fps}\n Aspect ratio of the video is {self.video_ratio}")
                    top_checkinfo_canvas =  tk.Canvas(self.top_checkinfo, width=800, height=300, bg='#d9d9d9',highlightthickness=0)
                    top_checkinfo_canvas.grid(row=10, column=3, padx=100, pady=5)
                    if self.gazecentered:
                        tk.Label(top_checkinfo_canvas, text =  f"Subject {subjectName}, Movie {movie}:\nDuration of eye tracking data in seconds: {self.eyetracking_duration}\nSampling rate of eyetracking data: {self.eyetracking_samplingrate}\nEyetracking (screen) height x width: {int(self.eyetracking_height)} x {int(self.eyetracking_width)}",bg = "#d9d9d9",font = ("Arial", 10)).grid(column = 0, row = 1,columnspan = 2)
                        if self.eyetracking_duration == self.video_duration:
                            self.ans_groundTruth = "Stretch to match"
                            
                    else:
                        tk.Label(top_checkinfo_canvas, text =  f"Screen-centered event extraction, Movie {movie}",bg = "#d9d9d9",font = ("Arial", 10)).grid(column = 0, row = 1,columnspan = 2)

                    tk.Label(top_checkinfo_canvas, text =  f"Duration of the video in seconds: {self.video_duration}\n Frame rate of the movie in Hz: {self.video_fps}\n Video height x width (in file): {self.video_height} x {self.video_width}\n Video height x width (on screen): {int(self.videoRealHeight)} x {int(self.videoRealWidth)}", bg = "#d9d9d9",font = ("Arial", 10)).grid(column = 0, row = 2,columnspan = 2,pady = 10)
                    
                   
                    video_duration = self.video_duration
                    video_ratio = self.video_ratio
                    
                    self.top_checkinfo_canvas = top_checkinfo_canvas
                    self.nextButton = ttk.Button(top_checkinfo_canvas,text = "Next", command = self.buttonFunc_show_featureextraction)
                    self.nextButton.grid(column = 1, row = 4)
                    self.exitButton = ttk.Button(top_checkinfo_canvas,text = "Exit", command = self.close)
                    self.exitButton.grid(column = 1, row = 5)
                    
                    self.top_checkinfo.wait_window()
        ######## Open a new window for modeling##############
        self.top_modeling = tk.Toplevel(self.window)
        self.top_modeling.geometry("400x300+0+0")
    
        self.top_modeling.title(f"Modeling {subjectName}")
        self.top_modeling.configure(bg = "#d9d9d9")
        label = tk.Label(self.top_modeling, text = "Event extraction has been done for all the subjects and movies.",font=('Arial', 10),bg = "#d9d9d9")
        label.grid(column = 0, row = 0,columnspan =2)
        t3 = threading.Thread(target=self.buttonFunc_modeling)
        t3.daemon = True
        self.button_modeling= ttk.Button(self.top_modeling,text='Start modeling', command = t3.start)
        self.button_modeling.grid(column = 0, row = 1, columnspan = 2)
        self.exitButton = ttk.Button(self.top_modeling,text = "Exit", command = self.close)
        self.exitButton.grid(column = 0, row = 2, columnspan = 2)
      
    def buttonFunc_ratioCheck(self):
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
                pic_dir = os.path.join(self.initialDir, "App_fig", "Screen_lower.jpg")
                choiceA = Image.open(self.resource_path(pic_dir))
                choiceA = choiceA.resize((300,160))
                choiceA_image = ImageTk.PhotoImage(choiceA, master=self.window)
                pic_dir = os.path.join(self.initialDir, "App_fig", "Screen_surrounding_lower.jpg")

                choiceB = Image.open(self.resource_path(pic_dir))
                choiceB = choiceB.resize((300,160))
                choiceB_image = ImageTk.PhotoImage(choiceB, master=self.window)
                tk.Label(self.top_ratioCheck, image = choiceA_image).grid(column = 0, row = 1,sticky = "we")
                tk.Label(self.top_ratioCheck, image = choiceB_image).grid(column = 1, row = 1,sticky = "we")
                ttk.Button(self.top_ratioCheck,text='Choose A',command = self.stretch).grid(column = 0, row = 2)
                ttk.Button(self.top_ratioCheck,text='Choose B', command = self.notStretch).grid(column = 1, row = 2)
            else:
                pic_dir = os.path.join(self.initialDir, "App_fig", "Screen_higher.jpg")
                choiceA = Image.open(self.resource_path(pic_dir))
                choiceA = choiceA.resize((300,160))
                choiceA_image = ImageTk.PhotoImage(choiceA, master=self.window)
                pic_dir = os.path.join(self.initialDir, "App_fig", "Screen_surrounding_higher.jpg")
                choiceB = Image.open(self.resource_path(pic_dir))
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
        t2 = threading.Thread(target=self.buttonFunc_feature_extraction)
        t2.daemon = True
        #t2 = self.threadingFunc(function = self.buttonFunc_feature_extraction)
        self.button_featureextraction= ttk.Button(self.top_checkinfo_canvas,text='Start event extraction',command=t2.start)
        self.button_featureextraction.grid(column = 1, row = 4)        
        self.nextButton.grid_forget()
            
    def buttonFunc_feature_extraction(self):    
        
        print("Extracting features...")
        label_info = tk.Label(self.top_checkinfo_canvas, text = "Please wait...", fg = "green")
        label_info.grid(column = 0, row = 4)

        eeObj = event_extraction()
        eeObj.label_info = label_info
        eeObj.button_featureextraction = self.button_featureextraction
        if self.gazecentered:
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
            eeObj.eye_to_screen = self.eye_to_screen
            
            eeObj.eyetracking_samplingrate = self.eyetracking_samplingrate
            eeObj.eyetracking_duration = self.eyetracking_duration
            
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
        eeObj.maxlum = self.maxlum
        eeObj.useApp = self.useApp
        eeObj.nWeight = self.nWeight
        eeObj.vidInfo = self.prepObj.vidInfo # extract vidInfo from preprocessing object
        eeObj.window = self.top_checkinfo_canvas
        eeObj.mapType = self.mapType
        eeObj.degVF = self.degVF
        eeObj.screen_width= self.screen_width
        eeObj.degVF_param= self.degVF_param
        self.eeObj = eeObj
        eeObj.event_extraction()
        #save new pickle dictionary
        self.window.update_idletasks() 
        nextMovie= ttk.Button(self.top_checkinfo_canvas,text='Next movie', command = self.close_top_checkinfo)
        nextMovie.grid(column = 1, row = 5)
        # self.nextButton.grid_forget()
        # self.exitButton.grid_forget()
        self.exitButton.grid(column = 1, row = 6)
        

    def buttonFunc_modeling(self):
        #load feature data
        os.chdir(self.outputDir)

        # create folder to save data
        # new folder for modeling results
        foldername = "Modeling result"
        if not os.path.exists(foldername):
           os.makedirs(foldername)
        os.chdir(foldername)
        #Create dictionaries to save results
        if os.path.exists(f"modelDataDict_nWeight{self.nWeight}.pickle"):
            with open(f"modelDataDict_nWeight{self.nWeight}.pickle", "rb") as handle:
                modelDataDict = pickle.load(handle)
                handle.close() 
        else:
            modelDataDict = {}
                
        if os.path.exists(f"modelResultDict_nWeight{self.nWeight}.pickle"):
            with open(f"modelResultDict_nWeight{self.nWeight}.pickle", "rb") as handle:
                modelResultDict = pickle.load(handle)
                handle.close() 
            #subjectProcessed = list(modelResultDict.keys())
            #subjects = [subject for subject in subjects if subject not in subjectProcessed]
        else:
            modelResultDict = {}
        # To-do: sameWeightFeature not work
        if "Eyetracking" in os.listdir(self.inputDir):
            subjects = os.listdir(self.eyetrackingDir) 
            
        else:
            subjects = ["NoEyetrackingData"] # name the events file as NoEyetrackingData
            self.gazecentered = False # not use gaze data
        for subjectName in subjects:
            if "Eyetracking" in os.listdir(self.inputDir):
            
                # if subject is already in the dictionary, skip
                if subjectName in modelResultDict.keys() and 'modelRegularization' in modelResultDict[subjectName].keys():
                    print(f"Modeling already done for subject {subjectName}")
                    showinfo(title = "", message = f"Modeling already done for subject {subjectName}\nnumber of weights: {self.nWeight}, map type: {self.mapType}")
                    #label.grid(column = 0, row = 2,columnspan =2)
                else:
                    self.exitButton.grid_forget()
                    modeling_progress = tk.Label(self.top_modeling, text = f"Modeling in progress for subject {subjectName}\n(number of weights: {self.nWeight}, map type: {self.mapType})",fg = "green")
                    modeling_progress.grid(column = 0, row = 3,columnspan = 2)
                    self.button_modeling.grid_forget()
                    # modeling_info = tk.Label(self.top_modeling, text = f"nWeight = {self.nWeight}, mapType = {self.mapType}",fg = "green")
                    # modeling_progress.grid(column = 0, row = 4)
                    self.exitButton.grid(column = 1, row = 5,columnspan = 2)
                    
                    self.subjectDir = self.eyetrackingDir + f"\\{subjectName}"
                    csvFiles = [file for file in os.listdir(self.subjectDir) if file.endswith('.csv')]
                    movieList = [file.replace('.csv','') for file in csvFiles]
                
                    # pupil prediction class
                    modelObj = pupil_prediction()
                    modelObj.window = self.top_modeling
                    modelObj.eyetrackingDir = self.eyetrackingDir
                    modelObj.subjectDir = self.subjectDir
                    modelObj.outputDir = self.outputDir
                    modelObj.feature_pickle_directory = f"{self.outputDir}\\visual events"
    
                    modelObj.nWeight =self.nWeight
                    modelObj.subject = subjectName
                    modelObj.sameWeightFeature =self.sameWeightFeature
                    modelObj.RF =self.RF 
                    modelObj.skipNFirstFrame =self.skipNFirstFrame 
                    modelObj.useBH = self.useBH
                    modelObj.niter = self.niter
                    
                    modelObj.useApp = self.useApp
                    if self.ans_groundTruth == "Stretch to match":
                        modelObj.stretchToMatch = True
                    else:
                        modelObj.stretchToMatch = False
                    modelObj.nFramesSeqImageDiff = self.nFramesSeqImageDiff
                    modelObj.mapType = self.mapType
                    # load eyetracking data
                    modelObj.useEtData = True
                    modelObj.gazecentered = self.gazecentered
                    modelObj.pupil_zscore = self.pupil_zscore
                    modelObj.modelDataDict = modelDataDict
                    modelObj.modelResultDict = modelResultDict
                    modelObj.movieList = movieList
                    # start modeling
                    modelObj.connect_data(movieList)
                    modelObj.pupil_prediction()
                    
                    self.sampledTimeStamps = modelObj.sampledTimeStamps
                    self.sampledpupilData = modelObj.sampledpupilData
                    self.r = modelObj.r
                    self.rmse = modelObj.rmse
                    # move two result dictionaries to this level of class
                    self.modelResultDict = modelObj.modelResultDict
                    self.modelDataDict = modelObj.modelDataDict
                    self.sampledFps = modelObj.sampledFps
                    # automatically save csvfile
                    if self.saveParams:
                        foldername = "csv results"
                        os.chdir(self.outputDir) 
                        if not os.path.exists(foldername):
                           os.makedirs(foldername)
                        os.chdir(foldername)
                        params = self.modelResultDict[subjectName]["modelContrast"]["parameters"]
                        if self.RF == "HL":
                            paramNames =  ['r', 'rmse'] + self.modelResultDict[subjectName]["modelContrast"]["parametersNames"]
                            params = np.insert(params,0,modelObj.r)
                            params = np.insert(params,1,modelObj.rmse)
                            df = pd.DataFrame(np.vstack([paramNames,params]).T)
                            df.columns = ["parameterName", "value"]
                            df.to_csv(f"{subjectName}_parameters_nWeight{self.nWeight}.csv")    
                        elif self.RF == "KB":
                            paramNames =  ['r', 'rmse'] + self.modelResultDict[subjectName]["modelContrast"]["parametersNames"]
                            params = np.insert(params,0,modelObj.r)
                            params = np.insert(params,1,modelObj.rmse)
                            df = pd.DataFrame(np.vstack([paramNames,params]).T)
                            df.columns = ["parameterName", "value"]
                            df.to_csv(f"{subjectName}_parameters_nWeight{self.nWeight}.csv")
                        # save modeling data
                        if self.saveData:
                            foldername = "csv results"
                            os.chdir(self.outputDir) 
                            if not os.path.exists(foldername):
                               os.makedirs(foldername)
                            os.chdir(foldername)
                            y_pred = self.modelResultDict[subjectName]["modelContrast"]["predAll"] 
                            lumConv = self.modelResultDict[subjectName]["modelContrast"]["lumConvAll"] 
                            contrastConv = self.modelResultDict[subjectName]["modelContrast"]["contrastConvAll"] 
                            sampledpupilData_z = modelObj.sampledPupilDataAll 
                            df = pd.DataFrame(np.vstack([sampledpupilData_z, y_pred,lumConv,contrastConv]).T)
                            df.columns = ["Actual pupil (z)", "Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]
                            
                            df.to_csv(f"{subjectName}_modelPrediction_nWeight{self.nWeight}.csv")
                    ###################################
                    # Do regularization
                    if self.regularizationType == "ridge": # This is the only type of regularization tested
                        # modeling_progress.grid_forget()
                        # modeling_progress = tk.Label(self.top_modeling, text = f"Regularization for subject {subjectName}...",fg = "green")
                        # modeling_progress.grid(column = 0, row = 2,columnspan = 2)
                        foldername = "Modeling result"
                        os.chdir(self.outputDir)
                        os.chdir(foldername)
                        modelObj.regularizationType = self.regularizationType
                        modelObj.params = self.modelResultDict[subjectName]["modelContrast"]["parameters"]
                        modelObj.regularization()
                        self.r = modelObj.r
                        self.rmse = modelObj.rmse
                        self.modelDataDict = modelObj.modelDataDict
                        self.modelResultDict = modelObj.modelResultDict
                        self.window.lower() # try to make the main window at the bottom of other window
                        showinfo(title = f"Model performance", message = f"Modeling done for subject {subjectName}\nR = {round(self.r,2)}; rmse = {round(self.rmse,2)}")
                        
                        # save the modeling results (data do not need to save because they are not different from the model without regularization)
                        if self.saveParams:
                            foldername = "csv results"
                            os.chdir(self.outputDir) 
                            if not os.path.exists(foldername):
                               os.makedirs(foldername)
                            os.chdir(foldername)
                            params = self.modelResultDict[subjectName]["modelRegularization"]["parameters"]
                            if self.RF == "HL":
                                paramNames =  ['r', 'rmse'] + self.modelResultDict[subjectName]["modelRegularization"]["parametersNames"]
                                params = np.insert(params,0,modelObj.r)
                                params = np.insert(params,1,modelObj.rmse)
                                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                                df.columns = ["parameterName", "value"]
                                df.to_csv(f"{subjectName}_parameters_regularization_nWeight{self.nWeight}.csv")    
                            elif self.RF == "KB":
                                paramNames =  ['r', 'rmse'] + self.modelResultDict[subjectName]["modelRegularization"]["parametersNames"]
                                params = np.insert(params,0,modelObj.r)
                                params = np.insert(params,1,modelObj.rmse)
                                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                                df.columns = ["parameterName", "value"]
                                df.to_csv(f"{subjectName}_parameters_regularization_nWeight{self.nWeight}.csv")
                        if self.saveData:
                            foldername = "csv results"
                            os.chdir(self.outputDir) 
                            if not os.path.exists(foldername):
                               os.makedirs(foldername)
                            os.chdir(foldername)
                            y_pred = self.modelResultDict[subjectName]["modelRegularization"]["predAll"] 
                            lumConv = self.modelResultDict[subjectName]["modelRegularization"]["lumConvAll"] 
                            contrastConv = self.modelResultDict[subjectName]["modelRegularization"]["contrastConvAll"] 
                            sampledpupilData_z = modelObj.sampledPupilDataAll 
                            df = pd.DataFrame(np.vstack([sampledpupilData_z, y_pred,lumConv,contrastConv]).T)
                            df.columns = [ "Actual pupil (z)", "Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]
                            
                            df.to_csv(f"{subjectName}_modelPrediction_regularization_nWeight{self.nWeight}.csv")
            else:
                # new folder for modeling results
                foldername = "Modeling result"
                os.chdir(self.outputDir) 
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
                    showinfo(title = "", message = "Modeling result existed.")
                else:
                    # pupil prediction class
                    modelObj = pupil_prediction()
                    modelObj.useEtData = False
                    if self.RF == 'HL':
                        params = [9.67,0.19,0.8,0.52,0.3] + [1] * self.nWeight # Those are the parameters gained from the our data
                    
                    else:
                        params = [0.12,4.59,0.14,6.78,0.28]+ [1] * self.nWeight
                    #modelObj.sampledTimeStamps = timeStamps
                    #modelObj.sampledFps = 1/(modelObj.sampledTimeStamps [-1]/(len(modelObj.sampledTimeStamps)))
                    movieList = [file.split(".")[0] for file in os.listdir(self.movieDir)]

                    modelObj.numRemoveMovFrame = 0
                    modelObj.outputDir = self.outputDir
                    modelObj.modelDataDict = modelDataDict
                    modelObj.modelResultDict = modelResultDict
                    modelObj.sameWeightFeature = self.sameWeightFeature
                    modelObj.RF = self.RF
                    modelObj.subject = subjectName
                    modelObj.nWeight= self.nWeight
                    modelObj.movieList = movieList
                    modelObj.feature_pickle_directory =  f"{self.outputDir}\\visual events"
                    modelObj.gazecentered = self.gazecentered
                    modelObj.mapType = self.mapType
                    modelObj.pupil_zscore = self.pupil_zscore
                    
                    modelObj.connect_data(movieList)
                    modelObj.pupil_predictionNoEyetracking(params)
                    ####################################
                    # save model results
                    foldername = "csv results"
                    os.chdir(self.outputDir) 
                    if not os.path.exists(foldername):
                       os.makedirs(foldername)
                    os.chdir(foldername)
                    # save data used for pupil prediction
                    if self.saveData:
                        y_pred = modelResultDict[subjectName]["modelContrast"]["predAll"] 
                        lumConv = modelResultDict[subjectName]["modelContrast"]["lumConvAll"] 
                        contrastConv = modelResultDict[subjectName]["modelContrast"]["contrastConvAll"] 
                        
                        df = pd.DataFrame(np.vstack([y_pred,lumConv,contrastConv]).T)
                        df.columns = ["Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]
                        
                        df.to_csv(f"{subjectName}_modelPrediction.csv")
        # interactive plot
        ######## Open a new window for plotting##############
        self.top_modeling.destroy()
        self.top_plotting = tk.Toplevel(self.window)
        self.top_plotting.geometry("500x300+0+0")
        self.top_plotting.title(f"Plotting")
        self.top_plotting.configure(bg = "#d9d9d9")
        label = tk.Label(self.top_plotting, text = "Modeling has been done for all the subjects and movies.",font=('Arial', 10),bg = "#d9d9d9" )
        label.grid(column = 0, row = 0,columnspan = 2)
        
        label_info = tk.Label(self.top_plotting, text = f"Select one movie to plot (plotting can take a few seconds):",  font = ("Arial", 10),bg = "#d9d9d9")
        label_info.grid(column = 0, row = 1,columnspan = 2)    
        subjects = list(modelDataDict.keys())
        label_info_s = tk.Label(self.top_plotting, text = f"Subject:",  font = ("Arial", 10), bg = "#d9d9d9")
        label_info_s.grid(column = 0, row = 2) 
        choices_s= tuple(subjects)
        def callback(*arg):
            self.subjectName_plot = choice_gt_subject.get()
        choice_gt_subject= tk.StringVar()
        choice_gt_subject.set(subjects[0])
        self.subjectName_plot = subjects[0]
        choicebox_gt_subject= ttk.Combobox(self.top_plotting, textvariable= choice_gt_subject, width = 20,font = ("Arial", 10))
        choicebox_gt_subject['values']= choices_s
        choicebox_gt_subject['state']= 'readonly'
        choicebox_gt_subject.grid(column = 1, row =2)
        choice_gt_subject.trace('w', callback)
        
        label_info_m = tk.Label(self.top_plotting, text = f"Movie:",  font = ("Arial", 10),  bg = "#d9d9d9")
        label_info_m.grid(column = 0, row = 3) 
        movies = list(modelDataDict[self.subjectName_plot].keys())
        choices_m= tuple(movies)
        def callback(*arg):
            self.movieName_plot = choice_gt_movie.get()
        choice_gt_movie= tk.StringVar()
        choice_gt_movie.set(movies[0])
        self.movieName_plot = movies[0]
        choicebox_gt_movie= ttk.Combobox(self.top_plotting, textvariable= choice_gt_movie, width = 20,font = ("Arial", 10))
        choicebox_gt_movie['values']= choices_m
        choicebox_gt_movie['state']= 'readonly'
        choicebox_gt_movie.grid(column = 1, row =3)
        choice_gt_movie.trace('w', callback)
        if "Eyetracking" in os.listdir(self.inputDir): 
            button_interactivePlot= ttk.Button(self.top_plotting,text='Plot', command = self.plot)
        else:
            button_interactivePlot= ttk.Button(self.top_plotting,text='Plot', command = self.plot_NoEyetracking)

        #label_info_plot = tk.Label(self.top_plotting, text = "Plotting can take a few second!", fg = "Black")
        #label_info_plot.grid(column = 1, row = 9)

        button_interactivePlot.grid(column = 1, row = 4)
        self.exitButton = ttk.Button(self.top_plotting, text = "Exit", command = self.close)
        self.exitButton.grid(column = 1, row = 5)
        

    def buttonFunc_save_params(self):
        foldername = "csv results"
        os.chdir(self.dataDir) 
        if not os.path.exists(foldername):
           os.makedirs(foldername)
        os.chdir(foldername)
        params = self.modelResultDict[self.subjectName]["modelContrast"]["parameters"]
        if self.RF == "HL":
            if self.gazecentered:
                paramNames =  ['r', 'rmse'] + self.modelResultDict[self.subjectName]["modelRegularization"]["parametersNames"]              
                params = np.insert(params,0,self.r)
                params = np.insert(params,1,self.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.subjectName}_parameters.csv")
            else:
                paramNames = ["r","rmse","n_luminance", "tmax_luminance", "n_contrast", "tmax_contrast", "weight_contrast"]
                params = np.insert(params,0,self.r)
                params = np.insert(params,1,self.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.subjectName}_parameters.csv")
        elif self.RF == "KB":
            if self.gazecentered:
                paramNames =  ['r', 'rmse'] + self.modelResultDict[self.subjectName]["modelRegularization"]["parametersNames"]                
                params = np.insert(params,0,self.r)
                params = np.insert(params,1,self.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.subjectName}_parameters.csv")
            else:
                paramNames = ["r","rmse","theta_luminance", "k_luminance", "theta_contrast", "k_contrast", "weight_contrast"]
                params = np.insert(params,0,self.r)
                params = np.insert(params,1,self.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.subjectName}_parameters.csv")
                
        showinfo(title = "", message = "Parameters saved!")
        # save the modeling results (data do not need to save because they are not different from the model without regularization)
        if self.regularizationType == "ridge":
            foldername = "csv results"
            os.chdir(self.dataDir) 
            if not os.path.exists(foldername):
               os.makedirs(foldername)
            os.chdir(foldername)
            params = self.modelResultDict[self.subjectName]["modelRegularization"]["parameters"]
            if self.RF == "HL":
                paramNames =  ['r', 'rmse'] + self.modelResultDict[self.subjectName]["modelRegularization"]["parametersNames"]
                params = np.insert(params,0,self.r)
                params = np.insert(params,1,self.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.subjectName}_parameters_regularization.csv")    
            elif self.RF == "KB":
                paramNames =  ['r', 'rmse'] +self.modelResultDict[self.subjectName]["modelRegularization"]["parametersNames"]
                params = np.insert(params,5,1)
                params = np.insert(params,0,self.r)
                params = np.insert(params,1,self.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.subjectName}_parameters_regularization.csv")
            

        #tk.Label(text = "Parameters saved!", fg = "green").grid(column = 0, row = 18)
        
    def buttonFunc_save_model_prediction(self):
        self.nextButton.grid_forget()
        self.exitButton.grid_forget()
        foldername = "csv results"
        os.chdir(self.dataDir) 
        if not os.path.exists(foldername):
           os.makedirs(foldername)
        os.chdir(foldername)
        # modeling result (without regularization)
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

        df.to_csv(f"{self.subjectName}_modelPrediction.csv")
        # with regularization
        y_pred = self.modelResultDict[self.subjectName]["modelRegularization"]["predAll"] 
        lumConv = self.modelResultDict[self.subjectName]["modelRegularization"]["lumConv"] 
        contrastConv = self.modelResultDict[self.subjectName]["modelRegularization"]["contrastConv"] 
        if hasattr(self, "filename_csv"):
            self.sampledpupilData_z = (self.sampledpupilData -np.nanmean(self.sampledpupilData)) /np.nanstd(self.sampledpupilData)
            df = pd.DataFrame(np.vstack([self.sampledTimeStamps,self.sampledpupilData, y_pred,lumConv,contrastConv]).T)
            df.columns = ["timeStamps", "Actual pupil (z)", "Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]
        else:
            df = pd.DataFrame(np.vstack([self.sampledTimeStamps,y_pred,lumConv,contrastConv]).T)
            df.columns = ["timeStamps", "Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]

        df.to_csv(f"{self.subjectName}_modelPrediction_regularization.csv")
        showinfo(title = "", message = "Model prediction result saved!")
        #tk.Label(text = "Model prediction result saved!", fg = "green").grid(column = 1, row = 18,sticky=tk.E)
        self.exitButton.grid(column = 1, row = 20)

        
    def plot(self):
        # This step have to be done after pupil prediction
        eeObj = event_extraction()
        eeObj.mapType = self.mapType
        eeObj.degVF = self.degVF
        eeObj.eye_to_screen =self.eye_to_screen
        eeObj.eyetracking_width =self.eyetracking_width
        eeObj.eyetracking_height =self.eyetracking_height
        eeObj.screen_width =self.screen_width
        eeObj.nWeight = self.nWeight
        eeObj.createMapMask()
        plotObj = interactive_plot()
        # select a subject to plot
        plotObj.window = self.top_plotting
        plotObj.subject = self.subjectName_plot
        plotObj.outputDir = self.outputDir
        plotObj.movieName = self.movieName_plot.split(".")[0]
        # other parameters
        plotObj.useApp = self.useApp
        plotObj.dataDir = self.dataDir
        plotObj.movieDir = self.movieDir
        plotObj.finalImgWidth = eeObj.finalImgWidth
        plotObj.finalImgHeight = eeObj.finalImgHeight

        plotObj.skipNFirstFrame =self.skipNFirstFrame
        plotObj.eyetracking_height = self.eyetracking_height
        plotObj.eyetracking_width = self.eyetracking_width
        plotObj.videoRealHeight = self.videoRealHeight
        plotObj.videoRealWidth = self.videoRealWidth
        plotObj.screenBgColorR = self.screenBgColorR
        plotObj.screenBgColorG = self.screenBgColorG
        plotObj.screenBgColorB = self.screenBgColorB
        plotObj.videoScreenSameRatio = self.videoScreenSameRatio 
        plotObj.videoStretched = self.videoStretched
        plotObj.nWeight = self.nWeight
        plotObj.gazecentered = self.gazecentered
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        plotObj.plot()
        
    def plot_NoEyetracking(self):
        # This step have to be done after pupil prediction
        plotObj = interactive_plot()
        plotObj.window = self.top_plotting
        # subject and movie to plot
        plotObj.subjectName = self.subjectName_plot
        plotObj.movieName = self.movieName_plot.split('.')[0]
        # other parameters
        plotObj.useApp = self.useApp
        plotObj.dataDir = self.dataDir
        plotObj.A = 1
        plotObj.skipNFirstFrame =self.skipNFirstFrame
        
        plotObj.outputDir = self.outputDir
        plotObj.movieDir = self.movieDir
        # plotObj.videoScreenSameRatio = videoScreenSameRatio 
        # plotObj.videoStretched = videoStretched
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
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
        self.entry_visualanglemap.grid_forget()
        self.entry_screenwidthcm.grid_forget()
        self.entry_screeneyecm.grid_forget()
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
        os.chdir(self.subjectDir)
        self.df_eyetracking.to_csv(f"{self.movie}.csv",index=False, header=False)
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
    
    