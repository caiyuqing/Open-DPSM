# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:36:42 2023

@author: 7009291
"""

#%% For starting the GUI, change the initialDir and run the code
# Import packages
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import os
# import tkfunctions from classes.App
from classes.App import tkfunctions
# initial director and data director
initialDir = "D:\\Users\\7009291\\Desktop\\Movie pupil perimetry\\codes for both data\\Open-DPSM"# This should be the folder of the Open-DPSM
dataDir = initialDir + '/Example' # This should be the folder for the eyetracking data and the video data (does not have to be under the Open-DPSM directory)

os.chdir(initialDir)
# Create an object for tkinter
tkObj = tkfunctions()
tkObj.initialDir = initialDir
tkObj.dataDir = dataDir()
# run_tk()

tkObj.run_tk()

