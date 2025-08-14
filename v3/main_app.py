# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:36:42 2023

@author: 7009291
"""

#%% For starting the GUI, change the initialDir and run the code
# initial director and data director
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

initialDir =os.getcwd()# "D:\\Users\\7009291\\Desktop\\Movie pupil perimetry\\codes for both data\\Open-DPSM"# This should be the folder of the Open-DPSM

# Import packages
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
# import tkfunctions from classes.App
from classes.App import tkfunctions

# Create an object for tkinter
tkObj = tkfunctions()
tkObj.initialDir = initialDir
tkObj.run_tk()

