# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:15:02 2023

@author: 7009291
"""

# Pre-determined parameters: Don't change unless absolutely sure
# boolean indicating whether or not to show frame-by-frame of each video with analysis results
showVideoFrames = False

# boolean indicating whether or not to skip feature extraction of already analyzed videos.
# If skipped, then video information is loaded (from pickle file with same name as video).
# If not skipped, then existing pickle files (one per video) are overwritten.
skipAlrAnFiles = True

# array with multiple levels [2, 4, 8], with each number indicating the number of vertical image parts (number of horizontal parts will be in ratio with vertical); the more levels, the longer the analysis
nVertMatPartsPerLevel = [3, 6]  # [4, 8, 16, 32]
aspectRatio = 0.75
imageSector = "6x8" # number of visual field regions (used for naming the new visual feature)
# integer indicating number of subsequent frames to calculate the change in features at an image part.
# it is recommended to set this number such that 100ms is between the compared frames.
# e.g. for a video with 24fps (50ms between subsequent frames), the variable should be set at 2.
nFramesSeqImageDiff = 2

# string indicating color space in which a 3D vector is calculated between frames
# Options: 'RGB', 'LAB', 'HSV', 'HLS', 'LUV'
colorSpace = "LAB"

# list with strings indicating which features to analyze
# If colorSpace = 'LAB', options: ["Luminance","Red-Green","Blue-Yellow","Lum-RG","Lum-BY","Hue-Sat","LAB"]
# If colorSpace = 'HSV', options: ["Hue","Saturation","Luminance","Hue-Sat","Hue-Lum","Sat-Lum","HSV"]
featuresOfInterest = [
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
scrGamFac = 2.2
# What is the ratio between gaze-centered coordinate system and the screen
A = 2
# Number of movie frames skipped at the beginning of the movie
skipNFirstFrame = 0
# gaze-contingent
gazecentered = True
############pupil prediction parameters##############
# Response function type
RF = "HL"
# same regional weights for luminance and contrast
sameWeightFeature = True 
# Basinhopping or minimizing
useBH = True
# iteration number for basinhopping
niter = 5
