# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:44:34 2025

@author: 7009291
"""
# Instruction: change the value for weight1 and weight 2 and run the script to until you the best combination of the two weights
weight1 = 0
weight2 = 0
# Hint2: drag to the end of the see the correct answer :)

##############################################Keep the folowing unchanged#####################################
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
# import tkinter as tk
# from tkinter import ttk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import scipy
# Initialize data and parameters
t_rf = np.linspace(0, 3, 11)
n = 10.1
tmax = 0.93
rf = t_rf**n * np.exp(-n * t_rf / tmax)
rf = rf / max(rf)
t_rf2 = np.linspace(0,3,100)
rf2 = t_rf2**n*np.exp(-n*t_rf2/tmax)
rf2 = rf2/max(rf2)
noise_array = np.random.normal(0, 0.25, 44)
linewidth = 3
s = 40
np.random.seed(42)
events1 = np.array([0, -4, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0])
events2 = np.array([0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0])

t = np.linspace(0, 12, 44)
t2_ = np.linspace(t.min(), t.max(), 500)
fontsize = 15
# Define plot update function
def zscore(x):
    zscore = (x-np.nanmean(x))/np.nanstd(x)
    return zscore

    
evnets1_plot = events1 + 0.01
evnets2_plot = events2 - 0.01

convRaw_event1 = np.convolve(-1 * events1, rf)[:len(events1)]
convRaw_smooth_event1 = make_interp_spline(t, convRaw_event1)(t2_)

convRaw_event2 = np.convolve(-1 * events2, rf)[:len(events2)]
convRaw_smooth_event2 = make_interp_spline(t, convRaw_event2)(t2_)

convRaw_event_all = convRaw_event1 * weight1 + convRaw_event2 * weight2
convRaw_smooth_event_all = convRaw_smooth_event1 * weight1 + convRaw_smooth_event2 * weight2
if weight1 == 0 and weight2 == 0:
    weight1_text = "?"
    weight2_text = "?"
else:
    weight1_text = str(weight1)[0:3]
    weight2_text = str(weight2)[0:3]

# add response function
fig, axes = plt.subplots(5, 2, figsize=(16, 13),gridspec_kw={'width_ratios': [0.25,1.3], "height_ratios": [1,1,1,1,0.2]})

axes[0,0].scatter(t_rf,rf,s = s, color = "black")
axes[0,0].plot(t_rf2, rf2, color = "black", linewidth = linewidth)
axes[0,0].spines['top'].set_visible(False)
axes[0,0].spines['right'].set_visible(False)
axes[0,0].spines["bottom"].set_linewidth(linewidth)
axes[0,0].spines["left"].set_linewidth(linewidth)
axes[0,0].set_xlabel("Time from event onset (s)", fontsize = fontsize-2)
axes[0,0].set_ylabel("Pupil size $(z)$",fontsize = fontsize-2)
axes[0,0].set_yticks([0,0.5,1])
axes[0,0].set_yticklabels(['0', '0.5', '1'])
axes[0,0].set_xticks([0,1,2,3])
axes[0,0].set_xlim(0,3.2)
axes[0,0].tick_params(axis='x', which='major', labelsize=fontsize-2)
axes[0,0].tick_params(axis='y', which='major', labelsize=fontsize-2)
# add illustration for weights
# Create a figure and axes
img = mpimg.imread('frame00132.png')  # Example: 'image.jpg'

axes[1,0].imshow(img,zorder = 0,extent=[30,  img.shape[1]-30, img.shape[0]-50,80])

# Add a blue square
blue_square = Rectangle((25, 80), img.shape[1]/2-40, img.shape[0]-130, facecolor="none", edgecolor="blue", linewidth = 5,zorder = 20)
axes[1,0].add_patch(blue_square)

# Add a red square next to the blue square
red_square = Rectangle(( img.shape[1]/2, 80),  img.shape[1]/2-35, img.shape[0]-130, facecolor="none", edgecolor="red",linewidth = 5,zorder =20)
axes[1,0].add_patch(red_square)

# Add text at the center of the blue square
axes[1,0].text(img.shape[1]/4,img.shape[0]/2, weight1_text, ha="center", va="center", fontsize=fontsize+10, color="blue",zorder = 30)
axes[1,0].text(img.shape[1]/4, img.shape[0]+35, "Region1", ha="center", va="center", fontsize=fontsize, color="blue",zorder = 30)

# Add text at the center of the red square
axes[1,0].text(img.shape[1]*3/4, img.shape[0]/2, weight2_text, ha="center", va="center", fontsize=fontsize+10, color="Red",zorder = 30)
axes[1,0].text(img.shape[1]*3/4, img.shape[0]+35, "Region2", ha="center", va="center", fontsize=fontsize, color="Red",zorder = 30)
# Set limits to ensure both squares are fully visible

axes[1,0].set_xlim(0, img.shape[1])  # Width of the image
axes[1,0].set_ylim(img.shape[0], 0)  # Height of the image (invert y-axis)
# Remove axes for a cleaner look
axes[1,0].axis("off")
#axes[4,1].text(0,0, s= "Instruction: Change the value of weight1 and weight2 (left) and press 'Update plot' to find the best predicted pupil size changes",  wrap = True, ha="left", va="top", fontsize=fontsize, color="Black",zorder = 30)
axes[4,0].axis("off")
axes[4,1].axis("off")

axes[3,0].axis("off")
axes[2,0].axis("off")

# Original first plot （luminance change)
for i in range(len(events1)):
    axes[0,1].vlines(x=t[i], ymin=0, ymax=evnets1_plot[i], color="blue", linewidth=linewidth)
    axes[0,1].scatter(t[i], evnets1_plot[i], s=s, facecolors='blue', edgecolors='blue')
    axes[0,1].vlines(x=t[i], ymin=0, ymax=evnets2_plot[i], color="red", linewidth=linewidth)
    axes[0,1].scatter(t[i], evnets2_plot[i], s=s, facecolors='red', edgecolors='red')

axes[0,1].axhline(y=0, color='black', linestyle='--')
axes[0,1].set_ylabel("Luminance change", fontsize=fontsize)
axes[0,1].set_title("")
#plt.axhline(y=0, color='black', linestyle='--')
axes[0,1].spines['top'].set_visible(False)
axes[0,1].spines['right'].set_visible(False)
axes[0,1].spines["bottom"].set_linewidth(linewidth)
axes[0,1].spines["left"].set_linewidth(linewidth)
axes[0,1].set_ylabel("Luminance change",fontsize = fontsize-2)
axes[0,1].yaxis.set_label_coords(-0.04, 0.5)
axes[0,1].tick_params(axis='y', which='major', labelsize=fontsize-2)
axes[0,1].set_ylim((-7.5,5.5))
axes[0,1].set_yticks([-6, -3,0,3,6])
axes[0,1].set_xticks([])

# Dynamic second plot (convolved result)
axes[1,1].scatter(t, convRaw_event1 * weight1, color="blue", s=s)
line1, = axes[1,1].plot(t2_, convRaw_smooth_event1 * weight1, color="blue", linewidth=linewidth)
axes[1,1].scatter(t, convRaw_event2 * weight2, color="red", s=s)
line2, = axes[1,1].plot(t2_, convRaw_smooth_event2 * weight2, color="red", linewidth=linewidth)
axes[1,1].legend(handles = [line1, line2], labels = ["Region1-convoluted", "Region2-convoluted"],frameon=False, fontsize = fontsize,loc = "lower center")#bbox_to_anchor=(0.75,0.45))
#axes[1,1].set_ylim((-8,8))
axes[1,1].spines['top'].set_visible(False)
axes[1,1].spines['right'].set_visible(False)
axes[1,1].spines["bottom"].set_linewidth(linewidth)
axes[1,1].spines["left"].set_linewidth(linewidth)
axes[1,1].set_xlabel("", fontsize = fontsize)
axes[1,1].set_ylabel("Convolved result",fontsize = fontsize-2)
axes[1,1].yaxis.set_label_coords(-0.04, 0.5)
axes[1,1].tick_params(axis='y', which='major', labelsize=fontsize-2)
axes[1,1].tick_params(axis='x', which='major', labelsize=fontsize-2)
if weight1 == weight2 ==0:
    axes[1,1].set_yticks([-2,0,2])

#axes[1,1].set_yticks([-14,0,14])
axes[1,1].set_xticks([])
# Dynamic third plot
axes[2,1].scatter(t, convRaw_event_all, color="green", s=s)
line, = axes[2,1].plot(t2_, convRaw_smooth_event_all, color="green", linewidth=linewidth)
axes[2,1].legend(handles=[line], labels = ["Sum of all regions"],frameon=False, loc = "lower center",fontsize = fontsize)
#plt.ylim((-14,14))

axes[2,1].spines['top'].set_visible(False)
axes[2,1].spines['right'].set_visible(False)
axes[2,1].spines["bottom"].set_linewidth(linewidth)
axes[2,1].spines["left"].set_linewidth(linewidth)

axes[2,1].set_xlabel("", fontsize = fontsize)
axes[2,1].set_ylabel("Combine convolved result",fontsize = fontsize-2)
#axes[0,1].set_title("Combine convolved result")

axes[2,1].yaxis.set_label_coords(-0.04, 0.5)
axes[2,1].tick_params(axis='y', which='major', labelsize=fontsize-2)
axes[2,1].tick_params(axis='x', which='major', labelsize=fontsize-2)
#ax.set_ylim((-2,2))
if weight1 == weight2 ==0:
    axes[2,1].set_yticks([-2,0,2])
axes[2,1].set_xticks([])
# final plot (result)
real_all = convRaw_smooth_event1*1 + convRaw_smooth_event2*3

convRaw_smooth_event_all_cum = zscore(np.nancumsum(real_all))
convRaw_event_all_cum = convRaw_smooth_event_all_cum[[int(i) for i in np.linspace(0,499,44)]]

convRaw_event_all_cum_real = convRaw_event_all_cum+ noise_array
convRaw_smooth_event_all_cum_real = make_interp_spline(t, convRaw_event_all_cum_real)
convRaw_smooth_event_all_cum_real = convRaw_smooth_event_all_cum_real(t2_)
axes[3,1].scatter(t, convRaw_event_all_cum_real, s = s, color = "black")
line1,= axes[3,1].plot(t2_,convRaw_smooth_event_all_cum_real,linewidth =linewidth, color = "black")

predicted_all =  convRaw_smooth_event1*weight1 + convRaw_smooth_event2*weight2

convRaw_smooth_event_all_cum = zscore(np.nancumsum(predicted_all))
convRaw_event_all_cum = convRaw_smooth_event_all_cum[[int(i) for i in np.linspace(0,499,44)]]
axes[3,1].scatter(t, convRaw_event_all_cum, s = s, color = "brown")
line2, =  axes[3,1].plot(t2_,convRaw_smooth_event_all_cum,linewidth =linewidth, color = "brown")
#axes[0,2].text(-1,0.5, s = f'Response function (left) is convolved with luminance changes in two regions separately', ha="left", va="center", fontsize=fontsize, color="black",zorder = 40,wrap=True)
if weight1 == 0 and weight2 == 0:
    axes[3,1].legend(handles = [line1],labels = ['real'],frameon=False, loc = "lower center",fontsize = fontsize)
else:
    axes[3,1].legend(handles = [line1, line2],labels = ['real', 'predicted'],frameon=False, loc = "lower center",fontsize = fontsize)
#axes[3,1].ylim((-4,4))
axes[3,1].set_yticks([-3,0,3])
axes[3,1].spines['top'].set_visible(False)
axes[3,1].spines['right'].set_visible(False)
axes[3,1].spines["bottom"].set_linewidth(linewidth)
axes[3,1].spines["left"].set_linewidth(linewidth)

axes[3,1].set_xlabel("Time(s)", fontsize = fontsize)
axes[3,1].set_ylabel("Pupil size $(z)$",fontsize = fontsize)
axes[3,1].yaxis.set_label_coords(-0.04, 0.5)
axes[3,1].tick_params(axis='y', which='major', labelsize=fontsize-2)
axes[3,1].tick_params(axis='x', which='major', labelsize=fontsize-2)
#ax.set_ylim((-2,2))
#ax.set_yticks([-6,0,6])
axes[3,1].set_xticks([0,2,4,6,8,10,12])

# add r and rmse
rmse= np.sqrt(np.nanmean((convRaw_smooth_event_all_cum - convRaw_smooth_event_all_cum_real) ** 2))
r,p=scipy.stats.pearsonr(convRaw_smooth_event_all_cum,convRaw_smooth_event_all_cum_real)    
 
# print congradulations
if r.round(2) == 0.98:
    axes[3,0].text(0.3,0.5, s = f'r = {r.round(2)}\nRMSE = {rmse.round(2)}', ha="left", va="center", fontsize=fontsize, color="brown",zorder = 40)
    axes[3,0].text(0.3,0, s = f'Correct answer!', ha="left", va="center", fontsize=fontsize, color="green",zorder = 40)

else:
    axes[3,0].text(0.3,0.5, s = f'r = {r.round(2)}\nRMSE = {rmse.round(2)}', ha="left", va="center", fontsize=fontsize, color="brown",zorder = 40)

plt.show()

# The correct answer is: weight1 =1 and weight2 = 3