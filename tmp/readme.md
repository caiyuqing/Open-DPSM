# Open Dynamic Pupil Size Modeling (Open-DPSM) Toolbox #
**Please read this page carefully for the instruction of each step before using the toolbox.**
## Table of Contents
- [Introduction](#introduction)
- [GUI](#gui)
- [Code](#code)
- [Example data](#example-data)
- [No eyetracking data](#no-eyetracking-data)

## Introduction ##
If you encounter any issue, feel free to contact me: y.cai2[at]uu.nl. 

Please cite: 

- Cai, Y., Strauch, C., Van der Stigchel, S., & Naber, M. (2023). Open-DPSM: An open-source toolkit for modeling pupil size changes to dynamic visual inputs. Behavior Research Methods. https://doi.org/10.3758/s13428-023-02292-1

- Yuqing Cai, Stefan Van der Stigchel, Julia Ganama, et al. Uncovering covert attention in complex environments with pupillometry. Authorea. November 08, 2024.

**Summary for features of v3**

Same as v1 and v2:

- The toolbox provides functions for (1) Visual event extraction from video input; (2) Pupil response prediction/modeling; (3) Interactive plotting.
  
- Incorporated different features to estimate regional weights, including the shape and number of weights (see below).

- Can also generate expected pupil trace if there is no eyetracking data

Updated features:

- Previous version only allowed to extract visual events and perform modeling for one movie. The current version allows processing multiple subjects and movies at the same time.

- The modeling is performed on all the movies for individual subjects

**Data folder structure**

Please arrange your data folder as the following structure.  
```
Example
|
└───Input
│     │
|     └───Eyetracking
│     │     │
│     │     └───p01
│     │     |     |
|     │     │     └───01.csv
|     │     │     └───02.csv
|     │     │     └───...
│     │     └───p02
│     │     |     |
|     │     │     └───01.csv
|     │     │     └───02.csv
|     │     │     └───...
│     │     └───...
|     └───Movies
│     │     │
│     │     └───01.mp4
│     │     └───02.mp4
│     │     └───...
└───Output
```

- Example: is the head folder. Can be any name.

- Input, Output, Eyetracking, Movies: Those folders have to be arranged in this exact structure and name beforehand

- p01, p02,etc: Participant name. Can be anything.

- 01.csv, 02.csv,etc: Eyetracking data for movie "01", "02". The names must be the same as movie names ("01.mp4", "02.mp4"). Different participant can have different number of movies but they must all exist in the "Movies" folder 

- 01.mp4, 02.mp4,etc: Movie files. The names must be the same as the eyetracking data.

- The toolbox will automatically model all the movies together under one participant's folder. If you don't want to model all the movies, just remove the files from the participant folder.

**Eye-tracking data**

   - Load a .csv file with four columns, in the **_exact order_** of:

  1. Timestamps 
  2. Gaze position (x) 
  3. Gaze position (y) 
  4. Pupil size
   
   - See Eyetracking folder in [Example](Example) for exemplary eye-tracking csv files.

   - The toolbox assume that the eyetracking data starts and ends together with the movie. If not, please cut if first.
      
   - It is important that the **left corner of the screen** should have the gaze position coordinates x = 0 and y =0. Please convert the gaze positions if it is not the case.

   - Pupil size should be measured in diameter. We therefore recommend conversion if data are not given in millimeters before entering data into the toolbox (see PsPM, Korn et al., 2017 for a number of conversion techniques).
   
   - It is recommended that the gaze and pupil data are preprocessed: blinks should be removed, possible foreshortening errors corrected before loading into Open-DPSM. A function for blink removal is provided in the classes.preprocessing, but not incorporated in the toolbox. For pupil foreshortening errors, the current toolbox has not incorporated any correction function. One generally accepted method is from Hayes & Petrov (2016) and a function to implement their method has been provided in PsPM (https://bachlab.github.io/PsPM/).”
 
   - Gaze position and pupil size can be data of the left eye or the right eye or an average of both eyes, depending on the preference of the user.

   - The unit of Timestamps should be in seconds.  
     
   - No index_col and header should be included in the csv file 

     > Note: Eye-tracking data is not a must for the toolbox. If no eye-tracking data is loaded, the toolbox will extract the visual events from the movie and generate a predicted pupil trace based on the parameters we obtained from our data.

**Movies**

  - Formats tested: .mp4 (recommanded),.avi,.mkv,.mwv,.mov,.flv,.webm. As we use OpenCV to process the video, in principle, any format that can be used in OpenCV can be used.

  - The movies do not have to be presented full screen. Further information will be required later.

  - However, the movies must be displayed with the same size and it must be presented at the center of the screen. Please modify the movie if it is not.

  - The movies can also be screen recordings of the experiment or video recordings of the screen.

**Regional weights**

One purpose of the toolbox is to predict luminance changes (visual events) in which part of the movie contribute more to pupil size changes (i.e., Regional weights). Refer to our papers for details.

The default map for regional weights in this version of toolbox is circular and the default number of regional weights is 44. Users can choose other map type (mapType) and number of weight (nWeight)

Currently, two 'mapType' are available: circular and square

- For 'circular' map, users can choose between 2 weights, 20 weights or 44 weights.

- For 'square' map, users can choose between 2 weights, 6 weights (original version, also used in v1) or 48 weights.

Allocation of regions:
  
<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/aaa74ec3-7aba-459a-9ead-03ebb547733e" />


**Open-DPSM can be used in two formats:**

- [GUI](#gui): [main_app.py](main_app.py) (For those who don't use Python, a .exe form of the GUI can be found on https://osf.io/qvn64/. Download *Open-DPSM.zip* and unzip it. The GUI version of Open-DPSM can be started directly by running *Open-DPSM.exe* without Python. Please note that the *App_fig* folder should be in the same directory as the .exe. The executable file will take about 10 seconds to open. Also, using this form means that the user accepts all the default parameters as they cannot be changed.)
- [Code](#code): [main.py](main.py) (Contain notes and instructions and should be mostly self-explanatory)

Also see: [Example data](#example-data) for details of the data provided as an exemplary user-case


**Loading the toolbox**

No installation is required. Simply clone or download the current repository.

**Python environment**

We recommand Spyder IDE because the toolbox has been built and tested with the Spyder IDE (version 5) with Python 3.9.7. 

Besides Spyder, Jupiter Notebook (6.4.5)/JupiterLab (3.2.1) and PyCharm (2013.1.4) have also been also tested. 

**Packages**

The toolbox depends on those packages: [numpy](https://numpy.org/install/), [pandas](https://pandas.pydata.org/docs/getting_started/install.html), [scipy](https://scipy.org/install/), [OpenCV](https://pypi.org/project/opencv-python/), [moviepy](https://zulko.github.io/moviepy/install.html), [matplotlib](https://matplotlib.org/stable/users/installing/index.html), [pillow](https://pillow.readthedocs.io/en/stable/installation.html), [sklearn](https://scikit-learn.org/1.6/install.html)

Please refer to their installation instructions and make sure that they have been correctly installed before using it.

## GUI

- Run *main_app.py* to start GUI

Note: With Jupiter notebook/JupiterLab, create a new .ipynb file under the same directory and run the following codes to start the GUI:
```python
import os
script_path = "main_app.py"
os.system(f'python {script_path}')
```
### Welcome page: Loading eye-tracking data and movie & select parameters for event extraction and modeling
<img width="745" height="444" alt="image" src="https://github.com/user-attachments/assets/685d7ae0-f636-4c58-a524-98ae11bf1096" />

**Open folder**: select the head folder of the data (e.g., "Example" folder)

**Select parameters** (If you are not sure, please use the default ones):

(1) Event extraction mode: gaze-centered or screen-centered (default: gaze-centered)

(2) Map type: circular or square (default: circular; Exception: When screen-centered, map type can only be square.) 

(3) Number of weights: 2 (circular or square), 6 (square), 20 (circular), 44 (circular), 48 (square) (default: 44)


### Entering the information page
<img width="743" height="446" alt="image" src="https://github.com/user-attachments/assets/3338b951-a921-4342-ba6b-907a8768d4d6" />

Information that needs to be entered manually by the user:

- Maximum luminance of the screen: This is the physical luminance level of the luminance when the color white is shown and can be measured with a photometer

- Resolution for the coordinate system of eye-tracking data (also the resolution of the screen): Height and width of the eye-tracking coordinate system. Note that it should be the absolute length, not the maximum coordinates of the eye-tracking coordinate system (in pixels)

- Resolution for the movies: Height and width of the video displayed on the screen. It can be the same or smaller than the resolution for eye-tracking data/screen (if movies were not shown full screen) (see Figure below)

- Color (r,g,b values) of the screen outside the movie (if the movies were not shown full screen) (see Figure below)

- Physical width of the screen (in cm)

- Physical distance between the eye and the monitor (in cm)

- Factor for visual angle of the regional weight map relative to the visual angle of the movie (degVF_param): The default value is 1. It means that if your movie is displayed with a visual angle of 60°, then the size of the map used for event extraction is also 60° (see Figure below)

- Difference in durations: The toolbox will extract the duration of the movie from the movie file and the duration of the eye-tracking data based on the last row of the timestamps in csv files. It is highly possible that the two lengths are not exactly the same (But do make sure that their difference is not exceeding 1s). If this is the case, there are two options:

  (1) Stretch to match: This means that the two lengths will be considered the same (no matter which one is longer) and the eye-tracking data will be downsampled to the framerate of the video with its original length (default)
  
  (2) Cut the last part of the longer file: This means that the file with a longer length will be chopped at the end.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/ae1fcfe5-5a03-46ff-acd1-2f02878d48b0" />

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/94d95987-243b-482f-a663-b5ef6de5523c" />


### Check information & event extraction
<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/87ec9359-16b0-4409-97b7-7d8c354b1e0d" />

- Display basic information about the eye-tracking data and movie data for one movie from one subject. If anything is not correct, please exit and modify the data.

- Press `Next` then `Start event extraction` to do event extraction for one movie

- When it is completed, event trace (luminance changes) per image region will be automatically saved as a pickle file in "Visual events" folder (under "Output" folder). Press `Next movie` to move on to the next eye-tracking file.

- If use the screen-centered mode, event extraction will only be performed to one "participant" named "sc". All the participants will use the same events extracted in modeling step.

- If event extraction has done for this movie, this step will be skipped and a pop up window will show to remind the user.
    
### Pupil prediction

After all the movies were extracted, modeling window will pop up. 

<img width="297" height="83" alt="image" src="https://github.com/user-attachments/assets/8f13ae26-fe51-42b3-9090-359e96f29833" />

- `Start modeling`: Model pupil size changes to the visual events (all the existing movies together for each participant iteratively). (If no eye-tracking data are available, it will generate a prediction of pupil trace with a set of free parameters acquired with our data. When it is completed, the model performance will be printed on the left.)

- If event extraction has done for this participant, this step will be skipped and a pop up window will show to remind the user.

When the modeling for one participant is done:

- A pop-up window will shown to display the model performance (Correlation coefficient and root-mean squared error between predicted and observed pupil size changes)

<img width="179" height="107" alt="image" src="https://github.com/user-attachments/assets/791d9fee-766c-4e98-bcf1-add39127255b" />

- Free parameters found by the model and the model performance for each participant will be saved as two columns (first column: names; second column: values) in a .csv file named *"[subjectName]_parameters.csv"* in "csv results" folder under "Output" folder(Only when eye-tracking data is available)

- Observed pupil size and predicted pupil size will be saved as a .csv file named *"[subjectName]_modelPrediction.csv"*. Predicted pupil size (z-standardized) will be provided with three columns, one for the combined prediction with both luminance and contrast change, one for prediction with luminance change only, and one for prediction with contrast change only.

### Interactive plot

After all the movies were modeled, plotting window will pop up. Select one participant and one movie to make an interactive plot.
 
<img width="269" height="135" alt="image" src="https://github.com/user-attachments/assets/6588f859-3430-40ec-a5c7-9847a5723c5f" />

Press `plot` will open an interactive plot window:

- Drag the scale at the bottom to select one frame to display.

<img width="548" height="364" alt="image" src="https://github.com/user-attachments/assets/090ab43c-e3dc-4cd8-a2ca-7399f1ecdd7a" />

And an button window:

- Select traces to display

- Save figure or parts of figure in "Figure" folder (under Output folder)

<img width="95" height="191" alt="image" src="https://github.com/user-attachments/assets/46082595-b96b-478e-a37c-6e56a334ba19" />

## Code

- The code version uses the same three classes of functions as the GUI version. Hence, all the steps are nearly identical. Refer to [GUI](#gui) for details. 

- Similar to the GUI pages, Code version is divided into different sections.
  
- To start, open *main.py* and change all the things under the section "Information entered by the user" 
  
- If no eye tracking data are provided, it is important that the line ```eyetrackingDir``` line is commented out.
  

### Preprocessing and visual even extraction

- Run this part to perform visual event extraction iteratively for each participant and each movie (see [Check information & event extraction](#Check-information-&-event-extraction)) for more information
  
- The main codes of this section are:
  
```eeObj = event_extraction()```: create an object with the class event_extraction
 
```eeObj.event_extraction()```: call function *event_extraction* in the class event_extraction
 
All the other codes are to load data and predetermined parameters to the *eeObj* object

Similar to the GUI version, if the visual events pickle file is already in the "Visual events" folder, then this step can be skipped.

### Pupil modeling 
- Run this part only when eye-tracking data is available. The pupil size changes will be modeled with the visual events extracted in the previous step.
  
- The main codes of this section are:
  
```modelObj = pupil_prediction()```: create an object with the class pupil_prediction

```modelObj.pupil_prediction()```: call function *pupil_prediction* in the class pupil_prediction

All the other codes are for the purpose to load data and predetermined parameters to the *modelObj* object
 
- When it is completed, the model performance will be printed. Parameters selected by the model, model performance and model prediction will be saved (see [Pupil prediction](#pupil-prediction) for more information).

### Interactive plot
- This part of the code can be run together with the *Pupil modeling* part

- Run it will open a window with the interactive plot (see [Interactive plot](#interactive-plot) for more information)

- The main codes of this section are:
  
```plotObj = interactive_plot()```: create an object with the class interactive_plot

```plotObj.plot()```: call function *plot* in the class interactive_plot

All the other codes are to load data and predetermined parameters to the *plotObj* object

## No eyetracking data ##

If there is no eyetracking data, don't create the folder "Eyetracking" in "Input" folder

### Pupil prediction (no eye-tracking data)
- Run this part when eye-tracking data is not available

- The main codes of this section are:
  
```modelObj = pupil_prediction()```: create an object with the class pupil_prediction

```
if RF == 'HL':
    params = [9.67,0.19,0.8,0.52,0.3, 1,1,1,1,1] 
else:
    params = [0.12,4.59,0.14,6.78,0.28,1,1,1,1,1]
```
Load the parameters found with our data. RF = response function; HL = "Erlang gamma function". The first four parameters are free parameters in response functions (2 for luminance change and 2 for contrast change). The fifth parameter is the weight of the contrast response relative to the luminance response. The last 5 parameters are regional weights, which are set to 1 because we do not consider regional weights here as the visual angles in different datasets can be very different

```modelObj.pupil_predictionNoEyetracking(params)```: Calculate predicted pupil size with the parameters 

All the other codes are for the purpose to load data and predetermined parameters to the *modelObj* object

- When it is completed, pupil prediction will be saved (see [Pupil prediction](#pupil-prediction) for more information).
  
### Interactive plot

- This part of the code can be run after the *Pupil prediction (no eye-tracking data)* part

- Run it to open a window with the interactive plot (see [Interactive plot](#interactive-plot) for more information)

- The main codes of this section are:
  
```plotObj = interactive_plot()```: create an object with the class interactive_plot

```plotObj.plot_NoEyetracking()```: call function *plot_NoEyetracking* in the class interactive_plot

All the other codes are for the purpose to load data and predetermined parameters to the *plotObj* object

### Final note for code version
Similar to the GUI, we recommend keeping all predetermined parameters at default. However, if the users want to change any of them, those parameters can be found and adjusted in *settings.py*

## Example data
The folder "example" contains a sample eye-tracking data of a participant watching a 5-minute video of driving on the road. This clip is not one of the clips from our dataset and only serves as an example for a possible user case. 

To run the example, the user needs to first download the video from: https://www.youtube.com/watch?v=sIsegSg5tps (with a youtube downloader such as: https://en.savefrom.net/1-youtube-video-downloader-528en/). 

After downloading, please run:
```python
# import moviepy
from moviepy.editor import *
import os
# change path
MoviePath = [change the path to the folder you save the movie]
MovieName = [change the name to the name of the movie file]
# read movie
os.chdir(MoviePath)
clip = VideoFileClip(MovieName)
# cut the movie: from 00:17:00 to 00:22:00
clip_cut = clip1.subclip(1020,1320)
# check the movie resolution
w1 = clip1.w
h1 = clip1.h
ratio = w1/h1
print(ratio) # The ratio should be 1920/1080
clip_cut.write_videofile("driving.mp4") 
```
> Note: For those who are using the executable version of the toolbox, cut the movie into 5 minutes from 00:17:00 to 00:22:00 using any video editor application and rename it as "driving.mp4". If possible, also check the resolution of the video in the property of the file. It should be 1920x1080.

Then move *driving.mp4* to the example folder and the example is ready to go. 

As the visual event extraction has been done (saved in the "Visual event" folder), the user can skip the "Visual events extraction" section in the code version. In other folders, there are also exemplary results and figures generated with the toolbox. 

