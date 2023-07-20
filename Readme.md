# Open Dynamic Pupil Size Modeling (Open-DPSM) Toolbox

## Table of contents
pending

## Python environment
The toolbox has been tested with the Spyder IDE (Pending: other IDE?)
? pending: how to install spyder

## Loading the toolbox
No installation is required. Simply clone the current repository by following the instructions of: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository

## Start Open-DPSM
There are two format of Open-DPSM:
- [GUI](#GUI): open main_app.py
- [Code](#Code): open main.py

Both of them use the three classes of functions in the "classes" folder

## GUI (pending: insert plots)
### Welcome page: Loading eye tracking data and movie
![Welcome figure](App_fig/Fig_welcome_page_App.PNG)

**Eye-tracking data**

   - Should be in the format of a .csv file with four columns, in the **_exact order_** of:

  1. Timestamps 
  2. Gaze position (x)
  3. Gaze position (y)
  4. Pupil size
 
   - It is recommended that the gaze and pupil data has been preprocessed for blink removal, foreshortening error etc. A function for blink removal is also provided in the classes.preprocessing file but not incorporated in the toolbox yet. You can choose to use it if necessary.
 
   - Gaze position and pupil size can be data of the left eye or the right eye or an average of both eyes, depending on the preference of the user

   - The unit of Timestamps should be in seconds or milliseconds. The default set-up is in seconds. Therefore, if in milliseconds, it will be converted to seconds in later steps. The unit of gaze and pupil data can be anything. They will be z-standardized in the later steps.
     
   - The first row should be the header

   - *Note: Eye-tracking data is not a must for the toolbox. If no eye-tracking data is loaded, the toolbox will extract the visual events from the movie and generate a predicted pupil trace based on the parameters we obtained from our data.*

**Movie**

  - The possible formats are listed in the welcome page. Those are the format that has been tested. As we use OpenCV to read the video, in principle, any format that can be used in OpenCV can be used.
  - The movie can be a screen recording of the experiment or a video recording of the screen. It is recommended that the video is full-screen, which also means that it will have the same aspect ratio as the eye-tracking data.

  - If the movie is not full-screen, further information will be required in the later pages. However, it is a must that the video is centered on the screen. Please cut it by yourself if it is not.

### Checking the information page
![Check the information figure](App_fig/Fig_check_information.PNG)

- Some basic information of the eye-tracking data and movie data will be extracted and shown. Note that one important feature of the movie is the aspect ratio, which will be used in the later data processing

- If anything is not correct, please exit and check the data.

### Entering the information page
![Enter the information figure](App_fig/Fig_enter_info.PNG)
Information that cannot be extracted from the files need to be enter manually by the user:

- The toolbox will extract the length of the movie based on its framerate and its number of frames available (ret = True in OpenCV) and the length of the eye tracking data based on the last row of the timestamp. It is highly possible that the two lengths are not exactly the same. If this is the case, there are two options:

  (1) Stretch to match: This means that the two lengths will be considered as the same (no matter which one is longer) and the eye-tracking data will be downsampled to the framerate of the video with its original length
  
  (2) Cut the last part of the longer file: This means that the file with a longer length will be chopped at the end before downsampling of the eye-tracking data.

- Maximum luminance of the screen: This is the physical luminance level of the luminance when the color white is shown and can be measured with photometer

- Spatial resolution of eye tracking data: Width and height of the eye tracking data. Note that it should be the absolute length, not the maximum number of the coordinate system.

### Select the position of the video relative to the screen 
If the aspect ratio (Height/width) of the video and the eye-tracking data are not the same, then the user needs to choose which of the following four positions is used: **A&B** Aspect ratio of the video is smaller than that of eye-tracking data; **C&D** Aspect ratio of the video is bigger than that of eye-tracking data; **A&C** The video is full-screen, which means that either height or width of the video is the same as the screen and the actual height and width of the video (relative to the eye-tracking data resolution) will be calculated; **B&D** The video is not full-screen, which means that the actual height and the width of the video are unknown and needs to be entered on the next page
![Screen video relation figure](App_fig/Fig_screen_video_relationship_all.png)

### Entering more information page (optional)
Based on the relative position of the video and screen, some extra information may be acquired
![Enter the information figure](App_fig/Fig_enter_info_extra.PNG)
- If the video is not full-screen, then the actual height and width of the video (relative to the eye-tracking data resolution) need to be entered. *Note that it is not the resolution in the video file.* For example, if the resolution of the eye-tracking data is 1000x500 and the physical height and width of the video displayed is half of the physical height and width of the screen, then 500 and 250 should be entered.


`Note: We recommend the users not to change any predetermined parameters. However, if the users want to change any, those parameters can be found in classes.App: tkfunctions.__init__
`
## Code
