# What do regional weights mean and how do they calculated?

This illustration is made for readers for are interested in the mechanism of how the model extracts regional weights. 

You are acting as the model. Your task is to find the best combination of regional weights in a artifical pupil and luminance change data. The procedure is exactly the same what the model does! 

It has two formats. Both will open the same plot. 

- For those who don't use Python, a .exe form of the GUI can be found on https://osf.io/qvn64/. Download *interactive_illustration.zip* and unzip it. Then run *Open-DPSM.exe*. The executable file will take about 10 seconds to open.
  
- A code form: interactive_illustration_simple.py

## Packages
The illustration depends on those packages: [numpy](https://numpy.org/install/), [scipy](https://scipy.org/install/), [matplotlib](https://matplotlib.org/stable/users/installing/index.html). Install them first.

## Constitutes of the plot

The initial plot looks like this. This shows the inputs of the model:

(1) Luminance change extracted from the movie, respectively for the left and right visual field (1st row, right side)

(2) Observed pupil response (4th row, left side)

(3) A predetermined shape of response function for pupil responses to luminance changes (1st row, left side)

![image](https://github.com/user-attachments/assets/e59db993-b608-4350-87d4-545d813d6a43)

Then the model (or you in the current illustration) starts to select values for regional weights. Here is a random example for the case of weight1 = 2, weight2 = 1. 

**(2nd row, right side)** The luminance changes are first convolved with the response function, resulting in two convolved traces for the left and right separately. In addition, the two traces were multiplied with regional weights. 

**(3nd row, right side)** Then the two convolved traces are summed up to a single result.

**(4th row, right side)** The single convolved result is then accumulated because pupil responses to luminance changes are sustained. And this is the predicted pupil size changes with this combination of weights (brown line). The model performance evaluation (r and RMSE) is also presented on the left. Note that as this is a fake data, the best weights will give us a r of 0.98.

Now it's your turn! Can you change the combination of weights to find answer? (Also observe that the regional weights are not determined by the relative strengths of luminance changes in the two sides).

After you find the answer, can you multiple weight1 and weight2 with a random number (above 0) and test the answer again? You will find that any number would work. This means that the absolute values of regional weights do not mean anything. Only the relative values of them matters!

![image](https://github.com/user-attachments/assets/226df435-ec98-4dec-9ab3-f80b8bd2a0cc)


