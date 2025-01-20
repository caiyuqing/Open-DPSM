# What do regional weights mean and how are they extracted?

This illustration is made for readers for are interested in the mechanism of how the model extracts regional weights. 

You will act as the model and your task is to find the best combination of regional weights in a artifical pupil and luminance change data. The procedure is exactly the same what the model does! 

It has two formats. Both will open the same plot. 

- For those who don't use Python, a .exe form of the GUI can be found on https://osf.io/qvn64/. Download *interactive_illustration.zip* and unzip it. Then run *interactive_illustration.exe*. The executable file will take about 10 seconds to open.
  
- A code form: interactive_illustration_simple.py in this folder.

## Packages
The illustration depends on those packages: [numpy](https://numpy.org/install/), [scipy](https://scipy.org/install/), [matplotlib](https://matplotlib.org/stable/users/installing/index.html). Install them first.

## Constitutes of the plot

The initial plot looks like this. This shows the inputs of the model:

**(1st row, right side)** Luminance change extracted from the movie, respectively for the left and right visual field

**(4th row, right side)** Observed pupil response (4th row, left side)

**(1st row, left side)** A predetermined shape of response function for pupil responses to luminance changes 

![image](https://github.com/user-attachments/assets/93afd008-c4cb-482d-b31d-7a7894b520b1)


Then the model (or you) starts to select values for regional weights. Here is a random example for the combination of weight1 = 2, weight2 = 1. 

**(2nd row, right side)** The luminance changes are first convolved with the response function, resulting in two convolved traces for the left and right separately. In addition, the two traces were multiplied with regional weights. 

**(3nd row, right side)** Then the two convolved traces are summed up to a single result.

**(4th row, right side)** The single convolved result is then accumulated because pupil responses to luminance changes are sustained. And this is the predicted pupil size changes with this combination of weights (brown line). The model performance evaluation (r and RMSE) is also presented on the left. Note that as this is a fake data, the best weights will give us a r of 0.98.

![image](https://github.com/user-attachments/assets/dded4f94-1368-4f20-8aea-65dc01518471)

This looks quite off. Let's try to decrease weight1.

![image](https://github.com/user-attachments/assets/409d621e-d01f-4d49-a08b-f0619df5f034)

This time the result looks better but not yet perfect. Can you try other combinations to find the answer? 

Please also observe that the regional weights are not determined by the relative strengths of luminance changes in the two sides.

After you find the answer, can you multiply weight1 and weight2 with a random number (above 0) and test the answer again? You will find that any number would work. This means that the absolute values of regional weights do not mean anything. Only the relative values of them matters!
