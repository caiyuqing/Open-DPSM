# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 18:19:13 2025

@author: 7009291
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
import os
import pickle
import scipy
import math
import pandas as pd
import tkinter as tk
from tkinter import ttk
from scipy.optimize import basinhopping,minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from matplotlib.lines import Line2D
from scipy import stats
from matplotlib import pyplot as plt

class pupil_prediction:
    def __init__(self):
        self.useBH = True
    def pupil_prediction(self):
        # Bounds
        if self.useApp:
            self.modelingLabel = tk.Label(self.window,text="Modeling in progress. Please wait...",fg = "green")
            self.modelingLabel.grid(column = 0, row = 15)
        else:
            print("Modeling in progress. Please wait...")

        if self.RF == "HL": 
            if self.sameWeightFeature:
                bounds1 = [(5, 30), (0.1,5), (5,30),(0.1,5),(0.0001,2)] + [(0.0000001,2)] * self.nWeight
                x0 =  [15,1,15,1,1]+[1] * self.nWeight 
            else:
                bounds1 = [(5, 30), (0.1,1.5), (5,30),(0,1.5),(0.0001,2),(0,2),(0,2)]
                x0 =  [15,1,15,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
            
        elif self.RF == "KB":
            if self.sameWeightFeature:
                bounds1 = [(0.00001,0.35),(2,10),(0.00001,0.35),(2,10),(0,3)]+ [(0,2)] * self.nWeight
                x0 =  [0.2,5,0.2,5,1] + [0.5] * self.nWeight            
            else:
                bounds1 = [(0.00001,0.35),(2,10),(0.00001,0.35),(2,10),(0,3),(0,2),(0,2)]
                x0 =  [0.2,5,0.2,5,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
            
        print(f"bounds: {bounds1}")
        # res0 = minimize(self.root_mean_of_squares_modelContrast, x0=x0, bounds = bounds1, method="Nelder-Mead")
        # res1 = minimize(self.root_mean_of_squares_modelContrast, x0=x1, bounds = bounds1, method="Nelder-Mead")
        # res2 = minimize(self.root_mean_of_squares_modelContrast, x0=x2, bounds = bounds1, method="Nelder-Mead")
        # res0 = minimize(self.r_modelContrast, x0=x0, bounds = bounds1, method="Nelder-Mead")
        # res1 = minimize(self.r_modelContrast, x0=x1, bounds = bounds1, method="Nelder-Mead")
        # res2 = minimize(self.r_modelContrast, x0=x2, bounds = bounds1, method="Nelder-Mead")
        if self.useBH:
            res = basinhopping(self.root_mean_of_squares_modelContrast, x0,minimizer_kwargs={'method':'nelder-mead','bounds':bounds1},  niter = self.niter)
        else:
            res = minimize(self.root_mean_of_squares_modelContrast, x0 = x0, bounds = bounds1, method="Nelder-Mead")
        self.modelContrast(res.x)
        r,p = self.correlation(self.y_pred,  self.sampledpupilData)
        rmse = self.root_mean_of_squares_modelContrast(res.x)
        self.res = res
        
        # self.modelContrast(res0.x)
        # r0,p0 = self.correlation(self.y_pred,  self.sampledpupilData)
        # self.modelContrast(res1.x)
        # r1,p1 = self.correlation(self.y_pred,  self.sampledpupilData)
        # self.modelContrast(res2.x)
        # r2,p2 = self.correlation(self.y_pred,  self.sampledpupilData)

        # rmse0 = self.root_mean_of_squares_modelContrast(res0.x)
        # rmse1 = self.root_mean_of_squares_modelContrast(res1.x)
        # rmse2 = self.root_mean_of_squares_modelContrast(res2.x)
        # if rmse0 <rmse1 and rmse0<rmse2:
        #     res = res0
        #     print("res0 chosen")
        # elif rmse1 < rmse2 and rmse1< rmse0:
        #     res = res1
        #     print("res1 chosen")

        # elif rmse2 < rmse1 and rmse2< rmse0:
        #     res = res2
        #     print("res2 chosen")
        self.list_modelResults = self.calculalte_parameters(self.y_pred,  self.sampledpupilData, len(bounds1), len(self.sampledpupilData[~np.isnan(self.sampledpupilData)]))
        print(self.list_modelResults)
       
        
        self.save_modelResults(self.subject)
        self.save_modelData(self.subject)
        if self.useApp:

            self.modelingLabel.grid_forget()
            tk.Label(self.window,text=f'Regularization in progress...',fg = "green").grid(column = 0, row = 15)
        else:
            print(f'Modeling done. R = {round(r,2)}; rmse = {round(rmse,2)}')
        self.r = r
        self.rmse = rmse
    def regularization(self, plot_reg = False):
        # Generate X and y data used for regularization
        ## response function parameters
        if self.sameWeightFeature:
            rf_params = self.params[0:5]
            self.rf_params = rf_params
            rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast= map(float, rf_params)
        else:
            rf_params = self.params[0:5]
            rf_params = rf_params+ [1]
            self.rf_params = rf_params 
            rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast= map(float, rf_params)

        if self.RF == "HL":
            rf_lum = self.responsefunction(rf_lum_param1, rf_lum_param2, 1)
            rf_contrast = self.responsefunction(rf_contrast_param1, rf_contrast_param2, 1)
        else:
            rf_lum = self.responsefunction_bach(theta = rf_lum_param1, k = rf_lum_param2, c = 1)
            rf_contrast = self.responsefunction_bach(theta = rf_contrast_param1, k =  rf_contrast_param2, c =  1)
            t = np.linspace(0,3,76)

            rf_lum[t<0.2] = 0
            rf_contrast[t<0.2] = 0
            
            rf_lum[np.isnan(rf_lum)] = 0
            rf_contrast[np.isnan(rf_contrast)] = 0
        # convolve luminance changes and contrast changes with RF to create the X for regularization
        y_pred = np.empty(self.luminanceMagnPerImPartTime.shape)
        y_pred_lum =np.empty(self.luminanceMagnPerImPartTime.shape)
        y_pred_contrast =np.empty(self.luminanceMagnPerImPartTime.shape)
        for i in range(self.luminanceMagnPerImPartTime.shape[0]):
            lumData = self.luminanceMagnPerImPartTime[i,:]
            contrastData = np.abs(self.luminanceMagnPerImPartTime)[i,:]
            lumConv_onewedge = self.convolve_cum(rf_lum, lumData)
            contrastConv_onewedge = self.convolve_spike(rf_contrast, contrastData)
            
            lumConv_onewedge = self.zscore(lumConv_onewedge)
            contrastConv_onewedge = self.zscore(contrastConv_onewedge)
            lumConv_onewedge[~np.isfinite(lumConv_onewedge)] =0
            contrastConv_onewedge[~np.isfinite(contrastConv_onewedge)] =0
            y_pred_onewedge = self.zscore(contrastConv_onewedge*amplitudeWeightContrast+lumConv_onewedge)
            
            if np.isnan(y_pred_onewedge).all():
                y_pred_onewedge = np.zeros(y_pred_onewedge.shape[0])
                # plt.figure()
                # plt.plot(y_pred_onewedge)
                # plt.title(movieNum+str(i))
            y_pred[i,:] = y_pred_onewedge
            y_pred_lum[i,:] = lumConv_onewedge
            y_pred_contrast[i,:] = contrastConv_onewedge
        # if some of the weights are assigned with the same value, first combine some regions together
        if self.nWeight < self.magnPerImPart["Luminance"].shape[1]:
            y_pred_combined = np.empty(y_pred.shape[1])#pd.DataFrame({})
            for i in list(np.unique(self.weightRegionArr)):
                mask = self.weightRegionArr == i
                selected = y_pred[list(mask),:]
                selected_mean = selected.mean(axis = 0)
                y_pred_combined = np.vstack((y_pred_combined,selected_mean))
            y_pred = y_pred_combined[1:,:]  
            #luminance
            y_pred_lum_combined = np.empty(y_pred_lum.shape[1])#pd.DataFrame({})
            for i in list(np.unique(self.weightRegionArr)):
                mask = self.weightRegionArr == i
                selected = y_pred_lum[list(mask),:]
                selected_mean = selected.mean(axis = 0)
                y_pred_lum_combined = np.vstack((y_pred_lum_combined,selected_mean))
            y_pred_lum = y_pred_lum_combined[1:,:]  
            #contrast
            y_pred_contrast_combined = np.empty(y_pred_contrast.shape[1])#pd.DataFrame({})
            for i in list(np.unique(self.weightRegionArr)):
                mask = self.weightRegionArr == i
                selected = y_pred_contrast[list(mask),:]
                selected_mean = selected.mean(axis = 0)
                y_pred_contrast_combined = np.vstack((y_pred_contrast_combined,selected_mean))
            y_pred_contrast = y_pred_contrast_combined[1:,:]  
        # independent variable is the convolved result for each region
        X = y_pred.T
        X_train = y_pred[:,0:int(y_pred.shape[1]*0.7)].T
        X_test = y_pred[:,int(y_pred.shape[1]*0.7):].T

        # dependent variable is the sampled pupil data
        y  = self.sampledpupilData
        y_train = y[0:int(len(y)*0.7)]
        y_test = y[int(len(y)*0.7):]
        X_train = X_train[~np.isnan(y_train),:]
        y_train = y_train[~np.isnan(y_train)]
        X_test = X_test[~np.isnan(y_test),:]
        y_test = y_test[~np.isnan(y_test)]
        # X = X[~np.isnan(y),:]
        # y_pred_lum = y_pred_lum.T[~np.isnan(y),:]
        # y_pred_contrast = y_pred_contrast.T[~np.isnan(y),:]
        # y = y[~np.isnan(y)]
        if self.regularizationType == 'ridge':
            # Define the number of elements in the array
            num_elements = 200
            
            # Generate an array with non-linear spacing (gradual increase in the beginning, faster later)
            non_linear_space = np.linspace(0.032, 1, num_elements) **4
            
            # Scale the non-linear space to the desired range (0 to 1000000)
            scaled_space = non_linear_space * 1000000
            
            # Take the logarithm of the scaled values
            alphas = np.log(scaled_space)*10000
            alphas = alphas-alphas[0]
            
            mse_train_all = []
            mse_test_all = []
            mse_all = []
            n_alpha = 0
            for alpha in alphas:
                ridge = Ridge(alpha = alpha)
                # use train and test
                ridge.fit(X_train,y_train)
                predictions_train = ridge.predict(X_train)
                predictions_test = ridge.predict(X_test)
                #coefficients_test = ridge.coef_
                # not use train and test
                # ridge = Ridge(alpha = alpha,positive = True)
                # ridge.fit(X,y)
                # predictions = ridge.predict(X)
                #coefficients = ridge.coef_

                # df_coef = pd.DataFrame(coefficients).T
                # if self.sameWeightFeature:
                #     df_coef.columns = ["weight"+ str(num+1) for num in range(44)]
                # else:
                #     df_coef.columns = ["weight_lum"+ str(num+1) for num in range(44)] + ["weight_contrast"+ str(num+1) for num in range(44)]
                # Calculate mean squared error (MSE) on the test set
                mse_train = mean_squared_error(y_train, predictions_train)
                mse_test = mean_squared_error(y_test, predictions_test)
                #mse = mean_squared_error(y, predictions)

                #print("Mean Squared Error (train): ", mse_train)
                #print("Mean Squared Error (test): ", mse_test)
                mse_train_all.append(mse_train)
                mse_test_all.append(mse_test)
                #mse_all.append(mse)
                n_alpha = n_alpha+1
                print(n_alpha)
            # if minimum mse_test for test is smaller than train, then use the minimal alpha as the final alpha
            index_min = mse_test_all.index(min(mse_test_all))
            #alpha_simplified = mse_all.index(min(mse_all))
            alpha_select1 = alphas[index_min]
            difference_test_train = np.array(mse_test_all)- np.array(mse_train_all)
            if min(mse_test_all) < mse_train_all[index_min]:
                ind_final = index_min
            else:
                weight1 = 1.2
                weight2 = 1.1
                weight3 = 1.4
                combined = np.array(mse_test_all)*weight1 + np.array(mse_train_all)*weight2+difference_test_train*weight3
                ind_final =list(combined).index(min(combined))
            alpha_final = alphas[ind_final]
            mse_final = mse_test_all[ind_final]
            # plot the regularization result
            if plot_reg:
                
                plt.subplots(1,1,figsize = (12,8))
                plt.subplot(1,1,1)
                plt.plot(alphas,mse_train_all,color = "blue")
                plt.plot(alphas,mse_test_all,color = "red")
                legend_elements = [Line2D([0], [0], color='blue', label="train", linestyle = "solid"),
                                    Line2D([0], [0], color='red',  label="test", linestyle = "solid")]
                plt.legend(handles=legend_elements)
    
                plt.axvline(x = alpha_select1, color = "black") # minial mse test
                plt.xlabel("Alpha\nGreen line")
                plt.ylabel("mse")
    
                #plt.axvline(x = alpha_select2, color = "green") # significant drop mse test
                #plt.axvline(x = alpha_select3, color = "orange") # minimum mse train-test
                
                plt.title("Subj"+str(self.subject) + " Regularization result")
               
                
                plt.scatter(alpha_final, mse_final, s = 50, color = "purple")
                currentDir = os.getcwd()
                plt.savefig(f"CheckReg_Subj{self.subject}.jpg")
                
            # calculate final prediction result
            # Note: This is exactly the same as that used in the paper. If the amount of data is not enough, may lead to some weired result.
            ridge = Ridge(alpha = alpha_final)
            ridge.fit(X_train, y_train)
            # replace the modeling result without regularization with the regularization result
            self.y_pred = self.zscore(ridge.predict(X))
            self.lumConv = self.zscore(ridge.predict(y_pred_lum.T))
            self.contrastConv = self.zscore(ridge.predict(y_pred_contrast.T))
            self.reg_coefficients = ridge.coef_
            self.alpha_final = alpha_final
            self.list_modelResults = self.calculalte_parameters(self.y_pred, y, self.nWeight, len(self.y_pred))
            self.save_modelResults_reg(self.subject)
            self.r = self.list_modelResults[1]
            self.rmse = self.list_modelResults[0]
            if self.useApp:

                self.modelingLabel.grid_forget()
                tk.Label(self.window,text=f'Modeling done. R = {round(self.r,2)}; rmse = {round(self.rmse,2)}',fg = "green").grid(column = 0, row = 15)
            else:
                print(f"Regularization done. R = {self.r.round(2)}; rmse={self.rmse.round(2)}")

    def pupil_predictionNoEyetracking(self, params):
        self.modelContrast(params)
        self.save_modelResults(self.subject)
        self.save_modelData(self.subject)
    def modelContrast(self, params):
        # for Predicted_Installation
        if self.sameWeightFeature:
            if self.nWeight == 2:
                rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast,weight1,weight2= map(float, params)
            elif self.nWeight ==6:
                rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast,weight1,weight2,weight3,weight4,weight5,weight6= map(float, params)
            elif self.nWeight ==44:
                rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast,weight1,weight2,weight3,weight4,weight5,weight6,weight7,weight8,weight9,weight10,weight11,weight12,weight13,weight14,weight15,weight16,weight17,weight18,weight19,weight20,weight21,weight22,weight23,weight24,weight25,weight26,weight27,weight28,weight29,weight30,weight31,weight32,weight33,weight34,weight35,weight36,weight37,weight38,weight39,weight40,weight41,weight42,weight43,weight44= map(float, params)
            elif self.nWeight ==48:
                rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast,weight1,weight2,weight3,weight4,weight5,weight6,weight7,weight8,weight9,weight10,weight11,weight12,weight13,weight14,weight15,weight16,weight17,weight18,weight19,weight20,weight21,weight22,weight23,weight24,weight25,weight26,weight27,weight28,weight29,weight30,weight31,weight32,weight33,weight34,weight35,weight36,weight37,weight38,weight39,weight40,weight41,weight42,weight43,weight44,weight45,weight46,weight47,weight48= map(float, params)

            self.paramNames = ["rf_lum_param1","rf_lum_param2","rf_contrast_param1","rf_contrast_param2","amplitudeWeightContrast"] + ['weight'+str(i+1) for i in range(self.nWeight)]      
        else:
            rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast,weight_lum2,weight_lum3,weight_lum4,weight_lum5,weight_lum6,weight_contrast2,weight_contrast3,weight_contrast4,weight_contrast5,weight_contrast6= map(float, params)
            paramNames = "rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast,weight_lum2,weight_lum3,weight_lum4,weight_lum5,weight_lum6,weight_contrast2,weight_contrast3,weight_contrast4,weight_contrast5,weight_contrast6"
        #self.paramNames = paramNames.split(",")
        # else:
        #     rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast= map(float, params)
        #     paramNames = "rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast"
        #     self.paramNames = paramNames.split(",")
        
        if self.RF == "HL":
            rf_lum = self.responsefunction(rf_lum_param1, rf_lum_param2, 1)
            rf_contrast = self.responsefunction(rf_contrast_param1, rf_contrast_param2, 1)
        elif self.RF == "KB":
            rf_lum = self.responsefunction_bach(theta = rf_lum_param1, k = rf_lum_param2, c = 1)
            rf_contrast = self.responsefunction_bach(theta = rf_contrast_param1, k =  rf_contrast_param2, c =  1)
        rf_lum[~np.isfinite(rf_lum)] =0
        rf_contrast[~np.isfinite(rf_contrast)] =0

        
        if self.sameWeightFeature:
            if self.nWeight == 2:
                weightList_lum = [weight1, weight2]
                weightList_contrast = [weight1, weight2]
            elif self.nWeight == 6:
                weightList_lum = [weight1, weight2, weight3,weight4,weight5,weight6]
                weightList_contrast = [weight1, weight2, weight3,weight4,weight5,weight6]
            elif self.nWeight ==44:
                weightList_lum = [weight1,weight2,weight3,weight4,weight5,weight6,weight7,weight8,weight9,weight10,weight11,weight12,weight13,weight14,weight15,weight16,weight17,weight18,weight19,weight20,weight21,weight22,weight23,weight24,weight25,weight26,weight27,weight28,weight29,weight30,weight31,weight32,weight33,weight34,weight35,weight36,weight37,weight38,weight39,weight40,weight41,weight42,weight43,weight44]
                weightList_contrast = [weight1,weight2,weight3,weight4,weight5,weight6,weight7,weight8,weight9,weight10,weight11,weight12,weight13,weight14,weight15,weight16,weight17,weight18,weight19,weight20,weight21,weight22,weight23,weight24,weight25,weight26,weight27,weight28,weight29,weight30,weight31,weight32,weight33,weight34,weight35,weight36,weight37,weight38,weight39,weight40,weight41,weight42,weight43,weight44]
            elif self.nWeight ==48:
                weightList_lum = [weight1,weight2,weight3,weight4,weight5,weight6,weight7,weight8,weight9,weight10,weight11,weight12,weight13,weight14,weight15,weight16,weight17,weight18,weight19,weight20,weight21,weight22,weight23,weight24,weight25,weight26,weight27,weight28,weight29,weight30,weight31,weight32,weight33,weight34,weight35,weight36,weight37,weight38,weight39,weight40,weight41,weight42,weight43,weight44,weight45,weight46,weight47,weight48]
                weightList_contrast = [weight1,weight2,weight3,weight4,weight5,weight6,weight7,weight8,weight9,weight10,weight11,weight12,weight13,weight14,weight15,weight16,weight17,weight18,weight19,weight20,weight21,weight22,weight23,weight24,weight25,weight26,weight27,weight28,weight29,weight30,weight31,weight32,weight33,weight34,weight35,weight36,weight37,weight38,weight39,weight40,weight41,weight42,weight43,weight44,weight45,weight46,weight47,weight48]

        else: 
            weightList_lum = [1, weight_lum2, weight_lum3,weight_lum4,weight_lum5,weight_lum6]
            weightList_contrast = [1, weight_contrast2, weight_contrast3,weight_contrast4,weight_contrast5,weight_contrast6]
    
        
        self.prepareRegionalWeightArr()
        matShape = np.shape(self.magnPerImPart["Luminance"])
        #prepare movie features
        if self.mapType == "circular": 
            if self.magnPerImPart["Luminance"].ndim ==3:
                luminanceMagnPerImPartTime = self.magnPerImPart["Luminance"][0,:,:]
            else:
                luminanceMagnPerImPartTime = self.magnPerImPart["Luminance"].T
        else:
            
            luminanceMagnPerImPartTime = (
                self.magnPerImPart["Luminance"]#[1:5,1:7,:]
                .transpose(0, 1, 2)
                .reshape(
                    matShape[0] * matShape[1],
                    matShape[2],
                )
            )
        if hasattr(self, "numRemoveFrame"):
            luminanceMagnPerImPartTime = luminanceMagnPerImPartTime[:,self.numRemoveFrame:]
        self.luminanceMagnPerImPartTime = luminanceMagnPerImPartTime
        lumData = self.prepare_feature(luminanceMagnPerImPartTime,weightList_lum, self.weightRegionArr)
        #contrastData = self.prepare_feature(luminanceMagnPerImPartTime,self.skipNFirstFrame,weightList_contrast, self.weightRegionArr)
        contrastData = np.abs(lumData)            
        
        self.lumConv = self.convolve_cum(rf_lum, lumData)
        self.contrastConv = self.convolve_spike(rf_contrast, contrastData)
        
        self.rf_lum = rf_lum
        self.rf_contrast = rf_contrast
        self.lumConv[~np.isfinite(self.lumConv)] =0
        self.contrastConv[~np.isfinite(self.contrastConv)] =0
        if np.nanstd(self.lumConv !=0) and np.nanstd(self.contrastConv!=0):
            self.lumConv = self.zscore(self.lumConv)
            self.contrastConv = self.zscore(self.contrastConv)
        
        self.y_pred = self.zscore(self.contrastConv*amplitudeWeightContrast+self.lumConv)
        self.y_pred[~np.isfinite(self.y_pred)] =0
        self.lumData = luminanceMagnPerImPartTime.mean(axis = 0)
        self.contrastData = np.abs(self.lumData)

    def save_modelResults(self,subject):
        # save model result to dictionary
        self.modelResultDict[subject] = {}
        self.modelResultDict[subject]["modelContrast"] = {}
        self.modelResultDict[subject]["modelContrast"]["predAll"] = self.y_pred
        self.modelResultDict[subject]["modelContrast"]["lumConv"] = self.lumConv
        self.modelResultDict[subject]["modelContrast"]["contrastConv"] = self.contrastConv
        #save modeling result in dictionary (only if eyetracking data is used)
        if self.useEtData:
            self.modelResultDict[subject]["modelContrast"]["modelResults"] = self.list_modelResults
            # save the results of the parameters selected in this model 
            self.modelResultDict[subject]["modelContrast"]["parameters"] = self.res.x
            self.modelResultDict[subject]["modelContrast"]["parametersNames"] = self.paramNames
        with open(f"modelResultDict_{self.subject}_nWeight{self.nWeight}.pickle", "wb") as f:
            pickle.dump(self.modelResultDict, f)
    def save_modelResults_reg(self, subject):
        self.modelResultDict[subject]["modelRegularization"] = {}
        self.modelResultDict[subject]["modelRegularization"]["parameters"]= {}
        self.modelResultDict[subject]["modelRegularization"]["parametersNames"]= {}
        self.modelResultDict[subject]["modelRegularization"]["predAll"] = self.y_pred
        self.modelResultDict[subject]["modelRegularization"]["lumConv"] = self.lumConv
        self.modelResultDict[subject]["modelRegularization"]["contrastConv"] = self.contrastConv
        self.modelResultDict[subject]["modelRegularization"]["modelResults"] = self.list_modelResults
        # save the results of the parameters selected in this model 
        parameters = list(self.rf_params) + list(self.reg_coefficients)
        self.modelResultDict[subject]["modelRegularization"]["parameters"] = parameters
        #print(f"number of parameters: {len(parameters)}")
        self.modelResultDict[subject]["modelRegularization"]["parametersNames"] = self.paramNames
        self.modelResultDict[subject]["modelRegularization"]["alpha"] = self.alpha_final
        with open(f"modelResultDict_{self.subject}_nWeight{self.nWeight}.pickle", "wb") as f:
            pickle.dump(self.modelResultDict, f)
        
    def save_modelData(self, subject):
        self.modelDataDict[subject] = {}
        if self.useEtData:
            self.modelDataDict[subject]["pupil"] = self.sampledpupilData
            self.modelDataDict[subject]["gazex"] = self.sampledgazexData
            self.modelDataDict[subject]["gazey"] = self.sampledgazeyData
        self.modelDataDict[subject]["lumData"] = self.lumData
        self.modelDataDict[subject]["contrastData"] = self.contrastData
        self.modelDataDict[subject]["timeStamps"] = self.sampledTimeStamps

        with open(f"modelDataDict_{self.subject}_nWeight{self.nWeight}.pickle", "wb") as f:
            pickle.dump(self.modelDataDict, f)
    def responsefunction(self, n, tmax, amplitude_weight,):
        rf_t_num = int(round((3*self.sampledFps))+1)
        gapNum = 0.2 / (1/self.sampledFps)
        t = np.linspace(0,3,rf_t_num)
        rf = t**n*np.exp(-n*t/tmax)
        rf = rf/max(rf)
        gap = int(gapNum) * [0]
        rf = np.insert(rf,0, gap)[0:len(rf)]
  
        rf = amplitude_weight *rf
        return rf
  
    def responsefunction_bach(self, theta, k,c):
        rf_t_num = int(round((3*self.sampledFps))+1)
        gapNum = 0.2 / (1/self.sampledFps)

        t = np.linspace(0,3,rf_t_num)

        rf = (c/((theta**k)*math.gamma(k)))*(t**(k-1))*np.exp(-t/theta)
        gap = int(gapNum) * [0]
        rf = np.insert(rf,0, gap)[0:len(rf)]
        #rf = rf/max(rf)
        #rf = amplitude_weight *rf
        return rf
    def prepareRegionalWeightArr(self):
        if self.nWeight == 6:
            weight6_regions = np.array([24,25,30,31,32,33,38,39,40,41,42,43,44,45,46,47])
            weight5_regions = np.array([26,29,34,35,36,37])
            weight4_regions = np.array([27,28])
            weight3_regions = np.array([0,1,2,3,4,5,6,7,8,9,14,15,16,17,22,23])
            weight2_regions = np.array([10,11,12,13,18,21])
            weight1_regions = np.array([19,20])
            weightRegionArr = np.empty(48)
            weightRegionArr[weight6_regions] = 5
            weightRegionArr[weight5_regions] = 4
            weightRegionArr[weight4_regions] = 3
            weightRegionArr[weight3_regions] = 2
            weightRegionArr[weight2_regions] = 1
            weightRegionArr[weight1_regions] = 0
            self.weightRegionArr = weightRegionArr
        elif self.nWeight ==2:
            if self.mapType == "circular":
                right = ['weight2','weight7','weight8','weight15','weight16','weight24','weight25','weight26','weight36','weight37','weight38','weight3','weight9','weight10','weight17','weight18','weight27','weight28','weight29','weight39','weight40','weight41']
                left = ['weight1','weight5','weight6','weight13','weight14','weight21','weight22','weight23','weight33','weight34','weight35','weight4','weight11','weight12','weight19','weight20','weight30','weight31','weight32','weight42','weight43','weight44']
                self.weightLeft_regions = [int(x.replace("weight",''))-1 for x in left]
                self.weightRight_regions = [int(x.replace("weight",''))-1 for x in right]
            elif self.mapType == "square":
                self.weightLeft_regions = np.array([0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27,32,33,34,35,40,41,42,43])
                self.weightRight_regions = np.array([4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31,36,37,38,39,44,45,46,47])
           
            weightRegionArr = np.empty(len(self.weightLeft_regions) + len(self.weightRight_regions))
            
            weightRegionArr[self.weightLeft_regions] = 1
            weightRegionArr[self.weightRight_regions] = 0
            self.weightRegionArr = weightRegionArr
        elif self.nWeight == 44 or self.nWeight==48:
            self.weightRegionArr = np.arange(0,self.nWeight)
        
    def prepare_feature(self, feautreMagnPerImPartTime,weightList, weightRegionArr,skipNFirstFrame =0):
        # calculate weighted data luminance
        if np.unique(self.weightRegionArr).shape[0] < self.weightRegionArr.shape[0]:
            weightArr = np.empty(feautreMagnPerImPartTime.shape[0])
            for i in range(len(weightArr)):
                weightArr[i] = weightList[int(self.weightRegionArr[i])]
        
            weightArr = np.reshape(weightArr, (-1, 1))
        else:
            weightArr = np.reshape(np.array(weightList), (-1, 1))
        if self.useEtData:
            feautreMagnPerImPartTime_weighted = feautreMagnPerImPartTime[:, skipNFirstFrame:(len(self.sampledpupilData)+1)] * weightArr
        else:
            feautreMagnPerImPartTime_weighted = feautreMagnPerImPartTime[:, skipNFirstFrame:] * weightArr

        featureData = np.nanmean(feautreMagnPerImPartTime_weighted , axis=0)
        
        return featureData
    
        
                                        
    def root_mean_of_squares_modelContrast(self,params):
        self.modelContrast(params)
        obj = np.sqrt(np.nanmean((self.y_pred - self.sampledpupilData) ** 2))
        return obj
    def r_modelContrast(self,params):
        self.modelContrast(params)
        r,p=scipy.stats.pearsonr(self.y_pred[~np.isnan(self.sampledpupilData)],self.sampledpupilData[~np.isnan(self.sampledpupilData)])
        r = r*(-1)
        return r
    def calculalte_parameters(self, predict, data, num_params, n_data):
        rmse= np.sqrt(np.nanmean((predict - data) ** 2))
        mse = np.nanmean((predict - data) ** 2)
        r, p = self.correlation(predict, data)
        mae = self.mae(predict, data)
        aic = self.calculate_aic(n_data, mse, num_params)
        bic = self.calculate_bic(n_data, mse, num_params)
        modelResults = [rmse, r, p, mae, aic, bic]
        #names_parameters = ["rmse","r","p","mae","aic","bic"]
        return modelResults
    def mae(self,predict, data):
        mae = np.nanmean(np.abs(predict-data))
        return mae
    def correlation(self,predict, data):
        r,p=scipy.stats.pearsonr(predict[~np.isnan(data)],data[~np.isnan(data)])
        return r, p
    
    def calculate_aic(self,n, mse, num_params):
    	aic = n * np.log(mse) + 2 * num_params
    	return aic
    def calculate_bic(self,n, mse, num_params):
    	bic = n * np.log(mse) + num_params * np.log(n)
    	return bic
    def convolve_cum(self, rf, featureData):
        featureData_ConvRaw = np.convolve(-1 * featureData, rf)
        featureData_ConvRaw = np.cumsum(featureData_ConvRaw[0:len(featureData)])
        featureDataConv = self.zscore(featureData_ConvRaw)
        return featureDataConv
    def convolve_spike(self,rf, featureData):
        featureData_ConvRaw = np.convolve(-1 * featureData, rf)
        featureData_ConvRaw = featureData_ConvRaw[0:len(featureData)]
        featureDataConv = self.zscore(featureData_ConvRaw)
        return featureDataConv
    def zscore(self,x):
        zscore = (x-np.nanmean(x))/np.nanstd(x)
        return zscore
    # eyetracking data processing
    def prepare_sampleData(self, dataBeforeSample,nsample):
        ## do nothing if choose the stretch to match
        if self.stretchToMatch:
            pass
        ## cut the longer one
        else:
            if self.video_duration > self.eyetracking_duration:
                diff = self.video_duration -self.eyetracking_duration
                nFrameDiff = np.floor(diff/(1/self.video_fps))
                nsample = int(nsample-nFrameDiff)
            else: 
                diff = np.abs(self.video_duration -self.eyetracking_duration)
                nFrameDiff = np.floor(diff/(1/self.eyetracking_samplingrate))
                eyetracking_nsample = int(len(dataBeforeSample)- nFrameDiff)
                dataBeforeSample = dataBeforeSample[:eyetracking_nsample]
        sampleTimes = np.linspace(0, len(dataBeforeSample) - 1, nsample).astype("int")
        sampledData = dataBeforeSample[sampleTimes]
        return sampledData
    def synchronize(self, dataBeforeSynchronize):
        # synchronize eyetracking data to match the change of the movie feature
        self.numRemoveFrame = int(self.nFramesSeqImageDiff -1)
        numFrame = int(len(dataBeforeSynchronize)-self.numRemoveFrame)
        # remove the the last data of eyetracking according to nFramesSeqImageDiff
        dataSynchronized = dataBeforeSynchronize[:numFrame]
        return dataSynchronized