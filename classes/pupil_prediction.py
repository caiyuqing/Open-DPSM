# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:44:00 2023

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
                bounds1 = [(5, 30), (0.001,1.5), (5,30),(0.001,1.5),(0.0001,2),(0,2),(0,2),(0,2),(0,2),(0,2)]
                x0 =  [15,1,15,1,1,1,1,1,1,1] 
            else:
                bounds1 = [(5, 30), (0,1.5), (5,30),(0,1.5),(0.0001,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2)]
                x0 =  [15,1,15,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
            
        elif self.RF == "KB":
            if self.sameWeightFeature:
                bounds1 = [(0.00001,0.35),(2,10),(0.00001,0.35),(2,10),(0,3),(0,2),(0,2),(0,2),(0,2),(0,2)]
                x0 =  [0.2,5,0.2,5,1,0.5,0.5,0.5,0.5,0.5]           
            else:
                bounds1 = [(0.00001,0.35),(2,10),(0.00001,0.35),(2,10),(0,3),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2)]
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
            tk.Label(self.window,text=f'Modeling done. R = {round(r,2)}; rmse = {round(rmse,2)}',fg = "green").grid(column = 0, row = 15)
        else:
            print(f'Modeling done. R = {round(r,2)}; rmse = {round(rmse,2)}')
        self.r = r
        self.rmse = rmse
    def pupil_predictionNoEyetracking(self, params):
        self.modelContrast(params)
        self.save_modelResults(self.subject)
        self.save_modelData(self.subject)
    def modelContrast(self, params):
        # for Predicted_Installation
        if self.sameWeightFeature:
            rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast,weight2,weight3,weight4,weight5,weight6= map(float, params)
            paramNames = "rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast,weight2,weight3,weight4,weight5,weight6"
        else:
            rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast,weight_lum2,weight_lum3,weight_lum4,weight_lum5,weight_lum6,weight_contrast2,weight_contrast3,weight_contrast4,weight_contrast5,weight_contrast6= map(float, params)
            paramNames = "rf_lum_param1,rf_lum_param2,rf_contrast_param1,rf_contrast_param2,amplitudeWeightContrast,weight_lum2,weight_lum3,weight_lum4,weight_lum5,weight_lum6,weight_contrast2,weight_contrast3,weight_contrast4,weight_contrast5,weight_contrast6"
        self.paramNames = paramNames.split(",")
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
            weightList_lum = [1, weight2, weight3,weight4,weight5,weight6]
            weightList_contrast = [1, weight2, weight3,weight4,weight5,weight6]
        else: 
            weightList_lum = [1, weight_lum2, weight_lum3,weight_lum4,weight_lum5,weight_lum6]
            weightList_contrast = [1, weight_contrast2, weight_contrast3,weight_contrast4,weight_contrast5,weight_contrast6]
    
        

        #prepare movie features
        self.prepareRegionalWeightArr()
        matShape = np.shape(self.magnPerImPart["Luminance"])
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
        self.lumData = lumData
        self.contrastData = contrastData
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
        with open("modelResultDict.pickle", "wb") as f:
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

        with open("modelDataDict.pickle", "wb") as f:
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
    def prepare_feature(self, feautreMagnPerImPartTime,weightList, weightRegionArr,skipNFirstFrame =0):
        # calculate weighted data luminance
        weightArr = np.empty(48)
        for i in range(len(weightArr)):
            weightArr[i] = weightList[int(weightRegionArr[i])]
        
        weightArr = np.reshape(weightArr, (-1, 1))
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