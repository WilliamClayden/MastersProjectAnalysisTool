# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:45:45 2020

@author: willi
"""

import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import math as m
import tkinter as gui
import pandas as pd
import statistics

if __name__ == '__main__':
    os.chdir('C:/Users/willi/OneDrive - Lancaster University/4th Year/MSc Data Science/Masters Project Placement/Image analysis work/Images/Test_images/demo')
    testdata = pd.read_csv("testdata.csv")

"""This method finds all vessels that are complete (without a significant straight edge)"""
def Find_Complete_Xylem(xylem_list, Image_dims):
    
    xylem_classification = []
    
    for cnt in xylem_list:
        contour_perimeter = cv2.arcLength(cnt, True)
        # Set the minimum and maximum x values for each contour to be in the middle of the image for each contour
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

        """If there is a value on the edge of the image it must do a few things
        1. save that value
        2. find the length of the boundary line if there is one
        """

        if leftmost[0] <= 1 or topmost[1] <= 1 or rightmost[0] >= Image_dims[1]-2 or bottommost[1] >= Image_dims[0]-2:
            
            for i in range(0,len(cnt)):
                if (cnt[i,0,:][0] <= 1 or cnt[i,0,:][0] >= Image_dims[1]-2) or (cnt[i,0,:][1] <= 1 or cnt[i,0,:][1] >= Image_dims[0]-2):
                    if 'border_contour' not in locals():
                        border_contour = np.array(cnt[i,0,:], ndmin=3)
                    else:
                        border_contour = np.append(border_contour, [[cnt[i,0,:]]], axis=0)

            # Check that the border length for the given xylem is worth identifying

            if len(border_contour) > 1:

                # Remove the error you get if the last value is between the first one and second/any other (as the contour can loop and have similar end values)
                validity_check = False
                while validity_check == False:
                    try:
                        if (abs(border_contour[0, 0,:][0] - border_contour[-1, 0,:][0]) <= 2) and (abs(border_contour[0, 0,:][1] - border_contour[-1, 0,:][1]) <= 2):
                            border_contour = np.delete(border_contour, -1,axis=0)
                        else:
                            validity_check = True
                    except:
                        validity_check = True
                border_length = cv2.arcLength(border_contour, False)
                
            else:
                border_length = 0

            if border_length/contour_perimeter >= 0.075:
                xylem_classification.append("incomplete")
            else:
                xylem_classification.append("complete")
            if border_contour is not None:
                del border_contour
        else:
            xylem_classification.append("complete")

    return xylem_classification

""""""




"""Finds the total xylem area for all images of each species identified"""
def calc_total_areas(dataset):

    total_areas = []
    updated_dataset = dataset.copy()
    """For each set of images from each sample we want to sum up their areas"""
    for sample in dataset['sample.ID'].unique():
        current_sample = dataset.loc[dataset['sample.ID']==sample]
        total_areas.extend([current_sample['Xylem.Areas'].sum()]*len(current_sample))
            
    updated_dataset['Total area'] = total_areas
    
    """Should this be appended to the existing dataset?"""
    return updated_dataset

def get_density(dataset):
    """ To get density
    I. Separate into study area
    II. Separate into complete or not
    1. Calculate median xylem are (of complete xylem) in the sample
    2. Using this value determine which edge xylem are less than 30% of this size (too inaccurate to reconstruct)
    4. Determine number of border edges of remaining xylem (1 or 2) #
    5. Calculate longest and shortest diameters
    6. Calculate edge diameter
    7. Find point furthest away from that edge midpoint and calculate radius
    8. If edge length greater than radius then calculate using Intersecting chords theorem
    9. Else if edge length less than radius, calculate using 1/2 edge diameter
    10. # If neither of these calculate ellipse from edge contour info
    11. If estimated area greater median (or maybe 75th percentile) take 75th percentile
    12. IF FAILS, insert median
    13. Calculate area fraction of estimated xylem area
    """
    updated_dataset = dataset.copy()
    for sample in dataset['sample.ID'].unique():
        current_sample = dataset.loc[dataset['sample.ID']==sample]
        complete_xyl = current_sample.loc[current_sample['Xylem.type']=='complete']
        incomp_xyl = current_sample.loc[current_sample['Xylem.type']=='incomplete']
        sample_median = complete_xylem.loc[:,'Xylem.Areas'].median()
            for i in range(len(incomp_xyl.index)):
      
    
    
    
    return updated_dataset


# calculate data information
"""import statistics
print(statistics.mean(areas))
print(statistics.median(areas))
small_areas = list(filter(lambda x: x < 1, areas))
print(np.percentile(areas,25))
print(np.percentile(areas,50))
print(np.percentile(areas,75))

"""

