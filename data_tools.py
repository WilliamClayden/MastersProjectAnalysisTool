# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:45:45 2020

@author: willi
"""

import cv2 
import matplotlib.pyplot as plt
import numpy as np
import math as m
import tkinter as gui
import pandas as pd

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

        if leftmost[0] <= 2 or topmost[1] <= 2 or rightmost[0] >= Image_dims[1]-2 or bottommost[1] >= Image_dims[0]-2:
            
            for i in range(0,len(cnt)):
                if (cnt[i,0,:][0] <= 1 or cnt[i,0,:][0] >= Image_dims[1]-2) or (cnt[i,0,:][1] <= 1 or cnt[i,0,:][1] >= Image_dims[0]-2):
                    if 'border_contour' not in locals():
                        border_contour = np.array(cnt[i,0,:], ndmin=3)
                    else:
                        border_contour = np.append(border_contour, [[cnt[i,0,:]]], axis=0)

            # Check that the border length for the given xylem is worth identifying

            if len(border_contour) > 1:

                # Remove the error you get if the last value is between the first one and second/any other (as the contour can loop and have similar end values)
                while True:
                    try:
                        if (abs(border_contour[0, 0,:][0] - border_contour[-1, 0,:][0]) <= 2) and (abs(border_contour[0, 0,:][1] - border_contour[-1, 0,:][1]) <= 2):
                            border_contour = np.delete(border_contour, -1,axis=0)
                        else:
                            break
                    except:
                        break
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

"""Finds the total xylem area for all images of each species identified"""
def calc_total_areas(dataset):

    total_xyl_areas = []
    mean_xyl_area = []
    updated_dataset = dataset.copy()
    """For each set of images from each sample we want to sum up their areas"""
    for sample in dataset['sample.ID'].unique():
        current_sample = dataset.loc[dataset['sample.ID']==sample]
        total_xyl_areas.extend([current_sample['Xylem.Areas'].sum()]*len(current_sample))
        complete_xyl = current_sample.loc[current_sample['Xylem.type']=='complete'].copy()
        mean_area = complete_xyl['Xylem.Areas'].mean()
        mean_xyl_area.extend([mean_area]*len(current_sample)) ##### UNIT
        
    
    updated_dataset['Tot.xyl.area'] = total_xyl_areas
    updated_dataset['Mean.xyl.area'] = mean_xyl_area
    """Should this be appended to the existing dataset?"""
    return updated_dataset

def get_density(dataset):
    """ To get density
    I. Separate into study area -----
    II. Separate into complete or not -----
    1. Calculate median xylem are (of complete xylem) in the sample -----
    2. depricated and moved. Using this value determine which edge xylem are less than 30% of this size (too inaccurate to reconstruct)
    3. Determine number of border edges of remaining xylem (1 or 2) # -----
    4. Calculate longest and shortest diameters
    5. Calculate edge diameter ------
    6. Find point furthest away from that edge midpoint and calculate radius
    7. If edge length greater than radius then calculate using Intersecting chords theorem
    8. Else if edge length less than radius, calculate using 1/2 edge diameter
    9. # If neither of these calculate ellipse from edge contour info
    10. If estimated area greater than  80th percentile, take 80th percentile
    11. IF FAILS, insert median
    12. Calculate area fraction of estimated xylem area
    13. calculate density based on total fraction
    """
    updated_dataset = dataset.copy()
    density_series = []
    lumen_fraction=[]
    xyl_counts = []
    for sample in dataset['sample.ID'].unique(): # Part I.
        current_sample = dataset.loc[dataset['sample.ID']==sample]
        complete_xyl = current_sample.loc[current_sample['Xylem.type']=='complete'].copy() # Part II.
        incomp_xyl = (current_sample.loc[current_sample['Xylem.type']=='incomplete'].copy()).reset_index(drop=True)
        fraction_list = []
        for i in range(len(incomp_xyl.index)):
            current_area = incomp_xyl['Xylem.Areas'][i]
            # Part 2
            if current_area >= complete_xyl.loc[:, 'Xylem.Areas'].quantile(0.7): # Set an upper cutoff so area not estimated incorrectly (can do maths to show this)
                fraction_list.append(1)
            else:
                area_est = get_area_est(incomp_xyl['Xylem.Contours'][i], incomp_xyl['Image.Dims'][i])
                if complete_xyl.loc[:, 'Xylem.Areas'].quantile(0.1) < area_est < complete_xyl.loc[:, 'Xylem.Areas'].quantile(0.9):
                    fraction_list.append(current_area/area_est)
                elif area_est > complete_xyl.loc[:, 'Xylem.Areas'].quantile(0.9):
                    fraction_list.append(current_area/complete_xyl.loc[:, 'Xylem.Areas'].quantile(0.9))
                elif area_est < complete_xyl.loc[:, 'Xylem.Areas'].quantile(0.1):
                    fraction_list.append(current_area/complete_xyl.loc[:, 'Xylem.Areas'].quantile(0.1))
                else:
                    fraction_list.append(1) # If fails set area fraction to be one average xylem
        total_xyl_count = sum(fraction_list) + len(complete_xyl)
        """Calculate the area of all the images in the sample"""
        total_area=0
        for image in current_sample['Image_ID'].unique():
            height = ((current_sample[current_sample['Image_ID'] == image]).reset_index(drop=True))['Image.Dims'][0][0]
            width = ((current_sample[current_sample['Image_ID'] == image]).reset_index(drop=True))['Image.Dims'][0][1]
            total_area += (height*width)/(2.98**2) # Per micro meter square as in fortunel paper (can be changed later) ############ UNIT HERE
        density = total_xyl_count/total_area
        density_series.extend([density]*len(current_sample))
        xyl_counts.extend([total_xyl_count]*len(current_sample))
        fraction = (total_xyl_count*complete_xyl['Xylem.Areas'].mean())/total_area
        lumen_fraction.extend([fraction]*len(current_sample))
        
    updated_dataset['xyl.density'] = density_series
    updated_dataset['totXylCount'] = xyl_counts
    updated_dataset['Vessel.Fraction']=lumen_fraction
    
    return updated_dataset

def get_area_est(contour, img_dims):
    cnt = contour.copy()
    contour_perimeter = cv2.arcLength(cnt, True)
    # Set the minimum and maximum x values for each contour to be in the middle of the image for each contour
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    # Work out which edge the contour is on
    if leftmost[0] <= 2:
        if topmost[1] <= 2:
            edge = 'top-left'
        elif bottommost[1] >= img_dims[0] - 2:
            edge = 'bot-left'
        else:
            edge = 'left'
    elif rightmost[0] >= img_dims[1] - 2:
        if topmost[1] <= 2:
            edge = 'top-right'
        elif bottommost[1] >= img_dims[0] - 2:
            edge = 'bot-right'
        else:
            edge = 'right'
    elif topmost[1] <= 2:
        edge = 'top'
    elif bottommost[1] >= img_dims[0] - 2:
        edge = 'bot'
    else:
        print('exception occurred')
    
    # Find bordering contour
    for i in range(0,len(cnt)):
        if (cnt[i,0,:][0] <= 1 or cnt[i,0,:][0] >= img_dims[1]-2) or (cnt[i,0,:][1] <= 1 or cnt[i,0,:][1] >= img_dims[0]-2):
            if 'border_contour' not in locals():
                border_contour = np.array(cnt[i,0,:], ndmin=3)
            else:
                border_contour = np.append(border_contour, [[cnt[i,0,:]]], axis=0)
    while True:
        try:
            if (abs(border_contour[0, 0,:][0] - border_contour[-1, 0,:][0]) <= 2) and (abs(border_contour[0, 0,:][1] - border_contour[-1, 0,:][1]) <= 2):
                border_contour = np.delete(border_contour, -1,axis=0)
            else:
                break
        except:
            break
    
    # Using the edge type calculate the bordering lines
    if 'edge' in locals(): # Check that edge actually exists
        if edge in ['bot', 'top', 'left', 'right']: # For single line edges
            edge_length = m.sqrt((border_contour[0,:,0][0] - border_contour[-1,:,0][0])**2 + (border_contour[0,:,1][0] - border_contour[-1,:,1][0])**2)
            midpoint = ((border_contour[0,:,0][0] + border_contour[-1,:,0][0])/2, (border_contour[0,:,1][0] + border_contour[-1,:,1][0])/2)
            # We now want to calculate the distance from the midpoint to the furthest edge of the contour, this lets us gauge what we are dealing with
            if edge == 'left':
                sagitta = m.sqrt((midpoint[0]-rightmost[0])**2 + (midpoint[1]-rightmost[1])**2)
            elif edge == 'top':
                sagitta = m.sqrt((midpoint[0]-bottommost[0])**2 + (midpoint[1]-bottommost[1])**2)
            elif edge == 'right':
                sagitta = m.sqrt((midpoint[0]-leftmost[0])**2 + (midpoint[1]-leftmost[1])**2)
            elif edge == 'bot':
                sagitta = m.sqrt((midpoint[0]-topmost[0])**2 + (midpoint[1]-topmost[1])**2)
            else:
                return 0
            if sagitta < edge_length:
                xyl_radius = edge_length/2 + ((sagitta)**2)/(8*edge_length)
                xyl_area = (m.pi*(xyl_radius)**2)/(2.98**2)
            elif sagitta > edge_length:
                if sagitta > 1.25*edge_length:
                    xyl_area = cv2.contourArea(cnt)/(2.98**2)
                else:
                    xyl_area = (m.pi*(edge_length/2)**2)/(2.98**2)
        elif edge in ['bot-right', 'top-right', 'bot-left', 'top-left']:
            #find corner point and distances from the corner to the other respective border points
            if edge == 'bot-right':
                corner = img_dims
                len_one = m.sqrt((border_contour[0,:,0][0] - corner[1])**2 + (border_contour[0,:,1][0] - corner[0])**2)
                len_two = m.sqrt((border_contour[-1,:,0][0] - corner[1])**2 + (border_contour[-1,:,1][0] - corner[0])**2)
                xyl_area = (m.pi*len_one *len_two)/(2.98**2) ####### UNIT HERE
            elif edge == 'top-right':
                corner = (0,img_dims[1])
                len_one = m.sqrt((border_contour[0,:,0][0] - corner[1])**2 + (border_contour[0,:,1][0] - corner[0])**2)
                len_two = m.sqrt((border_contour[-1,:,0][0] - corner[1])**2 + (border_contour[-1,:,1][0] - corner[0])**2)
                xyl_area = (m.pi*len_one *len_two)/(2.98**2) ########## UNIT HERE
            elif edge == 'bot-left':
                corner = (img_dims[0], 0)
                len_one = m.sqrt((border_contour[0,:,0][0] - corner[1])**2 + (border_contour[0,:,1][0] - corner[0])**2)
                len_two = m.sqrt((border_contour[-1,:,0][0] - corner[1])**2 + (border_contour[-1,:,1][0] - corner[0])**2)
                xyl_area = (m.pi*len_one*len_two)/(2.98**2) ######### UNIT HERE
            elif edge == 'top-left':
                corner = (0, 0)
                len_one = m.sqrt((border_contour[0,:,0][0] - corner[1])**2 + (border_contour[0,:,1][0] - corner[0])**2)
                len_two = m.sqrt((border_contour[-1,:,0][0] - corner[1])**2 + (border_contour[-1,:,1][0] - corner[0])**2)
                xyl_area = (m.pi*len_one*len_two)/(2.98**2) ######### UNIT HERE
            else:
                return 0
    else:
        return 0
                
                    
    
    return xyl_area 

def vessel_diameters(dataset):
    updated_dataset = dataset.copy()
    dia_type = ['simple', 'axis','ideal']
    """Offer different ways to calculate"""
    if 'simple' in dia_type:
        simple_dia = []
        for sample in dataset['sample.ID'].unique(): # Part I.
            current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
            for i in range(len(current_sample)):
                if current_sample['Xylem.type'][i] == 'complete':
                    area = cv2.contourArea(current_sample['Xylem.Contours'][i])/(2.98**2)
                    simple_dia.append(np.sqrt(4*area/np.pi))
                else:
                    simple_dia.append(0)
        updated_dataset['simple.diam'] = simple_dia
    if 'axis' in dia_type:
        long_axis = []
        short_axis =[]
        ideal_vessel_diam_list = []
        for sample in dataset['sample.ID'].unique(): # Part I.
            current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
            for i in range(len(current_sample)):
                if current_sample['Xylem.type'][i] == 'complete':
                    rect = cv2.minAreaRect(current_sample['Xylem.Contours'][i])
                    long_axis.append(max(rect[1])/(2.98))
                    short_axis.append(min(rect[1])/2.98)
                else:
                    long_axis.append(0)
                    short_axis.append(0)
        updated_dataset['long.diam'] = long_axis
        updated_dataset['short.diam'] = short_axis
        
    if 'ideal' in dia_type:
        long_axis = []
        short_axis =[]
        ideal_vessel_diam_list = []
        for sample in dataset['sample.ID'].unique(): # Part I.
            current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
            for i in range(len(current_sample)):
                if current_sample['Xylem.type'][i] == 'complete':
                    rect = cv2.minAreaRect(current_sample['Xylem.Contours'][i])
                    long_axis.append(max(rect[1])/(2.98))
                    short_axis.append(min(rect[1])/2.98)
                else:
                    long_axis.append(0)
                    short_axis.append(0)
                 # to the 4th power to save calculation later
            for i in range(0,len((current_sample))):
                if short_axis[i] != 0:
                    ideal_vessel_diam_list.append((((32*((long_axis[i]*short_axis[i])**3))/(long_axis[i]**2 + short_axis[i]**2)))**0.25)
                else:
                    ideal_vessel_diam_list.append(0)
        if len(ideal_vessel_diam_list) > 0:
            updated_dataset['ideal.diam'] = ideal_vessel_diam_list

        
    return updated_dataset

def hydraulic_diameter(dataset):
    updated_dataset = dataset.copy()
    if 'simple.diam' in dataset.columns:
        hydraulic_diam = []
        for sample in dataset['sample.ID'].unique():
            current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
            current_sample['simple.diam4'] = current_sample['simple.diam']**4
            hydraulic_diam.extend([((current_sample['simple.diam4'].sum())/len(current_sample))**0.25]*len(current_sample))
        updated_dataset['sim.hydr.diam'] = hydraulic_diam

    if 'long.diam' in dataset.columns: # Fortunel method but altered as it didnt work
        hydraulic_diam = []
        for sample in dataset['sample.ID'].unique():
            current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
            #### implement part of the formula
            d1_list = []
            d2_list = []
            for row in range(0,len(current_sample['Image_ID'])):
                d1_list.append(current_sample['long.diam'][row])
                d2_list.append(current_sample['short.diam'][row])

            ideal_vessel_diam_list = [] # to the 4th power to save calculation later
            for i in range(0,len((current_sample))):
                if d1_list[i] != 0:
                    ideal_vessel_diam_list.append(((32*((d1_list[i]*d2_list[i])**3))/(d1_list[i]**2 + d2_list[i]**2)))
            #### calculate mean hydraulic
            current_hydraulic_diam = (((sum(ideal_vessel_diam_list)))/len((current_sample.loc[current_sample['Xylem.type']=='complete'])))**0.25
            hydraulic_diam.extend([current_hydraulic_diam]*len(current_sample))
        updated_dataset['2axis.hydr.diam'] = hydraulic_diam
    elif 'ideal.diam' in dataset.columns:
        hydraulic_diam = []
        for sample in dataset['sample.ID'].unique():
            current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
            #### implement part of the formula
            #### calculate mean hydraulic
            current_hydraulic_diam = ((((current_sample['ideal.diam'])))/len((current_sample.loc[current_sample['Xylem.type']=='complete'])))**0.25
            hydraulic_diam.extend([current_hydraulic_diam]*len(current_sample))
        updated_dataset['2axis.hydr.diam'] = hydraulic_diam
    
    return updated_dataset

def spec_conductivity(dataset):
    updated_dataset=dataset.copy()
    if 'long.diam' in dataset.columns:
        spec_xyl_hydraulic = [] # specific hydraulic index
        for sample in dataset['sample.ID'].unique():
            diam_sqred = [] # Product of squared diameters
            current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
            for i in range(0,len((current_sample))):
                if current_sample['long.diam'][i] != 0:
                    diam_sqred.append(((current_sample['short.diam'][i]/(10**6))*(current_sample['long.diam'][i]/(10**6))**2))
                else:
                    diam_sqred.append(0)
            cross_section_area = 0
            for image in current_sample['Image_ID'].unique():
                current_image = (current_sample .loc[current_sample['Image_ID']==image].copy()).reset_index(drop=True)
                cross_section_area += (current_image['Image.Dims'][0][0] * current_image['Image.Dims'][0][1])/((2.98**2)*(10**12))
            spec_xyl_hydraulic.extend([(m.pi * sum(diam_sqred))/(128*(1.002*10**-9)*(cross_section_area))]*len(current_sample))
        updated_dataset['Specific.Conductivity'] = spec_xyl_hydraulic
    elif 'simple.diam' in dataset.columns:
        spec_xyl_hydraulic = [] # specific hydraulic index
        for sample in dataset['sample.ID'].unique():
            diam_4th = [] # Diameters to 4th power
            current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
            for i in range(0,len((current_sample))):
                if current_sample['simple.diam'][i] != 0:
                    diam_4th.append((current_sample['simple.diam'][i]/(10**6))**4)
                else:
                    diam_4th.append(0)
            cross_section_area = 0
            for image in current_sample['Image_ID'].unique():
                current_image = (current_sample .loc[current_sample['Image_ID']==image].copy()).reset_index(drop=True)
                cross_section_area += (current_image['Image.Dims'][0][0] * current_image['Image.Dims'][0][1])/((2.98**2)*(10**12))
            spec_xyl_hydraulic.extend([(m.pi * sum(diam_4th))/(128*(1.002*10**-9)*(cross_section_area))]*len(current_sample))
            updated_dataset['Specific.Conductivity'] = spec_xyl_hydraulic
    return updated_dataset

def cond_index(dataset):
    updated_dataset = dataset.copy()
    index_list = []
    for sample in dataset['sample.ID'].unique():
        current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
        for i in range(len(current_sample)):
            if '2axis.hydr.diam' in dataset:
                index_list.append(current_sample['xyl.density'][i]*current_sample['2axis.hydr.diam'][i])
            elif 'sim.hydr.diam' in dataset:
                index_list.extend(current_sample['xyl.density'][i]*current_sample['sim.hydr.diam'][0])
    updated_dataset['condIndex'] = index_list
    return updated_dataset

def sizeToNum(dataset): # Vessel size to number ratio
    updated_dataset = dataset.copy()
    ratio_list = []
    for sample in dataset['sample.ID'].unique():
        current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
        complete_sample = current_sample.loc[current_sample['Xylem.type']=='complete'].copy()
        ratio_list.extend([complete_sample['Xylem.Areas'].mean()/current_sample['xyl.density'][0]]*len(current_sample))
    
    updated_dataset['sizeToNum_Ratio'] = ratio_list
    return updated_dataset



def vuln_index(dataset):
    updated_dataset = dataset.copy()
    index_list = []
    for sample in dataset['sample.ID'].unique():
        current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
        complete_sample = (current_sample.loc[current_sample['Xylem.type']=='complete'].copy()).reset_index(drop=True)
        vuln_index = complete_sample['simple.diam'].mean()/complete_sample['xyl.density'][0]
        index_list.extend([vuln_index]*len(current_sample))
     
    updated_dataset['vulnerabilityIndex'] = index_list
    return updated_dataset

def group_index(dataset):
    updated_dataset= dataset.copy()
    group_index = []
    for sample in dataset['sample.ID'].unique():
        current_sample = (dataset.loc[dataset['sample.ID']==sample].copy()).reset_index(drop=True)
        n_xylem = current_sample['totXylCount'][0]
        n_groups = []
        for image in current_sample['Image_ID'].unique():
            current_image = (current_sample.loc[current_sample['Image_ID']==image].copy()).reset_index(drop=True)

            median_dist = current_image['simple.diam'].median()
                
            n_groups.append(countGroups(current_sample['Xylem.Contours'], current_sample['simple.diam'],median_dist))
        group_index.extend([len(current_sample)/sum(n_groups)]*len(current_sample))
    updated_dataset['groupIndex'] = group_index
            

    return updated_dataset

def findNearbyCnt(contours, diameter_list, mean_diameter):
    near_points_list = []
    center_list = []
    no_neighbours = set(list(range(0,len(contours))))
    has_neighbours = set([])
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        center_list.append((int((M['m10']/M['m00'])),int((M['m01']/M['m00']))))
    
    for i in range(0, len(center_list)-1):
        set_count = 0
        dist_list = []
        for j in range(i+1, len(center_list)):
            distance_between = np.sqrt((center_list[i][0]-center_list[j][0])**2 + (center_list[i][1]-center_list[j][1])**2) - (2.98*mean_diameter)
            dist_list.append(distance_between)
            if distance_between < (1.49*mean_diameter): # if they are closer than 1 median radius
                near_points_list.append([i,j])
                set_count += 1
                has_neighbours.add(i)
                has_neighbours.add(j)
    no_neighbours = no_neighbours - has_neighbours

    return near_points_list, has_neighbours, no_neighbours

def countGroups(contours, contour_diameters,mean_diameter):
    related_xylem_data = findNearbyCnt(contours, contour_diameters, mean_diameter)
    unsorted_groups = related_xylem_data[0]
    merged_groups = [unsorted_groups[0]]
    
    for pair in unsorted_groups:
        disjoint_count = 0
        for group_index in range(0, len(merged_groups)):
            if not set(merged_groups[group_index]).isdisjoint(pair):
                merged_groups[group_index] = list(set(merged_groups[group_index]).union(pair))
                break
            else:
                disjoint_count += 1
        if disjoint_count == len(merged_groups):
            merged_groups.append(pair)
                
                    
    n_groups = len(merged_groups) + len(related_xylem_data[2])
    return n_groups


