# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:53:01 2020

@author: willi
"""

# Image reading multiple files


import glob
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_tools as dtool
import xylem_detector as img_tool
import save_data as sv
import read_data as rd
import image_editor as edit




def edit_images():
    image_get = rd.images_from_folder(True)
    images, parent_dir, file_names = image_get[0], image_get[1], image_get[2]
    path = os.path.join(parent_dir, "EditedImages")
    if os.path.isdir("./EditedImages") == False:
        os.mkdir(path)
    edit.drawCorrections(images, file_names)
    
    
    return 
edit_images()
"""The xylem find method does the image processing and handles creation of an interpretable database"""
def xylem_find():
    images = rd.images_from_folder(False)[0]

    # This is for the database creation such that you have already identified the plot area of the images before 
    location = input("Enter the plot location for these images: ")

    image_counter = 0 # Image ID counter, simply counts how many images have been processed for numbering 
    
    save = False # Save is a conditional statement such that you can save and exit at the end of any image process, whether it is the 1st the 5th etc.
    

    while save == False: # It doesnt analyse the images until TRUE
        for item in images:    # Iterates through every image in the loaded dataset
            image_counter += 1 # Increase the image ID count by 1 for every image passed through so they are uniquely ID'd (May not be necessary if species is used instead)
            image_species = input("What species tree is the image (e.g. dipterocarpus actuangulus)? ")
            try:
                image_species = image_species.lower()
            except:
                print("The image species was not a string and will have to be reentered later")
        
            # species_list.append(image_species.lower()) Incorporate species selection later
            sample_ID = input("Enter a unique sample ID to refer to this specific sample series: ") # Each sample is from a set of several
            try:
                sample_ID = sample_ID.lower()
            except:
                print("The sample ID was not a string and will have to be reentered later")
        
            # Create a window that is of adjustable size (this enables larger images to be viewed on any display)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            # This enables the user to manually crop a rectangluar area out of the image for analysis
            print("Please manually select the area required for analysis (press return/enter to continue)")
            print("Press return/enter without selecting to use the whole image")
            print("\n")
            roi = cv2.selectROI("Image",item) # For manual area selection
            # Save the area selection
        
            if not all(roi):
                current_selection = item
            else:
                current_selection = item[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] # Crop the photo to the desired area
        
        
            cv2.destroyAllWindows() # Refreshing the windows
        
            # This shows the area selected
            cv2.namedWindow("Cropped Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Cropped Image", current_selection)
            cv2.waitKey(0) # The image closes on key press
            cv2.destroyAllWindows()
        
        
            results = img_tool.FindXylem(current_selection) # Using the FindXylem function defined in the other file

            # add the image data to
            
            temp_dataframe = pd.DataFrame()
            """Add each column one at a time because the data is not complete"""
            # Convert needed ones to categorical data
            temp_dataframe['location'] = pd.Series([location]*len(results[0]))
            temp_dataframe['location'] = pd.Categorical(temp_dataframe['location'])
            temp_dataframe['species'] = pd.Series([image_species]*len(results[0]))
            temp_dataframe['species'] = pd.Categorical(temp_dataframe['species'] )
            temp_dataframe['sample.ID'] = pd.Series([sample_ID]*len(results[0]))
            temp_dataframe['sample.ID'] = pd.Categorical(temp_dataframe['sample.ID'])
            temp_dataframe['Image_ID'] = pd.Series([image_counter]*len(results[0])) # Include a unique image ID for any analysis on individual images
            temp_dataframe['Xylem.Contours'] = pd.Series(results[0])
            xy_areas = np.array(results[1])/(2.98**2) # Convert xylem areas to micro metres
            temp_dataframe['Xylem.Areas'] = pd.Series(xy_areas)
            temp_dataframe['Image.Dims'] = pd.Series([results[2].shape]*len(results[0]))

        
            if 'dataset' not in locals():
                dataset = temp_dataframe.copy()
            else:
                dataset = pd.concat([dataset, temp_dataframe.copy()])
           # Check image equivalence
            try:
                diff = cv2.subtract(item, images[-1])
                b,g,r = cv2.split(diff)
                if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                    save = True
                else:
                    save_check = input("Is all the needed data extracted (minimum of all images processed for a given sample)? Yes or No \n")
                    if save_check.lower() == "yes":
                        save = True
            except:
                pass
        #save = True # Save if all the images have been analysed
    sv.save_xylem(dataset) # Save the dataframe as a pickle file to retain 
    
    return


"""The feature extraction handles getting all the needed data out of the given datast"""
def feature_extraction():
    dataset = rd.xylem_from_pkl()
    xylem_class = []
    
    # We need a list of all the contour complete/incomplete values
    # This is done by sample so the list is updated correctly
    
    for sample in dataset['sample.ID'].unique():
        subset_data = dataset.loc[dataset['sample.ID']==sample]
        xylem_contours = subset_data['Xylem.Contours'].to_list()
        xylem_class.extend(dtool.Find_Complete_Xylem(xylem_contours, subset_data['Image.Dims'].iloc[0])) # Use the find complete xylem function defined in the other file
    dataset['Xylem.type'] = xylem_class
    
    """Next we want to find area"""
    dataset = dtool.calc_total_areas(dataset)
    """Then we want to find the density and fraction (hard task)"""
    dataset = dtool.get_density(dataset)
    """Then get the dataset diameters"""
    dataset = dtool.vessel_diameters(dataset)
    """Then find the hydraulic diameter"""
    dataset = dtool.hydraulic_diameter(dataset)
    """Then get the theoretical specific conductivity"""
    dataset = dtool.spec_conductivity(dataset)
    """Then get the conductivity index"""
    dataset = dtool.cond_index(dataset)
    """Then get the size to number ratio"""
    dataset = dtool.sizeToNum(dataset)
    """Then get the vulnerability index"""
    dataset = dtool.vuln_index(dataset)
    """Then finally get grouping index"""
    dataset = dtool.group_index(dataset)
    # Remove unwanted rows such as contour
    dataset = dataset.drop(['Xylem.Contours', 'Image.Dims'], axis=1)
    sv.save_analysis(dataset)
    
    return

#xylem_find()
feature_extraction()