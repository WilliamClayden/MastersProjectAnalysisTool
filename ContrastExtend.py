# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:25:30 2020

@author: willi
"""
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


#"""ADD IMAGE BLUR/FIND LIBRARY VERSION"""

# The contrast extending method, Including a basic image transform then contrast limited adaptive histogra estimation
def ExtContrast(img, saturation = 10):
    #img = cv2.imread('SEP102x10photo2.TIF')

    enhanced_img = img.copy()
    

    # Formula based on https://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
    #https://www.mathworks.com/help/vision/ref/contrastadjustment.html
    for colour in range(3):
        
        upperbound = int(np.percentile(img[:,:,colour], 100-(saturation/2))) # Take upper limit as the 95th percentile
        lowerbound = int(np.percentile(img[:,:,colour], saturation/2)) # Take upper limit as the 95th percentile
        
        # Apply the transformation
        transform = np.identity(img.shape[0])
        transform = 255*(transform/(upperbound-lowerbound))
        enhanced_img[:,:,colour] = np.matmul(transform, enhanced_img[:,:,colour]-lowerbound)
        
        enhanced_img[:,:,colour][img[:,:,colour] >= upperbound] = 255
        enhanced_img[:,:,colour][img[:,:,colour] <= lowerbound] = 0
        #if colour != 0:
        #    enhanced_img[:,:,colour] = enhanced_img[:,:,colour]
        
        imgray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    
        
        # Contrast limited adaptive histrogram estimation https://docs.opencv.org/3.4/d5/daf/tutorial_py_histogram_equalization.html
        # Makes a second slight adjustment to even out the distribution and give a little more even contrast to the images
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)) 
        climg = clahe.apply(imgray)
    
        
    
    return climg

"""The click_contour will allow the user to selectively remove contours
This is more effective to the previous method as it is able to identify them all rapidly 
THEN remove any unwanted ones using a series of techniques.

"""
drawing = False  # The drawing variable is global as it can apply to any image
mode = True
ix,iy=-1,-1 # default coordinate position
add_list = []
rem_list = []

def click_contour(event,x,y,flags,param): 
    global ix,iy,drawing,mode, add_list, rem_list
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = False
        ix,iy = x,y
        print(ix,iy, mode)
        if mode == True:
            add_list.append([ix,iy])
            print(add_list)
        elif mode == False:
            rem_list.append([ix,iy])
            print(rem_list)

        
def FindXylem(img): # The function to extract xylem areas
    
    climg = ExtContrast(img) # Add contrast to the image

    plt.hist(climg.ravel(),245,[5,250]) # Dont use the full range because the edge values skew it
    plt.show()
    
    set_threshold = True # Include truth to determine whether the threshold will need to be changed (Useful for the end part of this code)
    xylem_count = 0
    while True:
        
        if set_threshold == True:
            BW_limit = int(input("Select the threshold limit based on the images and the histogram (Integer between 100 and 210, See user manual for details) \n"))
            
        ret, thresh = cv2.threshold(climg, BW_limit, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
        print("Number of Contours found = " + str(len(contours))) 
        
        if set_threshold==True: # This has to come after the thresholding as that is needed every loop but only needs to be shown if wanted
            cv2.destroyAllWindows()
            cv2.namedWindow("My Image", cv2.WINDOW_NORMAL)
            cv2.imshow('My Image', thresh) # Generate the image after thresholding to show any errors in black and white
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            """Overlay the contours on the original image"""
            newcolor = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            cv2.destroyAllWindows()
            cv2.drawContours(newcolor, contours, -1, (0,255,0), 3)
            
            cv2.namedWindow("My Image", cv2.WINDOW_NORMAL)
            cv2.imshow('My Image', newcolor)
            cv2.waitKey(0)
            set_threshold= False # Update the threshold value so if you need to change anything it won't automatically repeat each loop
            
            areas = []
            for contour in contours:
                areas.append(cv2.contourArea(contour))

            set_percentile = 98.5 # Use a default 98.5th percentile, allowing for error cells to be removed
            xylem_index = [i for i, v in enumerate(areas) if v > np.percentile(areas, set_percentile)] # Find the index locations of the xylem
            xylem = [contours[i] for i in tuple(xylem_index)] # Find all the contours above this threshold
            xylem_count = len(xylem)
            print("There are currently " + str(xylem_count) + " cells identified as xylem")
                
        updated = cv2.drawContours(img.copy(), xylem, -1, (0,255,0), 3) # Draw the contours that are correct based on the current percentiles
        cv2.namedWindow("Selected Contour Image", cv2.WINDOW_NORMAL)
        cv2.imshow('Selected Contour Image', updated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        print("Is the current selection correctly identifying only xylem?")
        satisfied = input("Enter yes or no (Uninteiligible answers assumed as no): \n")
        if satisfied.lower() == "yes":
            satisfied_bool = True
        else:
            satisfied_bool = False
        while satisfied_bool == False:
            print("Would you like to change the percentile (whole number) to add or remove a large number of detected cells")
            percentile_tweak = input("Enter \"yes\" to change the percentile threshold \n")
            if percentile_tweak.lower() == "yes":
                new_percentile = input("Set the percentile threshold to select the xylem (recommended between 98 and 99, default 98): \n")
                xylem_index = []
                xylem = []
            
                """Use exception handling to process whether a number is passed
                Replace with if statements to handle correctly e.g. if type == x then do this, else do this
                """
                try: 
                    new_percentile = float(new_percentile)
                except:   
                    print("Number unreadable, the percentile has been set at the default (99th)")
                    new_percentile = 98

                
                xylem_index = [i for i, v in enumerate(areas) if v > np.percentile(areas, new_percentile)] # Find the index locations of the xylem
                xylem = [contours[i] for i in tuple(xylem_index)] # Find all the contours above this treshold
                xylem_count = len(xylem)
                
                updated = cv2.drawContours(img.copy(), xylem, -1, (0,255,0), 3) # Draw the contours that are correct based on the current percentiles
                cv2.namedWindow('Selected Contour Image', cv2.WINDOW_NORMAL)
                cv2.imshow('Selected Contour Image', updated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                print("Is the current selection correctly identifying only xylem?")
                satisfied = input("Enter yes or no (Uninteiligible answers assumed as no): \n")
                if satisfied.lower() == "yes":
                    break
        
            print("Since the current selection is not complete, click to add or remove contours from the found list. ")
            print("You are in contour adding mode by default, press t to change modes between adding and removing ")
            print("If some Xylem won't remove or add by clicking, go through to t")
            cv2.namedWindow('Selected Contour Image', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Selected Contour Image',click_contour)
            global ix,iy,drawing,mode
            while(1):
                global add_list, rem_list
                updated = cv2.drawContours(img.copy(), xylem, -1, (0,255,0), 3)
                
                cv2.imshow('Selected Contour Image', updated)
                k = cv2.waitKey(800) & 0xFF
                
                if k == ord('t'):# T for toggle
                    mode = not mode
                    if mode == True:
                        print("You are now in adding contour mode")
                    elif mode == False:
                        print("You are now in remove contour mode")
                elif k == 27: # 27 is the unicode for the esc key
                    break
                
                # Remove unwanted cellst
                for point in rem_list:
                    closest_xylem=[]
                    dist_to_closest = 10**8
                    for cell in xylem:    
                         # Find the closest contour to the selected point
                        new_dist_to_closest = abs(cv2.pointPolygonTest(cell, tuple(point), True))
                        if dist_to_closest > new_dist_to_closest:
                            dist_to_closest = abs(cv2.pointPolygonTest(cell, tuple(point), True))
                            closest_xylem = cell
                    try:
                        xylem.remove(closest_xylem)
                    except:
                        pass
                
                # Add wanted cells
                for point in add_list:
                    for cell in contours:
                            if cv2.pointPolygonTest(cell, tuple(point), False) == 1:
                                # Check against the existing list of xylem so a contour is never added twice
                                count = 0
                                try:
                                    for item in xylem:
                                        if item == cell:
                                            count += 1
                                except:
                                    pass
                                if count == 0: # only append if no instances
                                    xylem.append(cell)
                add_list, rem_list = [], []

            cv2.destroyAllWindows()
            updated = cv2.drawContours(img.copy(), xylem, -1, (0,255,0), 3) # Draw the contours that are correct based on the current percentiles
            cv2.namedWindow('Final selection', cv2.WINDOW_NORMAL)
            cv2.imshow('Final selection', updated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                
            
            print("Is the current selection correctly identifying only xylem with minimal imperfections?")
            satisfied = input("Enter yes or no (Uninteiligible answers assumed as no): \n")
            if satisfied.lower() == "yes":
                break
        

                
                
        updated = cv2.drawContours(img.copy(), xylem, -1, (0,255,0), 3)
        cv2.namedWindow('Final selection', cv2.WINDOW_NORMAL)
        cv2.imshow('Final selection', updated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        satisfy_check = input("Is the current image selection appropriate? (Are the Xylem fully identified without parenchyma or other breakdown); Enter Yes or No ")
        if satisfy_check.lower() == "yes":
            break
        else:
            print("\n Is the current image colour threshold too low or is the number of Xylem detected incorrect?")
            problem_check = input("Enter X for Xylem or T for threshold and xylem:  ") # Use T to do both as if you change the treshold you will need to check the xylem
            if problem_check.lower() == "t": # Handle the case where the user enters a lower case T too
                set_threshold = True         
            else:
                pass

    # Find wanted xylem areas
    xylem_areas = []
    for contour in xylem:
        xylem_areas.append(cv2.contourArea(contour))

    return xylem, xylem_areas, thresh


"""There is a bug which some tiny xylem may not be noticed correctly"""
