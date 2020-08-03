# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:28:29 2020

@author: willi
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import copy


global img_cache
global colour
global drawsize
img_cache = []
colour = (255,255,255) # Colour set to red as default
drawing = False  # The drawing variable is global as it can apply to any image
drawsize = 10

def drawCorrections(images, file_names):
    global ix,iy,drawing,colour, img_cache, image,colour, drawsize

    for i in range(len(images)):
        img_cache = []
        image = images[i]
        img_cache.append(copy.deepcopy(image))

        ix,iy=-1,-1 # default coordinate position
        
        cv2.namedWindow('Image to Edit', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Image to Edit', draw)

        while(1):
            cv2.imshow('Image to Edit', image)
            k = cv2.waitKey(1) & 0xFF
            
            if k == ord("u"):
                if len(img_cache) > 1:
                    img_cache.pop(-1)
                    image = copy.deepcopy(img_cache[-1])
                    cv2.imshow('Image to Edit', image)
    
            if k == ord("t"):
                if drawsize >= 10:
                    drawsize = 6
                    print("The pen size is now small")
                elif drawsize < 10:
                    drawsize = 12
                    print("The pen size is now large")
            if k == 27: # 27 is the unicode for the esc key
                print("Have you finished making corrections to this image?")
                finished = input("Enter yes or no: ")
                if finished.lower() == "yes":
                    cv2.imwrite("EditedImages/"+"Edited"+file_names[i], image)
                    break
                else:
                    print("Continue editing the image")
        
        cv2.destroyAllWindows()
  
    return



def draw(event,x,y,flags,param): 
    global ix,iy,drawing,colour, image,img_cache, colour, drawsize
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(image,(ix,iy),(x,y),colour,drawsize)
            ix=x
            iy=y
                
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img_cache.append(copy.deepcopy(image))
 
        
    elif event == cv2.EVENT_RBUTTONDOWN: # Allow the user to select the colour
        drawing = False
        colour = (int(image[y,x,0]), int(image[y,x,1]), int(image[y,x,2]))
     
    return x,y
        


