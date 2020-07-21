# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 10:47:06 2020

@author: willi
"""

import os
import pandas
import numpy

"""This function takes the output dataframe from the initial image analysis and saves it to a user specified folder.

If they have previously used a folder then it is remembered and saved there. 

If the path is not recognised it saves into a sub folder of the code location

"""
def save_xylem(dataframe):
    print("What name would you like to give the output data file?")
    print(" (Do not use duplicate names as they will be overwritten without warning) \n")
    file_name = input() # Name the output file
    
    print("Enter the path to the folder you want to use")
    print(" (If the same as previous use or the same as current folder press enter to skip)")
    save_path = input()
    
    save_path = save_path.replace("\\", "/")
    if len(save_path) > 0:
        if save_path[-1] == "/":
            save_path = save_path[:-1]
    
    # Use the try form because if the path is entered incorrectly it won't run
    try:
        if os.path.isfile(str(os.path.dirname(__file__))+ "/save_location.txt") == True:
            with open(str(os.path.dirname(__file__))+ "/save_location.txt", "r+") as file:
                old_path = file.readline()
                old_path = old_path.lstrip('\x00')
                if save_path.lower() ==  old_path:
                    print("dataframe saved to original folder")
                    dataframe.to_pickle(old_path +"/"+ file_name + ".pkl")
                    file.close()
                else:
                    print("data frame saved to new folder")
                    dataframe.to_pkl(save_path +"/"+ file_name + ".pkl")
                    file.truncate(0)
                    file.write(save_path)
                    file.close()
                    # Update folder for next time       
        else:
            with open(str(os.path.dirname(__file__))+"/save_location.txt","w+") as file:
                file.write(save_path)
                file.close()
                print("dataframe saved to new folder")
                dataframe.to_pickle(save_path +"/"+file_name + ".pkl")
    except:
        print("Path not recognised, database saved to the datadump folder in the code folder")
        print("dataframe saved to: ")
        print(os.path.dirname(__file__)+"/datadump") # print the path
        if not os.path.exists(os.path.dirname(__file__)+"/datadump"):
            os.makedirs(os.path.dirname(__file__)+"/datadump")
        dataframe.to_pickle(os.path.dirname(__file__)+"/datadump/" + file_name + ".pkl")

    return