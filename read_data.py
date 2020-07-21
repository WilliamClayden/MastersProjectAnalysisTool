# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:57:00 2020

@author: willi
"""
import pandas as pd
import os
import glob

def xylem_from_pkl():
    print("Which folder would you like to import pkl from?")
    print("The default folder is the save location for the pkl")
    new_loc = input("Input the folder path or hit enter to use the save location:")
    new_loc = new_loc.replace("\\", "/")
    if len(new_loc) > 0:
        if new_loc[-1] == "/":
            new_loc = new_loc[:-1]
    
    try:
        with open(str(os.path.dirname(__file__))+ "/save_location.txt") as file:
            default = file.readline()
            default = default.lstrip('\x00')
            if new_loc == default:
                os.chdir(default)
                xylem_data = [pd.read_pickle(file) for file in glob.glob("*.pkl")]
                dup_rem = [] # Remove any duplicate dataframe entires
                for df1 in range(len(xylem_data)-1):
                    for df2 in range(df1, len(xylem_data)):
                        if df1 != df2 and (xylem_data[df1].drop('Xylem.Contours',axis=1)).equals(xylem_data[df2].drop('Xylem.Contours', axis=1)):
                            if df2 not in dup_rem:
                                dup_rem.append(df2)
                n=0
                for duplicate in dup_rem:
                    del xylem_data[duplicate-n]
                    n+=1
                            
                dataset = pd.concat(xylem_data)

            else:
                os.chdir(new_loc)
                xylem_data = [pd.read_pickle(file) for file in glob.glob("*.pkl")]
                dup_rem = [] # Remove any duplicate dataframe entires
                for df1 in range(len(xylem_data)):
                    for df2 in range(df1, len(xylem_data)):
                        if df1 != df2 and (xylem_data[df1].drop('Xylem.Contours',axis=1)).equals(xylem_data[df2].drop('Xylem.Contours', axis=1)):
                            if df2 not in dup_rem:
                                dup_rem.append(df2)
                n=0
                for duplicate in dup_rem:
                    del xylem_data[duplicate-n]
                    n+=1
                
                dataset = pd.concat(xylem_data)

        dataset=dataset.reset_index(drop=True)    
                
    except:
        try:
            os.chdir(os.path.dirname(__file__)+"/datadump/")
            xylem_data = [pd.read_pickle(file) for file in glob.glob("*.pkl")]
            dup_rem = [] # Remove any duplicate dataframe entires
            for df1 in range(len(xylem_data)):
                for df2 in range(df1, len(xylem_data)):
                    if df1 != df2 and (xylem_data[df1].drop('Xylem.Contours',axis=1)).equals(xylem_data[df2].drop('Xylem.Contours', axis=1)):
                        if df2 not in dup_rem:
                            dup_rem.append(df2)
            n=0
            for duplicate in dup_rem:
                del xylem_data[duplicate-n]
                n+=1
                
            dataset = pd.concat(xylem_data)

            dataset=dataset.reset_index(drop=True)
        except:    
            print("Pkl load failed. Please input a correct directory path")
            dataset = []

    dataset['sample.ID'] = pd.Categorical(dataset['sample.ID'], categories=dataset['sample.ID'].unique(), ordered=True)
    dataset.sort_values('sample.ID')
    return dataset

#test= xylem_from_pkl()
