# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:35:21 2019

Process CMJ, DJ, and Pogos .txt files -- CURRENTLY ONLY DJ

@author: BStone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import openpyxl
import tkinter as tk
from tkinter import messagebox, filedialog

import warnings 
warnings.filterwarnings("ignore")

#%% Functions
def locations_of_substring(string, substring):
    """Return a list of locations of a substring."""
    substring_length = len(substring)
    
    def recurse(locations_found, start):
        location = string.find(substring, start)
        if location != -1:
            return recurse(locations_found + [location], location + substring_length)
        else:
            return locations_found

    return recurse([], 0)

#%% Select files to loop through
root = tk.Tk()
root.withdraw()
filez = filedialog.askopenfilenames(parent=root, initialdir='D:/Rob/Data', title='Choose files to process')

rn = len(filez)  # Define number of files to loop through

try:
    if rn > 1:
        for filepath in filez:  # Loop through multiple files if more than one is selected
            print(f'File path being used: {filepath}')
            # Process each file as needed
    else:
        filepath = filez[0]  # Use a single file selection
        print(f'Single file path being used: {filepath}')

    with open(filepath, 'r') as f:
        data = f.read()

    # EXTRACT ADDED DATA    
    Fs = 1000
    Ts = 1 / Fs
    locs = locations_of_substring(data, 'Date')  # Find date of trial
    date = data[locs[-1] + 5:locs[-1] + 17]
    
    Dates = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
    mth = Dates.index(date[0:3]) + 1  # Find month number
    mth = f'{mth:02}'  # Format month to two digits

    day = date[4:6]  # Find day
    yr = date[-4:]  # Find year
    DATE = f'{day}/{mth}/{yr}'

    locs = locations_of_substring(filepath, '/')
    ath = filepath[locs[-1] + 1:-1]
    
    locs = locations_of_substring(ath, ' ')
    athlete = ath[0:locs[0]]

    if "cmj" in ath.lower():
        jump = "CMJ"
    elif "dj" in ath.lower():
        jump = "DJ"
    elif "pogos" in ath.lower():
        jump = "POGO"
    else:
        jump = "UNKNOWN"

    if ' r ' in ath.lower():
        leg = 'Right'
    elif ' l ' in ath.lower():
        leg = 'Left'
    else:
        leg = 'Double'

    #%% DJ PROCESSING
    if jump == 'DJ':
        if len(locs) == 4:
            BoxH = int(ath[locs[-3] + 1:locs[-2]])
        else:
            BoxH = int(ath[locs[-2] + 1:locs[-1]])
        
        # EXTRACT FORCE DATA
        locs = locations_of_substring(data, 'N')
        F = data[locs[-1] + 2:-1]  # Pull force data from complete txt file
        F = F.split()

        # Define time and force vectors as numpy arrays
        t = np.asarray(F[0::7], dtype=float)
        F1x = np.asarray(F[1::7], dtype=float)
        F1y = np.asarray(F[2::7], dtype=float)
        F1z = np.asarray(F[3::7], dtype=float)
        F2x = np.asarray(F[4::7], dtype=float)
        F2y = np.asarray(F[5::7], dtype=float)
        F2z = np.asarray(F[6::7], dtype=float)

        P1F = F1z[:5000]
        P2F = F2z[:5000]
        BW = np.mean(P1F[:1501])
        BWkg = BW / 9.812

        plt.figure(figsize=(15, 10))
        plt.rc('legend', fontsize=20)
        plt.rc('figure', titlesize=20)
        plt.rc('axes', labelsize=20)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.plot(P1F)
        plt.plot(P2F)
        plt.xlabel('Data Point')
        plt.ylabel('Force (N)')
        plt.title(ath[:-3])
        plt.legend(['Plate 1', 'Plate 2'])
        plt.show()

        window = tk.Tk()
        window.withdraw()
        messagebox.showinfo('Question', 'Press okay when you are happy to continue')
        plt.close('all')

        # Calculate residuals for P1 and P2
        res1P1 = np.sqrt(np.power(P1F - BW, 2))
        resP1 = np.sqrt(np.power(P1F, 2))
        resP2 = np.sqrt(np.power(P2F, 2))

        # Calculate max residuals for P1 and P2
        maxresP1 = [np.amax(resP1[j:j + 501]) if j + 500 < len(resP1) else np.amax(resP1[j:]) for j in range(len(resP1))]
        maxresP2 = [np.amax(resP2[j:j + 501]) if j + 500 < len(resP2) else np.amax(resP2[j:]) for j in range(len(resP2))]

        # Calculate additional metrics for DJ
        init = np.amax(res1P1[0:1001]) * 1.75
        OF1F = np.amin(maxresP1[0:4501])
        OF1loc = np.where(maxresP1 == OF1F)[0][0]
        TO1 = maxresP1[OF1loc] * 2
        P1init = P1F < TO1
        Mv = (P1F > (BW - init)) & (P1F < (BW + init))
        SoM = np.where(Mv == False)[0][0]

        # Calculate acceleration, velocity, and displacement
        FzJ = P1F[SoM:]
        acc = (FzJ - BW) / (BW / 9.812)
        vel = np.cumsum(acc) * Ts
        disp = np.cumsum(vel) * Ts

        # Calculate power and work done
        pwr = FzJ * vel
        WD = FzJ * disp

        # Prepare data for export
        results = {
            'Date': [DATE],
            'Athlete': [athlete],
            'Leg': [leg],
            'Box Height': [BoxH],
            'Body Weight (kg)': [BWkg],
            'Start of Movement': [SoM],
            'Max Residual P1': [np.amax(maxresP1)],
            'Max Residual P2': [np.amax(maxresP2)],
            'Power': [np.amax(pwr)],
            'Work Done': [np.sum(WD)],
        }

        df_results = pd.DataFrame(results)

        # Export to Excel
        output_path = 'D:/Rob/Results/'
        output_file = os.path.join(output_path, f'{athlete}_DJ_Results.xlsx')

        if not os.path.exists(output_file):
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df_results.to_excel(writer, index=False, sheet_name='DJ Results')
        else:
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
                df_results.to_excel(writer, index=False, sheet_name='DJ Results', startrow=writer.sheets['DJ Results'].max_row, header=False)

except Exception as e:
    print(f'An error occurred: {e}')
