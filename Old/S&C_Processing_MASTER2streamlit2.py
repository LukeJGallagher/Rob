# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:35:21 2019

Process CMJ, DJ and Pogos .txt files -- CURRENTLY ONLY DJ

@author: BStone

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import openpyxl
from io import BytesIO

# Set the title of the app
st.title("S&C Data Processing")

# Functions
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

# File upload section
uploaded_files = st.file_uploader("Upload your .txt files", accept_multiple_files=True, type="txt")

# Process files if uploaded
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read file
        data = uploaded_file.read().decode("utf-8")

        # Extract athlete's name from file name
        file_name = uploaded_file.name
        base_name = os.path.basename(file_name)  # This extracts just the file name without directories
        locs = locations_of_substring(base_name, ' ')
        athlete = base_name[:locs[0]] if locs else base_name  # Extract the athlete's name

        # Determine the jump type based on the file name
        if "cmj" in base_name.lower():
            jump = "CMJ"
        elif "dj" in base_name.lower():
            jump = "DJ"
        elif "pogos" in base_name.lower():
            jump = "POGO"
        else:
            jump = "Unknown"

        # Determine the leg information based on the file name
        if ' r ' in base_name.lower() or ' r.' in base_name.lower():
            leg = 'Right'
        elif ' l ' in base_name.lower() or ' l.' in base_name.lower():
            leg = 'Left'
        else:
            leg = 'Double'

        # Extract date from the data
        Fs = 1000
        Ts = 1 / Fs
        locs = locations_of_substring(data, 'Date')  # Find date of trial
        if locs:
            date = data[locs[-1] + 5:locs[-1] + 17]
            Dates = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
            mth = Dates.index(date[0:3]) + 1
            mth = f"{mth:02}"  # Format month to two digits
            day = date[4:6]
            yr = date[-4:]
            DATE = f"{day}/{mth}/{yr}"
        else:
            DATE = "Unknown Date"

        # Display extracted information
        st.write(f"**File Name:** {file_name}")
        st.write(f"**Athlete:** {athlete}")
        st.write(f"**Jump Type:** {jump}")
        st.write(f"**Leg:** {leg}")
        st.write(f"**Date of Trial:** {DATE}")

        try:
            # CMJ PROCESSING
            if jump == 'CMJ':  
                # Extract Force Data
                locs = locations_of_substring(data, 'N')
                F = data[locs[-1] + 2:-1]  # Pull force data from complete txt file
                F = F.split()
                # Define time and force vectors as numpy arrays
                t = F[0::7]; t = np.asarray(t) 
                F1x = F[1::7]; F1x = np.asarray(F1x).astype(float)
                F1y = F[2::7]; F1y = np.asarray(F1y).astype(float)
                F1z = F[3::7]; F1z = np.asarray(F1z).astype(float)
                F2x = F[4::7]; F2x = np.asarray(F2x).astype(float)
                F2y = F[5::7]; F2y = np.asarray(F2y).astype(float)
                F2z = F[6::7]; F2z = np.asarray(F2z).astype(float)

                Fz = F1z + F2z
                BW = np.mean(Fz[0:1200])
                BWkg = BW / 9.812

                # Calculate impulse, velocity, displacement, etc...
                imp = (((Fz[0] + Fz[1]) / 2) - BW) * Ts
                for n in range(1, len(Fz) - 1):
                    imp = np.append(imp, (((Fz[n] + Fz[n + 1]) / 2) - BW) * Ts)
                
                vel = np.cumsum(imp / (BW / 9.812))
                disp = np.cumsum(vel * Ts)
                pwr = Fz * vel

                # Data Visualization
                plt.figure(figsize=(10, 6))
                plt.plot(Fz)
                plt.title(f"Force Data - {DATE}")
                plt.xlabel('Data Point')
                plt.ylabel('Force (N)')
                st.pyplot(plt)

                # Export to Excel
                results = {
                    'Date': [DATE],
                    'Athlete': [athlete],
                    'Leg': [leg],
                    'Body Weight (kg)': [BWkg],
                    'Max Force': [np.max(Fz)],
                    'Impulse': [np.sum(imp)],
                    'Max Velocity': [np.max(vel)],
                    'Max Power': [np.max(pwr)]
                }

                df_results = pd.DataFrame(results)

                # Provide download option
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_results.to_excel(writer, index=False, sheet_name='Results')
                output.seek(0)
                
                st.download_button(
                    label="Download Results",
                    data=output,
                    file_name=f"results_{DATE}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
