# -*- coding: utf-8 -*-

"""

Process DJ .txt files using Streamlit for visualization and data processing.



"""



# =============================
#        IMPORT SECTION       
# =============================

import streamlit as st

# =============================
#        IMPORT SECTION       
# =============================

import pandas as pd

# =============================
#        IMPORT SECTION       
# =============================

import numpy as np

# =============================
#        IMPORT SECTION       
# =============================

import matplotlib.pyplot as plt

# =============================
#        IMPORT SECTION       
# =============================

import os

from io import BytesIO



# Set the title of the app

st.title("DJ Data Processing")



# Functions


# =============================
#       FUNCTION DEFINITION    
# =============================

def locations_of_substring(string, substring):

    """Return a list of locations of a substring."""


# =============================
#       CALCULATION SECTION 1 
# =============================

# Calculation: substring_length = len(substring)

    substring_length = len(substring)    


# =============================
#       FUNCTION DEFINITION    
# =============================

    def recurse(locations_found, start):

# Calculation: location = string.find(substring, start)

        location = string.find(substring, start)

# Calculation: if location != -1:

        if location != -1:

            return recurse(locations_found + [location], location + substring_length)

        else:

            return locations_found



    return recurse([], 0)



# File upload section


# =============================
#       CALCULATION SECTION 2 
# =============================

# Calculation: uploaded_files = st.file_uploader("Upload your DJ .txt files", accept_multiple_files=True, type="txt")

uploaded_files = st.file_uploader("Upload your DJ .txt files", accept_multiple_files=True, type="txt")



# Process files if uploaded

if uploaded_files:


# =============================
#       CALCULATION SECTION 3 
# =============================

# Calculation: tabs = st.tabs([f"File {i+1}" for i in range(len(uploaded_files))])  # Create tabs for each file

    tabs = st.tabs([f"File {i+1}" for i in range(len(uploaded_files))])  # Create tabs for each file

    

    for idx, uploaded_file in enumerate(uploaded_files):

        # Read file


# =============================
#       CALCULATION SECTION 4 
# =============================

# Calculation: data = uploaded_file.read().decode("utf-8")

        data = uploaded_file.read().decode("utf-8")



        # Extract athlete's name from file name


# =============================
#       CALCULATION SECTION 5 
# =============================

# Calculation: file_name = uploaded_file.name

        file_name = uploaded_file.name

# Calculation: base_name = os.path.basename(file_name)  # This extracts just the file name without directories

        base_name = os.path.basename(file_name)  # This extracts just the file name without directories

# Calculation: locs = locations_of_substring(base_name, ' ')

        locs = locations_of_substring(base_name, ' ')

# Calculation: athlete = base_name[:locs[0]] if locs else base_name  # Extract the athlete's name

        athlete = base_name[:locs[0]] if locs else base_name  # Extract the athlete's name



 # Determine Box Height (BoxH) from the file name


# =============================
#       CALCULATION SECTION 6 
# =============================

# Calculation: if len(locs) == 4:

        if len(locs) == 4:

# Calculation: BoxH = int(base_name[locs[-3]+1:locs[-2]])

                BoxH = int(base_name[locs[-3]+1:locs[-2]])

        else:

# Calculation: BoxH = int(base_name[locs[-2]+1:locs[-1]])

                BoxH = int(base_name[locs[-2]+1:locs[-1]])



        # Determine the jump type (Only DJ here)


# =============================
#       CALCULATION SECTION 7 
# =============================

# Calculation: jump = "DJ"

        jump = "DJ"

        

        # Determine the leg information based on the file name

        if ' r ' in base_name.lower() or ' r.' in base_name.lower():


# =============================
#       CALCULATION SECTION 8 
# =============================

# Calculation: leg = 'Right'

            leg = 'Right'

        elif ' l ' in base_name.lower() or ' l.' in base_name.lower():

# Calculation: leg = 'Left'

            leg = 'Left'

        else:

# Calculation: leg = 'Double'

            leg = 'Double'



        # Extract date from the data


# =============================
#       CALCULATION SECTION 9 
# =============================

# Calculation: Fs = 1000

        Fs = 1000

# Calculation: Ts = 1 / Fs

        Ts = 1 / Fs

# Calculation: locs = locations_of_substring(data, 'Date')  # Find date of trial

        locs = locations_of_substring(data, 'Date')  # Find date of trial

        if locs:

# Calculation: date = data[locs[-1] + 5:locs[-1] + 17]

            date = data[locs[-1] + 5:locs[-1] + 17]

# Calculation: Dates = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

            Dates = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

# Calculation: mth = Dates.index(date[0:3]) + 1

            mth = Dates.index(date[0:3]) + 1

# Calculation: mth = f"{mth:02}"  # Format month to two digits

            mth = f"{mth:02}"  # Format month to two digits

# Calculation: day = date[4:6]

            day = date[4:6]

# Calculation: yr = date[-4:]

            yr = date[-4:]

# Calculation: DATE = f"{day}/{mth}/{yr}"

            DATE = f"{day}/{mth}/{yr}"

        else:

# Calculation: DATE = "Unknown Date"

            DATE = "Unknown Date"



        # Display extracted information in the corresponding tab

        with tabs[idx]:

            st.header(f"Details for File {idx + 1}")

            st.write(f"**File Name:** {file_name}")

            st.write(f"**Athlete:** {athlete}")

            st.write(f"**Jump Type:** {jump}")

            st.write(f"**Leg:** {leg}")

            st.write(f"**Date of Trial:** {DATE}")




# =============================
#       CALCULATION SECTION 10 
# =============================

            # Visualization and calculations for DJ

# Calculation: chart_tab, calc_tab = st.tabs(["Chart", "Calculations"])

            chart_tab, calc_tab = st.tabs(["Chart", "Calculations"])

            

        with chart_tab:

               

                # Extract force data for DJ


# =============================
#       CALCULATION SECTION 11 
# =============================

# Calculation: locs = locations_of_substring(data, 'N')

                locs = locations_of_substring(data, 'N')

# Calculation: F = data[locs[-1] + 2:-1]  # Pull force data from complete txt file

                F = data[locs[-1] + 2:-1]  # Pull force data from complete txt file

# Calculation: F = F.split()

                F = F.split()

                # Define time and force vectors as numpy arrays

                


# =============================
#       CALCULATION SECTION 12 
# =============================

# Calculation: t = F[0::7]; t=np.asarray(t)

                t = F[0::7]; t=np.asarray(t) 

# Calculation: F1x = F[1::7]; F1x=np.asarray(F1x); F1x = F1x.astype(float)

                F1x = F[1::7]; F1x=np.asarray(F1x); F1x = F1x.astype(float)

# Calculation: F1y = F[2::7]; F1y=np.asarray(F1y); F1y = F1y.astype(float)

                F1y = F[2::7]; F1y=np.asarray(F1y); F1y = F1y.astype(float)

# Calculation: F1z = F[3::7]; F1z=np.asarray(F1z); F1z = F1z.astype(float)

                F1z = F[3::7]; F1z=np.asarray(F1z); F1z = F1z.astype(float)

# Calculation: F2x = F[4::7]; F2x=np.asarray(F2x); F2x = F2x.astype(float)

                F2x = F[4::7]; F2x=np.asarray(F2x); F2x = F2x.astype(float)

# Calculation: F2y = F[5::7]; F2y=np.asarray(F2y); F2y = F2y.astype(float)

                F2y = F[5::7]; F2y=np.asarray(F2y); F2y = F2y.astype(float)

# Calculation: F2z = F[6::7]; F2z=np.asarray(F2z); F2z = F2z.astype(float)

                F2z = F[6::7]; F2z=np.asarray(F2z); F2z = F2z.astype(float)

                


# =============================
#       CALCULATION SECTION 13 
# =============================

# Calculation: P1F = F1z[0:5000]; P2F = F2z[0:5000];

                P1F = F1z[0:5000]; P2F = F2z[0:5000];

                

# Check if the data is available



# Simple Moving Average function for filtering


# =============================
#       FUNCTION DEFINITION    
# =============================

def moving_average(data, window_size):


# =============================
#       CALCULATION SECTION 14 
# =============================

# Calculation: return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')



# Check if the data is available

if 'F1z' in locals() and 'F2z' in locals() and len(F1z) > 0 and len(F2z) > 0:

    # Apply smoothing to force data with a window size of 50 (adjust as needed)


# =============================
#       CALCULATION SECTION 15 
# =============================

# Calculation: smoothed_F1z = moving_average(F1z, window_size=50)

    smoothed_F1z = moving_average(F1z, window_size=50)

# Calculation: smoothed_F2z = moving_average(F2z, window_size=50)

    smoothed_F2z = moving_average(F2z, window_size=50)



    # Plot the smoothed force data for visual inspection


# =============================
#       CALCULATION SECTION 16 
# =============================

# Calculation: fig, ax = plt.subplots(figsize=(12, 6))

    fig, ax = plt.subplots(figsize=(12, 6))

# Calculation: ax.plot(smoothed_F1z, label='Smoothed Plate 1 Force (F1z)')

    ax.plot(smoothed_F1z, label='Smoothed Plate 1 Force (F1z)')

# Calculation: ax.plot(smoothed_F2z, label='Smoothed Plate 2 Force (F2z)')

    ax.plot(smoothed_F2z, label='Smoothed Plate 2 Force (F2z)')

    ax.set_xlabel('Data Point')

    ax.set_ylabel('Force (N)')

    ax.set_title('Smoothed Force Data from Plates 1 and 2')

    ax.legend()

    st.pyplot(fig)



    # Slider inputs for data range selection

    st.write("### Select Data Range for Body Weight Calculation")


# =============================
#       CALCULATION SECTION 17 
# =============================

# Calculation: start_point = st.slider('Select Start Point', min_value=0, max_value=len(smoothed_F1z)-1, value=0)

    start_point = st.slider('Select Start Point', min_value=0, max_value=len(smoothed_F1z)-1, value=0)

# Calculation: end_point = st.slider('Select End Point', min_value=0, max_value=len(smoothed_F1z)-1, value=1500)

    end_point = st.slider('Select End Point', min_value=0, max_value=len(smoothed_F1z)-1, value=1500)



    # Ensure the end point is after the start point

    if start_point < end_point:

        # Slice the smoothed data based on the selected range


# =============================
#       CALCULATION SECTION 18 
# =============================

# Calculation: P1F = smoothed_F1z[start_point:end_point]

        P1F = smoothed_F1z[start_point:end_point]

# Calculation: P2F = smoothed_F2z[start_point:end_point]

        P2F = smoothed_F2z[start_point:end_point]



        # Calculate body weight from the selected range


# =============================
#       CALCULATION SECTION 19 
# =============================

# Calculation: BW1 = np.mean(P1F) if len(P1F) > 0 else 0

        BW1 = np.mean(P1F) if len(P1F) > 0 else 0

# Calculation: BW2 = np.mean(P2F) if len(P2F) > 0 else 0

        BW2 = np.mean(P2F) if len(P2F) > 0 else 0



        # Validate body weight values

        if BW1 > 500 or BW2 > 500:  # Use 'or' to allow fallback to a single plate


# =============================
#       CALCULATION SECTION 20 
# =============================

# Calculation: BW = (BW1 + BW2) / 2 if BW1 > 500 and BW2 > 500 else max(BW1, BW2)

            BW = (BW1 + BW2) / 2 if BW1 > 500 and BW2 > 500 else max(BW1, BW2)

# Calculation: BWkg = BW / 9.812

            BWkg = BW / 9.812

            st.write(f"**Body Weight (N):** {BW:.2f}")

            st.write(f"**Body Weight (kg):** {BWkg:.2f}")

        else:

            st.write("Error: Calculated body weight is below expected range (> 500 N). Please adjust the selection or check the data.")

    else:

        st.write("Error: The end point must be greater than the start point.")

else:

    st.write("Error: `F1z` or `F2z` does not have sufficient data.")






# =============================
#       CALCULATION SECTION 21 
# =============================

    with calc_tab:

              

            # Ensure F1z and F2z have the expected data


# =============================
#       CALCULATION SECTION 22 
# =============================

            # Extract and perform calculations for DJ

            # Validate if F1z and F2z have enough data points

# Calculation: if 'F1z' in locals() and 'F2z' in locals() and len(F1z) >= 5000 and len(F2z) >= 5000:

            if 'F1z' in locals() and 'F2z' in locals() and len(F1z) >= 5000 and len(F2z) >= 5000:

# Calculation: P1F = F1z[0:5000]

                P1F = F1z[0:5000]

# Calculation: P2F = F2z[0:5000]

                P2F = F2z[0:5000]

                

                # Check if both P1F and P2F have the minimum required data points


# =============================
#       CALCULATION SECTION 23 
# =============================

# Calculation: if len(P1F) >= 1501 and len(P2F) >= 1501:

                if len(P1F) >= 1501 and len(P2F) >= 1501:

                    # Calculate body weight from both plates

# Calculation: BW1 = np.mean(P1F[0:1501])  # Body weight from Plate 1

                    BW1 = np.mean(P1F[0:1501])  # Body weight from Plate 1

# Calculation: BW2 = np.mean(P2F[0:1501])  # Body weight from Plate 2

                    BW2 = np.mean(P2F[0:1501])  # Body weight from Plate 2

                    

                    # Check if both BW1 and BW2 are within a reasonable range

                    if BW1 > 500 and BW2 > 500:


# =============================
#       CALCULATION SECTION 24 
# =============================

# Calculation: BW = (BW1 + BW2) / 2  # Use the average of both plates if valid

                        BW = (BW1 + BW2) / 2  # Use the average of both plates if valid

                    else:

                        st.write("Error: Calculated body weight is below expected range (> 500 N).")

                    

                    # Convert to kg


# =============================
#       CALCULATION SECTION 25 
# =============================

# Calculation: BWkg = BW / 9.812

                    BWkg = BW / 9.812

                else:

                    st.write("Error: `P1F` or `P2F` does not have the minimum required data points (1501).")

            else:

                st.write("Error: `F1z` or `F2z` does not have sufficient data.")



               

               

                               # Debugging outputs to verify values

                st.write("### Debug Information")

                st.write(f"First 10 values of F1z: {F1z[:10]}")

                st.write(f"First 10 values of P1F: {P1F[:10]}")

                st.write(f"Body Weight (N): {BW}")

                st.write(f"Body Weight (kg): {BWkg}")

         


# =============================
#       CALCULATION SECTION 26 
# =============================

                 # Proceed to display the calculated value

                st.write("### Calculations for DJ")

                st.write(f"**Body Weight (kg):** {BWkg:.2f}")

                st.write(f"**Mean Force (Plate 1):** {np.mean(P1F):.2f} N")

                st.write(f"**Mean Force (Plate 2):** {np.mean(P2F):.2f} N")

                st.write("Error: `P1F` does not have the minimum required data points (1501).")

                      


# =============================
#       CALCULATION SECTION 27 
# =============================

# Calculation: res1P1 = np.sqrt(np.power(P1F-BW,2))

    res1P1 = np.sqrt(np.power(P1F-BW,2))

# Calculation: resP1 = np.sqrt(np.power(P1F,2)); resP2 = np.sqrt(np.power(P2F,2));

    resP1 = np.sqrt(np.power(P1F,2)); resP2 = np.sqrt(np.power(P2F,2));

               

               #Calc residuals for P1 and P2 


# =============================
#       CALCULATION SECTION 28 
# =============================

# Calculation: maxresP1 = np.amax(resP1[0:501])

    maxresP1 = np.amax(resP1[0:501])

# Calculation: maxresP2 = np.amax(resP2[0:501])

    maxresP2 = np.amax(resP2[0:501])

for j in range(1,len(resP1)):

                   if j+500<len(resP1):

# Calculation: maxresP1 = np.append(maxresP1, np.amax(resP1[j:j+501]))

                       maxresP1 = np.append(maxresP1, np.amax(resP1[j:j+501]))

# Calculation: maxresP2 = np.append(maxresP2, np.amax(resP2[j:j+501]))

                       maxresP2 = np.append(maxresP2, np.amax(resP2[j:j+501]))

                   else:

# Calculation: maxresP1 = np.append(maxresP1, np.amax(resP1[j:len(resP1)]))

                       maxresP1 = np.append(maxresP1, np.amax(resP1[j:len(resP1)]))

# Calculation: maxresP2 = np.append(maxresP2, np.amax(resP2[j:len(resP2)]))

                       maxresP2 = np.append(maxresP2, np.amax(resP2[j:len(resP2)]))

                               

               #Calc moving averag (P1 and P2)


# =============================
#       CALCULATION SECTION 29 
# =============================

# Calculation: MAP1 = np.std(P1F[0:501], ddof=1)

                       MAP1 = np.std(P1F[0:501], ddof=1)

# Calculation: MAP2 = np.std(P2F[0:501], ddof=1)

                       MAP2 = np.std(P2F[0:501], ddof=1)

for j in range(1,len(P1F)):

                   if j+500<len(P1F):

# Calculation: MAP1 = np.append(MAP1, np.std(P1F[j:j+501], ddof=1))

                       MAP1 = np.append(MAP1, np.std(P1F[j:j+501], ddof=1))

# Calculation: MAP2 = np.append(MAP2, np.std(P2F[j:j+501], ddof=1))

                       MAP2 = np.append(MAP2, np.std(P2F[j:j+501], ddof=1))

                   else:

# Calculation: MAP1 = np.append(MAP1, np.std(P1F[j:len(P1F)], ddof=1))

                       MAP1 = np.append(MAP1, np.std(P1F[j:len(P1F)], ddof=1))

# Calculation: MAP2 = np.append(MAP2, np.std(P2F[j:len(P1F)], ddof=1))

                       MAP2 = np.append(MAP2, np.std(P2F[j:len(P1F)], ddof=1))

                  


# =============================
#       CALCULATION SECTION 30 
# =============================

# Calculation: init = np.amax(res1P1[0:1001])*1.75;

                       init = np.amax(res1P1[0:1001])*1.75;        

                            


# =============================
#       CALCULATION SECTION 31 
# =============================

# Calculation: OF1F = np.amin(MAP1[0:4501]) # offset 1

                       OF1F = np.amin(MAP1[0:4501]) # offset 1

# Calculation: OF1loc = np.where(MAP1==OF1F); OF1loc=OF1loc[0][0]

                       OF1loc = np.where(MAP1==OF1F); OF1loc=OF1loc[0][0]

# Calculation: TO1 = maxresP1[OF1loc]*2

                       TO1 = maxresP1[OF1loc]*2

# Calculation: P1init = P1F<TO1

                       P1init = P1F<TO1

# Calculation: Mv = (P1F > (BW-init)) & (P1F< (BW+init))

                       Mv = (P1F > (BW-init)) & (P1F< (BW+init))

# Calculation: P1FBW = P1F<BW

                       P1FBW = P1F<BW

               

               # Find start of movement

               


# =============================
#       CALCULATION SECTION 32 
# =============================

# Calculation: SoM = np.where(Mv==False); SoM = SoM[0][0]

SoM = np.where(Mv==False); SoM = SoM[0][0]

# Calculation: SoMF = P1F[SoM]

SoMF = P1F[SoM]

                  

if BW>SoMF:


# =============================
#       CALCULATION SECTION 33 
# =============================

# Calculation: PosNeg = 1

                   PosNeg = 1

else:

# Calculation: PosNeg = 0

                   PosNeg = 0

                        


# =============================
#       CALCULATION SECTION 34 
# =============================

# Calculation: if P1FBW[0]==PosNeg:

if P1FBW[0]==PosNeg:

# Calculation: avP1FBW=1

           avP1FBW=1

else:

# Calculation: avP1FBW=0

           avP1FBW=0    

           

for j in range(1,SoM):


# =============================
#       CALCULATION SECTION 35 
# =============================

# Calculation: if np.mean(P1FBW[j:SoM])==PosNeg:

       if np.mean(P1FBW[j:SoM])==PosNeg:

# Calculation: avP1FBW=np.append(avP1FBW,1)

           avP1FBW=np.append(avP1FBW,1)

       else:

# Calculation: avP1FBW=np.append(avP1FBW,0)

           avP1FBW=np.append(avP1FBW,0)

   

for j in range(SoM+1, len(P1FBW)):


# =============================
#       CALCULATION SECTION 36 
# =============================

# Calculation: if np.mean(P1FBW[SoM:j])== PosNeg:

       if np.mean(P1FBW[SoM:j])== PosNeg:

# Calculation: avP1FBW=np.append(avP1FBW,1)

           avP1FBW=np.append(avP1FBW,1)

       else:

# Calculation: avP1FBW=np.append(avP1FBW,0)

           avP1FBW=np.append(avP1FBW,0)

           

 


# =============================
#       CALCULATION SECTION 37 
# =============================

# Calculation: OF2F = np.amin(MAP2[0:4501]) # Offset 2

OF2F = np.amin(MAP2[0:4501]) # Offset 2

# Calculation: OF2loc = np.where(MAP2 == OF2F); OF2loc=OF2loc[0][0]

OF2loc = np.where(MAP2 == OF2F); OF2loc=OF2loc[0][0]

# Calculation: if isinstance(OF2loc, (np.ndarray))=='True':

if isinstance(OF2loc, (np.ndarray))=='True':

# Calculation: OF2loc=OF2loc[-1]

       OF2loc=OF2loc[-1]

# Calculation: LTOP2 = maxresP2[OF2loc]*2

LTOP2 = maxresP2[OF2loc]*2

   


# =============================
#       CALCULATION SECTION 38 
# =============================

# Calculation: SoJ = np.where(avP1FBW==1)

SoJ = np.where(avP1FBW==1)

   

if SoJ[0][0]>0:


# =============================
#       CALCULATION SECTION 39 
# =============================

# Calculation: SoJ = SoJ [0][0]-1

    SoJ = SoJ [0][0]-1

else:

# Calculation: SoJ = SoJ [0][1]-1

    SoJ = SoJ [0][1]-1




# =============================
#       CALCULATION SECTION 40 
# =============================

# Calculation: SoJF = P1F[SoJ]

SoJF = P1F[SoJ]

      


# =============================
#       CALCULATION SECTION 41 
# =============================

# Calculation: TO2 = maxresP2[OF2loc]*2

TO2 = maxresP2[OF2loc]*2

# Calculation: TOnum1 = np.where(P1init==1); TOnum1=TOnum1[0][0]

TOnum1 = np.where(P1init==1); TOnum1=TOnum1[0][0]




# =============================
#       CALCULATION SECTION 42 
# =============================

# Calculation: P2L = ((P2F[0] > TO2) & (P2F[1] > TO2)& (P2F[2] > TO2) & (P2F[3] > TO2) & (P2F[4] > TO2))

P2L = ((P2F[0] > TO2) & (P2F[1] > TO2)& (P2F[2] > TO2) & (P2F[3] > TO2) & (P2F[4] > TO2))

for j in range (1,len(P2F)):

    if j+5<len(P2F):

# Calculation: P2L = np.append(P2L, ((P2F[j] > TO2) & (P2F[j+1] > TO2)& (P2F[j+2] > TO2) & (P2F[j+3] > TO2) & (P2F[j+4] > TO2)))

        P2L = np.append(P2L, ((P2F[j] > TO2) & (P2F[j+1] > TO2)& (P2F[j+2] > TO2) & (P2F[j+3] > TO2) & (P2F[j+4] > TO2)))




# =============================
#       CALCULATION SECTION 43 
# =============================

# Calculation: Lnum1 = np.where(P2L == 1); Lnum1 = Lnum1[0][0]

Lnum1 = np.where(P2L == 1); Lnum1 = Lnum1[0][0]



# Make Fz array that has both force plates in divided at right point


# =============================
#       CALCULATION SECTION 44 
# =============================

# Calculation: Fz = P1F[0:TOnum1]

Fz = P1F[0:TOnum1]

# Calculation: Fz = np.append(Fz,0)

Fz = np.append(Fz,0)

# Calculation: Fz = np.append(Fz, P2F[(TOnum1+1):len(P2F)])

Fz = np.append(Fz, P2F[(TOnum1+1):len(P2F)])

        

# Calc vertical F, acc, vel and disp during jump


# =============================
#       CALCULATION SECTION 45 
# =============================

# Calculation: FzJ = Fz[SoJ:len(Fz)]

FzJ = Fz[SoJ:len(Fz)]

# Calculation: acc = (FzJ-BW)/(BW/9.812)

acc = (FzJ-BW)/(BW/9.812)

# Calculation: vel = 0

vel = 0

# Calculation: vel=np.append(vel,(vel+((acc[0]*(1/Fs)))))

vel=np.append(vel,(vel+((acc[0]*(1/Fs)))))

for j in range(2,(len(acc)+1)):

# Calculation: vel=np.append(vel,(vel[j-1]+((acc[j-1]*(1/Fs)))))

    vel=np.append(vel,(vel[j-1]+((acc[j-1]*(1/Fs)))))




# =============================
#       CALCULATION SECTION 46 
# =============================

# Calculation: imp=0

imp=0

for j in range(1,len(FzJ)):

# Calculation: imp=np.append(imp, (FzJ[j]-BW)*(1/Fs)+(0.5*((FzJ[j]-FzJ[j-1])*(1/Fs))))

    imp=np.append(imp, (FzJ[j]-BW)*(1/Fs)+(0.5*((FzJ[j]-FzJ[j-1])*(1/Fs))))




# =============================
#       CALCULATION SECTION 47 
# =============================

# Calculation: disp=0

disp=0

# Calculation: disp=np.append(disp, (disp+(vel[0]*(1/Fs))))

disp=np.append(disp, (disp+(vel[0]*(1/Fs))))

for j in range(2,len(vel)+1):

# Calculation: disp=np.append(disp, (disp[j-1]+(vel[j-1]*(1/Fs))))

    disp=np.append(disp, (disp[j-1]+(vel[j-1]*(1/Fs))))

    


# =============================
#       CALCULATION SECTION 48 
# =============================

# Calculation: vel=vel[0:-1]

vel=vel[0:-1]    

# Calculation: pwr = FzJ * vel

pwr = FzJ * vel

# Calculation: WD = FzJ*disp[0:-2]

WD = FzJ*disp[0:-2]

# Calculation: FBW = FzJ/BW

FBW = FzJ/BW




# =============================
#       CALCULATION SECTION 49 
# =============================

# Calculation: MAF = np.std(FzJ[0:11], ddof=1)

MAF = np.std(FzJ[0:11], ddof=1)

for j in range(1,len(FzJ)):

    if j+11<len(FzJ):

# Calculation: MAF = np.append(MAF,np.std(FzJ[j:j+11], ddof=1) )

        MAF = np.append(MAF,np.std(FzJ[j:j+11], ddof=1) )

    else:

# Calculation: MAF=np.append(MAF, np.std(FzJ[j:-1], ddof=1))

        MAF=np.append(MAF, np.std(FzJ[j:-1], ddof=1))



    

# Ensure P2L has the required length before proceeding

if len(P2L) > Lnum1:


# =============================
#       CALCULATION SECTION 50 
# =============================

# Calculation: TO2 = np.where(P2L[Lnum1:-1] == 0)

    TO2 = np.where(P2L[Lnum1:-1] == 0)

    if len(TO2[0]) > 0:

# Calculation: TO2 = TO2[0][0] + Lnum1

        TO2 = TO2[0][0] + Lnum1

    else:

        st.write("Warning: No valid match found for TO2. Setting default value.")

# Calculation: TO2 = Lnum1  # Default to Lnum1 if no match found

        TO2 = Lnum1  # Default to Lnum1 if no match found



    # Check if P2L[TO2:-1] has enough data

    if len(P2L) > TO2:


# =============================
#       CALCULATION SECTION 51 
# =============================

# Calculation: Lnum2 = np.where(P2L[TO2:-1] == 1)

        Lnum2 = np.where(P2L[TO2:-1] == 1)

        if len(Lnum2[0]) > 0:

# Calculation: Lnum2 = Lnum2[0][0] + TO2 + 1

            Lnum2 = Lnum2[0][0] + TO2 + 1

        else:

            st.write("Warning: No valid match found for Lnum2. Setting default value.")

# Calculation: Lnum2 = TO2 + 1  # Use a default or alternative value

            Lnum2 = TO2 + 1  # Use a default or alternative value

    else:

        st.write("Error: P2L does not have sufficient data after TO2.")

# Calculation: Lnum2 = TO2 + 1

        Lnum2 = TO2 + 1

else:

    st.write("Error: P2L does not have sufficient data for indexing.")

# Calculation: TO2 = Lnum1

    TO2 = Lnum1

# Calculation: Lnum2 = Lnum1 + 1

    Lnum2 = Lnum1 + 1



        

# Displacement @ key points


# =============================
#       CALCULATION SECTION 52 
# =============================

# Calculation: maxnegdisp = np.amin(disp[(Lnum1-SoJ):(TO2-SoJ)])

maxnegdisp = np.amin(disp[(Lnum1-SoJ):(TO2-SoJ)])

# Calculation: maxnegdisploc=np.where(disp==maxnegdisp) + SoJ

maxnegdisploc=np.where(disp==maxnegdisp) + SoJ

# Calculation: maxnegdisploc=maxnegdisploc[0][0]+1

maxnegdisploc=maxnegdisploc[0][0]+1

# Calculation: dispTO1 = disp[(TOnum1-SoJ)-1]

dispTO1 = disp[(TOnum1-SoJ)-1]

# Calculation: dispL1 = disp[(Lnum1-SoJ)-1]

dispL1 = disp[(Lnum1-SoJ)-1]

# Calculation: dispCT = maxnegdisp-dispL1

dispCT = maxnegdisp-dispL1

# Calculation: dispTO2 = disp[(TO2-SoJ)-1]

dispTO2 = disp[(TO2-SoJ)-1]

# Calculation: dispL2 = disp[(Lnum2-SoJ)-1]

dispL2 = disp[(Lnum2-SoJ)-1]



#Vel @ certain points


# =============================
#       CALCULATION SECTION 53 
# =============================

# Calculation: VTO1 = vel[(TOnum1-SoJ)-1]

VTO1 = vel[(TOnum1-SoJ)-1]

# Calculation: VL1 = vel [(Lnum1-SoJ)-1]

VL1 = vel [(Lnum1-SoJ)-1]

# Calculation: VTO2 = vel[(TO2-SoJ)-1]

VTO2 = vel[(TO2-SoJ)-1]            

# Times of different phases

# Calculation: CT = (TO2-Lnum1)/1000    # Contact time

CT = (TO2-Lnum1)/1000    # Contact time

# Calculation: EP = (maxnegdisploc-Lnum1)/1000  # Eccentric phase

EP = (maxnegdisploc-Lnum1)/1000  # Eccentric phase

# Calculation: CP = (TO2 - maxnegdisploc)/1000  # Concentric phase

CP = (TO2 - maxnegdisploc)/1000  # Concentric phase

# Calculation: FT = (Lnum2 - TO2)/1000   # Flight time

FT = (Lnum2 - TO2)/1000   # Flight time

# Jump height based on velocity

# Calculation: JHv = (np.power(VTO2,2)/(9.81*2))

JHv = (np.power(VTO2,2)/(9.81*2))

# Calculation: JHt = (9.812*(np.power(FT,2)))/8

JHt = (9.812*(np.power(FT,2)))/8

# Calc forces at key points

# Calculation: maxnegdispF = FzJ[(maxnegdisploc-SoJ)-1]

maxnegdispF = FzJ[(maxnegdisploc-SoJ)-1]

# Calculation: EccF = np.amax(FzJ[Lnum1-SoJ:maxnegdisploc-SoJ])

EccF = np.amax(FzJ[Lnum1-SoJ:maxnegdisploc-SoJ])

# Calculation: EccFloc = np.where(FzJ==EccF); EccFloc=EccFloc+SoJ; EccFloc = EccFloc[0][0]+1

EccFloc = np.where(FzJ==EccF); EccFloc=EccFloc+SoJ; EccFloc = EccFloc[0][0]+1

# Calculation: ConF = np.amax(FzJ[maxnegdisploc-SoJ:TO2-SoJ])

ConF = np.amax(FzJ[maxnegdisploc-SoJ:TO2-SoJ])

# Calculation: ConFloc = np.where(FzJ==ConF); ConFloc=ConFloc[0][0]+SoJ+1

ConFloc = np.where(FzJ==ConF); ConFloc=ConFloc[0][0]+SoJ+1

# Calculation: AvEccF=np.mean(FzJ[((Lnum1-SoJ)-1):((maxnegdisploc-SoJ))])

AvEccF=np.mean(FzJ[((Lnum1-SoJ)-1):((maxnegdisploc-SoJ))])

# Calculation: AvConF = np.mean(FzJ[((maxnegdisploc-SoJ)-1):TO2-SoJ])

AvConF = np.mean(FzJ[((maxnegdisploc-SoJ)-1):TO2-SoJ])

# Calculation: EccconF=np.amax(EccF)

EccconF=np.amax(EccF)

# Calculation: EccconF=np.append(EccconF, np.amax(ConF))

EccconF=np.append(EccconF, np.amax(ConF))

# Calculation: PkF = np.amax(EccconF)

PkF = np.amax(EccconF)

# Calculation: PkFloc = np.where(FzJ==PkF); PkFloc=PkFloc[0][0]+SoJ+1

PkFloc = np.where(FzJ==PkF); PkFloc=PkFloc[0][0]+SoJ+1

   


# =============================
#       CALCULATION SECTION 54 
# =============================

# Calculation: EccFBW = EccF/BW

EccFBW = EccF/BW

# Calculation: ConFBW = ConF/BW

ConFBW = ConF/BW

# Time to key points

# Calculation: EccFt = (EccFloc-Lnum1)/1000

EccFt = (EccFloc-Lnum1)/1000

# Calculation: ConFt = ((ConFloc-Lnum1)/1000)-EccFt

ConFt = ((ConFloc-Lnum1)/1000)-EccFt

# Phase ratios

# Calculation: EccConP=CP/EP

EccConP=CP/EP

# Calculation: FTCT = FT/CT

FTCT = FT/CT

# Peak Power

# Calculation: PkEccpwr = np.amin(pwr[Lnum1-SoJ:maxnegdisploc-SoJ])

PkEccpwr = np.amin(pwr[Lnum1-SoJ:maxnegdisploc-SoJ])

# Calculation: PkConpwr = np.amax(pwr[maxnegdisploc-SoJ:TO2-SoJ])

PkConpwr = np.amax(pwr[maxnegdisploc-SoJ:TO2-SoJ])

# Calculation: PkConpwrloc = np.where(pwr==PkConpwr); PkConpwrloc=PkConpwrloc[0][0]+SoJ+1

PkConpwrloc = np.where(pwr==PkConpwr); PkConpwrloc=PkConpwrloc[0][0]+SoJ+1

# Calculation: Vk = np.abs(maxnegdispF/(maxnegdisp-dispL1))   # Vertical stiffness

Vk = np.abs(maxnegdispF/(maxnegdisp-dispL1))   # Vertical stiffness

# Calculation: RSt = JHt/CT     # Reactive strength based on FT

RSt = JHt/CT     # Reactive strength based on FT

# Calculation: RSv = JHv/CT     # Reactive strength based on velocity

RSv = JHv/CT     # Reactive strength based on velocity

# Actual drop height

# Calculation: DH = disp[(TOnum1-SoJ)-1]

DH = disp[(TOnum1-SoJ)-1]

# Calculation: actDH = (DH*100)+BoxH

actDH = (DH*100)+BoxH

# Calculation: effDH = (np.power(VL1,2)/(2*9.81))   # Effective drop height

effDH = (np.power(VL1,2)/(2*9.81))   # Effective drop height

# Calculation: Pkpwrt = (PkConpwrloc-Lnum1+1)/1000;

Pkpwrt = (PkConpwrloc-Lnum1+1)/1000;

# Calculation: avRPD = PkConpwr/Pkpwrt

avRPD = PkConpwr/Pkpwrt

# Calc Impulse

      


# =============================
#       CALCULATION SECTION 55 
# =============================

# Calculation: Conimp = np.sum(imp[((maxnegdisploc-SoJ)-1):TO2-SoJ])

Conimp = np.sum(imp[((maxnegdisploc-SoJ)-1):TO2-SoJ])

# Calculation: Eccimp = np.sum(imp[((Lnum1-SoJ)-1):maxnegdisploc-SoJ])

Eccimp = np.sum(imp[((Lnum1-SoJ)-1):maxnegdisploc-SoJ])

# Calculation: Totimp = Eccimp+Conimp

Totimp = Eccimp+Conimp

# Calculation: imprat = Eccimp/Conimp

imprat = Eccimp/Conimp

# Calculation: ms50 = Lnum1+50

ms50 = Lnum1+50

# Calculation: ms100 = ms50+50

ms100 = ms50+50

# Calculation: imp50ms= np.sum(imp[((Lnum1-SoJ)-1):ms50-SoJ])

imp50ms= np.sum(imp[((Lnum1-SoJ)-1):ms50-SoJ])

# Calculation: imp100ms= np.sum(imp[((Lnum1-SoJ)-1):ms100-SoJ])

imp100ms= np.sum(imp[((Lnum1-SoJ)-1):ms100-SoJ])



#Average Eccentric power


# =============================
#       CALCULATION SECTION 56 
# =============================

# Calculation: avEccpwr = np.mean(pwr[((Lnum1-SoJ)-1):maxnegdisploc-SoJ])

avEccpwr = np.mean(pwr[((Lnum1-SoJ)-1):maxnegdisploc-SoJ])

# Calculation: avConpwr = np.mean(pwr[((maxnegdisploc-SoJ)-1):TO2-SoJ])

avConpwr = np.mean(pwr[((maxnegdisploc-SoJ)-1):TO2-SoJ])

# Calculation: relpwr = PkConpwr/BWkg

relpwr = PkConpwr/BWkg

# Calculation: pwrut = avEccpwr+avConpwr

pwrut = avEccpwr+avConpwr



#Write data

st.write("### Calculations for DJ")



st.write(f"**Mean Force (Plate 1):** {np.mean(P1F):.2f} N")

st.write(f"**Mean Force (Plate 2):** {np.mean(P2F):.2f} N")



# Prepare and export results


# =============================
#       CALCULATION SECTION 57 
# =============================

# Calculation: results = {

results = {

                    'Date': [DATE],

                    'Athlete': [athlete],

                    'Leg': [leg],

                    'Box Height': [BoxH],

                    'Body Weight (kg)': [BWkg],

                    'Contact Time (s)': [CT],

                    'Eccentric Time (s)': [EP],

                    'Concentric Time (s)': [CP],

                    'Flight Time (s)': [FT],

                    'Jump Height (m)': [JHv],

                    'Disp during Contact (m)': [dispCT],

                    'Actual Drop Height (m)': [actDH],

                    'Reactive Strength Index': [RSv],

                    'Velocity @ TO (m/s)': [VTO2],

                    'Vertical Stiffness (N/m)': [Vk],

                    'Relative Power (W/kg)': [relpwr],

                    'Peak Power (W)': [PkConpwr],

                    'Mean Eccentric Power (W)': [avEccpwr],

                    'Mean Concentric Power (W)': [avConpwr],

                    'Power Utilisation (W)': [pwrut],

                    'Peak Eccentric Force (N)': [EccF],

                    'Peak Concentric Force (N)': [ConF],

                    'Eccentric Force/BW': [EccFBW],

                    'Concentric Force/BW': [ConFBW],

                    'Total Impulse (N.s)': [Totimp],

                    'Eccentric Impulse (N.s)': [Eccimp],

                    'Concentric Impulse (N.s)': [Conimp],

                    'Impulse @ 50 ms (N.s)': [imp50ms],

                    'Eccentric:Concentric Impulse Ratio': [imprat],

                    'Max Residual P1': [np.amax(maxresP1)],

                    'Max Residual P2': [np.amax(maxresP2)],

                    'Power': [np.amax(pwr)],

                    'Work Done': [np.sum(WD)],

                }

# Calculation: df_results = pd.DataFrame(results)

df_results = pd.DataFrame(results)



                # Provide download option for results


# =============================
#       CALCULATION SECTION 58 
# =============================

# Calculation: output = BytesIO()

output = BytesIO()

# Calculation: with pd.ExcelWriter(output, engine='openpyxl') as writer:

with pd.ExcelWriter(output, engine='openpyxl') as writer:

# Calculation: df_results.to_excel(writer, index=False, sheet_name='DJ Results')

 df_results.to_excel(writer, index=False, sheet_name='DJ Results')

output.seek(0)

                

st.download_button(


# =============================
#       CALCULATION SECTION 59 
# =============================

# Calculation: label="Download DJ Results",

                    label="Download DJ Results",

# Calculation: data=output,

                    data=output,

# Calculation: file_name=f"{athlete}_DJ_Results.xlsx",

                    file_name=f"{athlete}_DJ_Results.xlsx",

# Calculation: mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                )
