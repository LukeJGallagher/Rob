# -*- coding: utf-8 -*-
"""
Process DJ .txt files using Streamlit for visualization and data processing.

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO

# Set the title of the app
st.title("DJ Data Processing")

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
uploaded_files = st.file_uploader("Upload your DJ .txt files", accept_multiple_files=True, type="txt")

# Process files if uploaded
if uploaded_files:
    tabs = st.tabs([f"File {i+1}" for i in range(len(uploaded_files))])  # Create tabs for each file
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Read file
        data = uploaded_file.read().decode("utf-8")

        # Extract athlete's name from file name
        file_name = uploaded_file.name
        base_name = os.path.basename(file_name)  # This extracts just the file name without directories
        locs = locations_of_substring(base_name, ' ')
        athlete = base_name[:locs[0]] if locs else base_name  # Extract the athlete's name

 # Determine Box Height (BoxH) from the file name
        if len(locs) == 4:
                BoxH = int(base_name[locs[-3]+1:locs[-2]])
        else:
                BoxH = int(base_name[locs[-2]+1:locs[-1]])

        # Determine the jump type (Only DJ here)
        jump = "DJ"
        
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

        # Display extracted information in the corresponding tab
        with tabs[idx]:
            st.header(f"Details for File {idx + 1}")
    st.write(f"**File Name:** {file_name}")
    st.write(f"**Athlete:** {athlete}")
    st.write(f"**Jump Type:** {jump}")
    st.write(f"**Leg:** {leg}")
    st.write(f"**Date of Trial:** {DATE}")

            # Visualization and calculations for DJ
            chart_tab, calc_tab = st.tabs(["Chart", "Calculations"])
            
        with chart_tab:
               
                # Extract force data for DJ
                locs = locations_of_substring(data, 'N')
                F = data[locs[-1] + 2:-1]  # Pull force data from complete txt file
                F = F.split()
                # Define time and force vectors as numpy arrays
                
                t = F[0::7]; t=np.asarray(t) 
                F1x = F[1::7]; F1x=np.asarray(F1x); F1x = F1x.astype(float)
                F1y = F[2::7]; F1y=np.asarray(F1y); F1y = F1y.astype(float)
                F1z = F[3::7]; F1z=np.asarray(F1z); F1z = F1z.astype(float)
                F2x = F[4::7]; F2x=np.asarray(F2x); F2x = F2x.astype(float)
                F2y = F[5::7]; F2y=np.asarray(F2y); F2y = F2y.astype(float)
                F2z = F[6::7]; F2z=np.asarray(F2z); F2z = F2z.astype(float)
                
                P1F = F1z[0:5000]; P2F = F2z[0:5000];
                
# Check if the data is available

# Simple Moving Average function for filtering
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Check if the data is available
if 'F1z' in locals() and 'F2z' in locals() and len(F1z) > 0 and len(F2z) > 0:
    
    
    # Apply smoothing to force data with a window size of 50 (adjust as needed)
    smoothed_F1z = moving_average(F1z, window_size=50)
    smoothed_F2z = moving_average(F2z, window_size=50)

    # Plot the smoothed force data for visual inspection
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(smoothed_F1z, label='Smoothed Plate 1 Force (F1z)')
    ax.plot(smoothed_F2z, label='Smoothed Plate 2 Force (F2z)')
    ax.set_xlabel('Data Point')
    ax.set_ylabel('Force (N)')
    ax.set_title('Smoothed Force Data from Plates 1 and 2')
    ax.legend()
    st.pyplot(fig)

    # Slider inputs for data range selection
    st.write("### Select Data Range for Body Weight Calculation")
    start_point = st.slider('Select Start Point', min_value=0, max_value=len(smoothed_F1z)-1, value=0)
    end_point = st.slider('Select End Point', min_value=0, max_value=len(smoothed_F1z)-1, value=1500)

    # Ensure the end point is after the start point
    if start_point < end_point:
        # Slice the smoothed data based on the selected range
        P1F = smoothed_F1z[start_point:end_point]
        P2F = smoothed_F2z[start_point:end_point]

        # Calculate body weight from the selected range
    BW1 = np.mean(P1F) if len(P1F) > 0 else 0
        BW2 = np.mean(P2F) if len(P2F) > 0 else 0

        # Validate body weight values
        if BW1 > 200 or BW2 > 200:  # Use 'or' to allow fallback to a single plate
            BW = (BW1 + BW2) / 2 if BW1 > 200 and BW2 > 200 else max(BW1, BW2)
            BWkg = BW / 9.812
    st.write(f"**Body Weight (N):** {BW:.2f}")
    st.write(f"**Body Weight (kg):** {BWkg:.2f}")
        else:
    st.write("Error: Calculated body weight is below expected range (> 500 N). Please adjust the selection or check the data.")
    else:
    st.write("Error: The end point must be greater than the start point.")
else:
    st.write("Error: `F1z` or `F2z` does not have sufficient data.")





calc_tab = st.container()  # Define calc_tab to avoid NameError
with calc_tab:
              
            # Ensure F1z and F2z have the expected data
            # Extract and perform calculations for DJ
            # Validate if F1z and F2z have enough data points
            if 'F1z' in locals() and 'F2z' in locals() and len(F1z) >= 5000 and len(F2z) >= 5000:
                P1F = F1z[0:5000]
                P2F = F2z[0:5000]
                
                # Check if both P1F and P2F have the minimum required data points
                if len(P1F) >= 1501 and len(P2F) >= 1501:
                    # Calculate body weight from both plates
    BW1 = np.mean(P1F[0:1501])  # Body weight from Plate 1
                    BW2 = np.mean(P2F[0:1501])  # Body weight from Plate 2
                    
                    # Check if both BW1 and BW2 are within a reasonable range
                    if BW1 > 200 and BW2 > 200:
                        BW = (BW1 + BW2) / 2  # Use the average of both plates if valid
                        
                    # Perform the residual calculations using the validated BW
                    res1P1 = np.sqrt(np.power(P1F - BW, 2))
                    resP1 = np.sqrt(np.power(P1F, 2))
                    resP2 = np.sqrt(np.power(P2F, 2))
                    # Calculate maximum residuals
                    maxresP1 = np.amax(resP1[0:501])
                    maxresP2 = np.amax(resP2[0:501])
                    
                    # Convert to kg
                    BWkg = BW / 9.812
                    
           

               
               
                               # Debugging outputs to verify values
    st.write("### Debug Information")
    st.write(f"First 10 values of F1z: {F1z[:10]}")
    st.write(f"First 10 values of P1F: {P1F[:10]}")
    st.write(f"Body Weight (N): {BW}")
    st.write(f"Body Weight (kg): {BWkg}")
         
                 # Proceed to display the calculated value
    st.write("### Calculations for DJ")
    st.write(f"**Body Weight (kg):** {BWkg:.2f}")
    st.write(f"**Mean Force (Plate 1):** {np.mean(P1F):.2f} N")
    st.write(f"**Mean Force (Plate 2):** {np.mean(P2F):.2f} N")
if len(P1F) < 1501:
    st.write('Error: P1F does not have the minimum required data points (1501).')
    P1F = None  # Set P1F to None if it doesn't meet the requirement
else:
    # Proceed with calculations only if P1F is valid
    BW1 = np.mean(P1F[0:1501])
    st.write("Error: `P1F` does not have the minimum required data points (1501).")
                      
  
                
    
                

    
    
if resP1 is not None:
    for j in range(1,len(resP1)):
                   if j+500<len(resP1):
                       maxresP1 = np.append(maxresP1, np.amax(resP1[j:j+501]))
                       maxresP2 = np.append(maxresP2, np.amax(resP2[j:j+501]))
                   else:
                       maxresP1 = np.append(maxresP1, np.amax(resP1[j:len(resP1)]))
                       maxresP2 = np.append(maxresP2, np.amax(resP2[j:len(resP2)]))
                               
               #Calc moving averag (P1 and P2)
                       MAP1 = np.std(P1F[0:501], ddof=1)
                       MAP2 = np.std(P2F[0:501], ddof=1)
for j in range(1,len(P1F)):
                   if j+500<len(P1F):
                       MAP1 = np.append(MAP1, np.std(P1F[j:j+501], ddof=1))
                       MAP2 = np.append(MAP2, np.std(P2F[j:j+501], ddof=1))
                   else:
                       MAP1 = np.append(MAP1, np.std(P1F[j:len(P1F)], ddof=1))
                       MAP2 = np.append(MAP2, np.std(P2F[j:len(P1F)], ddof=1))
                  
                       init = np.amax(res1P1[0:1001])*1.75;        
                            
                       OF1F = np.amin(MAP1[0:4501]) # offset 1
                       OF1loc = np.where(MAP1==OF1F); OF1loc=OF1loc[0][0]
                       TO1 = maxresP1[OF1loc]*2
                       P1init = P1F<TO1
                       Mv = (P1F > (BW-init)) & (P1F< (BW+init))
                       P1FBW = P1F<BW
               
               # Find start of movement
               
SoM = np.where(Mv==False); SoM = SoM[0][0]
SoMF = P1F[SoM]
                  
if BW>SoMF:
                   PosNeg = 1
else:
                   PosNeg = 0
                        
if P1FBW[0]==PosNeg:
           avP1FBW=1
else:
           avP1FBW=0    
           
for j in range(1,SoM):
       if np.mean(P1FBW[j:SoM])==PosNeg:
           avP1FBW=np.append(avP1FBW,1)
       else:
           avP1FBW=np.append(avP1FBW,0)
   
for j in range(SoM+1, len(P1FBW)):
       if np.mean(P1FBW[SoM:j])== PosNeg:
           avP1FBW=np.append(avP1FBW,1)
       else:
           avP1FBW=np.append(avP1FBW,0)
           
 
OF2F = np.amin(MAP2[0:4501]) # Offset 2
OF2loc = np.where(MAP2 == OF2F); OF2loc=OF2loc[0][0]
if isinstance(OF2loc, (np.ndarray))=='True':
       OF2loc=OF2loc[-1]
LTOP2 = maxresP2[OF2loc]*2
   
SoJ = np.where(avP1FBW==1)
   
if SoJ[0][0]>0:
    SoJ = SoJ [0][0]-1
else:
    SoJ = SoJ [0][1]-1

SoJF = P1F[SoJ]
      
TO2 = maxresP2[OF2loc]*2
TOnum1 = np.where(P1init==1); TOnum1=TOnum1[0][0]

P2L = ((P2F[0] > TO2) & (P2F[1] > TO2)& (P2F[2] > TO2) & (P2F[3] > TO2) & (P2F[4] > TO2))
for j in range (1,len(P2F)):
    if j+5<len(P2F):
        P2L = np.append(P2L, ((P2F[j] > TO2) & (P2F[j+1] > TO2)& (P2F[j+2] > TO2) & (P2F[j+3] > TO2) & (P2F[j+4] > TO2)))

Lnum1 = np.where(P2L == 1); Lnum1 = Lnum1[0][0]

# Make Fz array that has both force plates in divided at right point
Fz = P1F[0:TOnum1]
Fz = np.append(Fz,0)
Fz = np.append(Fz, P2F[(TOnum1+1):len(P2F)])
        
# Calc vertical F, acc, vel and disp during jump
FzJ = Fz[SoJ:len(Fz)]
acc = (FzJ-BW)/(BW/9.812)
vel = 0
vel=np.append(vel,(vel+((acc[0]*(1/Fs)))))
for j in range(2,(len(acc)+1)):
    vel=np.append(vel,(vel[j-1]+((acc[j-1]*(1/Fs)))))

imp=0
for j in range(1,len(FzJ)):
    imp=np.append(imp, (FzJ[j]-BW)*(1/Fs)+(0.5*((FzJ[j]-FzJ[j-1])*(1/Fs))))

disp=0
disp=np.append(disp, (disp+(vel[0]*(1/Fs))))
for j in range(2,len(vel)+1):
    disp=np.append(disp, (disp[j-1]+(vel[j-1]*(1/Fs))))
    
vel=vel[0:-1]    
pwr = FzJ * vel
WD = FzJ*disp[0:-2]
FBW = FzJ/BW

MAF = np.std(FzJ[0:11], ddof=1)
for j in range(1,len(FzJ)):
    if j+11<len(FzJ):
        MAF = np.append(MAF,np.std(FzJ[j:j+11], ddof=1) )
    else:
        MAF=np.append(MAF, np.std(FzJ[j:-1], ddof=1))

    
# Ensure P2L has the required length before proceeding
if len(P2L) > Lnum1:
    TO2 = np.where(P2L[Lnum1:-1] == 0)
    if len(TO2[0]) > 0:
        TO2 = TO2[0][0] + Lnum1
    else:
    st.write("Warning: No valid match found for TO2. Setting default value.")
        TO2 = Lnum1  # Default to Lnum1 if no match found

    # Check if P2L[TO2:-1] has enough data
    if len(P2L) > TO2:
        Lnum2 = np.where(P2L[TO2:-1] == 1)
        if len(Lnum2[0]) > 0:
            Lnum2 = Lnum2[0][0] + TO2 + 1
        else:
    st.write("Warning: No valid match found for Lnum2. Setting default value.")
            Lnum2 = TO2 + 1  # Use a default or alternative value
    else:
    st.write("Error: P2L does not have sufficient data after TO2.")
        Lnum2 = TO2 + 1
else:
    st.write("Error: P2L does not have sufficient data for indexing.")
    TO2 = Lnum1
    Lnum2 = Lnum1 + 1

        
# Displacement @ key points
maxnegdisp = np.amin(disp[(Lnum1-SoJ):(TO2-SoJ)])
maxnegdisploc=np.where(disp==maxnegdisp) + SoJ
maxnegdisploc=maxnegdisploc[0][0]+1
# Check if the index is within the bounds of the 'disp' array
if 0 <= (TOnum1 - SoJ) - 1 < len(disp):
    dispTO1 = disp[(TOnum1 - SoJ) - 1]
else:
    st.write('Error: Calculated index for dispTO1 is out of bounds.')
# Check if the index is within the bounds of the 'disp' array
if 0 <= (Lnum1 - SoJ) - 1 < len(disp):
    dispL1 = disp[(Lnum1 - SoJ) - 1]
else:
    st.write('Error: Calculated index for dispL1 is out of bounds.')
# Ensure dispL1 is defined before using it
if dispL1 is not None:
    dispCT = maxnegdisp - dispL1
else:
    st.write('Error: dispL1 is not defined, cannot calculate dispCT.')
    dispCT = None
dispTO2 = disp[(TO2-SoJ)-1]
dispL2 = disp[(Lnum2-SoJ)-1]

#Vel @ certain points
VTO1 = vel[(TOnum1-SoJ)-1]
VL1 = vel [(Lnum1-SoJ)-1]
VTO2 = vel[(TO2-SoJ)-1]            
# Times of different phases
CT = (TO2-Lnum1)/1000    # Contact time
EP = (maxnegdisploc-Lnum1)/1000  # Eccentric phase
CP = (TO2 - maxnegdisploc)/1000  # Concentric phase
FT = (Lnum2 - TO2)/1000   # Flight time
# Jump height based on velocity
JHv = (np.power(VTO2,2)/(9.81*2))
JHt = (9.812*(np.power(FT,2)))/8
# Calc forces at key points
maxnegdispF = FzJ[(maxnegdisploc-SoJ)-1]
EccF = np.amax(FzJ[Lnum1-SoJ:maxnegdisploc-SoJ])
EccFloc = np.where(FzJ==EccF); EccFloc=EccFloc+SoJ; EccFloc = EccFloc[0][0]+1
ConF = np.amax(FzJ[maxnegdisploc-SoJ:TO2-SoJ])
ConFloc = np.where(FzJ==ConF); ConFloc=ConFloc[0][0]+SoJ+1
AvEccF=np.mean(FzJ[((Lnum1-SoJ)-1):((maxnegdisploc-SoJ))])
AvConF = np.mean(FzJ[((maxnegdisploc-SoJ)-1):TO2-SoJ])
EccconF=np.amax(EccF)
EccconF=np.append(EccconF, np.amax(ConF))
PkF = np.amax(EccconF)
PkFloc = np.where(FzJ==PkF); PkFloc=PkFloc[0][0]+SoJ+1
   
EccFBW = EccF/BW
ConFBW = ConF/BW
# Time to key points
EccFt = (EccFloc-Lnum1)/1000
ConFt = ((ConFloc-Lnum1)/1000)-EccFt
# Phase ratios
EccConP=CP/EP
FTCT = FT/CT
# Peak Power
PkEccpwr = np.amin(pwr[Lnum1-SoJ:maxnegdisploc-SoJ])
PkConpwr = np.amax(pwr[maxnegdisploc-SoJ:TO2-SoJ])
PkConpwrloc = np.where(pwr==PkConpwr); PkConpwrloc=PkConpwrloc[0][0]+SoJ+1
Vk = np.abs(maxnegdispF/(maxnegdisp-dispL1))   # Vertical stiffness
RSt = JHt/CT     # Reactive strength based on FT
RSv = JHv/CT     # Reactive strength based on velocity
# Actual drop height
# Check if the index is within the bounds of the 'disp' array
if 0 <= (TOnum1 - SoJ) - 1 < len(disp):
    dispTO1 = disp[(TOnum1 - SoJ) - 1]
else:
    st.write('Error: Calculated index for dispTO1 is out of bounds.')
actDH = (DH*100)+BoxH
effDH = (np.power(VL1,2)/(2*9.81))   # Effective drop height
Pkpwrt = (PkConpwrloc-Lnum1+1)/1000;
avRPD = PkConpwr/Pkpwrt
# Calc Impulse
      
Conimp = np.sum(imp[((maxnegdisploc-SoJ)-1):TO2-SoJ])
Eccimp = np.sum(imp[((Lnum1-SoJ)-1):maxnegdisploc-SoJ])
Totimp = Eccimp+Conimp
imprat = Eccimp/Conimp
ms50 = Lnum1+50
ms100 = ms50+50
imp50ms= np.sum(imp[((Lnum1-SoJ)-1):ms50-SoJ])
imp100ms= np.sum(imp[((Lnum1-SoJ)-1):ms100-SoJ])

#Average Eccentric power
avEccpwr = np.mean(pwr[((Lnum1-SoJ)-1):maxnegdisploc-SoJ])
avConpwr = np.mean(pwr[((maxnegdisploc-SoJ)-1):TO2-SoJ])
relpwr = PkConpwr/BWkg
pwrut = avEccpwr+avConpwr

#Write data
    st.write("### Calculations for DJ")

    st.write(f"**Mean Force (Plate 1):** {np.mean(P1F):.2f} N")
    st.write(f"**Mean Force (Plate 2):** {np.mean(P2F):.2f} N")

# Prepare and export results
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
df_results = pd.DataFrame(results)

                # Provide download option for results
output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
 df_results.to_excel(writer, index=False, sheet_name='DJ Results')
output.seek(0)
                
st.download_button(
                    label="Download DJ Results",
                    data=output,
                    file_name=f"{athlete}_DJ_Results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
