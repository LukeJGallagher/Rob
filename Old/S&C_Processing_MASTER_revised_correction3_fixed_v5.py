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

            res1P1 = np.sqrt(np.power(P1F-BW,2))
            resP1 = np.sqrt(np.power(P1F,2)); resP2 = np.sqrt(np.power(P2F,2));
        
            #Calc residuals for P1 and P2 
            maxresP1 = np.amax(resP1[0:501])
            maxresP2 = np.amax(resP2[0:501])
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
        
            
            TO2 = np.where(P2L[Lnum1:-1]==0); TO2 = TO2[0][0]+Lnum1
            Lnum2 = np.where(P2L[TO2:-1]==1); Lnum2 = Lnum2[0][0]+TO2+1
                
            # Displacement @ key points
            maxnegdisp = np.amin(disp[(Lnum1-SoJ):(TO2-SoJ)])
            maxnegdisploc=np.where(disp==maxnegdisp) + SoJ
            maxnegdisploc=maxnegdisploc[0][0]+1
            dispTO1 = disp[(TOnum1-SoJ)-1]
            dispL1 = disp[(Lnum1-SoJ)-1]
            dispCT = maxnegdisp-dispL1
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
            DH = disp[(TOnum1-SoJ)-1]
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

            # Prepare data for export
            DJ = np.hstack((DATE, f'{leg} {BoxH}', BWkg, CT, EP, CP, FT, JHv, dispCT, actDH, RSv, VTO2, Vk, relpwr,
                           PkConpwr, avEccpwr, avConpwr, pwrut, EccF, ConF, EccFBW, ConFBW, Totimp, Eccimp,
                           Conimp, imp50ms, imprat))
        
            header2aDJ = ['Date Collected', 'Leg/BoxH', 'Body Mass (kg)', 'Contact Time(s)', 'Ecc Time(s)', 'Con Time(s)',
                          'Flight Time(s)', 'Jump Height (m)', 'Disp during contact (m)', 'Actual Drop height (m)',
                          'Reactive Strength Index', 'Vel @ TO (m/s)', 'Vertical Stiffness (N/m)', 'Relative Power (W/kg)',
                          'Peak Power (W)', 'mean Ecc Power (W)', 'Mean Con Power (W)', 'Power Utilisation (W)',
                          'Peak Ecc Force (N)', 'Peak Con Force (N)', 'Ecc FxBW', 'Con FxBW', 'Total Impulse (N.s)',
                          'Ecc Impulse (N.s)', 'Con Impulse (N.s)', 'Impulse @ 50 ms (N.s)', 'Ecc:Con impulse ratio']
        
            # Ensure DJ and header2aDJ have the same number of elements
            if len(DJ) == len(header2aDJ):
                DJ = pd.DataFrame([DJ], columns=header2aDJ)
            else:
                raise ValueError("Number of data points in DJ does not match number of columns in header2aDJ")
        
            # Export to Excel
            output_path = 'D:/Rob/Results/'
            output_file = os.path.join(output_path, f'{athlete}_DJ_Results.xlsx')
            try:
                try:
                    if not os.path.exists(output_file):
                        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                            DJ.to_excel(writer, index=False, sheet_name='DJ Results')
                    else:
                        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
                            DJ.to_excel(writer, index=False, sheet_name='DJ Results', startrow=writer.sheets['DJ Results'].max_row, header=False)
                except Exception as e:
                    print(f'Error during Excel export: {e}')
