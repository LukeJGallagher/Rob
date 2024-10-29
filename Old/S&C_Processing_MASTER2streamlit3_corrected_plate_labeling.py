
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
st.title("DJ Data Processing with Corrected Plate Labeling")

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
        base_name = os.path.basename(file_name)
        locs = locations_of_substring(base_name, ' ')
        athlete = base_name[:locs[0]] if locs else base_name

        # Example force data for testing (swap F1z and F2z if labels were incorrect)
        F2z = np.random.randn(5000) * 1000  # Previously labeled as F1z
        F1z = np.random.randn(5000) * 1000  # Previously labeled as F2z

        # Initialize data variables
        if 'F1z' in locals() and len(F1z) >= 5000:
            # Now F1z correctly corresponds to Plate 1, and F2z to Plate 2
            P1F = F1z[0:5000]  # Corrected to represent Plate 1
            P2F = F2z[0:5000]  # Corrected to represent Plate 2
            
            # Check if P1F contains valid data before proceeding
            if len(P1F) >= 1501:
                BW = np.mean(P1F[0:1501])
                BWkg = BW / 9.812

                # Debugging outputs to verify values
                st.write("### Debug Information")
                st.write(f"First 10 values of corrected F1z (Plate 1): {F1z[:10]}")
                st.write(f"First 10 values of corrected P1F (Plate 1): {P1F[:10]}")
                st.write(f"Body Weight (N): {BW}")
                st.write(f"Body Weight (kg): {BWkg}")

                # Perform residual calculations
                res1P1 = np.sqrt(np.power(P1F - BW, 2))
                resP1 = np.sqrt(np.power(P1F, 2))
                resP2 = np.sqrt(np.power(P2F, 2))

                # Enhanced error handling for Lnum2 calculation
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

                # Proceed to display the corrected values
                st.write("### Calculations for DJ (Corrected Plates)")
                st.write(f"**Body Weight (kg):** {BWkg:.2f}")
                st.write(f"**Mean Force (Corrected Plate 1):** {np.mean(P1F):.2f} N")
                st.write(f"**Mean Force (Corrected Plate 2):** {np.mean(P2F):.2f} N")
            else:
                st.write("Error: `P1F` does not have the minimum required data points (1501).")
        else:
            st.write("Error: `F1z` or `F2z` does not have sufficient data.")
