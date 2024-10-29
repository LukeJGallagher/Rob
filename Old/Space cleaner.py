# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 10:34:48 2024

@author: Luke.Gallagher
"""

# Open the file and clean up any potential hidden whitespace issues
with open('D:\Rob\S&C_Processing_MASTER_fully_corrected_final.py', 'r') as file:
    lines = file.readlines()

# Strip leading and trailing whitespace, and replace tabs with spaces
cleaned_lines = [line.replace('\t', '    ').rstrip() + '\n' for line in lines]

# Save the cleaned file
with open('D:\Rob\S&C_Processing_MASTER_fully_corrected_final.py', 'w') as cleaned_file:
    cleaned_file.writelines(cleaned_lines)
