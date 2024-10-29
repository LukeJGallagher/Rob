
try:
    # Debugging checks before data preparation
    print(f"Checking critical data sizes before processing:")
    print(f"Length of maxresP1: {len(maxresP1) if 'maxresP1' in locals() else 'Not defined'}")
    print(f"Length of maxresP2: {len(maxresP2) if 'maxresP2' in locals() else 'Not defined'}")
    print(f"Length of pwr: {len(pwr) if 'pwr' in locals() else 'Not defined'}")
    print(f"Length of WD: {len(WD) if 'WD' in locals() else 'Not defined'}")

    # Check if critical arrays are empty before proceeding
    if 'maxresP1' in locals() and len(maxresP1) == 0:
        raise ValueError("maxresP1 is empty. Data extraction failed at some point.")
    if 'maxresP2' in locals() and len(maxresP2) == 0:
        raise ValueError("maxresP2 is empty. Data extraction failed at some point.")
    if 'pwr' in locals() and len(pwr) == 0:
        raise ValueError("pwr is empty. Data extraction failed at some point.")
    if 'WD' in locals() and len(WD) == 0:
        raise ValueError("WD is empty. Data extraction failed at some point.")

    # Prepare data for export
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
