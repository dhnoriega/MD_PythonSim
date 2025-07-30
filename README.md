# MD_PythonSim
Simulation and analysis of temperature and position dynamics of argon in gaseous phase.  
Completely built in and analyzed with python3.

SIMULATION:
1. Defines functions: Switching Funtion, Neighbor List, Potential Energy, Compute Forces, Apply Periodic BC, and Compute Pressure
2. Constants are defined and initial velocity matrix is computed to align with a specific temperature range
3. Generates initial position matrix, ensuring no overlapping particles. This is based on number of particles and desired density/length-scale
4. Initial forces and Neighbor list are computed, then equilibriation phase runs based on number of eq_frames stated to reach desired temperature
5. Production run for defined number of production frames at timescale dt: Energies, Temperatures, Positions, and Velocities are recorded to arrays
6. Arrays are written to disk as csv files for later analysis
7. Energy and Temperature plotted to ensure proper behavior
8. MP4 generated (Optional)

ANALYSIS:

!!!**DISCLAIMER**!!! ALL NECESSARY FILES MUST BE PLACED INTO A SPECIFIC STRUCURE IN ORDER TO FUNCTION
The file tree should be as follows: **Simulation_Group > RunX > necessary files**
    **The necessary files include: {all_vel.csv, energies.csv, positions_over_time.csv, temperatures.csv}**
The post sim analysis program will sort through the file tree of the entire group folder and turn csv files into npy files to manage RAM usage (files can get quite large)
1. Files read and converted
2. Data is preprocessed into arrays for averaging, averaged arrays are created
3. Average position and velocity over all simulations is plotted
4. Temperature Fluctuations are plotted
5. Radial Distribution function is plotted (downsampled for efficiency)
6. Mean Square displacement in plotted
7. Kinetic energy distribution is plotted
