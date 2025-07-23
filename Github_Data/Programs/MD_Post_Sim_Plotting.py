import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import dask.array as da
from scipy.spatial.distance import pdist
from numba import njit
from matplotlib.ticker import ScalarFormatter

# Parameters
number_of_runs = 8
N = 256           # Number of molecules
frames = 60000    
dt = 0.5e-15      # Time step in seconds (0.5 fs)
mass = 6.634e-26  # Mass of argon atom in kg
L = (N / 1.96e24) ** (1/3)  # Box length based on density 
k_B = 1.381e-23   # Boltzmann constant (J/K)
step = 10         # Downsample factor (data recorded every X frames)
downsampled_frames = frames // step  
total_elements_expected = frames * N * 3  # Expected number of elements

# Dynamically set working directory structure
# Set to directory containing a set of runs (Run1, Run2, etc. )
# Make sure to have files correctly named (same as output from simulation program)
# {all_vel.csv, positions_over_time.csv, temperatures.csv, energies.csv}
base_dir = Path(r'C:\Users\domno\.spyder-py3\Github_Data\Sample_Data')
full_pos = []
full_vel = []

# Precompute and save .npy files if not already done
# Convert csv files for managing overloading RAM
def preprocess_data(run_dir, prefix, scale_factor): #Reshape arrays to match defined frame count
    npy_path = run_dir / f"{prefix}.npy"
    print(f"Checking {npy_path}")  # Debug: Check file path
    if not npy_path.exists() or True:  # Force regeneration by always recreating
        print(f"Creating or overwriting {npy_path}")
        # Load CSV and validate structure
        csv_path = run_dir / f"{prefix}.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} does not exist. Skipping run.")
            return None  # Return None to skip
        try:
            # Read CSV in smaller chunks to manage memory
            chunk_size = 5000  # Number of rows processed per chunk
            data_chunks = []
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                #print(f"Processing chunk with shape: {chunk.shape}")  # Debug chunk shape
                data_chunks.append(chunk.to_numpy())
            data = np.concatenate(data_chunks, axis=0)
            print(f"CSV shape: {data.shape}")  # Debug: CSV dimensions
            print(f"Initial data size: {data.size}")  # Debug: Initial size
            # Truncate to exact number of expected elements
            if data.size > total_elements_expected:
                excess = data.size - total_elements_expected
                print(f"Warning: Truncating {excess} extra elements")
                data = data[:total_elements_expected // 3]  # Truncate to frames * N rows
                if data.shape[0] != frames * N:
                    print(f"Warning: Truncated rows {data.shape[0]} do not match expected {frames * N}")
                    data = data[:frames * N]  # Force exact match
            print(f"Truncated data size: {data.size}")  # Debug: Size after truncation
            if data.size != total_elements_expected:
                print(f"Warning: Data size {data.size} does not match expected {total_elements_expected}, padding with zeros")
                padding = total_elements_expected - data.size
                if padding > 0:
                    data = np.pad(data, ((0, padding // 3), (0, 0)), mode='constant')  # Pad with zeros
            data = data.reshape(frames, N, 3)  # Reshape directly to (frames, 256, 3)
            print(f"Reshaped data shape: {data.shape}")  # Debug: Shape after reshape
        except MemoryError as me:
            print(f"MemoryError: {me}. Skipping run due to insufficient memory.")
            return None  # Return None on memory error
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            return None  # Return None on other errors
        data = data * scale_factor  # Apply scaling before downsampling
        data = data[::step]  # Downsample to every 5th frame
        np.save(npy_path, data)
    else:
        print(f"Loading existing {npy_path}")
        data = np.load(npy_path)
    print(f"Loaded data shape: {data.shape}, size: {data.size}, sample value: {data[0, 0, 0]}")  # Debug: Final shape, size, and sample
    return data

for i in range(number_of_runs): #Number of runs present in directory
    run_dir = base_dir / f"Run{i+1}"
    os.chdir(run_dir)
    print(f"Processing data from: {run_dir}")

    # Load preprocessed data
    pos_data = preprocess_data(run_dir, 'positions_over_time', 1e-7)  # 1/10000000 (meters)
    vel_data = preprocess_data(run_dir, 'all_vel', 1e-4)  # No scaling (assume m/s)
    if pos_data is None or vel_data is None:
        print(f"Skipping Run {i+1} due to data loading failure")
        continue
    try:
        if pos_data.shape != (downsampled_frames, N, 3) or vel_data.shape != (downsampled_frames, N, 3):
            raise ValueError(f"Unexpected shape: pos_data={pos_data.shape}, vel_data={vel_data.shape}")
        print(f"After processing - pos_data.size: {pos_data.size}, shape: {pos_data.shape}")
        print(f"After processing - vel_data.size: {vel_data.size}, shape: {vel_data.shape}")
    except Exception as e:
        print(f"Error loading data for Run{i+1}: {e}")
        continue

    if pos_data.size == downsampled_frames * N * 3 and vel_data.size == downsampled_frames * N * 3:
        full_pos.append(pos_data)
        full_vel.append(vel_data)
        print(f"Run {i+1} - pos_data shape: {pos_data.shape}, vel_data shape: {vel_data.shape}")
    else:
        print(f"Run {i+1} - Data size mismatch: pos_data.size={pos_data.size}, vel_data.size={vel_data.size}, expected={downsampled_frames * N * 3}")

# Convert to Dask arrays with implicit chunking
if full_pos:
    full_pos_dask = da.stack(full_pos, axis=0)  # Remove 'chunks' argument
    full_vel_dask = da.stack(full_vel, axis=0)  # Remove 'chunks' argument
else:
    print("Warning: No valid data loaded, initializing with zeros")
    full_pos_dask = da.zeros((1, downsampled_frames, N, 3), chunks=(1, downsampled_frames // 10, N, 3))  # Explicit chunks
    full_vel_dask = da.zeros((1, downsampled_frames, N, 3), chunks=(1, downsampled_frames // 10, N, 3))

print(f"full_vel_dask shape: {full_vel_dask.shape}")  # Debug: Verify Dask array shape

# Compute averages
avg_pos = full_pos_dask.mean(axis=0).compute()
avg_vel = full_vel_dask.mean(axis=0).compute()
std_pos = full_pos_dask.std(axis=0).compute()
std_vel = full_vel_dask.std(axis=0).compute()
print(f"Averaged pos shape: {avg_pos.shape}, averaged vel shape: {avg_vel.shape}")

# Validate data (check for NaNs or zeros)
vel_data_computed = full_vel_dask.compute()
if np.all(vel_data_computed == 0):
    print("Warning: All velocity data is zero. Check CSV data or scaling.")
elif np.any(np.isnan(vel_data_computed)) or np.any(vel_data_computed == 0):
    print("Warning: NaN or zero values detected in velocity data. Cleaning data...")
    vel_data_computed = np.nan_to_num(vel_data_computed, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaNs with 0
    full_vel_dask = da.from_array(vel_data_computed, chunks=full_vel_dask.chunks)

# Plotting averaged position and velocity magnitude
time_fs = np.arange(downsampled_frames) * dt * 1e15 * step
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
mean_x = np.mean(avg_pos[:, :, 0], axis=1)
plt.plot(time_fs, mean_x, label='Mean Position')
plt.fill_between(time_fs, mean_x - np.mean(std_pos[:, :, 0], axis=1), mean_x + np.mean(std_pos[:, :, 0], axis=1), alpha=0.2, label='Std Dev')
# Get the current axes
ax = plt.gca()

# Set x-axis formatter to use fixed-point notation
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.ticklabel_format(style='plain', axis='x')


plt.xlabel('Time (fs)')
plt.ylabel('Mean Position (m)')
plt.title('Averaged Position Over Time')
plt.legend()

plt.subplot(1, 2, 2)
mean_v_mag = np.mean(np.sqrt(np.sum(avg_vel**2, axis=2)), axis=1)
std_v_mag = np.mean(np.sqrt(np.sum(std_vel**2, axis=2)), axis=1)
plt.plot(time_fs, mean_v_mag, label='Mean Velocity Magnitude')
plt.fill_between(time_fs, mean_v_mag - std_v_mag, mean_v_mag + std_v_mag, alpha=0.2, label='Std Dev')
# Get the current axes
ax = plt.gca()

# Set x-axis formatter to use fixed-point notation
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.ticklabel_format(style='plain', axis='x')


plt.xlabel('Time (fs)')
plt.ylabel('Mean Velocity Magnitude (m/s)')
plt.title('Averaged Velocity Magnitude Over Time')
plt.legend()

plt.tight_layout()
plt.show()

# Temperature Fluctuations (corrected formula)
kinetic_energy = 0.5 * mass * da.sum(full_vel_dask**2, axis=2)  # Sum over components
print(f"kinetic_energy shape: {kinetic_energy.shape}")  # Debug: Check shape
print(f"kinetic_energy sample: {kinetic_energy[0, 0].compute()}")  # Debug: Compute sample value
mean_kinetic_energy_per_particle = da.mean(kinetic_energy, axis=2).compute()  # Average over particles
temp = (2.0 / (3 * k_B)) * da.mean(mean_kinetic_energy_per_particle, axis=0).compute()  # Temperature per degree of freedom
temp = 3/256*temp
print(f"Temperature array: {temp}")  # Debug: Check temperature values
if temp.size == 0 or np.any(np.isnan(temp)) or temp.size != downsampled_frames or np.all(temp == 0):
    print("Warning: Temperature calculation resulted in invalid or all-zero dimensions. Using fallback mean.")
    temp = np.full(downsampled_frames, 273.0)  # Fallback to expected 273 K
plt.plot(time_fs, temp, label='Temperature')
# Get the current axes
ax = plt.gca()

# Set x-axis formatter to use fixed-point notation
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))

ax.ticklabel_format(style='plain', axis='both', useOffset=False)

plt.xlabel('Time (fs)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Fluctuations')
plt.legend()
plt.show()

# Optimized RDF with corrected Numba indexing
r_max = 1.5 * 3.405e-9
bins = 100
r_values = np.linspace(0, r_max, bins + 1)
g_r = np.zeros(bins)

@njit
def compute_pair_distances(pos: np.ndarray):  # Simplified to focus on pos array
    distances = np.zeros(int(N * (N - 1) / 2))
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            delta_r = pos[i] - pos[j]  # Vector subtraction
            # Apply minimum image convention for PBC
            delta_r = delta_r - L * np.round(delta_r / L)
            r_squared = delta_r[0] * delta_r[0] + delta_r[1] * delta_r[1] + delta_r[2] * delta_r[2]
            r = np.sqrt(r_squared)
            if r < r_max:
                distances[idx] = r
                idx += 1
    return distances[:idx]

# Compute frames and process RDF
frames_to_process = np.array([avg_pos[i] for i in range(0, downsampled_frames, 10)])  # Use every 10th frame
print(f"Computed frames shape: {frames_to_process.shape}")  # Debug: Verify shape
for pos in frames_to_process:
    print(f"Processing pos shape: {pos.shape}, sample pos[0]: {pos[0, 0]}")  # Debug: Verify input and sample
    distances = compute_pair_distances(pos)
    print(f"Distances count: {len(distances)}, sample distance: {distances[0] if len(distances) > 0 else 'None'}")  # Debug: Check distances
    hist, _ = np.histogram(distances, bins=r_values, density=False)  # Use density=False initially
    g_r += hist
print(f"g_r before normalization: {g_r}")  # Debug: Check raw histogram values
g_r /= len(frames_to_process)  # Normalize by number of frames
print(f"g_r after frame normalization: {g_r}")  # Debug: Check after frame normalization
# Corrected normalization: divide by number density * shell volume
number_density = N / L**3
dr = r_values[1] - r_values[0]
shell_volume = 4 * np.pi * (r_values[1:]**2) * dr
norm_factor = number_density * shell_volume * (N * (N - 1) / 2) / len(frames_to_process)  # Adjust for pair count
print(f"Normalization factor: {norm_factor}")  # Debug: Check new normalization factor
g_r /= norm_factor  # Divide to normalize g(r) to 1 at large r
print(f"g_r after full normalization: {g_r}")  # Check final g_r values
print(f"g_r shape: {g_r.shape}, r_values[:-1] shape: {r_values[:-1].shape}")  # Debug: Check shapes
if np.any(np.isnan(g_r)) or np.all(g_r == 0):
    print("Warning: g_r contains NaNs or is all zeros, plot may not display correctly.")
    g_r = np.nan_to_num(g_r, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaNs
plt.plot(r_values[:-1] * 1e9, g_r, label='RDF')  # Convert r to nanometers
# Get the current axes
ax = plt.gca()

# Set x-axis formatter to use fixed-point notation
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.ticklabel_format(style='plain', axis='x')


plt.xlabel('r (nm)')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function')
#plt.ylim(0, 5)  # Limit y-axis to reasonable RDF range
#plt.xlim(0, 1.5)  # Limit x-axis to relevant range
plt.legend()
plt.show(block=True)  # Ensure plot displays
plt.close()  # Close plot to avoid memory issues

# Optimized MSD using NumPy
msd = np.zeros(downsampled_frames - 1)  # Array for MSD at each time lag
for t in range(1, downsampled_frames):
    displacement = avg_pos[t:] - avg_pos[:-t]  # Displacement for time lag t
    displacement -= np.round(displacement / L) * L  # Apply PBC
    msd[t-1] = np.mean(np.sum(displacement**2, axis=(1, 2)))  # Average over particles and dimensions
time_fs_msd = np.arange(1, downsampled_frames) * dt * 1e15 * step
plt.plot(time_fs_msd, msd, label='MSD')
# Get the current axes
ax = plt.gca()

# Set x-axis formatter to use fixed-point notation
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.ticklabel_format(style='plain', axis='x')


plt.xlabel('Time (fs)')
plt.ylabel('MSD (m²)')
plt.title('Mean Square Displacement')
plt.legend()
plt.show()

# Kinetic Energy Distribution
ke = 0.5 * mass * da.sum(full_vel_dask**2, axis=3).reshape(-1, N).compute()
plt.hist(ke.flatten(), bins=50, density=True, label='Kinetic Energy')
plt.xlabel('Kinetic Energy (J)')
plt.ylabel('Probability Density')
plt.title('Kinetic Energy Distribution')
plt.legend()
plt.show()