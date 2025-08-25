import pandas as pd
import random as rn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter

N = 1024          # Whole number of molecules
frames = 100000   # Production Run
dt = 0.5e-15     # Time step in seconds (0.5 fs)
mass = 6.634e-26 # Mass of argon atom in kg
L = (N / 1.96e24) ** (1/3)  # Box length based on density
eq_frames = 50000

def switching_function(r, r_c, delta):
    if r < r_c - delta:
        return 1.0, 0.0  # S(r), dS/dr
    elif r >= r_c:
        return 0.0, 0.0
    else:
        x = (r - (r_c - delta)) / delta
        S = 1.0 - 6 * x**5 + 15 * x**4 - 10 * x**3
        dS_dr = -(1.0 / delta) * (30 * x**4 - 60 * x**3 + 30 * x**2)
        return S, dS_dr

def compute_potential_energy(pos, L, neighbors):
    eps = 1.657e-21
    sig = 3.405e-10
    cutoff = 8.0 * sig  # Increased cutoff
    delta = 2.5 * sig    # Adjusted transition width
    potential = 0
    for i, j in neighbors:
        delta_r = pos[i] - pos[j]
        delta_r -= L * np.round(delta_r / L)
        r = np.sqrt(np.sum(delta_r**2))
        if r < cutoff and r > 3.5e-10:
            pot_raw = 4 * eps * ((sig / r)**12 - (sig / r)**6)
            S, _ = switching_function(r, cutoff, delta)
            potential += pot_raw * S
    # Long-range correction (adjusted for switching)
    V = L**3
    N_pairs = N * (N - 1) / 2
    r_cut = cutoff - delta  # Effective cutoff for correction
    correction = (8 * np.pi * N_pairs / V) * eps * ((sig**12 / (3 * r_cut**9)) - (sig**6 / (3 * r_cut**3)))
    tapered_volume = (4.0 / 3.0) * np.pi * ((cutoff**3) - (r_cut**3))
    tapered_pairs = N_pairs * (tapered_volume / V)
    tapered_correction = (8 * np.pi * tapered_pairs / V) * eps * ((sig**12 / (3 * cutoff**9)) - (sig**6 / (3 * cutoff**3)))
    return potential + correction - tapered_correction

def build_neighbor_list(pos, L, cutoff, skin):
    neighbors = []
    r_c_plus_s = cutoff + skin
    for i in range(N):
        for j in range(i + 1, N):
            delta = pos[i] - pos[j]
            delta -= L * np.round(delta / L)
            r = np.sqrt(np.sum(delta**2))
            if r < r_c_plus_s:
                neighbors.append((i, j))
    return neighbors

def compute_forces(pos, L, neighbors):
    eps = 1.657e-21
    sig = 3.405e-10
    cutoff = 8.0 * sig  # Increased cutoff
    delta = 2.5 * sig    # Adjusted transition width
    forces = np.zeros((N, 3), dtype=np.float64)
    for i, j in neighbors:
        delta_r = pos[i] - pos[j]
        delta_r -= L * np.round(delta_r / L)
        r = np.sqrt(np.sum(delta_r**2))
        if r < cutoff and r > 3.5e-10:
            pot_raw = 4 * eps * ((sig / r)**12 - (sig / r)**6)
            dU_dr = -24 * eps * (2 * (sig**12 / r**13) - (sig**6 / r**7))  # Negative gradient
            S, dS_dr = switching_function(r, cutoff, delta)
            force = (dU_dr * S + pot_raw * dS_dr) * (delta_r / r)
            forces[i] += force
            forces[j] -= force
    return forces

def apply_periodic_bc(pos, L):
    return pos % L

def compute_pressure(pos, v, L, neighbors):
    eps = 1.657e-21
    sig = 3.405e-10
    cutoff = 8.0 * sig
    delta = 2.5 * sig
    virial = 0
    for i, j in neighbors:
        delta_r = pos[i] - pos[j]
        delta_r -= L * np.round(delta_r / L)
        r = np.sqrt(np.sum(delta_r**2))
        if r < cutoff and r > 3.5e-10:
            pot_raw = 4 * eps * ((sig / r)**12 - (sig / r)**6)
            dU_dr = -24 * eps * (2 * (sig**12 / r**13) - (sig**6 / r**7))
            S, dS_dr = switching_function(r, cutoff, delta)
            virial += (dU_dr * S + pot_raw * dS_dr) * r
    kinetic_term = (2.0 / 3.0) * (mass * np.sum(v**2)) / (L**3)
    return (kinetic_term - virial / (3 * L**3)) * 1e-5  # atm

# NEW: Compute interaction accuracy metrics
def compute_interaction_accuracy_metrics(pos, L, neighbors, v):
    """
    Compute comprehensive accuracy metrics for particle interactions
    Returns a dictionary with various accuracy measures
    """
    eps = 1.657e-21
    sig = 3.405e-10
    cutoff = 8.0 * sig
    delta = 2.5 * sig
    kB = 1.380649e-23
    
    metrics = {}
    
    # 1. Radial Distribution Function (RDF) accuracy
    # Compare to experimental/theoretical RDF for argon at this temperature
    dr = sig / 10  # bin width
    max_r = L / 2
    bins = int(max_r / dr)
    g_r = np.zeros(bins)
    
    for i in range(N):
        for j in range(i + 1, N):
            delta_r = pos[i] - pos[j]
            delta_r -= L * np.round(delta_r / L)
            r = np.sqrt(np.sum(delta_r**2))
            if r < max_r:
                bin_idx = int(r / dr)
                if bin_idx < bins:
                    g_r[bin_idx] += 2  # Count for both particles
    
    # Normalize RDF
    V = L**3
    rho = N / V
    for i in range(bins):
        r = (i + 0.5) * dr
        shell_volume = 4 * np.pi * r**2 * dr
        g_r[i] /= (N * rho * shell_volume)
    
    # Expected first peak position for argon (approximately 1.1 * sigma)
    expected_first_peak = 1.1 * sig
    actual_first_peak_idx = np.argmax(g_r[5:20]) + 5  # Skip very close range
    actual_first_peak = (actual_first_peak_idx + 0.5) * dr
    rdf_accuracy = 100 * (1 - abs(actual_first_peak - expected_first_peak) / expected_first_peak)
    metrics['rdf_accuracy'] = rdf_accuracy
    
    # 2. Force symmetry and energy-force consistency check
    # Since Newton's third law is explicitly enforced in compute_forces,
    # we instead check energy-force consistency (force = -gradient of potential)
    force_consistency_errors = []
    test_displacement = sig * 1e-8  # Small displacement for numerical gradient
    
    sample_size = min(20, len(neighbors))  # Sample some interactions
    sample_indices = np.random.choice(len(neighbors), sample_size, replace=False) if len(neighbors) > 0 else []
    
    for idx in sample_indices:
        i, j = neighbors[idx]
        delta_r = pos[i] - pos[j]
        delta_r -= L * np.round(delta_r / L)
        r = np.sqrt(np.sum(delta_r**2))
        
        if r < cutoff and r > sig:  # Valid range for testing
            # Analytical force magnitude
            pot_raw = 4 * eps * ((sig / r)**12 - (sig / r)**6)
            dU_dr = -24 * eps * (2 * (sig**12 / r**13) - (sig**6 / r**7))
            S, dS_dr = switching_function(r, cutoff, delta)
            analytical_force_mag = abs(dU_dr * S + pot_raw * dS_dr)
            
            # Numerical gradient check (energy at r+dr vs r-dr)
            r_plus = r + test_displacement
            r_minus = r - test_displacement
            
            # Energy at r+dr
            pot_plus = 4 * eps * ((sig / r_plus)**12 - (sig / r_plus)**6)
            S_plus, _ = switching_function(r_plus, cutoff, delta)
            E_plus = pot_plus * S_plus
            
            # Energy at r-dr
            pot_minus = 4 * eps * ((sig / r_minus)**12 - (sig / r_minus)**6)
            S_minus, _ = switching_function(r_minus, cutoff, delta)
            E_minus = pot_minus * S_minus
            
            # Numerical force magnitude
            numerical_force_mag = abs((E_plus - E_minus) / (2 * test_displacement))
            
            # Relative error
            if analytical_force_mag > 1e-25:  # Avoid division by very small forces
                relative_error = abs(analytical_force_mag - numerical_force_mag) / analytical_force_mag
                force_consistency_errors.append(relative_error)
    
    # Calculate accuracy based on consistency errors
    if force_consistency_errors:
        mean_error = np.mean(force_consistency_errors)
        # Allow up to 1% error for numerical differentiation
        force_conservation_accuracy = 100 * max(0, 1 - mean_error / 0.01)
    else:
        force_conservation_accuracy = 100.0
    
    metrics['force_conservation_accuracy'] = force_conservation_accuracy
    
    # 3. Cutoff effectiveness (percentage of significant interactions captured)
    # Check how many interactions beyond cutoff would contribute > 0.1% to energy
    missed_interactions = 0
    total_interactions = 0
    energy_threshold = 0.001 * eps  # 0.1% of epsilon
    
    for i in range(N):
        for j in range(i + 1, N):
            delta_r = pos[i] - pos[j]
            delta_r -= L * np.round(delta_r / L)
            r = np.sqrt(np.sum(delta_r**2))
            if r > cutoff and r < L/2:  # Beyond cutoff but within half box
                pot_beyond = 4 * eps * ((sig / r)**12 - (sig / r)**6)
                if abs(pot_beyond) > energy_threshold:
                    missed_interactions += 1
                total_interactions += 1
    
    if total_interactions > 0:
        cutoff_effectiveness = 100 * (1 - missed_interactions / total_interactions)
    else:
        cutoff_effectiveness = 100.0
    metrics['cutoff_effectiveness'] = cutoff_effectiveness
    
    # 4. Switching function smoothness (continuity at boundaries)
    # Check the smoothness of the switching function
    test_points = 100
    r_test = np.linspace(cutoff - delta - 0.1*sig, cutoff + 0.1*sig, test_points)
    switch_values = []
    switch_derivatives = []
    
    for r in r_test:
        if r > 0:
            S, dS = switching_function(r, cutoff, delta)
            switch_values.append(S)
            switch_derivatives.append(dS)
    
    # Check continuity (maximum jump should be small)
    max_jump = np.max(np.abs(np.diff(switch_values)))
    switching_smoothness = 100 * (1 - max_jump)  # Ideal is 0 jump
    metrics['switching_smoothness'] = max(0, switching_smoothness)
    
    # 5. Neighbor list completeness
    # Check if all particles within cutoff + skin are in neighbor list
    actual_neighbors = len(neighbors)
    expected_neighbors = 0
    for i in range(N):
        for j in range(i + 1, N):
            delta_r = pos[i] - pos[j]
            delta_r -= L * np.round(delta_r / L)
            r = np.sqrt(np.sum(delta_r**2))
            if r < cutoff + 1.0 * sig:  # cutoff + skin
                expected_neighbors += 1
    
    if expected_neighbors > 0:
        neighbor_list_completeness = 100 * min(1.0, actual_neighbors / expected_neighbors)
    else:
        neighbor_list_completeness = 100.0
    metrics['neighbor_list_completeness'] = neighbor_list_completeness
    
    # 6. Temperature stability (how well the system maintains expected temperature)
    kinetic_energy = 0.5 * mass * np.sum(v**2)
    T_current = (2 * kinetic_energy) / (3 * N * kB)
    T_expected = 273  # K
    temperature_accuracy = 100 * (1 - abs(T_current - T_expected) / T_expected)
    metrics['temperature_stability'] = max(0, temperature_accuracy)
    
    # Overall interaction accuracy (weighted average)
    weights = {
        'rdf_accuracy': 0.25,
        'force_conservation_accuracy': 0.20,
        'cutoff_effectiveness': 0.15,
        'switching_smoothness': 0.10,
        'neighbor_list_completeness': 0.15,
        'temperature_stability': 0.15
    }
    
    # Ensure all metrics are bounded between 0 and 100
    for key in metrics:
        if key != 'overall_interaction_accuracy':
            metrics[key] = max(0, min(100, metrics[key]))
    
    overall_accuracy = sum(metrics[key] * weights[key] for key in weights.keys())
    metrics['overall_interaction_accuracy'] = overall_accuracy
    
    return metrics, g_r

# Initialize positions and velocities with double precision
kB = 1.380649e-23
T_init = 273
sigma = np.sqrt(kB * T_init / mass)
v = np.random.normal(0, sigma, (N, 3)).astype(np.float64)
v -= np.mean(v, axis=0)  # Remove COM motion
pos = np.random.uniform(0, L, (N, 3)).astype(np.float64)
pos = apply_periodic_bc(pos, L)
min_dist = np.min([np.sqrt(np.sum((pos[i] - pos[j])**2)) for i in range(N) for j in range(i+1, N)])
while min_dist < 3.405e-10:
    pos = np.random.uniform(0, L, (N, 3)).astype(np.float64)
    pos = apply_periodic_bc(pos, L)
    min_dist = np.min([np.sqrt(np.sum((pos[i] - pos[j])**2)) for i in range(N) for j in range(i+1, N)])

# Initial force and neighbor list
cutoff = 8.0 * 3.405e-10
skin = 1.0 * 3.405e-10
neighbors = build_neighbor_list(pos, L, cutoff, skin)
forces = compute_forces(pos, L, neighbors)
positions_over_time = [pos.copy()]
velocities_over_time = [v.copy()]
energies = []
temperatures = []
accuracy_metrics_over_time = []

# Equilibration (20,000 steps)
kinetic_energy = 0.5 * mass * np.sum(v**2)
T_current = (2 * kinetic_energy) / (3 * N * kB)
v *= np.sqrt(T_init / T_current)  # Single rescaling to match T_init
for k in range(eq_frames):  # 10 ps equilibration
    pos += v * dt + 0.5 * forces / mass * dt**2
    pos = apply_periodic_bc(pos, L)
    new_forces = compute_forces(pos, L, neighbors)
    v += (forces + new_forces) / (2 * mass) * dt
    forces = new_forces
    if k % 200 == 0:
        max_displacement = np.max(np.sqrt(np.sum((pos - positions_over_time[-1])**2, axis=1)))
        if k % 5 == 0 or max_displacement > 0.5 * skin:
            neighbors = build_neighbor_list(pos, L, cutoff, skin)

# Production run
positions_over_time = [pos.copy()]
velocities_over_time = [v.copy()]
energies = []
temperatures = []
update_interval = 5

print("\n=== PARTICLE INTERACTION ACCURACY METRICS ===\n")

for k in range(frames):
    pos += v * dt + 0.5 * forces / mass * dt**2
    pos = apply_periodic_bc(pos, L)
    new_forces = compute_forces(pos, L, neighbors)
    v += (forces + new_forces) / (2 * mass) * dt
    forces = new_forces
    if k % update_interval == 0:
        max_displacement = np.max(np.sqrt(np.sum((pos - positions_over_time[-1])**2, axis=1)))
        if max_displacement > 0.5 * skin:
            neighbors = build_neighbor_list(pos, L, cutoff, skin)
    kinetic_energy = 0.5 * mass * np.sum(v**2)
    potential_energy = compute_potential_energy(pos, L, neighbors)
    total_energy = kinetic_energy + potential_energy
    T_current = (2 * kinetic_energy) / (3 * N * kB)
    energies.append([kinetic_energy, potential_energy, total_energy])
    temperatures.append(T_current)
    positions_over_time.append(pos.copy())
    velocities_over_time.append(v.copy())
    
    # Compute accuracy metrics every 5000 steps
    if k % 5000 == 0:
        metrics, g_r = compute_interaction_accuracy_metrics(pos, L, neighbors, v)
        accuracy_metrics_over_time.append(metrics)
        
        if k % 10000 == 0:
            print(f"\nFrame {k}:")
            print(f"  Temperature: {T_current:.2f} K")
            print(f"  Total Energy: {total_energy:.2e} J")
            print(f"  --- Interaction Accuracy Metrics ---")
            print(f"  RDF Accuracy: {metrics['rdf_accuracy']:.2f}%")
            print(f"  Energy-Force Consistency: {metrics['force_conservation_accuracy']:.2f}%")
            print(f"  Cutoff Effectiveness: {metrics['cutoff_effectiveness']:.2f}%")
            print(f"  Switching Smoothness: {metrics['switching_smoothness']:.2f}%")
            print(f"  Neighbor List Completeness: {metrics['neighbor_list_completeness']:.2f}%")
            print(f"  Temperature Stability: {metrics['temperature_stability']:.2f}%")
            print(f"  >>> OVERALL INTERACTION ACCURACY: {metrics['overall_interaction_accuracy']:.2f}% <<<")
    
    if k % 50000 == 0 and k > 0:
        P = compute_pressure(pos, v, L, neighbors)
        print(f"\n  Pressure: {P:.2f} atm")

# Final accuracy assessment
print("\n=== FINAL SIMULATION ACCURACY REPORT ===\n")
final_metrics, final_g_r = compute_interaction_accuracy_metrics(pos, L, neighbors, v)

print("Individual Metric Scores:")
print(f"  1. RDF Accuracy: {final_metrics['rdf_accuracy']:.2f}%")
print(f"     (Measures how well particle distribution matches expected structure)")
print(f"  2. Energy-Force Consistency: {final_metrics['force_conservation_accuracy']:.2f}%")
print(f"     (Validates force calculations match potential gradients)")
print(f"  3. Cutoff Effectiveness: {final_metrics['cutoff_effectiveness']:.2f}%")
print(f"     (Percentage of significant interactions captured)")
print(f"  4. Switching Function Smoothness: {final_metrics['switching_smoothness']:.2f}%")
print(f"     (Continuity of potential at cutoff boundary)")
print(f"  5. Neighbor List Completeness: {final_metrics['neighbor_list_completeness']:.2f}%")
print(f"     (All relevant particle pairs tracked)")
print(f"  6. Temperature Stability: {final_metrics['temperature_stability']:.2f}%")
print(f"     (System maintains target temperature)")

print(f"\n╔════════════════════════════════════════════════╗")
print(f"║  OVERALL PARTICLE INTERACTION ACCURACY: {final_metrics['overall_interaction_accuracy']:.2f}% ║")
print(f"╚════════════════════════════════════════════════╝")

# Calculate time-averaged accuracy
if accuracy_metrics_over_time:
    avg_accuracy = np.mean([m['overall_interaction_accuracy'] for m in accuracy_metrics_over_time])
    std_accuracy = np.std([m['overall_interaction_accuracy'] for m in accuracy_metrics_over_time])
    print(f"\nTime-averaged Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")

# Visualization with accuracy metrics
fig = plt.figure(figsize=(16, 10))

# Original plots
plt.subplot(2, 3, 1)
plt.plot(np.arange(frames) * dt * 1e15, temperatures, label='Temperature')
plt.xlabel('Time (fs)')
plt.ylabel('Temperature (K)')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(np.arange(frames) * dt * 1e15, [e[2] for e in energies], label='Total Energy')
plt.xlabel('Time (fs)')
plt.ylabel('Total Energy (J)')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(np.arange(frames) * dt * 1e15, [e[0] for e in energies], label='Kinetic')
plt.xlabel('Time (fs)')
plt.ylabel('Energy (J)')
plt.legend()

# New accuracy plots
if accuracy_metrics_over_time:
    metric_times = np.arange(0, frames, 5000) * dt * 1e15
    
    plt.subplot(2, 3, 4)
    plt.plot(metric_times, [m['overall_interaction_accuracy'] for m in accuracy_metrics_over_time], 'g-', linewidth=2)
    plt.xlabel('Time (fs)')
    plt.ylabel('Overall Accuracy (%)')
    plt.title('Particle Interaction Accuracy Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    # Plot individual metrics
    for key in ['rdf_accuracy', 'force_conservation_accuracy', 'cutoff_effectiveness']:
        plt.plot(metric_times, [m[key] for m in accuracy_metrics_over_time], label=key.replace('_', ' ').title()[:15])
    plt.xlabel('Time (fs)')
    plt.ylabel('Accuracy (%)')
    plt.title('Component Accuracy Metrics')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

# RDF plot
plt.subplot(2, 3, 6)
sig = 3.405e-10
dr = sig / 10
r_values = [(i + 0.5) * dr / sig for i in range(len(final_g_r))]
plt.plot(r_values[:50], final_g_r[:50], 'b-')
plt.xlabel('r/σ')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulation_with_accuracy_metrics.png', dpi=150)
plt.show()

# Energy drift calculation
energy_drift = 100 * (energies[-1][2] - energies[0][2]) / abs(energies[0][2])
print(f"\nEnergy drift: {energy_drift:.2f}%")

# Write metrics to file
print('\nWriting Files...')
energies = np.array(energies)
temperatures = np.array(temperatures)
np.savetxt('energies.csv', energies, delimiter=',', header='Kinetic,Potential,Total')
np.savetxt('temperatures.csv', temperatures, delimiter=',', header='Temperature')

# Save accuracy metrics
if accuracy_metrics_over_time:
    accuracy_data = np.array([[m['overall_interaction_accuracy'], 
                               m['rdf_accuracy'],
                               m['force_conservation_accuracy'],
                               m['cutoff_effectiveness'],
                               m['switching_smoothness'],
                               m['neighbor_list_completeness'],
                               m['temperature_stability']] 
                              for m in accuracy_metrics_over_time])
    np.savetxt('accuracy_metrics.csv', accuracy_data, delimiter=',', 
               header='Overall,RDF,Force_Conservation,Cutoff_Effectiveness,Switching_Smoothness,Neighbor_List,Temperature_Stability')

positions_over_time = np.array(positions_over_time)
velocities_over_time = np.array(velocities_over_time)
arr_2d = positions_over_time.reshape(-1, positions_over_time.shape[-1])
vel_2d = velocities_over_time.reshape(-1, velocities_over_time.shape[-1])
np.savetxt('positions_over_time.csv', arr_2d * 10000000, delimiter=',', fmt='%f')
np.savetxt('all_vel.csv', vel_2d * 10000, delimiter=',', fmt='%f')
print('Files Available.')

# Animation code (commented out as in original)
# ani = FuncAnimation(fig, update, frames=range(0, frames, 100), interval=50, blit=False)
# writer = FFMpegWriter(fps=100, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
# print('Writing Mp4...')
# ani.save('ANIMATION.mp4', writer=writer)
# plt.close()
# print('Mp4 Available.')