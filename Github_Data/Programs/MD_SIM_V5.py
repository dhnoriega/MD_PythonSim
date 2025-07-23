# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 21:55:06 2025

@author: domno
"""

import pandas as pd
import random as rn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter

N = 256          # Whole number of molecules
frames = 60000   # Production Run
dt = 0.5e-15     # Time step in seconds (0.5 fs)
mass = 6.634e-26 # Mass of argon atom in kg
L = (N / 1.96e24) ** (1/3)  # Box length based on density
eq_frames = 20000

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

    if k % 10000 == 0:
        print(f"Frame {k}, T = {T_current:.2f} K, E_total = {total_energy:.2e} J, KE = {kinetic_energy:.2e} J, PE = {potential_energy:.2e} J")
    if k % 50000 == 0:
        P = compute_pressure(pos, v, L, neighbors)
        print(f"Frame {k}, Pressure: {P:.2f} atm")

# Visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
frame_text = ax.text2D(0.25, 0.25, "", transform=ax.get_figure().transFigure, fontsize=32)
def update(frame):
    ax.clear()
    pos = positions_over_time[frame]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], marker='o')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    frame_text.set_text(f"Frame: {frame}")
    t = frame * dt
    t = "{:.12f}".format(t)
    ax.set_title('Frame: ' + str(frame) + '\n' + ' {' + str(t) + 's} ')
    return ()

print('Writing Files...')
energies = np.array(energies)
temperatures = np.array(temperatures)
np.savetxt('energies.csv', energies, delimiter=',', header='Kinetic,Potential,Total')
np.savetxt('temperatures.csv', temperatures, delimiter=',', header='Temperature')

positions_over_time = np.array(positions_over_time)
velocities_over_time = np.array(velocities_over_time)
arr_2d = positions_over_time.reshape(-1, positions_over_time.shape[-1])
vel_2d = velocities_over_time.reshape(-1, velocities_over_time.shape[-1])
np.savetxt('positions_over_time.csv', arr_2d * 10000000, delimiter=',', fmt='%f')
np.savetxt('all_vel.csv', vel_2d * 10000, delimiter=',', fmt='%f')
print('Files Available.')

plt.figure(figsize=(16, 7))
plt.subplot(1, 3, 1)
plt.plot(np.arange(frames) * dt * 1e15, temperatures, label='Temperature')
plt.xlabel('Time (fs)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(np.arange(frames) * dt * 1e15, [e[2] for e in energies], label='Total Energy')
plt.xlabel('Time (fs)')
plt.ylabel('Total Energy (J)')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(np.arange(frames) * dt * 1e15, [e[0] for e in energies], label='Kinetic')
#plt.plot(np.arange(frames) * dt * 1e15, [e[1] for e in energies], label='Potential')
plt.xlabel('Time (fs)')
plt.ylabel('Energy (J)')
plt.legend()
plt.tight_layout()
plt.show()

energy_drift = 100 * (energies[-1][2] - energies[0][2]) / abs(energies[0][2])
print(f"Energy drift: {energy_drift:.2f}%")

# ani = FuncAnimation(fig, update, frames=range(0, frames, 100), interval=50, blit=False)
# writer = FFMpegWriter(fps=100, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
# print('Writing Mp4...')
# ani.save('ANIMATION.mp4', writer=writer)
# plt.close()
# print('Mp4 Available.')
