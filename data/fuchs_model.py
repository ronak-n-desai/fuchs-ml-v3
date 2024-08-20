# Imports
import numpy as np
import scipy.special as special
#import csv
#import h5py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
from scipy.special import ellipe
from scipy.integrate import quad as integral
from numpy.random import RandomState

spectrum = False
bins_per_MeV = 5 
max_energy_MeV = 5 # MeV
max_energy_J = max_energy_MeV * 1.60218e-13 # Joules
seed = 2
pct_noise = 0

def calc_laser_energy(I0, w0, tFWHM):
    return (np.pi/2)*I0*w0**2 * tFWHM # For a Sine-Squared Pulse
    #return (np.pi/2)*I0*w0**2 * tFWHM * np.sqrt(np.pi / (4*np.log(2))) # For a Gaussian Shaped Pulse

# integrate dN/dE from eMin to eMax from Eq. (2) in Fuchs Paper
def calc_N_between(ne, cs, tacc, S, Tp, eMin=0, eMax=1):
    xmin = np.sqrt(2*eMin/Tp)
    xmax = np.sqrt(2*eMax/Tp)
    return ne*cs*tacc*S*(np.exp(-xmin) - np.exp(-xmax))

# integrate dN/dE * E from eMin to eMax from Eq. (2) in Fuchs Paper
def calc_E_between(ne, cs, tacc, S, Tp, eMin=0, eMax=1):
    xmin = np.sqrt(2*eMin/Tp)
    xmax = np.sqrt(2*eMax/Tp)
    return ne*cs*tacc*S*Tp/2*(np.exp(-xmin)*(2 + xmin*(2 + xmin)) - np.exp(-xmax)*(2 + xmax*(2 + xmax)))

# Calculate Max proton energy in terms of normalized acceleration time and hot temperature Tp from Eq. (1) in Fuchs Paper
def calc_max_E(omega_pi, tacc, Tp):
    tp = omega_pi*tacc / np.sqrt(2*np.exp(1))
    return 2*Tp*np.log(tp + np.sqrt(tp**2 + 1))**2

def calc_dNdE(E, ne, cs, tacc, S, Tp):
    x = np.sqrt(2*E/Tp)
    x = np.where(x==0, 1e-10, x)
    return (ne*cs*tacc*S/Tp) * np.exp(-x)/x

def gaussian_noise(energy_list, pct_noise):
    prng = RandomState(seed)
    α = pct_noise/100 # Fraction instead of percent
    noisy_energy_list = prng.normal(energy_list, energy_list*α)
    for i in range(len(noisy_energy_list)):
        while(noisy_energy_list[i] < 0):
            print('found (-) energy value at i = {} during dataset generation, resampling now ...'.format(i))
            noisy_energy_list[i] = prng.normal(energy_list[i], energy_list[i]*α)
    return noisy_energy_list

def log_gaussian_noise(energy_list, pct_noise, random_seed = False):
    if random_seed:
        prng = RandomState()
    else: 
        prng = RandomState(seed)
    α = pct_noise/100 # Fraction instead of percent
    noisy_energy_list = np.zeros(len(energy_list))
    energy_list = np.clip(energy_list, 1e-4, None)
    for i in range(len(noisy_energy_list)):
        mu = np.log(energy_list[i]/np.sqrt(1+α**2))
        σ = np.sqrt(np.log(1+α**2))
        noisy_energy_list[i] = prng.lognormal(mu, σ)
    return noisy_energy_list
    
def fuchs_model(I0, z, d, w0 = 1.5e-6, lmda = 0.8e-6, tFWHM = 40.0e-15, c1 = 25, c2 = 0.74, theta = 25, pct_noise = 0, const_f = False, max_array=[[1]], crit_array=[[1]], spectrum=False):
    c = 2.998e8
    m= 9.109e-31
    e=1.602e-19
    mi=1.673e-27
    Zi=1
    eps0=8.854e-12
    laser_energy = calc_laser_energy(I0, w0, tFWHM)
    
    omega = 2*np.pi*c/lmda
    zR = np.pi * w0**2 / lmda
    theta_rad = theta*np.pi/180
    Iz = I0 / (1 + (z/zR)**2)
    w = w0*np.sqrt(1 + (z/zR)**2)
    E0 = np.sqrt(2*Iz / (c*eps0))
    a0 = (e*E0)/(m*omega*c)
    gamma = np.sqrt(1+a0**2)

    # --- NEEDS JUSTIFICATION --- #
    Tp = 0.469 * a0**(2/3) * m * c**2
    # --------------------------- #

    # Different Types of Hot Electron Temperature Scalings:
    # T_Wilks = m*c**2*(γ-1) = 0.511 (sqrt(1+(Iλ)^2/1.37E18) - 1)
    # T_Beg = 215(I_18 * λ_μm^2)^(1/3) = 0.468 a_0^(2/3) mc^2 (Farhat Beg) induce focal depth dip
    #Tp = m * c**2 * (2 * ellipe(-a0**2) / np.pi - 1) #-> proposed temperature using elliptic integral instead
    #Tp = m*c**2 *  (gamma - 1)
    if const_f:
        f = 0.5 # Constant Hot Electron Fraction
    else:
        f = np.minimum(1.2e-15 * (Iz*1e-4)**c2, 0.5) # Hot Electron Fraction from Fuchs
    Ne = f*laser_energy / Tp  # Number of Hot Electrons
    Ne = np.nan_to_num(Ne, nan=0, posinf=0, neginf=0) # Replace NaNs with 0s
    r0 = w * np.sqrt(2*np.log(2))/2
    S = np.pi*(r0 + d*np.tan(theta_rad))**2   # Area of Sheath
    # ne = Ne / (S*c*tFWHM) # From Fuchs et. al.
    ne = Ne / (S*d)     # Modification to account for thickness of sheath instead of pulse length
    omega_pi = np.sqrt(Zi * e**2 *ne / (mi*eps0)) # Plasma Frequency
    tacc = c1 * tFWHM # Acceleration Time
    cs = np.sqrt(Zi*Tp/mi) # Sound Speed

    max_proton_energy = calc_max_E(omega_pi, tacc, Tp)  # Maximum Proton Energy from Fuchs
    max_proton_energy_MeV = max_proton_energy / (1.602e-13) # Convert to MeV
    noisy_max_proton_energy_MeV = log_gaussian_noise(max_proton_energy_MeV, pct_noise)

    if spectrum:
        # Calculate the Spectrum
        energy_array = np.linspace(0, max_energy_J, max_energy_MeV*bins_per_MeV+1)
        energy_array_MeV = energy_array / (1.602e-13)
        dNdE_array = np.array([(calc_dNdE(energy_array[i], ne, cs, tacc, S, Tp) + calc_dNdE(energy_array[i+1], ne, cs, tacc, S, Tp))/2 for i in range(len(energy_array)-1)])
        N_array = np.array([calc_N_between(ne, cs, tacc, S, Tp, eMin=energy_array[i], eMax=energy_array[i+1]) for i in range(len(energy_array)-1)])
        noisy_N_array = np.array([log_gaussian_noise(N_array[i], pct_noise, random_seed=True) for i in range(len(dNdE_array))])
        noisy_dNdE_array = np.array([log_gaussian_noise(dNdE_array[i], pct_noise, random_seed=True) for i in range(len(dNdE_array))])
        return energy_array_MeV, dNdE_array, noisy_dNdE_array, noisy_max_proton_energy_MeV, N_array, noisy_N_array

    else:
        # Calculate the Number of Protons and Total Proton Energy
        num_protons = calc_N_between(ne, cs, tacc, S, Tp, eMin=0, eMax=max_proton_energy)
        total_proton_energy = calc_E_between(ne, cs, tacc, S, Tp, eMin=0, eMax=max_proton_energy)
        average_proton_energy = total_proton_energy/num_protons

        # Convert Energies to MeV
        
        total_proton_energy_MeV = total_proton_energy / (1.602e-13)
        average_proton_energy_MeV = average_proton_energy / (1.602e-13)
    
        # Laser to Proton Energy Conversion Ratio
        laser_conversion_efficiency = total_proton_energy / laser_energy
        
        # Add Gaussian Noise to the Proton Energies
        noisy_total_proton_energy_MeV = log_gaussian_noise(total_proton_energy_MeV, pct_noise)
        noisy_average_proton_energy_MeV = log_gaussian_noise(average_proton_energy_MeV, pct_noise)
    
        return_array = np.column_stack((noisy_max_proton_energy_MeV, noisy_total_proton_energy_MeV, noisy_average_proton_energy_MeV,
                laser_conversion_efficiency, laser_energy,
                max_proton_energy_MeV, total_proton_energy_MeV, average_proton_energy_MeV, max_array, crit_array))
        #print(return_array)
        for i, row in enumerate(return_array):
            if row[-2] < row[-1]:
                print("Unphysical value found, replacing...")
                return_array[i] = np.zeros(10)
        print("Shape", np.shape(return_array))
        return (return_array[:, 0], return_array[:, 1], return_array[:, 2], return_array[:, 3], return_array[:, 4], return_array[:, 5], return_array[:, 6], return_array[:, 7])
    
def fuchs_function_with_prepulse(I_main = 1e23, z=0, d0 = 10e-6, w0= 1.5e-6, lmbda = 0.8e-6, tFWHM = 40.0e-15, c1=25, c2=0.74, theta=25, pct_noise=pct_noise, const_f = False, contrast=1e-7, n0 = 1e29, t0 = 0.05e-9, spectrum=False):
    """Wrapper for fuchs_function that accounts for pre-pulse effects.
    
    Keyword arguments:
    I_main -- the main pulse intensity (default 1e23 W/m^2)
    z -- the focal depth offset (default 0)
    d0 -- the initial target thickness (default 10 microns)
    w0 -- laser spot size (default 1.5 microns)
    lmbda -- laser wavelength (default 800 nm)
    tFWHM -- laser pulse duration (default 40 fs)
    c1 -- a constant (default 25)
    c2 -- another constant (default 0.74)
    theta -- angular separation of laser beam (default 25 degrees)
    pct_noise -- percentage gaussian noise to be added to sample (default 0)
    const_f -- boolean to control whether conversion efficiency is constant (default False)
    contrast -- the contrast factor of main intensity / pre intensity (default 1e-7)
    n0 -- the initial target density (default 1e29 m^-3)
    t0 -- the time between the pre- and main-pulses' arrivals (default 1 ns)
    """
    # Definining constants
    T_pre0 = 50 # (eV)
    I_pre0 = 1e16 # (W/m^2)
    n_crit = 1.74e27 # m^-3, based on 800nm laser wavelength
    Z = 1 # Effective Ion charge
    mi = 1 # Ion mass, both assuming the relevant ions are protons
    mp = 1.673e-27 # Proton Mass
    c = 2.998e8 # Speed of Light
    lmda = 0.8e-6 # Wavelength
    w0 = 1.5e-6 # Spot Size
    e = 1.602e-19 # Elementary Charge
    eps0 = 8.854e-12  # Permittivity of Free Space
    me = 9.109e-31 # Electron Mass
    

    # Calculated Quantities
    zR = np.pi * w0**2 / lmda
    omega = 2*np.pi*c/lmda
    theta_rad = theta*np.pi/180

    corr = 1
    n_crit *= corr
    T_scaling = 1
    a0_min = 0.0#0.7   # minimum a0 for depletion

    # Step 0: Find effective intensity on target
    Iz = I_main / (1 + (z/zR)**2) 
    # Step 1: find electron temperature
    T_e = T_pre0 * np.power(Iz * contrast / I_pre0, T_scaling) # eV
    # Step 2: find sound speed from electron temperature
    C_s = np.sqrt(Z * T_e * e / (mi * mp))  #m/s
    # Step 3: find new maximum density; if below critical density, return a spectrum of 0 energy
    n_max = n0 * d0 / (d0 + 2*C_s*t0)
    omega_pi = np.sqrt(Z * e**2 *n_max / (mp*eps0))
    #print("Omega_pi t: \n", omega_pi*t0)
    if (n_max < n_crit).any(): print('Found out of bounds value') 
    # Step 4 : solve for x using exponential decay, use it to find effective density
    x0 =  C_s * t0  * (np.log(n_max) - np.log(n_crit))
    #d_eff_2 = d0 + 2*x0
    d_eff = d0 + 2 * C_s * t0
    # d_eff_comp = d_eff - d_eff_2
    # num_greater = np.sum(d_eff_comp < 0)
    # print("Number of instances where d_eff is less than d_eff_2:", num_greater)
    # print("Percentage of times d_eff is less than d_eff_2:", num_greater/len(d_eff_comp))
    #num_instances = np.sum(d_eff < d0)
    #print("Number of instances where d_eff is less than d0:", num_instances)
    #num_nums = np.sum(n_max < n_crit)
    #print("Number of instances where n_max is less than n_crit:", num_nums)
    
    n_crit_array = n_crit * np.ones(len(n_max))

    # Modify effective intensity with Pulse Depletion in under-dense region
    
    #zp = zR*np.sqrt((I_main/1e4)*(lmda*1e6)**2/a0_min**2/(1.37e18)) # position of minimum a0
    #zm = -zp 
    #xf = np.inf*5.517 * C_s * t0  # End position of Plasma Expansion
    xf = (2*np.log(omega_pi*t0) - np.log(2) + 3) * C_s * t0 # Position of maximum density
    #x0 = (np.log(n_max) - np.log(n_crit)) * C_s * t0 # Position of critical density

    a = x0.copy() 
    b = xf.copy() 
    out_of_bounds = np.ones(len(z))

    # for i in range(len(z)):
    #     # Shorten a to location of minimum a0
    #     if z[i] - xf[i] < zp[i] < z[i] - x0[i]:
    #         a[i] = z[i] - zp[i]
    #     # Shorten b to location of minimum a0
    #     if z[i] - xf[i] < zm[i] < z[i] - x0[i]:
    #         b[i] = z[i] - zm[i]
    #     # Under-dense plasma is located in region where a_0 < a_0,min
    #     if zp[i] < z[i] - xf[i] or z[i] - x0[i] < zm[i]:
    #         out_of_bounds[i] = 0
    #     # Good, both xf and x0 are within valid a_0 bounds: no change
    #     if zm[i] < z[i] - xf[i] < z[i] - x0[i] < zp[i]:
    #         pass
    #     # If plasma expansion doesn't reach critical density, there is no underdense plasma
    #     if xf[i] < x0[i]:
    #         out_of_bounds[i] = 0
    # "Etching" Distance Integral
    dx_etch = e**2 * n_max * (C_s*t0) / (eps0 * me * omega**2) * (np.exp(-a/(C_s*t0))- np.exp(-b/(C_s*t0))) * out_of_bounds
    #print("COUNT", np.sum(dx_etch < 0))
    c_tau = c*tFWHM

    #decay_factor = decay(z)
    decay_factor = np.clip(1-dx_etch/c_tau, 1e-6, 1)
    #decay_factor = np.ones(len(z))
    neg = np.sum(decay_factor > 1)
    #print("Number of instances where decay factor is greater than 1:", neg)
    #print(decay_factor)
    #I_main = I_main * decay_factor
    tFWHM = tFWHM * decay_factor
    
    return fuchs_model(I0=I_main, tFWHM=tFWHM, z=z, d=d_eff, c1=c1, c2=c2, theta=theta, pct_noise = pct_noise, const_f = const_f,max_array = n_max, crit_array = n_crit_array, spectrum=spectrum)


if __name__== "__main__":
    z = np.linspace(-30e-6, 30e-6, 100)
    I_main = 1e23 *np.ones(len(z))
    d0 = 0.5e-6 * np.ones(len(z))

    max_energy = fuchs_function_with_prepulse(I_main=I_main, z=z, d0=d0, t0=100e-12, c1=50)[5]

    fig, ax = plt.subplots()
    ax.plot(z*1e6, max_energy)
    ax.set_xlabel(rf'z $(\mu m)$')
    ax.set_ylabel('Max Proton Energy (MeV)')
    plt.show()
