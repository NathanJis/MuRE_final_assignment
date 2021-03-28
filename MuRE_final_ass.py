# -*- coding: utf-8 -*-

# import libraries
# from scipy import integrate
from scipy import optimize
import numpy as np 
import matplotlib.pyplot as plt 


# define functions
def p_to_c_ideal_gas(p):
    return p / R / T


def c_to_p_ideal_gas(c):
    return c * R * T


# define equation system
def AES(x):
    U_large, j_CO_avg, j_H2_avg, C_CO_l, C_H2_l, U_l, eps_large_avg, R_CO, R_H2, kLa_CO_avg, kLa_H2_avg, U_large_avg, p_CO_l, p_H2_l, eps_slurry = x
    return [
        # balances
        (U_large_previous_stage - U_large) * (C_CO_large + C_H2_large) - H_stage * (j_CO_avg + j_H2_avg),
        H_stage * j_CO_avg + C_CO_l_previous_stage * U_l_previous_stage - C_CO_l * U_l - H_stage * eps_slurry * R_CO,
        H_stage * j_H2_avg + C_H2_l_previous_stage * U_l_previous_stage - C_H2_l * U_l - H_stage * eps_slurry * R_H2,
        # constituent equations
        j_CO_avg - (kLa_CO_avg * (C_CO_large / H_CO - C_CO_l)),
        j_H2_avg - (kLa_H2_avg * (C_H2_large / H_H2 - C_H2_l)),
        U_large_avg - (0.5 * (U_large + U_large_previous_stage)),
        eps_large_avg - (0.3 * abs(U_large_avg)**0.58),
        kLa_CO_avg - (eps_large_avg * 0.5 * np.sqrt(D_CO / D_ref)),
        kLa_H2_avg - (eps_large_avg * 0.5 * np.sqrt(D_H2 / D_ref)),
        U_l - (U_l_previous_stage + (H_stage * eps_slurry * A_reactor * R_CO * M_CO + H_stage * eps_slurry * A_reactor * R_H2 * M_H2) / rho_slurry / A_reactor),
        R_CO - (F * a * p_H2_l * p_CO_l / (1 + b * p_CO_l)**2 * eps_cat * rho_skeleton_cat),
        R_H2 - (2 * R_CO),
        p_CO_l - (H_CO * C_CO_l * R * T * 10**-5),
        p_H2_l - (H_H2 * C_H2_l * R * T * 10**-5),
        eps_slurry - (1 - eps_large_avg),
            ]

#####################################################################################################
# parameter and variable definition

# design variables
number_of_stages = 1    # number of stages
height_reactor = 5      # reactor diameter in m
diam_reactor = 7        # reactor diameter in m
U_in = 0.3              # inlet superficial gas velocity in m/s
P = 30 * 10**5          # pressure in Pa
T = 240 + 273.15        # temperature in K
dp = 50 * 10**-6        # catalyst particle size in m
eps_cat = 0.35          # catalyst concentration in vol_cat/vol_slurry
feed_ratio = 2          # ratio of C_H2/C_CO
# inlet conditions
U_large_previous_stage = U_in
C_CO_l_previous_stage = 0  # 0 for fresh slurry
C_H2_l_previous_stage = 0  # 0 for fresh slurry
U_l_previous_stage = 0  # 0 for semi-batch

# independent parameters
# constants
R = 8.314               # universal gas constant in J/mol/K
# species parameter
M_CO = 0.02801          # molar mass CO in kg/mol
M_H2 = 0.002            # molar mass CO in kg/mol
H_CO = 2.478            # henry coefficient CO at T=240 °C (maretto1999)
H_H2 = 2.964            # henry coefficient H2 at T=240 °C (maretto1999)
D_CO = 17.2 * 10**-9    # CO diffusivity in m2/s at T=240 °C (maretto2001)
D_H2 = 45.5 * 10**-9    # H2 diffusivity in m2/s at T=240 °C (maretto2001)
D_ref = 2 * 10**-9      # reference diffusivity in m2/s (maretto1999)
# solvent parameters (C16H24)
rho_solvent = 640               # solvent density in kg/m3 at T=240 °C (maretto1999)
visco_solvent = 2.9 * 10**-4    # solvent viscosity in Pas at T=240 °C (maretto1999)
surf_tension_solvent = 0.01     # solvent surface tenstion in N/m at T=240 °C (maretto1999)
# catalyst parameters
rho_cat = 647               # catalyst density (inluding pores) in kg/m3 (maretto1999)
V_pores = 0.00105           # catalyst pore volume in m3/kg (maretto1999)
rho_skeleton_cat = 2030     # catalyst skeleton density in kg/m3 (maretto1999)
# kinetic parameters
a0 = 8.88533 * 10**-3       # Reaction rate coefficient at T = 493.15 K  in mol/s/kgcat/bar2 (assignment pdf)
b0 = 2.226                  # Adsorption coefficient at T = 493.15 K in 1/bar (assignment pdf)
Ea = 3.737 * 10**4          # activation energy in J/mol (assignment pdf)
deltabH = -6.837 * 10**3    # adsorption enthalpy in J/mol (assignment pdf)
F = 3                       # catalyst activity multiplication factor (assignment pdf)

# calculate dependent parameters
# bubble concentrations
C_CO_large = p_to_c_ideal_gas(P / (1 + feed_ratio))
C_H2_large = p_to_c_ideal_gas(P / (1 + 1 / feed_ratio))
# kinetic parameters
a = a0 * np.exp(Ea/R * (1/493.15 - 1/T))        # langmuir hinshelwood parameter 1 mol/s/kgcat/bar2
b = b0 * np.exp(deltabH/R * (1/493.15 - 1/T))   # langmuir hinshelwood parameter 2 in 1/bar
# hydrodynamic parameters
A_reactor = np.pi * diam_reactor ** 2 / 4  # reactor cross section area in m2
H_stage = height_reactor/number_of_stages
U_large = 1  # superficial gas velocity for large bubbles in m/s
U_small = 1  # superficial gas velocity for small bubbles in m/s
# slurry parameters
rho_slurry = rho_solvent * (1 - rho_solvent/rho_skeleton_cat * eps_cat) + rho_cat * eps_cat # slurry density in kg/m3 (maretto2009, 23)
visco_slurry = visco_solvent * (1 + 4.5 * eps_cat)                                          # slurry viscosity in Pas (maretto2009, 24)


#####################################################################################################
# simulation

# initial guess definition
initial_guess = np.array([
    U_in,  # U_large
    1,  # j_CO_avg
    1,  # j_H2_avg
    C_CO_large,  # C_CO_l
    C_H2_large,  # C_H2_l
    1,  # U_l
    0.5,  # eps_large_avg
    1,  # R_CO
    1,  # R_H2
    1,  # k_La_CO_avg
    1,  # k_La_H2_avg
    U_in,  # U_large_avg
    c_to_p_ideal_gas(C_CO_large) * 10**-5,  # p_CO_l
    c_to_p_ideal_gas(C_H2_large) * 10**-5,  # p_H2_l
    0.5   # eps_slurry
])

stage_solutions = []
for stage_counter in range(number_of_stages):
    sol, output, ier, mesg = optimize.fsolve(AES, initial_guess, full_output=True)
    # print(ier)
    print(mesg)
    # root = optimize.root(AES, initial_guess)
    # sol = root.x
    stage_solutions.append(sol)  # save solutions of this stage
    initial_guess = sol  # the initial guess for the next stage

    print('residuals: ',AES(sol))
    print('sum of residuals = ', sum(AES(sol)))
    print()

    # define boundary inlet conditions for the next stage
    U_large_previous_stage = sol[0]
    C_CO_l_previous_stage = sol[3]
    C_H2_l_previous_stage = sol[4]
    U_l_previous_stage = sol[5]

sol_labels = ['U_large', 'j_CO_avg', 'j_H2_avg', 'C_CO_l', 'C_H2_l', 'U_l', 'eps_large_avg', 'R_CO', 'R_H2', 'kLa_CO_avg', 'kLa_H2_avg', 'U_large_avg', 'p_CO_l', 'p_H2_l', 'eps_slurry']
for i in range(len(sol_labels)):
    print('%s =\t %s' % (sol_labels[i], round(sol[i],5)))

exit()
################################################################
# sensitivity analysis
diam_arr = np.linspace(1,7,7)
diam_sens_lis_co = []
diam_sens_lis_h = []
for diam in diam_arr:
    num_big = (bub_split*(np.pi/4)*diam**2*e)/((1/6)*np.pi*big_bub_diam**3) #to calc total area of exchange in ODE
    num_small = ((1-bub_split)*(np.pi/4)*diam**2*e)/((1/6)*np.pi*small_bub_diam**3) #to calc total area of exchange in ODE
    simul = run_simul()
    diam_sens_lis_co.append(simul[4])
    diam_sens_lis_h.append(simul[5])

plt.figure(1)
plt.scatter(diam_arr, diam_sens_lis_co)
plt.scatter(diam_arr, diam_sens_lis_h)

diam = 7


height_arr = np.linspace(0.5,5,9)
height_sens_lis_co = []
height_sens_lis_h = []
for height in height_arr:
    height_stage = height/stages
    z = np.linspace(0,height_stage,100)
    simul = run_simul()
    height_sens_lis_co.append(simul[4])
    height_sens_lis_h.append(simul[5])

plt.figure(2)
plt.scatter(height_arr, height_sens_lis_co)
plt.scatter(height_arr, height_sens_lis_h)

height = 10

stage_arr = np.linspace(1,5,5)
stage_sens_lis_co = []
stage_sens_lis_h = []
for stage in stage_arr:
    height_stage = height/stage
    z = np.linspace(0,height_stage,100)
    simul = run_simul()
    stage_sens_lis_co.append(simul[4])
    stage_sens_lis_h.append(simul[5])

plt.figure(3)
plt.scatter(stage_arr, stage_sens_lis_co)
plt.scatter(stage_arr, stage_sens_lis_h)