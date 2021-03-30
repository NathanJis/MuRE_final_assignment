# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:10:46 2021
@author: natha
"""
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


# define functions
def p_to_c_ideal_gas(p):
    # input in mol/m3, output in Pa
    return p / R / T


def c_to_p_ideal_gas(c):
    # input in mol/m3, output in Pa
    return c * R * T


def generate_initial_guess(N):
    lis = []
    for i in range(N):
        lis.append(U_in)  # U_lb
        lis.append(1)  # j_CO
        lis.append(2)  # j_H2
        lis.append(0.5)  # eps_sl
    lis.append(C_CO_lb / H_CO)  # C_CO_l
    lis.append((C_H2_lb / H_H2))  # C_H2_l
    lis.append((1))  # U_l
    lis.append((1))  # R_CO
    lis.append((1))  # R_H2
    lis.append((c_to_p_ideal_gas(C_CO_lb / H_CO) * 10 ** -5))  # p_CO_l
    lis.append((c_to_p_ideal_gas(C_H2_lb / H_H2) * 10 ** -5))  # p_H2_l
    return np.array(lis)


def calculate_differential_slice(U_lb_slice_in, U_lb_slice_out, j_CO, j_H2, eps_sl, C_CO_l, C_H2_l):
    mass_balance_gas_residual = (U_lb_slice_out - U_lb_slice_in) / h + (j_CO + j_H2) / (C_CO_lb + C_H2_lb)
    mass_transfer_CO_residual = j_CO - (0.3 * U_lb_slice_out ** 0.58 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l))
    mass_transfer_H2_residual = j_H2 - (0.3 * U_lb_slice_out ** 0.58 * np.sqrt(D_H2 / D_ref) * (C_H2_lb / H_H2 - C_H2_l))
    slurry_holdup_residual = eps_sl - (1 - 0.3 * U_lb_slice_out ** 0.58)
    return mass_balance_gas_residual, mass_transfer_CO_residual, mass_transfer_H2_residual, slurry_holdup_residual


# define equation system
def AES(x):
    C_CO_l, C_H2_l, U_l, R_CO, R_H2, p_CO_l, p_H2_l = x[4 * n_slices:]
    residuals = []

    # differential slices
    for i in range(n_slices):
        if i == 0:
            # use a different U_lb_slice_in for the first slice
            slice_residuals = calculate_differential_slice(U_lb_previous_stage, x[0], x[1], x[2], x[3], C_CO_l, C_H2_l)
        else:
            slice_residuals = calculate_differential_slice(x[i * 4 - 4], x[i * 4], x[i * 4 + 1], x[i * 4 + 2], x[i * 4 + 3], C_CO_l, C_H2_l)
        residuals.append(slice_residuals[0])
        residuals.append(slice_residuals[1])
        residuals.append(slice_residuals[2])
        residuals.append(slice_residuals[3])

    # create sums
    sum_j_CO = sum(x[1:n_slices * 4:4])
    sum_j_H2 = sum(x[2:n_slices * 4:4])
    sum_eps_sl = sum(x[3:n_slices * 4:4])

    # slurry phase mass balances
    residuals.append(h * sum_j_CO + C_CO_l_previous_stage * U_l_previous_stage - C_CO_l * U_l - R_CO * h * sum_eps_sl)
    residuals.append(h * sum_j_H2 + C_H2_l_previous_stage * U_l_previous_stage - C_H2_l * U_l - R_H2 * h * sum_eps_sl)
    # constituent equations
    residuals.append(U_l - (U_l_previous_stage + (h * sum_eps_sl * (R_CO * M_CO + R_H2 * M_H2) / rho_solvent)))
    residuals.append(R_CO - (F * a * p_H2_l * p_CO_l / (1 + b * p_CO_l) ** 2 * eps_cat * rho_skel))
    residuals.append(R_H2 - (2 * R_CO))
    residuals.append(p_CO_l - (H_CO * C_CO_l * R * T * 10 ** -5))
    residuals.append(p_H2_l - (H_H2 * C_H2_l * R * T * 10 ** -5))

    return np.array(residuals)


#####################################################################################################
# parameter and variable definition
# design variables
number_of_stages = 2  # number of stages
n_slices = 10  # number of slices
height_reactor = 2  # reactor diameter in m
diam_reactor = 1  # reactor diameter in m
U_in = 0.3  # inlet superficial gas velocity in m/s
P = 30 * 10 ** 5  # pressure in Pa
T = 240 + 273.15  # temperature in K
dp = 50 * 10 ** -6  # catalyst particle size in m
eps_cat = 0.35  # catalyst concentration in vol_cat/vol_slurry
feed_ratio = 2  # ratio of C_H2/C_CO
h = height_reactor / number_of_stages / n_slices  # finite difference step size
# inlet conditions
U_lb_previous_stage = U_in
C_CO_l_previous_stage = 0  # 0 for fresh slurry
C_H2_l_previous_stage = 0  # 0 for fresh slurry
U_l_previous_stage = 0  # 0 for semi-batch

# independent parameters
# constants
R = 8.314  # universal gas constant in J/mol/K
# species parameter
M_CO = 0.02801  # molar mass CO in kg/mol
M_H2 = 0.002  # molar mass CO in kg/mol
H_CO = 2.478  # henry coefficient CO at T=240 °C (maretto1999)
H_H2 = 2.964  # henry coefficient H2 at T=240 °C (maretto1999)
D_CO = 17.2 * 10 ** -9  # CO diffusivity in m2/s at T=240 °C (maretto2001)
D_H2 = 45.5 * 10 ** -9  # H2 diffusivity in m2/s at T=240 °C (maretto2001)
D_ref = 2 * 10 ** -9  # reference diffusivity in m2/s (maretto1999)
# solvent parameters (C16H24)
rho_solvent = 640  # solvent density in kg/m3 at T=240 °C (maretto1999)
visco_solvent = 2.9 * 10 ** -4  # solvent viscosity in Pas at T=240 °C (maretto1999)
surf_tension_solvent = 0.01  # solvent surface tenstion in N/m at T=240 °C (maretto1999)
# catalyst parameters
rho_cat = 647  # catalyst density (inluding pores) in kg/m3 (maretto1999)
V_pores = 0.00105  # catalyst pore volume in m3/kg (maretto1999)
rho_skel = 2030  # catalyst skeleton density in kg/m3 (maretto1999)
# kinetic parameters
a0 = 8.88533 * 10 ** -3  # Reaction rate coefficient at T = 493.15 K  in mol/s/kgcat/bar2 (assignment pdf)
b0 = 2.226  # Adsorption coefficient at T = 493.15 K in 1/bar (assignment pdf)
Ea = 3.737 * 10 ** 4  # activation energy in J/mol (assignment pdf)
deltabH = -6.837 * 10 ** 3  # adsorption enthalpy in J/mol (assignment pdf)
F = 3  # catalyst activity multiplication factor (assignment pdf)

# calculate dependent parameters
# bubble concentrations
C_CO_lb = p_to_c_ideal_gas(P / (1 + feed_ratio))
C_H2_lb = p_to_c_ideal_gas(P / (1 + 1 / feed_ratio))
# kinetic parameters
a = a0 * np.exp(Ea / R * (1 / 493.15 - 1 / T))  # langmuir hinshelwood parameter 1 mol/s/kgcat/bar2
b = b0 * np.exp(deltabH / R * (1 / 493.15 - 1 / T))  # langmuir hinshelwood parameter 2 in 1/bar
# hydrodynamic parameters
A_reactor = np.pi * diam_reactor ** 2 / 4  # reactor cross section area in m2
H_stage = height_reactor / number_of_stages
U_lb = 1  # superficial gas velocity for lb bubbles in m/s
U_small = 1  # superficial gas velocity for small bubbles in m/s
# slurry parameters
rho_sl = rho_solvent * (1 - rho_solvent / rho_skel * eps_cat) + rho_cat * eps_cat  # slurry density in kg/m3 (maretto2009, 23)
visco_sl = visco_solvent * (1 + 4.5 * eps_cat)  # slurry viscosity in Pas (maretto2009, 24)


#####################################################################################################
# simulation

# initial guess definition
initial_guess = generate_initial_guess(n_slices)

# res.append(h * sum(x) + C_CO_l_previous_stage * U_l_previous_stage - C_CO_l * U_l - H_stage * eps_slurry * R_CO)
# res.append(h * sum(x[2:slices * 3:3]) + C_CO_l_previous_stage * U_l_previous_stage - C_CO_l * U_l - H_stage * eps_slurry * R_CO)

stage_solutions = []
for stage_counter in range(number_of_stages):
    # sol, output, ier, mesg = optimize.fsolve(AES, initial_guess, full_output=True)
    # print(ier)
    # print(mesg)
    root = optimize.root(AES, initial_guess)
    sol = root.x
    stage_solutions.append(sol)  # save solutions of this stage
    initial_guess = sol  # the initial guess for the next stage

    print('\nResults stage %s:' % (stage_counter + 1))
    # print('Solution vector', sol)
    # print('Residuals', AES(sol))
    print('sum of residuals = ', sum(AES(sol)))
    print('U_lb_out =\t', round(sol[n_slices * 4 - 4], 3))
    print('C_CO_l =  \t', round(sol[n_slices * 4], 3))
    print('C_H2_l =  \t', round(sol[n_slices * 4 + 1], 3))
    print('U_l =     \t', round(sol[n_slices * 4 + 2], 3))

    # define boundary inlet conditions for the next stage
    U_lb_previous_stage = sol[n_slices * 4 - 4]
    C_CO_l_previous_stage = sol[n_slices * 4]
    C_H2_l_previous_stage = sol[n_slices * 4 + 1]
    U_l_previous_stage = sol[n_slices * 4 + 2]

print('The final conversion is %s percent' % (round(1 - U_lb_previous_stage / U_in, 2) * 100))


U_lb_array = []
j_CO_array = []
j_H2_array = []
eps_sl_array = []
slice_height_array = []
for i in range(number_of_stages):
    for j in range(n_slices):
        U_lb_array.append(stage_solutions[i][j * 4])
        j_CO_array.append(stage_solutions[i][j * 4 + 1])
        j_H2_array.append(stage_solutions[i][j * 4 + 2])
        eps_sl_array.append(stage_solutions[i][j * 4 + 3])
        slice_height_array.append(i + h * (j + 1))


plt.figure()
plt.plot(0, U_in, 'o', color='blue')
plt.plot(slice_height_array, U_lb_array, 'o', color='blue')
plt.xlabel('Height in m')
plt.ylabel('Superficial gas velocity in m/s')
plt.show()

plt.figure()
plt.plot(slice_height_array, eps_sl_array, 'o', color='blue')
plt.xlabel('Height in m')
plt.ylabel('Slurry concentration in m3slurry/m3reactor')
plt.show()

exit()

plt.figure()
for i in range(number_of_stages):
    plt.plot(i+H_stage/2,stage_solutions[i][30], 'o', color='blue',label="CO")
    # plt.plot(i + H_stage / 2, stage_solutions[i][31], 'o', color='green', label="H2")
plt.xlabel('Height in m')
plt.ylabel('CO surry concentration in mol/m3')
plt.xlim(0,2)
plt.show()

plt.figure()
for i in range(number_of_stages):
    for j in range(n_slices):
        plt.plot(i + h * (j + 1), stage_solutions[i][1+j * 3], 'o', color='blue')
plt.xlabel('Height in m')
plt.ylabel('kLa_CO 1/s')
plt.show()

plt.figure()
for i in range(number_of_stages):
    for j in range(n_slices):
        plt.plot(i + h * (j + 1), stage_solutions[i][2+j * 3], 'o', color='blue')
plt.xlabel('Height in m')
plt.ylabel('kLa_H2 1/s')
plt.show()

print(stage_solutions)
# exit()
"""
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
"""