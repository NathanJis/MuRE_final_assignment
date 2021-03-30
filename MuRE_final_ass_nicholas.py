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
    return p / R / T


def c_to_p_ideal_gas(c):
    return c * R * T

def init_stage(U_lb_out, U_lb_in, j_CO, j_H2,C_CO_l,C_H2_l):
    gas = (U_lb_out - U_lb_in) / h + (j_CO + j_H2) / (C_CO_lb + C_H2_lb)
    liq_CO = j_CO - (0.3 * U_lb_out**0.58 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l))
    liq_H2 = j_H2 - (0.3 * U_lb_out**0.58 * np.sqrt(D_H2/ D_ref) * (C_H2_lb / H_H2 - C_H2_l))
    return gas, liq_CO, liq_H2

# define equation system
def AES(x):
    C_CO_l, C_H2_l, U_l, R_CO, R_H2, p_CO_l, p_H2_l = x[-8:-1]
    eps_slurry = x[-1]
    res = []
    first = init_stage(x[0], U_lb_previous_stage,x[1],x[2],C_CO_l,C_H2_l)
    res.append(first[0])
    res.append(first[1])
    res.append(first[2])
    for i in range(slices-1):
            stage_calc = init_stage(x[i*3+3], x[i*3], x[i*3+4], x[i*3+5], C_CO_l,C_H2_l)
            res.append(stage_calc[0])
            res.append(stage_calc[1])
            res.append(stage_calc[2])
    res.append(h * sum(x[1:slices*3:3]) + C_CO_l_previous_stage * U_l_previous_stage - C_CO_l * U_l - H_stage * eps_slurry * R_CO)
    res.append(h * sum(x[2:slices*3:3]) + C_CO_l_previous_stage * U_l_previous_stage - C_CO_l * U_l - H_stage * eps_slurry * R_CO)
    res.append(U_l - (U_l_previous_stage + (H_stage * eps_slurry * A_reactor * R_CO * M_CO + H_stage * eps_slurry * A_reactor * R_H2 * M_H2) / rho_slurry / A_reactor))
    res.append(R_CO - (F * a * p_H2_l * p_CO_l / (1 + b * p_CO_l)**2 * eps_cat * rho_skeleton_cat))
    res.append(R_H2 - (2 * R_CO))
    res.append(p_CO_l - (H_CO * C_CO_l * R * T * 10**-5))
    res.append(p_H2_l - (H_H2 * C_H2_l * R * T * 10**-5))
    res.append(eps_slurry - (1 - 0.3 * x[int(0.5*3*slices-3)]**0.58))
    return np.array(res)


#####################################################################################################
# parameter and variable definition
# design variables
number_of_stages = 5    # number of stages
slices = 10              #number of slices
height_reactor = 5     # reactor diameter in m
diam_reactor = 7        # reactor diameter in m
kg_per_day = 2000e3     #throughput per day       

P = 30 * 10**5          # pressure in Pa
T = 240 + 273.15        # temperature in K
dp = 50 * 10**-6        # catalyst particle size in m
eps_cat = 0.35          # catalyst concentration in vol_cat/vol_slurry
feed_ratio = 2          # ratio of C_H2/C_CO
h = height_reactor / number_of_stages / slices  # finite difference step size
# inlet conditions
U_lb_previous_stage = U_in
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
C_CO_lb = p_to_c_ideal_gas(P / (1 + feed_ratio))
C_H2_lb = p_to_c_ideal_gas(P / (1 + 1 / feed_ratio))
# kinetic parameters
a = a0 * np.exp(Ea/R * (1/493.15 - 1/T))        # langmuir hinshelwood parameter 1 mol/s/kgcat/bar2
b = b0 * np.exp(deltabH/R * (1/493.15 - 1/T))   # langmuir hinshelwood parameter 2 in 1/bar
# hydrodynamic parameters
A_reactor = np.pi * diam_reactor ** 2 / 4  # reactor cross section area in m2
U_in = (((kg_per_day/(24*3600))/(3*M_CO))*(R*T/P))/(A_reactor)  # inlet superficial gas velocity in m/s
U_lb_previous_stage = U_in 
H_stage = height_reactor/number_of_stages
U_lb = 1  # superficial gas velocity for lb bubbles in m/s
U_small = 1  # superficial gas velocity for small bubbles in m/s
# slurry parameters
rho_slurry = rho_solvent * (1 - rho_solvent/rho_skeleton_cat * eps_cat) + rho_cat * eps_cat # slurry density in kg/m3 (maretto2009, 23)
visco_slurry = visco_solvent * (1 + 4.5 * eps_cat)                                          # slurry viscosity in Pas (maretto2009, 24)


#####################################################################################################
# simulation
def generate_init(N):
    lis = []
    for i in range(N):
        lis.append(U_in)
        lis.append(1)
        lis.append(1)
    lis.append(C_CO_lb / H_CO)
    lis.append((C_H2_lb / H_H2))  # C_H2_l
    lis.append((1))  # U_l
    lis.append((1))  # R_CO
    lis.append((1))  # R_H2
    lis.append((c_to_p_ideal_gas(C_CO_lb / H_CO) * 10**-5))  # p_CO_l
    lis.append((c_to_p_ideal_gas(C_H2_lb / H_H2) * 10**-5))  # p_H2_l
    lis.append((0.5))
    return np.array(lis)
    
# initial guess definition
initial_guess = generate_init(slices)


stage_solutions = []
for stage_counter in range(number_of_stages):
    # sol, output, ier, mesg = optimize.fsolve(AES, initial_guess, full_output=True)
    # print(ier)
    # print(mesg)
    root = optimize.root(AES, initial_guess)
    sol = root.x
    stage_solutions.append(sol)  # save solutions of this stage
    initial_guess = sol  # the initial guess for the next stage

    # print('residuals: ',AES(sol))
    print('sum of residuals = ', sum(AES(sol)))
    print()
    print('U_lb_out =\t', round(sol[slices*3-3], 3))
    print('C_CO_l =  \t', round(sol[slices*3], 3))
    print('C_H2_l =  \t', round(sol[slices*3+1], 3))
    print('U_l =     \t', round(sol[slices*3+2], 3))

    # define boundary inlet conditions for the next stage
    U_lb_previous_stage = sol[slices*3-3]
    C_CO_l_previous_stage = sol[slices*3]
    C_H2_l_previous_stage = sol[slices*3+1]
    U_l_previous_stage = sol[slices*3+2]

# sol_labels = ['U_lb1', 'j_CO1', 'j_H21', 'U_lb2', 'j_CO2', 'j_H22', 'C_CO_l', 'C_H2_l', 'U_l', 'R_CO', 'R_H2', 'p_CO_l', 'p_H2_l', 'eps_slurry']
# for i in range(len(sol_labels)):
#     print('%s =\t %s' % (sol_labels[i], round(sol[i],5)))

print('The final conversion is %s percent' % (round(1 - U_lb_previous_stage/U_in, 2)*100))

plt.figure()
plt.plot(0, U_in, 'o', color='blue')
for i in range(number_of_stages):
    for j in range(slices):
        plt.plot(i + h * (j+1),stage_solutions[i][j*3], 'o', color='blue')
plt.xlabel('Height in m')
plt.ylabel('Superficial gas velocity in m/s')
plt.legend()
plt.show()


print(stage_solutions)
#exit()
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