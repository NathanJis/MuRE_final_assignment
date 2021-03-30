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
    U_lb_1, j_CO_1, j_H2_1, U_lb_2, j_CO_2, j_H2_2, U_lb_3, j_CO_3, j_H2_3, U_lb_4, j_CO_4, j_H2_4, U_lb_5, j_CO_5, j_H2_5, \
    U_lb_6, j_CO_6, j_H2_6, U_lb_7, j_CO_7, j_H2_7, U_lb_8, j_CO_8, j_H2_8, U_lb_9, j_CO_9, j_H2_9, U_lb_10, j_CO_10, j_H2_10, \
    C_CO_l, C_H2_l, U_l, R_CO, R_H2, p_CO_l, p_H2_l, eps_slurry = x
    return [
        # balances
        (U_lb_1 - U_lb_previous_stage) / h + (j_CO_1 + j_H2_1) / (C_CO_lb + C_H2_lb),
        j_CO_1 - (0.3 * U_lb_1**0.58 / diam_reactor**0.18 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l)),
        j_H2_1 - (0.3 * U_lb_1**0.58 / diam_reactor**0.18 * np.sqrt(D_H2/ D_ref) * (C_H2_lb / H_H2 - C_H2_l)),

        (U_lb_2 - U_lb_1) / h + (j_CO_2 + j_H2_2) / (C_CO_lb + C_H2_lb),
        j_CO_2 - (0.3 * U_lb_2**0.58 / diam_reactor**0.18 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l)),
        j_H2_2 - (0.3 * U_lb_2**0.58 / diam_reactor**0.18 * np.sqrt(D_H2 / D_ref) * (C_H2_lb / H_H2 - C_H2_l)),

        (U_lb_3 - U_lb_2) / h + (j_CO_3 + j_H2_3) / (C_CO_lb + C_H2_lb),
        j_CO_3 - (0.3 * U_lb_3 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l)),
        j_H2_3 - (0.3 * U_lb_3 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_H2 / D_ref) * (C_H2_lb / H_H2 - C_H2_l)),

        (U_lb_4 - U_lb_3) / h + (j_CO_4 + j_H2_4) / (C_CO_lb + C_H2_lb),
        j_CO_4 - (0.3 * U_lb_4 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l)),
        j_H2_4 - (0.3 * U_lb_4 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_H2 / D_ref) * (C_H2_lb / H_H2 - C_H2_l)),

        (U_lb_5 - U_lb_4) / h + (j_CO_5 + j_H2_5) / (C_CO_lb + C_H2_lb),
        j_CO_5 - (0.3 * U_lb_5 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l)),
        j_H2_5 - (0.3 * U_lb_5 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_H2 / D_ref) * (C_H2_lb / H_H2 - C_H2_l)),

        (U_lb_6 - U_lb_5) / h + (j_CO_6 + j_H2_6) / (C_CO_lb + C_H2_lb),
        j_CO_6 - (0.3 * U_lb_6 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l)),
        j_H2_6 - (0.3 * U_lb_6 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_H2 / D_ref) * (C_H2_lb / H_H2 - C_H2_l)),

        (U_lb_7 - U_lb_6) / h + (j_CO_7 + j_H2_7) / (C_CO_lb + C_H2_lb),
        j_CO_7 - (0.3 * U_lb_7 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l)),
        j_H2_7 - (0.3 * U_lb_7 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_H2 / D_ref) * (C_H2_lb / H_H2 - C_H2_l)),

        (U_lb_8 - U_lb_7) / h + (j_CO_8 + j_H2_8) / (C_CO_lb + C_H2_lb),
        j_CO_8 - (0.3 * U_lb_8 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l)),
        j_H2_8 - (0.3 * U_lb_8 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_H2 / D_ref) * (C_H2_lb / H_H2 - C_H2_l)),

        (U_lb_9 - U_lb_8) / h + (j_CO_9 + j_H2_9) / (C_CO_lb + C_H2_lb),
        j_CO_9 - (0.3 * U_lb_9 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l)),
        j_H2_9 - (0.3 * U_lb_9 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_H2 / D_ref) * (C_H2_lb / H_H2 - C_H2_l)),

        (U_lb_10 - U_lb_9) / h + (j_CO_10 + j_H2_10) / (C_CO_lb + C_H2_lb),
        j_CO_10 - (0.3 * U_lb_10 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l)),
        j_H2_10 - (0.3 * U_lb_10 ** 0.58 / diam_reactor**0.18 * np.sqrt(D_H2 / D_ref) * (C_H2_lb / H_H2 - C_H2_l)),

        h * (j_CO_1 + j_CO_2 + j_CO_3 + j_CO_4 + j_CO_5 + j_CO_6 + j_CO_7 + j_CO_8 + j_CO_9 + j_CO_10) + C_CO_l_previous_stage * U_l_previous_stage - C_CO_l * U_l - H_stage * eps_slurry * R_CO,
        h * (j_H2_1 + j_H2_2 + j_H2_3 + j_H2_4 + j_CO_5 + j_H2_6 + j_H2_7 + j_H2_8 + j_H2_9 + j_H2_10) + C_H2_l_previous_stage * U_l_previous_stage - C_H2_l * U_l - H_stage * eps_slurry * R_H2,
        # constituent equations
        U_l - (U_l_previous_stage + (H_stage * eps_slurry * A_reactor * R_CO * M_CO + H_stage * eps_slurry * A_reactor * R_H2 * M_H2) / rho_slurry / A_reactor),
        R_CO - (F * a * p_H2_l * p_CO_l / (1 + b * p_CO_l)**2 * eps_cat * rho_skeleton_cat),
        R_H2 - (2 * R_CO),
        p_CO_l - (H_CO * C_CO_l * R * T * 10**-5),
        p_H2_l - (H_H2 * C_H2_l * R * T * 10**-5),
        eps_slurry - (1 - 0.3 * U_lb_5**0.58 / diam_reactor**0.18),
            ]


#####################################################################################################
# parameter and variable definition
# design variables
number_of_stages = 1    # number of stages
height_reactor = 1     # reactor diameter in m
diam_reactor = 1        # reactor diameter in m
U_in = 0.3              # inlet superficial gas velocity in m/s
P = 30 * 10**5          # pressure in Pa
T = 240 + 273.15        # temperature in K
dp = 50 * 10**-6        # catalyst particle size in m
eps_cat = 0.35          # catalyst concentration in vol_cat/vol_slurry
feed_ratio = 2          # ratio of C_H2/C_CO
h = height_reactor / number_of_stages / 10  # finite difference step size
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
H_stage = height_reactor/number_of_stages
U_lb = 1  # superficial gas velocity for lb bubbles in m/s
U_small = 1  # superficial gas velocity for small bubbles in m/s
# slurry parameters
rho_slurry = rho_solvent * (1 - rho_solvent/rho_skeleton_cat * eps_cat) + rho_cat * eps_cat # slurry density in kg/m3 (maretto2009, 23)
visco_slurry = visco_solvent * (1 + 4.5 * eps_cat)                                          # slurry viscosity in Pas (maretto2009, 24)


#####################################################################################################
# simulation

# initial guess definition
initial_guess = np.array([
    U_in,  # U_lb
    1,  # j_CO
    1,  # j_H2
    U_in,  # U_lb
    1,  # j_CO
    1,  # j_H2
    U_in,  # U_lb
    1,  # j_CO
    1,  # j_H2
    U_in,  # U_lb
    1,  # j_CO
    1,  # j_H2
    U_in,  # U_lb
    1,  # j_CO
    1,  # j_H2
    U_in,  # U_lb
    1,  # j_CO
    1,  # j_H2
    U_in,  # U_lb
    1,  # j_CO
    1,  # j_H2
    U_in,  # U_lb
    1,  # j_CO
    1,  # j_H2
    U_in,  # U_lb
    1,  # j_CO
    1,  # j_H2
    U_in,  # U_lb
    1,  # j_CO
    1,  # j_H2
    C_CO_lb / H_CO,  # C_CO_l
    C_H2_lb / H_H2,  # C_H2_l
    1,  # U_l
    1,  # R_CO
    1,  # R_H2
    c_to_p_ideal_gas(C_CO_lb / H_CO) * 10**-5,  # p_CO_l
    c_to_p_ideal_gas(C_H2_lb / H_H2) * 10**-5,  # p_H2_l
    0.5   # eps_slurry
])

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
    print('U_lb_out =\t', round(sol[0], 3))
    print('U_lb_out =\t', round(sol[3], 3))
    print('U_lb_out =\t', round(sol[6], 3))
    print('U_lb_out =\t', round(sol[9], 3))
    print('U_lb_out =\t', round(sol[12], 3))
    print('U_lb_out =\t', round(sol[15], 3))
    print('U_lb_out =\t', round(sol[18], 3))
    print('U_lb_out =\t', round(sol[21], 3))
    print('U_lb_out =\t', round(sol[24], 3))
    print('U_lb_out =\t', round(sol[27], 3))
    print('C_CO_l =  \t', round(sol[30], 3))
    print('C_H2_l =  \t', round(sol[31], 3))
    print('U_l =     \t', round(sol[32], 3))
    print('R_CO =     \t', round(sol[33], 3))
    print('R_H2 =     \t', round(sol[34], 3))
    print('p_CO =     \t', round(sol[35], 3))
    print('p_H2 =     \t', round(sol[36], 3))

    # define boundary inlet conditions for the next stage
    U_lb_previous_stage = sol[27]
    C_CO_l_previous_stage = sol[30]
    C_H2_l_previous_stage = sol[31]
    U_l_previous_stage = sol[32]

# sol_labels = ['U_lb1', 'j_CO1', 'j_H21', 'U_lb2', 'j_CO2', 'j_H22', 'C_CO_l', 'C_H2_l', 'U_l', 'R_CO', 'R_H2', 'p_CO_l', 'p_H2_l', 'eps_slurry']
# for i in range(len(sol_labels)):
#     print('%s =\t %s' % (sol_labels[i], round(sol[i],5)))

print('The final conversion is %s percent' % (round(1 - U_lb_previous_stage/U_in, 2)*100))


plt.figure()
plt.plot(0, U_in, 'o', color='blue')
for i in range(number_of_stages):
    for j in range(10):
        plt.plot(i + h * (j+1),stage_solutions[i][j*3], 'o', color='blue')
plt.xlabel('Height in m')
plt.ylabel('Superficial gas velocity in m/s')
plt.ylim(0.25,0.3)
plt.show()
exit()

plt.figure()
for i in range(5):
    plt.plot(i+H_stage/2,stage_solutions[i][30], 'o', color='blue')
plt.xlabel('Height in m')
plt.ylabel('CO slurry concentration in mol/m3')
plt.show()

plt.figure()
for i in range(5):
    plt.plot(i+H_stage/2,stage_solutions[i][31], 'o', color='blue')
plt.xlabel('Height in m')
plt.ylabel('H2 concentration in mol/m3')
plt.show()

plt.figure()
for i in range(5):
    plt.plot(i+H_stage/2,stage_solutions[i][32], 'o', color='blue')
plt.xlabel('Height in m')
plt.ylabel('CO reaction rate in mol/m3s')
plt.show()

plt.figure()
for i in range(5):
    plt.plot(i+H_stage/2,stage_solutions[i][34], 'o', color='blue')
plt.xlabel('Height in m')
plt.ylabel('H2 reaction rate in mol/m3s')
plt.show()


print(stage_solutions)
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