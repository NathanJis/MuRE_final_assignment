# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:10:46 2021
@author: natha
"""
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

#####################################################################################################
# design variables and execution options
# design variable definition
number_of_stages = 5  # number of stages
height_reactor = 10  # reactor diameter in m
diam_reactor = 1  # reactor diameter in m
P = 30 * 1e5  # pressure in Pa
T = 240 + 273.15  # temperature in K
dp = 50 * 1e-6  # catalyst particle size in m
eps_cat = 0.35  # catalyst concentration in vol_cat/vol_slurry
feed_ratio = 2  # ratio of C_H2/C_CO
feed_mass_flow_rate = 2e6  # reactor feed mass flow rate of CO in kg/day
# code settings
n_slices = 100  # number of slices
n_slice_eqs = 4  # number of equations solved in the differential slices
plot_everything = False  # decide if everything or just the superficial gas velocities should be plotted
residual_sum_tol = 10e-4  # value above which a simulation is deemed failed
do_the_general_simulation = True  # decide if the first initial simulation needs to be done (can be skipped if only the sensitivity analysis is necessary)
do_the_sensitivity_analysis = False  # see above
# inlet conditions
C_CO_l_inlet = 0  # 0 for fresh slurry
C_H2_l_inlet = 0  # 0 for fresh slurry
U_l_inlet = 0  # 0 for semi-batch


# define functions
def p_to_c_ideal_gas(p):
    # input in mol/m3, output in Pa
    return p / R / T


def c_to_p_ideal_gas(c):
    # input in mol/m3, output in Pa
    return c * R * T


def get_inlet_conditions():
    return U_in, C_CO_l_inlet, C_H2_l_inlet, U_l_inlet


def calculate_dependent_parameters():
    # calculate dependent parameters
    h = height_reactor / number_of_stages / n_slices  # finite difference step size
    # bubble concentrations
    C_CO_lb = p_to_c_ideal_gas(P / (1 + feed_ratio))
    C_H2_lb = p_to_c_ideal_gas(P / (1 + 1 / feed_ratio))
    # kinetic parameters
    a = a0 * np.exp(Ea / R * (1 / 493.15 - 1 / T))  # langmuir hinshelwood parameter 1 mol/s/kgcat/bar2
    b = b0 * np.exp(deltabH / R * (1 / 493.15 - 1 / T))  # langmuir hinshelwood parameter 2 in 1/bar
    # hydrodynamic parameters
    A_reactor = np.pi * diam_reactor ** 2 / 4  # reactor cross section area in m2
    H_stage = height_reactor / number_of_stages  # stage heigh in m
    U_in = feed_mass_flow_rate / (24*3600) / M_CO * 3 * (R * T / P) / A_reactor  # inlet superficial gas velocity in m/s
    print(U_in)
    U_small = 1  # superficial gas velocity for small bubbles in m/s
    # slurry parameters
    rho_sl = rho_solvent * (
                1 - rho_solvent / rho_skel * eps_cat) + rho_cat * eps_cat  # slurry density in kg/m3 (maretto2009, 23)
    visco_sl = visco_solvent * (1 + 4.5 * eps_cat)  # slurry viscosity in Pas (maretto2009, 24)
    return h, C_CO_lb, C_H2_lb, a, b, A_reactor, H_stage, U_in, U_small, rho_sl, visco_sl


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
    lis.append((c_to_p_ideal_gas(C_CO_lb / H_CO) * 1e-5))  # p_CO_l
    lis.append((c_to_p_ideal_gas(C_H2_lb / H_H2) * 1e-5))  # p_H2_l
    return np.array(lis)


def simulate_reactor():
    global initial_guess, U_lb_previous_stage, C_CO_l_previous_stage, C_H2_l_previous_stage, U_l_previous_stage  # this line is necessary to make it easier to run the simulation with just one line
    stage_solutions = []
    for stage_counter in range(number_of_stages):
        # sol, output, ier, mesg = optimize.fsolve(AES, initial_guess, full_output=True)
        # print(ier)
        # print(mesg)
        root = optimize.root(AES, initial_guess)
        sol = root.x
        stage_solutions.append(sol)  # save solutions of this stage
        initial_guess = sol  # the initial guess for the next stage

        print('Results stage %s:' % (stage_counter + 1))
        # print('Solution vector', sol)
        # print('Residuals', AES(sol))
        residual_sum = sum(AES(sol))
        print('sum of residuals = ', residual_sum)
        # print('U_lb_out =\t', round(sol[n_slices * n_slice_eqs - n_slice_eqs], 3))
        # print('C_CO_l =  \t', round(sol[n_slices * n_slice_eqs], 3))
        # print('C_H2_l =  \t', round(sol[n_slices * n_slice_eqs + 1], 3))
        # print('U_l =     \t', round(sol[n_slices * n_slice_eqs + 2], 3))

        # define boundary inlet conditions for the next stage
        U_lb_previous_stage = sol[n_slices * 4 - 4]
        C_CO_l_previous_stage = sol[n_slices * 4]
        C_H2_l_previous_stage = sol[n_slices * 4 + 1]
        U_l_previous_stage = sol[n_slices * 4 + 2]
        if residual_sum > residual_sum_tol:
            print('The sum of residuals is kinda high. Seems like there was a problem in the calculation')
            break

    final_conversion = 1 - U_lb_previous_stage / U_in
    print('The final conversion is %s percent' % round(final_conversion * 100,2))
    return stage_solutions, final_conversion


def calculate_differential_slice(U_lb_slice_in, U_lb_slice_out, j_CO, j_H2, eps_sl, C_CO_l, C_H2_l):
    mass_balance_gas_residual = (U_lb_slice_out - U_lb_slice_in) / h + (j_CO + j_H2) / (C_CO_lb + C_H2_lb)
    mass_transfer_CO_residual = j_CO - (0.3 / diam_reactor**0.18 * U_lb_slice_out ** 0.58 * np.sqrt(D_CO / D_ref) * (C_CO_lb / H_CO - C_CO_l))
    mass_transfer_H2_residual = j_H2 - (0.3 / diam_reactor**0.18 * U_lb_slice_out ** 0.58 * np.sqrt(D_H2 / D_ref) * (C_H2_lb / H_H2 - C_H2_l))
    slurry_holdup_residual = eps_sl - (1 - 0.3 / diam_reactor**0.18 * U_lb_slice_out ** 0.58)
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
            slice_residuals = calculate_differential_slice(x[i * 4 - 4], x[i * n_slice_eqs], x[i * n_slice_eqs + 1], x[i * n_slice_eqs + 2], x[i * n_slice_eqs + 3], C_CO_l, C_H2_l)
        residuals.append(slice_residuals[0])
        residuals.append(slice_residuals[1])
        residuals.append(slice_residuals[2])
        residuals.append(slice_residuals[3])

    # create sums
    sum_j_CO = sum(x[1:n_slices * n_slice_eqs:n_slice_eqs])
    sum_j_H2 = sum(x[2:n_slices * n_slice_eqs:n_slice_eqs])
    sum_eps_sl = sum(x[3:n_slices * n_slice_eqs:n_slice_eqs])

    # slurry phase mass balances
    residuals.append(h * sum_j_CO + C_CO_l_previous_stage * U_l_previous_stage - C_CO_l * U_l - R_CO * h * sum_eps_sl)
    residuals.append(h * sum_j_H2 + C_H2_l_previous_stage * U_l_previous_stage - C_H2_l * U_l - R_H2 * h * sum_eps_sl)
    # constituent equations
    residuals.append(U_l - (U_l_previous_stage + (h * sum_eps_sl * (R_CO * M_CO + R_H2 * M_H2) / rho_solvent)))
    residuals.append(R_CO - (F * a * p_H2_l * p_CO_l / (1 + b * p_CO_l) ** 2 * eps_cat * rho_skel))
    residuals.append(R_H2 - (2 * R_CO))
    residuals.append(p_CO_l - (H_CO * C_CO_l * R * T * 1e-5))
    residuals.append(p_H2_l - (H_H2 * C_H2_l * R * T * 1e-5))

    return np.array(residuals)


def plot_results(stage_solution_vector):
    # sort variables into arrays for easier plotting
    U_lb_array, j_CO_array, j_H2_array, eps_sl_array, C_CO_l_array, C_H2_l_array, U_l_array, R_CO_array, R_H2_array, p_CO_l_array, p_H2_l_array, slice_position_array = [
        [], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(number_of_stages):

        # sort slice variables
        for j in range(n_slices):
            slice_position_array.append(i * height_reactor / number_of_stages + h * (j + 1))
            U_lb_array.append(stage_solutions[i][j * 4])
            j_CO_array.append(stage_solutions[i][j * 4 + 1])
            j_H2_array.append(stage_solutions[i][j * 4 + 2])
            eps_sl_array.append(stage_solutions[i][j * 4 + 3])

        # sort stage variables
        C_CO_l_array.append(stage_solutions[i][n_slices * n_slice_eqs])
        C_H2_l_array.append(stage_solutions[i][n_slices * n_slice_eqs + 1])
        U_l_array.append(stage_solutions[i][n_slices * n_slice_eqs + 2])
        R_CO_array.append(stage_solutions[i][n_slices * n_slice_eqs + 3])
        R_H2_array.append(stage_solutions[i][n_slices * n_slice_eqs + 4])
        p_CO_l_array.append(stage_solutions[i][n_slices * n_slice_eqs + 5])
        p_H2_l_array.append(stage_solutions[i][n_slices * n_slice_eqs + 6])

    # plot superficial gas velocities and conversion
    plt.figure()
    plt.plot(0, U_in, 'o', color='blue')
    plt.plot(slice_position_array, U_lb_array, 'o', color='blue')
    plt.xlabel('Height in m')
    plt.ylabel('Superficial gas velocity in m/s')
    plt.show()

    plt.figure()
    plt.plot(slice_position_array, 1 - np.array(U_lb_array)/U_in, 'o', color='blue')
    plt.xlabel('Height in m')
    plt.ylabel('Conversion')
    plt.show()

    # print other variables
    if plot_everything:
        # plot slice variables
        ylabels = [r'$j_{H2}$ in mol/m3s', r'$j_{H2}$ in mol/m3s', 'Slurry holdup in m3slurry/m3reactor']
        sorted_results = [j_CO_array, j_H2_array, eps_sl_array]
        for i in range(len(ylabels)):
            plt.figure()
            plt.plot(slice_position_array, sorted_results[i], 'o', color='blue')
            plt.xlabel('Height in m')
            plt.ylabel(ylabels[i])
            plt.show()

        # plot stage variables
        ylabels = ['CO slurry concentration in mol/m3', 'H2 slurry concentration in mol/m3',
                   'Exit superficial liquid velocity in m/s', 'CO reaction rate in mol/m3slurry/s',
                   'H2 reaction rate in mol/m3slurry/s', 'CO slurry partial pressure in bar',
                   'H2 slurry partial pressure in bar']
        sorted_results = [C_CO_l_array, C_H2_l_array, U_l_array, R_CO_array, R_H2_array, p_CO_l_array, p_H2_l_array]
        for i in range(len(ylabels)):
            plt.figure()
            plt.plot(np.linspace(1, number_of_stages, number_of_stages, endpoint=True), sorted_results[i], 'o-',
                     color='blue')
            plt.xlabel('Stage number')
            plt.ylabel(ylabels[i])
            plt.show()


def get_conversion_vector(stage_solution_vector):
    U_lb_array = []
    slice_position_array = []
    for i in range(number_of_stages):

        # sort slice variables
        for j in range(n_slices):
            slice_position_array.append(i * height_reactor / number_of_stages + h * (j + 1))
            U_lb_array.append(stage_solutions[i][j * 4])
    conversion = 1 - np.array(U_lb_array) / U_in
    return conversion, slice_position_array


#####################################################################################################
# parameter and dependent variable definition

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


#####################################################################################################
# simulation
if do_the_general_simulation:
    h, C_CO_lb, C_H2_lb, a, b, A_reactor, H_stage, U_in, U_small, rho_sl, visco_sl = calculate_dependent_parameters()
    U_lb_previous_stage, C_CO_l_previous_stage, C_H2_l_previous_stage, U_l_previous_stage = get_inlet_conditions()
    initial_guess = generate_initial_guess(n_slices)
    stage_solutions, conversion = simulate_reactor()
    plot_results(stage_solutions)

################################################################
# sensitivity analysis
fig_counter = 1
if do_the_sensitivity_analysis:
    # define original values (useful for easy reverting to the original value)
    original_number_of_stages, original_height_reactor, original_diam_reactor, original_eps_cat = [number_of_stages, height_reactor, diam_reactor, eps_cat]

    ###################################################################################################################
    # number of stages
    number_of_stages_sens_an = [number_of_stages - 2, number_of_stages - 1, number_of_stages, number_of_stages + 1, number_of_stages + 2]
    final_conversions, conversion_vector_array, slice_position_vector_array = [[],[],[]]
    for i in range(len(number_of_stages_sens_an)):
        print('Simulating %s stages' % number_of_stages_sens_an[i])
        number_of_stages = number_of_stages_sens_an[i]

        # simulate the reactor
        h, C_CO_lb, C_H2_lb, a, b, A_reactor, H_stage, U_in, U_small, rho_sl, visco_sl = calculate_dependent_parameters()
        U_lb_previous_stage, C_CO_l_previous_stage, C_H2_l_previous_stage, U_l_previous_stage = get_inlet_conditions()
        initial_guess = generate_initial_guess(n_slices)
        stage_solutions, final_conversion = simulate_reactor()
        conversion_vector, slice_position_vector = get_conversion_vector(stage_solutions)

        # get results
        final_conversions.append(final_conversion)
        conversion_vector_array.append(conversion_vector)
        slice_position_vector_array.append(slice_position_vector)
    # plot results
    plt.figure()
    plt.plot(number_of_stages_sens_an, final_conversions, 'o')
    plt.xlabel('Number of stages in reactor')
    plt.ylabel('Final conversion')
    plt.savefig('Figure%s' % fig_counter, dpi=500)
    fig_counter += 1
    plt.show()
    plt.figure()
    for i in range(len(number_of_stages_sens_an)):
        plt.plot(slice_position_vector_array[i],conversion_vector_array[i],label='%s stages in reactor'%number_of_stages_sens_an[i])
    plt.xlabel('Reactor height in m')
    plt.ylabel('Conversion')
    plt.legend()
    plt.savefig('Figure%s' % fig_counter, dpi=500)
    fig_counter += 1
    plt.show()
    # set the original value back
    number_of_stages = original_number_of_stages

    ###################################################################################################################
    # reactor height
    reactor_height_sens_an = [height_reactor - 2, height_reactor - 1, height_reactor, height_reactor + 1, height_reactor + 2]
    final_conversions, conversion_vector_array, slice_position_vector_array = [[],[],[]]
    for i in range(len(reactor_height_sens_an)):
        print('Simulating %s reactor height' % reactor_height_sens_an[i])
        height_reactor = reactor_height_sens_an[i]

        # simulate the reactor
        h, C_CO_lb, C_H2_lb, a, b, A_reactor, H_stage, U_in, U_small, rho_sl, visco_sl = calculate_dependent_parameters()
        U_lb_previous_stage, C_CO_l_previous_stage, C_H2_l_previous_stage, U_l_previous_stage = get_inlet_conditions()
        initial_guess = generate_initial_guess(n_slices)
        stage_solutions, final_conversion = simulate_reactor()
        conversion_vector, slice_position_vector = get_conversion_vector(stage_solutions)

        # get results
        final_conversions.append(final_conversion)
        conversion_vector_array.append(conversion_vector)
        slice_position_vector_array.append(slice_position_vector)
    # plot results
    plt.figure()
    plt.plot(reactor_height_sens_an, final_conversions, 'o')
    plt.xlabel('Total reactor height in m')
    plt.ylabel('Final conversion')
    plt.savefig('Figure%s' % fig_counter, dpi=500)
    fig_counter += 1
    plt.show()
    plt.figure()
    for i in range(len(reactor_height_sens_an)):
        plt.plot(slice_position_vector_array[i], conversion_vector_array[i],
                 label='%s m total reactor height' % reactor_height_sens_an[i])
    plt.xlabel('Reactor height in m')
    plt.ylabel('Conversion')
    plt.legend()
    plt.savefig('Figure%s' % fig_counter, dpi=500)
    fig_counter += 1
    plt.show()
    # set the original value back
    height_reactor = original_height_reactor

    ###################################################################################################################
    # reactor diameter
    reactor_diameter_sens_an = [diam_reactor - 2, diam_reactor - 1, diam_reactor, diam_reactor + 1, diam_reactor + 2]
    final_conversions, conversion_vector_array, slice_position_vector_array = [[],[],[]]
    for i in range(len(reactor_diameter_sens_an)):
        print('Simulating %s reactor diameter' % reactor_diameter_sens_an[i])
        diam_reactor = reactor_diameter_sens_an[i]

        # simulate the reactor
        h, C_CO_lb, C_H2_lb, a, b, A_reactor, H_stage, U_in, U_small, rho_sl, visco_sl = calculate_dependent_parameters()
        U_lb_previous_stage, C_CO_l_previous_stage, C_H2_l_previous_stage, U_l_previous_stage = get_inlet_conditions()
        initial_guess = generate_initial_guess(n_slices)
        stage_solutions, final_conversion = simulate_reactor()
        conversion_vector, slice_position_vector = get_conversion_vector(stage_solutions)

        # get results
        final_conversions.append(final_conversion)
        conversion_vector_array.append(conversion_vector)
        slice_position_vector_array.append(slice_position_vector)
    # plot results
    plt.figure()
    plt.plot(reactor_diameter_sens_an, final_conversions, 'o')
    plt.xlabel('Reactor diameter in m')
    plt.ylabel('Final conversion')
    plt.savefig('Figure%s' % fig_counter, dpi=500)
    fig_counter += 1
    plt.show()
    plt.figure()
    for i in range(len(reactor_diameter_sens_an)):
        plt.plot(slice_position_vector_array[i], conversion_vector_array[i],
                 label='%s m reactor diameter' % reactor_diameter_sens_an[i])
    plt.xlabel('Reactor height in m')
    plt.ylabel('Conversion')
    plt.legend()
    plt.savefig('Figure%s' % fig_counter, dpi=500)
    fig_counter += 1
    plt.show()
    # set the original value back
    diam_reactor = original_diam_reactor

    ###################################################################################################################
    # catalyst concentration
    eps_cat_sens_an = [eps_cat * 0.8, eps_cat * 0.9, eps_cat, eps_cat * 1.1, eps_cat * 1.2]
    final_conversions, conversion_vector_array, slice_position_vector_array = [[],[],[]]
    for i in range(len(eps_cat_sens_an)):
        print('Simulating %s catalyst concentration' % eps_cat_sens_an[i])
        eps_cat = eps_cat_sens_an[i]

        # simulate the reactor
        h, C_CO_lb, C_H2_lb, a, b, A_reactor, H_stage, U_in, U_small, rho_sl, visco_sl = calculate_dependent_parameters()
        U_in, U_small,_previous_stage, C_CO_l_previous_stage, C_H2_l_previous_stage, U_l_previous_stage = get_inlet_conditions()
        initial_guess = generate_initial_guess(n_slices)
        stage_solutions, final_conversion = simulate_reactor()
        conversion_vector, slice_position_vector = get_conversion_vector(stage_solutions)

        # get resutls
        final_conversions.append(final_conversion)
        conversion_vector_array.append(conversion_vector)
        slice_position_vector_array.append(slice_position_vector)
    # plot results
    plt.figure()
    plt.plot(eps_cat_sens_an, final_conversions, 'o')
    plt.xlabel('Catalyst concentration')
    plt.ylabel('Final conversion')
    plt.savefig('Figure%s' % fig_counter, dpi=500)
    fig_counter += 1
    plt.show()
    plt.figure()
    for i in range(len(eps_cat_sens_an)):
        plt.plot(slice_position_vector_array[i], conversion_vector_array[i], label='%s catalyst concentration' % round(eps_cat_sens_an[i],3))
    plt.xlabel('Reactor height in m')
    plt.ylabel('Conversion')
    plt.legend()
    plt.savefig('Figure%s' % fig_counter, dpi=500)
    fig_counter += 1
    plt.show()
    # set the original value back
    eps_cat = original_eps_cat