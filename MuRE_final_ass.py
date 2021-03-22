# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:23:54 2021

@author: natha
"""

#paper 1 = Design and optimisation of a multi-stage bubble column slurry reactor for Fischer-Tropsch synthesis
#paper 2  = Modeling of a slurry bubble column reactor for Fischer-Tropsch synthesis
#conc in liquid is way too hard here

from scipy import integrate 
import numpy as np 
import matplotlib.pyplot as plt 

def diff(x,init):
    ccog,chg, ccol, chl= init
    dccogdz = (-diffCO*big_area*(ccog-ccol)*num_big)/v_big_bub -(diffCO*small_area*(ccog-ccol)*num_small)/v_small_bub
    dchgdz = (-diffH*big_area*(chg-chl)*num_big)/v_big_bub -(diffH*small_area*(chg-chl)*num_small)/v_small_bub
    dccoldz = -dccogdz - arrhenius*(((ccog*R*T)*0.33)**1.88*((chg*R*T)*0.66)**-0.67) #paper 2
    dchldz = -dchgdz - arrhenius*(((ccog*R*T)*0.33)**1.88*((chg*R*T)*0.66)**-0.67) # paper 2
    return dccogdz, dchgdz, dccoldz,dchldz

diffH = 3.81e-7 #from data companion
diffCO = 1.60e-7 #from data companion
height = 5 #made up this is height per stage
diam = 7 #made up
P = 3000000 #paper 1
R = 8.314

bub_split = 0.7 #i made this up juet to make an artificial balance between small and large bubbles
e = 0.4 #voidage also made up for now
big_bub_diam = 45e-3 #paper 1
small_bub_diam = 7e-3 #paper 1
v_big_bub =2 #paper 1
v_small_bub = 4 #made up
big_area = np.pi*big_bub_diam
small_area = np.pi*small_bub_diam
num_big = (bub_split*(np.pi/4)*diam**2*e)/((1/6)*np.pi*big_bub_diam**3) #to calc total area of exchange in ODE
num_small = ((1-bub_split)*(np.pi/4)*diam**2*e)/((1/6)*np.pi*small_bub_diam**3) #to calc total area of exchange in ODE


T = 400 # paper 1
arrhenius = 5.04e6*np.exp((-108.67e3)/(8.314*T)) #paper 2

ccog0 = 0.33*(P/(R*T))
chg0 = 0.67*(P/(R*T))
init = [ccog0,chg0,0,0]
z = np.linspace(0,height,100)

 


sol = integrate.solve_ivp(diff,[0,height], init, method='BDF', t_eval=z)

plt.figure(1)
plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1])
plt.figure(2)
plt.plot(sol.t,sol.y[2])
plt.plot(sol.t,sol.y[3])
init1 = [sol.y[0][-1], sol.y[1][-1],sol.y[2][-1], sol.y[3][-1]]

for i in range(1): #this increases stages
    sol = integrate.solve_ivp(diff,[0,height], init1, method='BDF', t_eval=z)
    plt.figure(1)
    plt.plot(sol.t,sol.y[0])
    plt.plot(sol.t,sol.y[1])
    plt.figure(2)
    plt.plot(sol.t,sol.y[2])
    plt.plot(sol.t,sol.y[3])
    init1 = [sol.y[0][-1], sol.y[1][-1],sol.y[2][-1], sol.y[3][-1]]





