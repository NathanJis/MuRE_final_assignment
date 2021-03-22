# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:05:27 2021

@author: natha
"""
import numpy as np

sizes = [20e-6,40e-6,60e-6,80e-6,100e-6,120e-6]
kla = 0.6

rhos =2648 #siliciumdioxide
D=3e-9
for size in sizes:
    V = np.pi*(1/6)*size**3
    A= np.pi*size**2
    km = kla*V/A
    thiele = (V/A)*np.sqrt((km*rhos/D))
    #print(thiele)
    
rhog = 1.98
eg = 0.3
es = 0.35
rhol= 900
el = 1-eg-es
g =9.81
dp = (rhos*es+rhol*el+rhog*eg)*g #barghi
dps = -(1-eg)*9.81*(rhos*es+rhol*el)/(es+el) #schneider
print(dp,dps)

Dh_oil = 4e-9 #hydrogen diffusion in transformer oil as baseline
#maybe we can use dahmkohler to prove we can neglect liquid-solid, but i don't know if calculation is right
for size in sizes:
    Da_d = km*size/2
    Da_c = km**size**2/Dh_oil
    #print(Da_d, Da_c)
T =400
R =8.314
ccog0 = 297
chg0= 604    
Da = 3.24e-8*(((ccog0*R*T)*0.33)**1.88*((chg0*R*T)*0.66)**-0.67)/kla
print(Da)

diam = 6
flow = 2000e3 #tonne per day
vol_flow_day = flow/rhol #m3/day
vol_flow = vol_flow_day/(24*60*60) #m3/s
liq_area = el*(np.pi/4)*diam**2 #m2
v_liq = vol_flow/liq_area #m/s
print(v_liq)

