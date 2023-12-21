# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:03:48 2023

@author: James
"""

"""
Analysis of fluctuations in the Gross-Witten Wadia matrix model.

We find the correlation function or green's function who's diagonal elements give the density of eigenvalues
of the matrix model. We subtract off the average density and study the fluctuations about this.
We are able to find excellent fit with an ansatz derived from instanton-considerations.
"""

import numpy as np
from scipy.special import iv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import toeplitz
L =16 #define L, the number of eigenvalues
d = 101 #number of data points

#generating data in k space
x = np.linspace(-np.pi/10, np.pi/10, d);

#generating range of 
R = np.linspace(L/8,L/3,d)

#define green's function in k-space. It is made up of the Toeplitz determinant of Bessel functions, iv.
def g(k,x):
    a = [(iv(i, 2*(x))*2 - iv(i+1, 2*(x))*np.exp(-1j*k) - iv(i-1, 2*(x))*np.exp(1j*k)) for i in range(L-1)]
    m = toeplitz(a)
    return np.linalg.det(m)*np.exp(2*x*np.cos(k)-x**2)

#By symmetry, the peak in the fluctuations of the density occurs at x = 0 or at the d//2 value. 
#We find this peak for a range of the control parameter, R, and use this data to fit our ansatz.
q = []
for i in range(len(R)):
    h = []
    for n in range(len(x)):
        h.append((np.real(g(x[n],R[i])) - (L+ (2*R[i])*np.cos(x[n])))) #fluctuations = green function-average density.
    q.append(np.abs(h[d//2])) #append peak
 
#Ansatz for what the amplitude of the fluctuations looks like, with one degree of freedom to fit, b.
def func(x, b):
    c=L/(2*x)
    return  b*np.exp(-2*x*(c*np.log(c+np.sqrt(c**2 -1)) - np.sqrt(c**2 -1)))*np.sqrt(np.pi/np.sqrt(c**2 -1))

popt, pcov = curve_fit(func, R, q)

#Plotting data which shows a good fit over a large range, suggesting the ansatz is correct
plt.semilogy(R, func(R, *popt),'r-',label='fit: b =%5.3f' % tuple(popt))
plt.semilogy(R,q,  label = 'Oscillatory numeric')
plt.legend()
plt.show()

