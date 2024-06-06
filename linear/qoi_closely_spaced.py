import numpy as np
import os, sys; sys.path.append(os.path.join('..'))
from util import plt

# example from Sec 4 page 78:
# https://projecteuclid.org/journals/annals-of-statistics/volume-35/issue-1/Methodology-and-convergence-rates-for-functional-linear-regression/10.1214/009053606000000957.full

J = 60 - 1
al = 1.1

qoi = np.zeros(J)

qoi[0] = 1
for j in range(1, 4):
    jj = j + 1
    qoi[j] = 0.2*(-1.)**(jj + 1)*(1. - 0.0001*jj)
    
for k in range(4 + 1):
    for j in range(1, int(np.ceil((J - 4)/5)) + 1):
        qoi[5*j + k -1] = 0.2*(-1)**(5*j+k+1)*((5*j)**(-al/2) - 0.0001*k)

plt.semilogy(np.arange(1, J+1), np.abs(qoi), 'bo')