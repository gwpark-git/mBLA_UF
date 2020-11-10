from numpy import *
import matplotlib.pyplot as plt

dat_old = loadtxt('/Users/gunwoo/working_branch/code_mBLA_UF_collect/code_mBLA_UF_old/scripts/test.dat')
dat_new = loadtxt('test.dat')

plt.close()
plt.ion()
plt.plot(dat_old[:,0], dat_old[:,1], 'k-')
plt.plot(dat_new[:,0], dat_new[:,1], 'r.')
plt.show()
