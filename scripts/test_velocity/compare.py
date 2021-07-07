from numpy import *
import matplotlib.pyplot as plt

dat_BCP = loadtxt('BCP_test.dat')
dat_BCu = loadtxt('BCu_test_uin_from_BCP.dat')

dat_BCP_ref = loadtxt('../../ref_data/result_v2.dat')

plt.close()
plt.ion()
ind_plot = 4
# plt.plot(dat_BCP_ref[:,0], dat_BCP_ref[:,ind_plot], 'k.')
plt.plot(dat_BCP[:,0], dat_BCP[:,ind_plot], 'b-', linewidth=1)
# plt.plot(dat_BCu[:,0], dat_BCu[:,ind_plot], 'r-', linewidth=1)



plt.show()
