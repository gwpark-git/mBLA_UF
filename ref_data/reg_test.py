from numpy import *
import matplotlib.pyplot as plt

dat_new_Ny200 = loadtxt('test_arclength_Ny200.dat')
dat_new_Ny100 = loadtxt('test_arclength_Ny100.dat')

dat_ref_FEM = loadtxt('result_FEM.dat', skiprows=8)

dat_v1 = loadtxt('result_v1.dat')
dat_v2 = loadtxt('result_v2.dat')

dat_cpp_test = loadtxt('result_cpp_pre_test.dat')

N_dat = size(dat_ref_FEM[:,0])
ind_FEM = range(0, N_dat, int(N_dat/30.))

from scipy.interpolate import interp1d

int_f = interp1d(dat_cpp_test[:,0], dat_cpp_test[:,1])
int_old = interp1d(dat_v1[:,0], dat_v1[:,1])
int_new = interp1d(dat_v2[:,0], dat_v2[:,1])
int_test_Ny200 = interp1d(dat_new_Ny200[:,0], dat_new_Ny200[:,1])
int_test_Ny100 = interp1d(dat_new_Ny100[:,0], dat_new_Ny100[:,1])
z_ref = dat_ref_FEM[ind_FEM,0]

int_f_ref = int_f(z_ref)
int_old_ref = int_old(z_ref)
int_new_ref = int_new(z_ref)
int_test_Ny200_ref = int_test_Ny200(z_ref)
int_test_Ny100_ref = int_test_Ny100(z_ref)

N_f = size(dat_cpp_test[:,0])
ind_f = range(0, N_f, int(N_f/30.))
L_channel = 0.5 
plt.close()
plt.ion()
plt.figure(figsize=(8,12))
plt.subplot(211)
plt.plot(dat_ref_FEM[ind_FEM,0]/L_channel, dat_ref_FEM[ind_FEM,1], 'ko', markerfacecolor='white', markeredgecolor='black', label = 'FEM')

plt.plot(z_ref/L_channel, int_old_ref, 'b-', label = 'mBLA Python code v1.0 (used in manuscript)')
plt.plot(z_ref/L_channel, int_new_ref, 'r-', label = 'mBLA Python code v2.0')
plt.plot(z_ref/L_channel, int_f(z_ref), 'k.', label = 'mBLA C++ code (not published yet, used FDM for ODE)')
plt.plot(dat_new_Ny200[:,0]/L_channel, dat_new_Ny200[:,1], 'g-', label = 'refined yt_arr, Ny=200')
plt.plot(dat_new_Ny100[:,0]/L_channel, dat_new_Ny100[:,1], 'm-', label = 'refined yt_arr, Ny=100')
plt.legend(loc = 'lower right')
plt.xlabel(r'$\tilde{z}=z/L$', fontsize=15)
plt.ylabel(r'$\phi_w(\tilde{z})$', fontsize=15)
plt.title('regression test after revising Python code')

plt.subplot(212)
plt.plot(z_ref/L_channel, (dat_ref_FEM[ind_FEM, 1] - int_old_ref), 'b-', label = 'mBLA Python code v1.0 (used in manuscript)')
plt.plot(z_ref/L_channel, (dat_ref_FEM[ind_FEM, 1] - int_new_ref), 'r-', label = 'mBLA Python code v2.0')
plt.plot(z_ref/L_channel, (dat_ref_FEM[ind_FEM, 1] - int_f(z_ref)), 'k.', label = 'mBLA C++ code (not published yet, used FDM for ODE)')
plt.plot(z_ref/L_channel, (dat_ref_FEM[ind_FEM, 1] - int_test_Ny200(z_ref)), 'g-', label = 'refined yt_arr, Ny=200')
plt.plot(z_ref/L_channel, (dat_ref_FEM[ind_FEM, 1] - int_test_Ny100(z_ref)), 'm-', label = 'refined yt_arr, Ny=100')
plt.legend(loc = 'upper right', fontsize=8)
plt.xlabel(r'$\tilde{z}=z/L$', fontsize=15)
plt.ylabel(r'$\phi_w^{(FEM)}(\tilde{z}) - \phi_w^{(mBLA)}(\tilde{z})$', fontsize=15)
plt.title('regression test after revising Python code')
plt.grid()


plt.savefig('reg_test_result.pdf', bbox_inches='tight')
plt.show()
