from numpy import *
import matplotlib.pyplot as plt

dat = loadtxt('test_yt_arr.dat')
ind = range(0, size(dat[:,0]), 4)
plt.close()
plt.ion()
plt.plot(dat[ind,1], dat[ind,0], 'b.-')
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('arc length', fontsize=15)
plt.ylabel(r'$y/R$', fontsize=15)
plt.title('arc length vs yt_arr, stride=4 for visualization')
plt.savefig('arc_length_test.pdf', bbox_inches='tight')
plt.show()
