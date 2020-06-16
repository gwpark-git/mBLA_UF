from numpy import *
import matplotlib.pyplot as plt
import sys
# fn = 'test_PAR_modi_DLP5000'
fn_in = sys.argv[1]
fn_out = sys.argv[2]
dat = loadtxt(fn_in)
# fn_out = fn + '.pdf'

plt.close()
plt.figure(figsize=(16,10))
plt.ion()
plt.subplot(321)
plt.plot(dat[:,0], dat[:,1], 'k-')
plt.title(r'$\phi_w$', fontsize=15)

plt.subplot(323)
plt.plot(dat[:,0], dat[:,6], 'k-')
plt.title(r'$P-P_{perm}$', fontsize=15)

plt.subplot(322)
plt.plot(dat[:,0], dat[:,7], 'k-')
plt.title(r'$v_w/v^\ast$', fontsize=15)

plt.subplot(324)
plt.plot(dat[:,0], dat[:,8], 'k-')
plt.title(r'$u_{max}/u^\ast$', fontsize=15)

plt.subplot(325)
plt.plot(dat[:,0], dat[:,5], 'k-')
plt.title(r'$\Pi$', fontsize=15)

plt.subplot(326)
plt.plot(dat[:,0], dat[:,9]/dat[0,9], 'k-')
plt.title(r'$\Phi/\Phi_0$', fontsize=15)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.savefig(fn_out, bbox_inches='tight')
# plt.show()
