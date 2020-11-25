from numpy import *
import scipy.constants as const
import matplotlib.pyplot as plt
import sys
sys.path.append('../scripts/')
import sol_solvent as PS
from datetime import datetime
now_str = datetime.now().strftime("%H:%M (%d/%m/%Y)")

# fn = 'test_PAR_modi_DLP5000'
fn_cond = sys.argv[1]
fn_in = sys.argv[2]
fn_out = sys.argv[3]


exec(open(fn_cond).read())
dat = loadtxt(fn_in)
# fn_out = fn + '.pdf'

k_B = const.k                                                      # Boltzmann constant
kT = k_B*T                                                         # thermal energy

a_H = a_particle*gamma                                             # hydrodynamic radius
D0 = kT/(6.*pi*eta0*a_H)                                           # Stokes-Einstein-Sutherland
Va = (4./3.)*pi*a_particle**3.0                                    # volume measure is still using particle exclusion-size

k = 4.*sqrt(L_channel**2.0 * Lp * eta0 /R_channel**3.0)            # dimensionless parameter k

Pin = PS.get_Pin(DLP, ref_Pout)                                       # calculating Pin for the given DLP and Pout
Pper = PS.get_Pper(DLP, ref_DTP, k, ref_Pout)                         # calculating Pper for the given DLP, DTP_linear, k, and P_out

pre_cond = {'k':k, 'R':R_channel, 'L':L_channel, 'Lp':Lp, 'eta0':eta0}
cond_PS = PS.get_cond(pre_cond, Pin, ref_Pout, Pper)                  # allocating Blank Test (pure test) conditions

DTP_HP = (1/2.)*(Pin + ref_Pout) - Pper                            # length-averaged TMP with a linearly declined pressure approximation
vw0 = cond_PS['Lp']*DTP_HP                                         # v^\ast
epsilon_d = D0/(cond_PS['R']*vw0)                                  # 1/Pe_R


plt.close()
fig = plt.figure(figsize=(16,18))
plt.ion()

# First column
plt.subplot(321)
plt.plot(dat[:,0], dat[:,1], 'k-', linewidth=3)
plt.ylabel(r'$\phi_w$', fontsize=15)
plt.title(r'$\phi_{freeze} \approx $%.3f'%(phi_freeze), fontsize=15)

              
plt.subplot(323)
plt.plot(dat[:,0], dat[:,5], 'k-', linewidth=3)
plt.ylabel(r'$\Pi$ (Pa)', fontsize=15)


plt.subplot(325)
plt.plot(dat[:,0], dat[:,6], 'k-', linewidth=3)
plt.ylabel(r'$P-P_{perm} (Pa)$', fontsize=15)
plt.xlabel(r'$z$ (m)', fontsize=15)


# Second column
plt.subplot(322)
plt.plot(dat[:,0], dat[:,7], 'k-', linewidth=3)
plt.ylabel(r'$v_w/v^\ast$', fontsize=15)
plt.title(r'$v^\ast \approx $%4.3e (m/s)'%(cond_PS['vw0']), fontsize=15)

plt.subplot(324)
plt.plot(dat[:,0], dat[:,8], 'k-', linewidth=3)
plt.ylabel(r'$u_{max}/u^\ast$', fontsize=15)
plt.title(r'$u^\ast \approx $%4.3e (m/s)'%(cond_PS['u_HP']), fontsize=15)


plt.subplot(326)
plt.plot(dat[:,0], dat[:,9]/dat[0,9], 'k-', linewidth=3)
plt.ylabel(r'$\Phi/\Phi_0$  ', fontsize=15)
plt.xlabel(r'$z$ (m)', fontsize=15)
plt.title(r'$\max(\Phi/\Phi_0) - \min(\Phi/\Phi_0)$ = %4.3e (should be less than $\epsilon_\delta=%4.3e$)'%(max(dat[:,9]/dat[0,9]) - min(dat[:,9]/dat[0,9]), epsilon_d), fontsize=10)
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)


# fig.subplots_adjust(top=0.85)
st = plt.suptitle('Data plotted %s\nINPUT FILE = %s\nDATA FILE = %s\nOUTPUT GRAPH FILE = %s'%(now_str, fn_cond, fn_in, fn_out), fontsize=15)
st.set_x(0.1)
st.set_y(0.95)
st.set_horizontalalignment('left')
plt.savefig(fn_out, bbox_inches='tight')
# plt.show()
