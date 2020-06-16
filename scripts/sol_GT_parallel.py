#############################################################################
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################

from sol_GT import *
from pathos.multiprocessing import Pool
from functools import partial

def get_new_phiw_div_phib_arr_parallel(cond_GT, Pi_arr, fcn_D, fcn_eta, z_arr, phiw_div_phib_arr, weight, gp_arr, gm_arr, yt_arr):
    # Special function for the parallel computation
    
    phiw_new_arr = zeros(size(phiw_div_phib_arr))
    phi_b = cond_GT['phi_bulk']
    ed = cond_GT['epsilon_d']
    dr = cond_GT['dr']
    R = cond_GT['R']; L = cond_GT['L']
    dyt = dr/R
    dyp = ed*dyt

    Ny = size(yt_arr)
    phi_arr_z0 = zeros(Ny)
    Ieta_arr_z0 = zeros(Ny)
    ID_arr_z0 = zeros(Ny)

    vw_div_vw0_z0 = get_vw_GT_boost(0, cond_GT, Pi_arr, gp_arr[0], gm_arr[0])/cond_GT['vw0']        

    get_phi_with_fixed_z_GT(0, cond_GT, phiw_div_phib_arr[0]*phi_b, Pi_arr, fcn_D, vw_div_vw0_z0, yt_arr, phi_arr_z0)
    get_int_eta_phi(0, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr_z0, Ieta_arr_z0)
    get_int_D_phi(0, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr_z0, ID_arr_z0)
    uZ0 = get_u_center_GT_boost(0, cond_GT, Pi_arr, fcn_D, fcn_eta,  phi_arr_z0[-1], Ieta_arr_z0[-1], gp_arr[0], gm_arr[0])
    F2_0 = get_F2(0, cond_GT, Pi_arr, fcn_D, fcn_eta, uZ0, yt_arr, phi_arr_z0, Ieta_arr_z0)
    pool = Pool()

    func = partial(process_at_z, cond_GT=cond_GT, z_arr=z_arr, yt_arr=yt_arr, Pi_arr=Pi_arr, fcn_D=fcn_D, fcn_eta=fcn_eta, phiw_div_phib_arr=phiw_div_phib_arr, phiw_new_arr=phiw_new_arr, F2_0=F2_0, weight=weight, gp_arr=gp_arr, gm_arr=gm_arr)
    Nz = size(z_arr)
    phiw_new_arr = asarray(pool.map(func, range(Nz)))
    pool.close()
    pool.join()
    return phiw_new_arr#, uZ0, uZz, F1, F2_Z, F2_0# , bar_eta_with_fixed_z

def get_new_phiw_div_phib_modi_arr_parallel(cond_GT, Pi_arr, fcn_D, fcn_eta, z_arr, phiw_div_phib_arr, weight, gp_arr, gm_arr, yt_arr):
    import time
    phi_b = cond_GT['phi_bulk']
    phiw_new_arr = ones(size(phiw_div_phib_arr)) # normalized condition
    
    ed = cond_GT['epsilon_d']
    dr = cond_GT['dr']
    R = cond_GT['R']; L = cond_GT['L']
    dyt = dr/R
    dyp = ed*dyt

    Ny = size(yt_arr)
    phi_arr_z0 = zeros(Ny)
    Ieta_arr_z0 = zeros(Ny)
    ID_arr_z0 = zeros(Ny)
    
    vw_div_vw0_z0 = get_vw_GT_boost(0, cond_GT, Pi_arr, gp_arr[0], gm_arr[0])/cond_GT['vw0']        

    # phi_arr_z0 = phi_b*ones(size(yt_arr))
    get_phi_modi_with_fixed_z_GT(0, cond_GT, phiw_div_phib_arr[0]*phi_b, Pi_arr, fcn_D, vw_div_vw0_z0, yt_arr, phi_arr_z0)
    get_int_eta_phi(0, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr_z0, Ieta_arr_z0)
    get_int_D_phi(0, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr_z0, ID_arr_z0)
    uZ0 = get_u_center_GT_boost(0, cond_GT, Pi_arr, fcn_D, fcn_eta,  phi_arr_z0[-1], Ieta_arr_z0[-1], gp_arr[0], gm_arr[0])
    F2_0 = get_F2(0, cond_GT, Pi_arr, fcn_D, fcn_eta, uZ0, yt_arr, phi_arr_z0, Ieta_arr_z0)
    # F2_0 = uZ0/4.
    F2_0_modi = get_F_modi(0, cond_GT, Pi_arr, fcn_D, fcn_eta, uZ0, vw_div_vw0_z0, yt_arr, phi_arr_z0, Ieta_arr_z0, ID_arr_z0)
    # F2_0_modi = 0.
    # print 'F2_0, F2_0_expected, F2_0/F2_0_expted', F2_0, uZ0/4., F2_0/(uZ0/4.)
    # print 'F2_0_modi', F2_0_modi, 0
    # print 'F2_0_modi ', F2_0_modi, ' is forced to be zero'
    # F2_0 = uZ0/4.
    # F2_0_modi = 0.
    pool = Pool()
    func = partial(process_at_z_modi, cond_GT=cond_GT, z_arr=z_arr, yt_arr=yt_arr, Pi_arr=Pi_arr, fcn_D=fcn_D, fcn_eta=fcn_eta, phiw_div_phib_arr=phiw_div_phib_arr, phiw_new_arr=phiw_new_arr, F2_0=F2_0, F2_0_modi=F2_0_modi, weight=weight, gp_arr=gp_arr, gm_arr=gm_arr)
    Nz = size(z_arr)
    phiw_new_arr[1:] = asarray(pool.map(func, range(1,Nz)))

    pool.close()
    pool.join()
    return phiw_new_arr

