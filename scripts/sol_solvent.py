#############################################################################
#   Flow profile of pure solvent flow in CF UF                              #
#   This is the solution of pure solvent flow described in Sec. IV A        #
#   of the paper described below.                                           #
#                                                                           #
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################


from numpy import *


def get_cond(pre_cond, Pin, Pout, Pper): # conditions for pure solvent flow
    """ return dictionary "cond" of conditions for pure solvent flow (Sec. IV A)
    dictionary "cond":
    inherite from dictionary "pre_cond" parameter described below.
    'Ap' : [WIP]
    
    Parameters:
        pre_cond = {'k':k, 'R':R_channel, 'L':L_channel, 'Lp':Lp, 'eta0':eta0, 'preU':prefactor_U}
            'k'    : system parameter k in Eq. (26)   in the dimensionless unit
            'R'    : radius of membrane channel       in the unit of m
            'Lp'   : solvent permeability on the clean membrane in Eqs. (12) and (13) 
                                                      in the unit of m/(Pa sec)
            'eta0' : pure solvent viscosity at the operating temperature 
                                                      in the unit of Pa sec
            'preU' : auxiliary conversion factor sqrt(Lp*R/eta0) bet. dimensionless quantities  
                                                      in the dimensionless unit
        Pin      = Pressure inlet boundary condition  in the unit of Pa
        Pout     = Pressure outlet boundary condition in the unit of Pa
        Pper     = Pressure in permeate which affect to Darcy-Starling law in Eq. (12) 
                                                      in the unit of Pa
    
    """
    COND_TYPE = 'PS' # describe this condition dictionary is type of PS (pure solvent)
    
    k = pre_cond['k'] 
    
    DLP = Pin - Pout                    # Longitudinal pressure difference
    DTP_HP = (1/2.)*(Pin + Pout) - Pper # Length-averaged TMP with linear pressure approximation in Eq. (8)
    DTP = DTP_HP*2.*tanh(k/2.)/k        # Length-averaged TMP in Eq. (7)
    vw0 = pre_cond['Lp']*DTP_HP         # v^\ast in Eq. (21)
    alpha_ast = DTP_HP/DLP              # alpha^\ast in Eq. (23)
    beta_ast = k**2.0 * alpha_ast       # beta^\ast in Eq. (24) and (26)
    u_HP = pre_cond['R']**2.0 * DLP/(4.*pre_cond['eta0']*pre_cond['L'])
    
    print ('Pin, Pout, Pper in Pa : ', Pin, Pout, Pper)
    print ('DLP, DTP, DTP_HP in Pa : ', DLP, DTP, DTP_HP)
    
    Ap = get_Apm(+1.0, k, alpha_ast, Pper/DLP)
    Am = get_Apm(-1.0, k, alpha_ast, Pper/DLP)
    print ('A+ = %4.3e, A- = %4.3e'%(Ap, Am))
    cond = {'k':pre_cond['k'], 'Ap':Ap, 'Am':Am, 'Pin':Pin, 'Pout':Pout, 'Pper':Pper, 'DLP':DLP,\
           'R':pre_cond['R'], 'L':pre_cond['L'], 'Lp':pre_cond['Lp'], 'eta0':pre_cond['eta0'], \
            'u_HP':u_HP, 'vw0':vw0, 'alpha_ast':alpha_ast, 'beta_ast':beta_ast,\
            'Pper_div_DLP':Pper/DLP, 'COND':COND_TYPE}
    return cond


def get_Apm(pm, k, alpha_ast, Pper_div_DLP):
    """ Get dimensionless Apm using Eq. (32)
    """
    return pm*(1./(4.*sinh(k)))*(2.*alpha_ast - 1. - (2.*alpha_ast + 1)*exp(-pm*k))

def get_Apm_conv(pm, cond):
    """ Convinience version for get_Apm using cond
    """
    return get_Apm(pm, cond['k'], cond['alpha_ast'], cond['Pper_div_DLP'])

def get_P(z_div_L, k, Ap, Am, Pper_div_DLP):
    """ Using Eq. (31) (the first expression)
    """
    return Pper_div_DLP + Ap*exp(k*z_div_L) + Am*exp(-k*z_div_L)

def get_P_conv(z_div_L, cond):
    return get_P(z_div_L, cond['k'], cond['Ap'], cond['Am'], cond['Pper_div_DLP'])

# def get_P(r, z, cond):
#     k=cond['k']; Pper = cond['Pper']; Ap = cond['Ap']; Am = cond['Am']; L = cond['L']
#     return cond['Pper']/cond['DLP'] + cond['Ap']*exp(k*z/L) + cond['Am']*exp(-k*z/L)

def get_u(r_div_R, z_div_L, k, Ap, Am):
    """ Using Eq. (31) (the second expression)
    """
    uR_HP = 1. - r_div_R**2.0
    uZ_PS = -k*(exp( k*z_div_L)*Ap - exp(-k*z_div_L)*Am)
    return uZ_PS*uR_HP

def get_u_conv(r_div_R, z_div_L, cond):
    return get_u(r_div_R, z_div_L, cond['k'], cond['Ap'], cond['Am'])


def get_v(r_div_R, z_div_L, k, alpha_ast, Ap, Am):
    """ Using Eq. (31) (the third expression)
    """
    sign = +1.
    vR = 2.*r_div_R - r_div_R**3.0
    vw =(exp( k*z_div_L)*Ap + exp(-k*z_div_L)*Am)/alpha_ast

    return sign*vR*vw

def get_v_conv(r_div_R, z_div_L, cond):
    return get_v(r_div_R, z_div_L, cond['k'], cond['alpha_ast'], cond['Ap'], cond['Am'])


def get_Pin(DLP, Pout):
    return DLP + Pout

def get_Pper(DLP, DTP, k, Pout):
    return (1./2.)*DLP - DTP/(2.*tanh(k/2.)/k) + Pout
