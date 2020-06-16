#############################################################################
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################


from numpy import *

# generation condition structure with the given system condition
def get_cond(pre_cond, Pin, Pout, Pper):
    k = pre_cond['k']
    
    DLP = Pin - Pout
    DTP_HP = (1/2.)*(Pin + Pout) - Pper
    DTP = DTP_HP*2.*tanh(k/2.)/k
    vw0 = pre_cond['Lp']*DTP_HP
    
    print 'Pin, Pout, Pper in Pa : ', Pin, Pout, Pper
    print 'DLP, DTP, DTP_HP in Pa : ', DLP, DTP, DTP_HP
    
    Cp = get_Cpm(k, +1.0, Pin, Pout, Pper)
    Cm = get_Cpm(k, -1.0, Pin, Pout, Pper)
    print 'Cp, Cm : ', Cp, Cm
    cond = {'k':pre_cond['k'], 'Cp':Cp, 'Cm':Cm, 'Pin':Pin, 'Pout':Pout, 'Pper':Pper,\
           'R':pre_cond['R'], 'L':pre_cond['L'], 'Lp':pre_cond['Lp'], 'eta0':pre_cond['eta0'],
            'preU':pre_cond['preU'], 'vw0':vw0}    
    return cond

## coefficients
def get_Cpm(k, pm, P_in, P_out, P_per):
    return pm*(P_out - P_per - (P_in - P_per)*exp(-pm*k))/(2.*sinh(k))

## solutions
def get_P(r, z, cond):
    k=cond['k']; Pper = cond['Pper']; Cp = cond['Cp']; Cm = cond['Cm']; L = cond['L']
    return Pper + Cp*exp(k*z/L) + Cm*exp(-k*z/L)

def get_u(r, z, cond):
    k=cond['k']; Pper = cond['Pper']; Cp = cond['Cp']; Cm = cond['Cm']
    L = cond['L']; R = cond['R']; preU=cond['preU']
    return -preU * (1 - (r/R)**2.0)*(Cp*exp(k*z/L) - Cm*exp(-k*z/L))

def get_v(r, z, cond):
    k=cond['k']; Pper = cond['Pper']; Cp = cond['Cp']; Cm = cond['Cm']
    L = cond['L']; R = cond['R']; preU=cond['preU']; Lp=cond['Lp']    
    return Lp*(2.*(r/R) - (r/R)**3.0)*(Cp*exp(k*z/L)+Cm*exp(-k*z/L))


# different operating conditions

def get_Pin(DLP, Pout):
    return DLP + Pout

def get_Pper(DLP, DTP, k, Pout):
    return (1./2.)*DLP - DTP/(2.*tanh(k/2.)/k) + Pout
