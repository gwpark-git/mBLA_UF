
# Preface
The test in this folder aims to compare two different mono-dispersed hard spheres with radius a = 3.13nm and a=10nm. We tried to compare two different system in the similar Pe_R, which might have a different inlet velocity. Note that we aim to have Pe_R value for both systems about 78.

# condition for a=10nm
With the previous study (Park and N\"agele, JCP 2020), we uesed L_P = 6.7e-10 m/(Pa sec) and reference DTP was 5kPa. The corresponding inlet velocity is 0.022279 m/s, which, however, we adjust this value to u_inlet = 0.01625 m/s based on u_HP = R^2 DTP/(4 eta L) with DLP_linear = 130 Pa. In summary, the conditions are

## HF
(1) a = 10 nm
(2) DTP = 5 kPa
(3) u_inlet = 0.01625 m/s (corresponding Hagen-Poiseuille flow of DLP=130 Pa)
(4) L_P = 6.7e-10 m/(Pa sec)

## FMM: u_mean
Change only u_inlet = (3/4)u_inlet_HF, which is u_inlet = 0.01219. But the script will be used as (3/4)*0.01625. This corresponding the matching of mean-velocity at the inlet with HF.

## FMM: u_max
This matched u_inlet the same value with HF.

# condition for a=3.13nm with Pe_R~78
Now, we put the reduced radius 3.13nm, which makes 10/3.13~3.195 times higher D0 compared to a=10nm case. In order to keep the similar Pe_R values, we use the reference DTP as 16kPa, which gives us Pe_R about 78.3. Of course, we expect the particle has higher diffusivity whereas the viscosity does not really contribute to the system since the particle-size does not really contributed the viscosity, but volume fraction is. In summary, the parameters are

## HF
(1) a = 3.13 nm
(2) DTP = 16 kPa
(3) u_inlet = 0.1625 m/s (corresponding Hagen-Poiseuille flow of DLP=1300 Pa)
(4) L_P = 6.7e-10 m/(Pa sec)


## FMM: u_mean
Change only u_inlet = (3/4)u_inlet_HF, which is u_inlet = 0.1219. But the script will be used as (3/4)*0.1625

## FMM: u_max
This matched u_inlet the same value with HF.


# condition for a=3.13nm with Pi/DTP~0.5

