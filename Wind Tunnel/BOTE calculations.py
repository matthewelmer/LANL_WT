import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


########### CONSTANTS #############
# Ratio of specific heats
kair = 1.4
kHe = 1.66

# Specific Gas Constants
Rair = 53.353         # ft * lbf/lbm*{deg R}
RHe = 386.047	       # ft * lbf/lbm*{deg R}


############# ALTITUDE #################
# Import Standard Atmosphere table https://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
StdAtmo = pd.read_csv('StdAtmos.csv')

# Determine ambient pressure in the test section for upper and lower bounds of density altitude
def AmbientConditions(h:float, Atmo:pd.DataFrame):
    # Given an altitude, determine the Temperature, Acc. of Gravity, Abs pressure, Density, Dynamic Viscosity
    #h	t	g	p	ρ	μ  (ft)	(Deg F)	(ft/s2)	(lbf/in2)	(10-4 lbm/ft2)	(10-7 lbm s/ft2)
    n = Atmo.to_numpy()
    x = 0
    # find the two rows in the table to interpolate from
    for i in range(len(n)-1):
        if (h > n[i, 0]) & (h < n[i+1, 0]):
            x = i

    # Create a list for each quantity with the table values to interplolate from
    alt, temp, grav, press, dens, visc = [n[x, 0], n[x+1, 0]], [n[x, 1], n[x+1, 1]], [n[x, 2], n[x+1, 2]], [n[x, 3], n[x+1, 3]], [n[x, 4], n[x+1, 4]], [n[x, 5], n[x+1, 5]]

    # Interpolate properties at the given altitude
    T = np.interp(h, alt, temp)
    g = np.interp(h, alt, grav)
    p = np.interp(h, alt, press)
    rho = np.interp(h, alt, dens)
    mu = np.interp(h, alt, visc)
    return h, p, rho


# use funciton to get values for minimum and max altitude
hl, pl, rhol = AmbientConditions(7300, StdAtmo)
hu, pu, rhou = AmbientConditions(11000, StdAtmo)



############### ISENTROPIC EXPANSION ###################
# isentropic expansion ratios
pr = lambda M, k: (1 + (k-1)*0.5*M**2)**(-k/(k-1))
Tr = lambda M, k: (1 + (k-1)*0.5*M**2)**(-1)
denr = lambda M, k: (1 + (k-1)*0.5*M**2)**(-1/(k-1))

# Ratios for Mach 3
M = 3
d = np.array([[kair, pr(M, kair), Tr(M, kair), denr(M, kair)],
              [kHe, pr(M, kHe), Tr(M, kHe), denr(M, kHe)]])
expansionratiosM3 = pd.DataFrame(data=d, columns=['Gamma', 'Pressure', 'Temperature', 'Density'])
print('\nExpansion to Mach 3')
print(expansionratiosM3)

# Ratios for Mach 5
M = 5
d = np.array([[kair, pr(M, kair), Tr(M, kair), denr(M, kair)],
              [kHe, pr(M, kHe), Tr(M, kHe), denr(M, kHe)]])
expansionratiosM5 = pd.DataFrame(data=d, columns=['Gamma', 'Pressure', 'Temperature', 'Density'])
print('\nExpansion to Mach 5')
print(expansionratiosM5)

# Ratios for Mach 7
M = 7
d = np.array([[kair, pr(M, kair), Tr(M, kair), denr(M, kair)],
              [kHe, pr(M, kHe), Tr(M, kHe), denr(M, kHe)]])
expansionratiosM7 = pd.DataFrame(data=d, columns=['Gamma', 'Pressure', 'Temperature', 'Density'])
print('\nExpansion to Mach 7')
print(expansionratiosM7)

# Ratios for Mach 10
M = 10
d = np.array([[kair, pr(M, kair), Tr(M, kair), denr(M, kair)],
              [kHe, pr(M, kHe), Tr(M, kHe), denr(M, kHe)]])
expansionratiosM10 = pd.DataFrame(data=d, columns=['Gamma', 'Pressure', 'Temperature', 'Density'])
print('\nExpansion to Mach 10')
print(expansionratiosM10)


################ MACH AREA RELATION #################
# Area ratio
AoverA_t = lambda M, k: (((k+1)*0.5)**(-(k+1)/(2*(k-1))))*((1 + ((k-1)*0.5)*M**2)**((k+1)/(2*(k-1)))/M)

# Mach number range
Mach = np.arange(1, 10.1, 0.01)
# Set specific heat ratio
k = kair

# Plot Mach - Area Relation
fig = plt.figure()
plt.plot(Mach, AoverA_t(Mach, k))
plt.xlabel('Exit Mach Number')
plt.ylabel(r'$\frac{A_{e}}{A_{t}}$', size=20)
plt.title('Ratio of Exit Area over Throat Area for given Exit Mach Number \n Isentropic Expansion for Air')
# Plot dashed lines to show target mach
plt.hlines([AoverA_t(3, k), AoverA_t(5, k), AoverA_t(7, k), AoverA_t(10, k)], [1, 1, 1, 1], [3, 5, 7, 10], linestyles='--')
plt.vlines([3, 5, 7, 10], [0, 0, 0, 0], [AoverA_t(3, k), AoverA_t(5, k), AoverA_t(7, k), AoverA_t(10, k)], linestyles='--')
# Set plot bounds
plt.ylim([0, AoverA_t(10.1, k)])
plt.xlim([1, 10.1])
plt.xticks([1, 3, 5, 7, 10])
plt.yticks([AoverA_t(3, k), AoverA_t(5, k), AoverA_t(7, k), AoverA_t(10, k)])
fig.set_size_inches(7, 8)
#plt.savefig('Area Exit to Throat Ratio, Air', dpi=300, bbox_inches='tight')
plt.show()

