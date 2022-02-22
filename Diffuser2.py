import numpy as np
from GENozzle import GENozzle
import matplotlib.pyplot as plt
import pandas as pd

# Define Constant Values
k = 1.4
R_air = 287.058  # J/kgK

DataTable = pd.DataFrame(index=['M', 'Mns', 'T0', 'T', 'P0', 'P', 'Delta', 'Beta', 'Phi'],
                         columns=['1', '2', '3', '4', '5', '6'])



def beta2(delta, mach, n):
    # n = 0 for weak oblique shock, n = 1 for strong shock
    mu = np.arcsin(1 / mach)  # Mach wave angle
    b = -(((mach**2 + 2)/(mach**2)) + k*(np.sin(delta))**2)
    c = ((2*mach**2 + 1)/(mach**4)) + ((k+1)**2/(4) + (k-1)/(mach**2))*(np.sin(delta))**2
    d = -(np.cos(delta)**2/mach**4)
    v = (3*d - b**2)/9
    w = (9*b*c - 27*d - 2*b**2)/54
    D = v**3 + w**2
    phi = (1/3)*(np.arctan(np.sqrt(-D)/w))
    x_s = -b/3 + 2*np.sqrt(-v)*np.cos(phi)
    x_w = -b/3 - np.sqrt(-v)*(2*np.sqrt(-v)*np.cos(phi))
    
    if n == 0:
        B = np.arctan(np.sqrt(x_w/(1-x_w)))
    elif n == 1:
        B = np.arctan(np.sqrt(x_s/(1-x_s)))
    return B

    
# Delta-Beta-Mach Relation functions
def delta(beta, mach):
    delt = np.arctan(2*np.cot(beta)*((mach**2*(np.sin(beta)**2) - 1)/(mach**2*(k + 2*np.cos(2*beta) + 2))))
    return delt


def beta(delt, mach, n):
    # n = 0 for weak oblique shock, n = 1 for strong shock
    mu = np.arcsin(1 / mach) # Mach wave angle
    c = np.tan(mu) ** 2
    a = ((k - 1) / 2 + (k + 1) * c / 2) * np.tan(delt)
    b = ((k + 1) / 2 + (k + 3) * c / 2) * np.tan(delt)
    d = np.sqrt(4 * (1 - 3 * a * b) ** 3 / ((27 * a ** 2 * c + 9 * a * b - 2) ** 2) - 1)
    Beta = np.arctan((b + 9 * a * c) / (2 * (1 - 3 * a * b)) - (d * (27 * a ** 2 * c + 9 * a * b - 2)) / (
                6 * a * (1 - 3 * a * b)) * np.tan(n * np.pi / 3 + 1 / 3 * np.arctan(1 / d)))
    return Beta


# Isentropic expansion ratio functions
def ToTt(k, M):
    return (1 + ((k-1)/2)*M**2)**-1


def PoPt(k, M):
    return (1 + ((k-1)/2)*M**2)**(-k/(k-1))


def rhoorhot(k, M):
    return (1 + ((k-1)/2)*M**2)**(-1/(k-1))


def AoAt(k, M):
    return (((k+1)*0.5)**(-(k+1)/(2*(k-1))))*((1 + ((k-1)*0.5)*M**2)**((k+1)/(2*(k-1)))/M)


def P2oP1(k, M):
    return (2*k*M**2 - (k - 1))/(k + 1)


def rho2orho1(k, M):
    return ((k+1)*M**2)/((k - 1)*M**2 + 2)


def T2oT1(k, M):
    return ((2*k*M**2 - (k - 1))*((k - 1)*M**2 + 2))/((k+1)**2*M**2)


def Mach2(k, M):
    return ((k-1)*M**2 + 2)/(2*k*M**2 - (k-1))


def Pt20Pt1(k, M):
    return (((k+1)*M**2)/((k-1)*M**2 + 2))**(k/(k-1))*((k+1)/(2*k*M**2 - (k-1)))**(1/(k-1))


Me = 3
T1 = 222.2 #K
P1 = 3.5 # Kpa
k = 1.4
rho1 = 0.054879 # kg/m3
a1 = np.sqrt(k*R_air*T1)
v = Me*a1
p01 = (P1*1000 + 0.5*rho1*v**2)/1000
delt = np.deg2rad(8)

DataTable.at['Delta', '1'] = delt
DataTable.at['P', '1'] = P1
DataTable.at['P0', '1'] = p01
DataTable.at['T', '1'] = T1
DataTable.at['M', '1'] = Me





# Generate Mach 3 Nozzle using GE nozzle class
mesh_size = 75
radius_of_expansion = 1.4
Nozzle = GENozzle(k, Me, mesh_size, radius_of_expansion)
# Test section height and width
test_section_h_w = 1.9860        # m
# Test section length
test_section_length = 2.286
# get nozzle scaled throat height
throat_height = Nozzle.get_scaled_throat_height(test_section_h_w)
# Get nozzle points scaled to exit height
xnozz, ynozz = Nozzle.scale_wall_points(test_section_h_w)
# second throat height
throat2_height = throat_height/Pt20Pt1(1.4, Me) # m

plt.figure()
plt.axis('equal')
plt.grid()
plt.title('Mach 3 Wind Tunnel')
# plot the nozzle upper and lower contours
plt.plot(xnozz, -ynozz)
#plot the test section walls
plt.hlines(-ynozz[-1], xnozz[-1], xnozz[-1]+test_section_length)
plt.hlines(0, 0, 10, color='k')

# Save endpoints of test section walls
y0 = ynozz[-1]
x0 = xnozz[-1]+test_section_length


# Normal Shock at Mach 3 to get rough estimate for throat2 height
NS_relation = Pt20Pt1(1.4, Me)

# Calculate the height from the 'floor' to the bottom of the second throat
h2 = 0.5*test_section_h_w - 0.5*throat2_height
# Use that length and the wedge angle to find the length of the wedge section
L2 = h2/np.tan(delt)
# Plot the diffuser wedges
# x0, y0 = 0, 0
y1 = y0-h2
x1 = x0 + L2
# Plot the converging diffuser sections
plt.plot([x0, x0 + L2], [-y0, -y0 + h2])
plt.hlines((-y0 + h2), x1, 10, color='k')

# Calculate the first shockwaves
beta1 = beta(delt, Me, 0)
DataTable.at['Beta', '1'] = beta1


# Calculate the new properties across the first oblique shock, lower side
M1NS = Me*np.sin(beta1)
DataTable.at['Mns', '1'] = M1NS
M2NS = Mach2(k, M1NS)
P2 = P2oP1(k, M1NS)*P1
T2 = T2oT1(k, M1NS)*T1
p02_over_p01 = Pt20Pt1(k, M1NS)
p1_over_p01 = PoPt(k, Me)
T1_over_T01 = ToTt(k, Me)
p02 = (p02_over_p01/p1_over_p01)*P1
T02 = T1/T1_over_T01
M2 = M2NS/np.sin(beta1-delt)
DataTable.at['M', '2'] = M2
DataTable.at['Mns', '2'] = M2NS
DataTable.at['P0', '2'] = p02
DataTable.at['T0', '2'] = T02
DataTable.at['P', '2'] = P2
DataTable.at['T', '2'] = T2

# DataTable.at['Delta', '2'] = delt
# DataTable.at['Beta', '2'] =
# DataTable.at['Phi', '2'] =






# Calculate the angle of the first oblique shock
LS23 = (0.5*test_section_h_w)/np.tan(beta1)
# find the point at which the shocks interact
yA = 0
xA = x0 + LS23
# Plot the initial oblique shocks
plt.plot([x0, xA], [-y0, yA], linestyle='dashed')

# Calculate the new properties across the first oblique shock, lower side

# Section 2 to 3
M3NS = Mach2(k, M2NS)
P3 = P2oP1(k, M2NS)*P2
T3 = T2oT1(k, M2NS)*T2
p03 = (Pt20Pt1(k, M2NS)/PoPt(k, M2))*P2
T03 = T1/ToTt(k, M2)
beta2 = beta(delt, M2, 0)
M3 = M2NS/np.sin(beta2-delt)
PHI3 = beta2-delt
y3 = -y0
x3 = xA + (0.5*throat2_height)/np.tan(PHI3)
plt.plot([xA, x3], [0, -y1], linestyle='dashed')

# Section 3 to 4
# Section 2 to 3
M4NS = Mach2(k, M3NS)
p4 = P2oP1(k, M3NS)*P3
T4 = T2oT1(k, M3NS)*T3
p04 = (Pt20Pt1(k, M3NS)/PoPt(k, M3))*P3
T04 = T1/ToTt(k, M3)
beta3 = beta(delt, M3, 1)
M4 = M3NS/np.sin(beta3-delt)
PHI4 = beta3-delt
x4 = xA + (0.5*throat2_height)/np.tan(PHI4)
plt.plot([x3, x4], [-y1, 0], linestyle='dashed')

print(DataTable)
plt.show()
print()
