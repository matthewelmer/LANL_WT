import numpy as np
from GENozzle import GENozzle
import matplotlib.pyplot as plt
import pandas as pd

# Define Constant Values
k = 1.4
R_air = 287.058  # J/kgK

DataTable = pd.DataFrame(index=['M', 'Mns', 'T0', 'T', 'P0', 'P', 'Delta', 'Beta', 'Phi'],
                         columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])



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
T1 = 225 #K
P1 = 3.5 # Kpa
k = 1.4
rho1 = 0.0542 # kg/m3
# Me = 5
# T1 = 154 #K
# P1 = 3.0 # Kpa
# k = 1.4
# rho1 = 0.0679 # kg/m3
# Me = 7
# T1 = 86 #K
# P1 = 1.5 # Kpa
# k = 1.4
# rho1 = 0.0608 # kg/m3
# Me = 10
# T1 = 50 #K
# P1 = 0.5 # Kpa
# k = 1.4
# rho1 = 0.0348 # kg/m3
a1 = np.sqrt(k*R_air*T1)
v = Me*a1
p01 = (P1*1000 + 0.5*rho1*v**2)/1000
delt = np.deg2rad(9)

DataTable.at['Delta', '1'] = np.rad2deg(delt)
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

# Calculate the first shockwaves
beta1 = beta(delt, Me, 0)
DataTable.at['Beta', '1'] = np.rad2deg(beta1)
# Calculate the new properties across the first oblique shock, lower side
# Calculate the normal mach number
M1NS = Me*np.sin(beta1)
DataTable.at['Mns', '1'] = M1NS
# Mach number behind shock
M2 = Mach2(k, M1NS)/np.sin(beta1-delt)
DataTable.at['M', '2'] = M2
# Pressure and Temperature behind shock
P2 = P2oP1(k, M1NS)*P1
T2 = T2oT1(k, M1NS)*T1
DataTable.at['P', '2'] = P2
DataTable.at['T', '2'] = T2
# Total Pressure and temperature behind shock
p02 = ( Pt20Pt1(k, M1NS)/PoPt(k, Me))*P1
T02 = T1/ToTt(k, Me)
DataTable.at['P0', '2'] = p02
DataTable.at['T0', '2'] = T02
# Delta stays the same
DataTable.at['Delta', '2'] = np.rad2deg(delt)
# Calculate the angle of the first oblique shock
LS23 = (0.5*test_section_h_w)/np.tan(beta1)
# find the point at which the shocks interact
yA = 0
xA = x0 + LS23
# Plot the initial oblique shocks
plt.plot([x0, xA], [-y0, yA], linestyle='dashed')




# Section 2 to 3
# Beta for the first shock reflection (shock going back down)
beta2 = beta(delt, M2, 0)
DataTable.at['Beta', '2'] = np.rad2deg(beta2)
# Normal mach number
M2NS = M2*np.sin(beta2)
DataTable.at['Mns', '2'] = M2NS
# Mach number behind shock
M3 = Mach2(k, M2NS)/np.sin(beta2-delt)
DataTable.at['M', '3'] = M3
# Calculate temp and pressure after the shock
P3 = P2oP1(k, M2NS)*P2
T3 = T2oT1(k, M2NS)*T2
DataTable.at['P', '3'] = P3
DataTable.at['T', '3'] = T3
# Total temp and pressure after shock
p03 = (Pt20Pt1(k, M2NS)/PoPt(k, M2))*P2
T03 = T1/ToTt(k, M2)
DataTable.at['P0', '3'] = p03
DataTable.at['T0', '3'] = T03
# Angle that shock makes with the centerline
PHI2 = beta2-delt
DataTable.at['Phi', '2'] = np.rad2deg(PHI2)
# Plot second shock wave
y3 = -y0
x3 = xA + (0.5*throat2_height)/np.tan(PHI2)
plt.plot([xA, x3], [0, -y1], linestyle='dashed')
# Delta angle stays the same
DataTable.at['Delta', '3'] = np.rad2deg(delt)




# Section 3 to 4 ( third shock wave, reflection back up
beta3 = beta(0.01, M3, 0)
DataTable.at['Beta', '3'] = np.rad2deg(beta3)
# Normal mach number
M3NS = M3*np.sin(beta3)
DataTable.at['Mns', '3'] = M3NS
# Mach number after shock
M4 = Mach2(k, M3NS)/np.sin(beta3)
DataTable.at['M', '4'] = M4
# Temp and pressure after shock
P4 = P2oP1(k, M3NS)*P3
T4 = T2oT1(k, M3NS)*T3
DataTable.at['P', '4'] = P4
DataTable.at['T', '4'] = T4
# total temp and pressure after shock
p04 = (Pt20Pt1(k, M3NS)/PoPt(k, M3))*P3
T04 = T1/ToTt(k, M3)
DataTable.at['P0', '4'] = p04
DataTable.at['T0', '4'] = T04
# angle shock makes with lower wall
PHI3 = beta3
DataTable.at['Phi', '3'] = np.rad2deg(PHI3)
# Plot shock
x4 = x3 + (0.5*throat2_height)/np.tan(PHI3)
plt.plot([x3, x4], [-y1, 0], linestyle='dashed')
# Delta stays the same
DataTable.at['Delta', '4'] = 0


# Section 4 to 5 ( Fourth shock wave, reflection down again)
beta4 = beta(0.01, M4, 0)
DataTable.at['Beta', '4'] = np.rad2deg(beta4)
# Normal mach number
M4NS = M4*np.sin(beta4)
DataTable.at['Mns', '4'] = M4NS
# Mach number after shock
M5 = Mach2(k, M4NS)/np.sin(beta4)
DataTable.at['M', '5'] = M5
# Temp and pressure after shock
P5 = P2oP1(k, M4NS)*P4
T5 = T2oT1(k, M4NS)*T4
DataTable.at['P', '5'] = P5
DataTable.at['T', '5'] = T5
# total temp and pressure after shock
p05 = (Pt20Pt1(k, M4NS)/PoPt(k, M4))*P4
T05 = T1/ToTt(k, M4)
DataTable.at['P0', '5'] = p05
DataTable.at['T0', '5'] = T05
# angle shock makes with lower wall
PHI4 = beta4
DataTable.at['Phi', '4'] = np.rad2deg(PHI4)
# Plot shock
x5 = x4 + (0.5*throat2_height)/np.tan(PHI4)
plt.plot([x4, x5], [0, -y1], linestyle='dashed')
# Delta stays the same
DataTable.at['Delta', '5'] = 0



# Section 5 to 6 ( Fifth shock wave, reflection up again)
beta5 = beta(0.01, M5, 0)
DataTable.at['Beta', '5'] = np.rad2deg(beta5)
# Normal mach number
M5NS = M5*np.sin(beta5)
DataTable.at['Mns', '5'] = M5NS
# Mach number after shock
M6 = Mach2(k, M5NS)/np.sin(beta5)
DataTable.at['M', '6'] = M6
# Temp and pressure after shock
P6 = P2oP1(k, M5NS)*P5
T6 = T2oT1(k, M5NS)*T5
DataTable.at['P', '6'] = P6
DataTable.at['T', '6'] = T6
# total temp and pressure after shock
p06 = (Pt20Pt1(k, M5NS)/PoPt(k, M5))*P5
T06 = T1/ToTt(k, M5)
DataTable.at['P0', '6'] = p06
DataTable.at['T0', '6'] = T06
# angle shock makes with lower wall
PHI5 = beta5
DataTable.at['Phi', '5'] = np.rad2deg(PHI5)
# Plot shock
x6 = x5 + (0.5*throat2_height)/np.tan(PHI5)
plt.plot([x5, x6], [-y1, 0], linestyle='dashed')
# Delta stays the same
DataTable.at['Delta', '6'] = 0




# Section 6 to 7 (sixth shock wave, reflection up again)
beta6 = beta(0.01, M6, 0)
DataTable.at['Beta', '6'] = np.rad2deg(beta6)
# Normal mach number
M6NS = M6*np.sin(beta6)
DataTable.at['Mns', '6'] = M6NS
# Mach number after shock
M7 = Mach2(k, M6NS)/np.sin(beta6)
DataTable.at['M', '7'] = M7
# Temp and pressure after shock
P7 = P2oP1(k, M5NS)*P6
T7 = T2oT1(k, M5NS)*T6
DataTable.at['P', '7'] = P7
DataTable.at['T', '7'] = T7
# total temp and pressure after shock
p07 = (Pt20Pt1(k, M6NS)/PoPt(k, M6))*P6
T07 = T1/ToTt(k, M6)
DataTable.at['P0', '7'] = p07
DataTable.at['T0', '7'] = T07
# angle shock makes with lower wall
PHI6 = beta6
DataTable.at['Phi', '6'] = np.rad2deg(PHI6)
# Plot shock
x7 = x6 + (0.6*throat2_height)/np.tan(PHI6)
plt.plot([x6, x7], [0, -y1], linestyle='dashed')
# Delta stays the same
DataTable.at['Delta', '7'] = 0






# Section 7 to 8 ( seventh shock wave, reflection up again)
beta7 = beta(0.01, M7, 0)
DataTable.at['Beta', '7'] = np.rad2deg(beta7)
# Normal mach number
M7NS = M7*np.sin(beta7)
DataTable.at['Mns', '7'] = M7NS
# Mach number after shock
M8 = Mach2(k, M7NS)/np.sin(beta7)
DataTable.at['M', '8'] = M8
# Temp and pressure after shock
P8 = P2oP1(k, M5NS)*P7
T8 = T2oT1(k, M5NS)*T7
DataTable.at['P', '8'] = P8
DataTable.at['T', '8'] = T8
# total temp and pressure after shock
p08 = (Pt20Pt1(k, M7NS)/PoPt(k, M7))*P7
T08 = T1/ToTt(k, M7)
DataTable.at['P0', '8'] = p08
DataTable.at['T0', '8'] = T08
# angle shock makes with lower wall
PHI7 = beta7
DataTable.at['Phi', '7'] = np.rad2deg(PHI7)
# Plot shock
x8 = x7 + (0.7*throat2_height)/np.tan(PHI7)
plt.plot([x7, x8], [-y1, 0], linestyle='dashed')
# Delta stays the same
DataTable.at['Delta', '8'] = 0




# Section 8 to 9 ( eighth shock wave, reflection up again)
beta8 = beta(0.01, M8, 0)
DataTable.at['Beta', '8'] = np.rad2deg(beta8)
# Normal mach number
M8NS = M8*np.sin(beta8)
DataTable.at['Mns', '8'] = M8NS
# Mach number after shock
M9 = Mach2(k, M8NS)/np.sin(beta8)
DataTable.at['M', '9'] = M9
# Temp and pressure after shock
P9 = P2oP1(k, M5NS)*P8
T9 = T2oT1(k, M5NS)*T8
DataTable.at['P', '9'] = P9
DataTable.at['T', '9'] = T9
# total temp and pressure after shock
p09 = (Pt20Pt1(k, M8NS)/PoPt(k, M8))*P8
T09 = T1/ToTt(k, M8)
DataTable.at['P0', '9'] = p09
DataTable.at['T0', '9'] = T09
# angle shock makes with lower wall
PHI8 = beta8
DataTable.at['Phi', '8'] = np.rad2deg(PHI8)
# Plot shock
x9 = x8 + (0.8*throat2_height)/np.tan(PHI8)
plt.plot([x8, x9], [0, -y1], linestyle='dashed')
# Delta stays the same
DataTable.at['Delta', '9'] = 0




# Section 9 to 10 ( Ninth shock wave, reflection up again)
beta9 = beta(0.01, M9, 0)
DataTable.at['Beta', '9'] = np.rad2deg(beta9)
# Normal mach number
M9NS = M9*np.sin(beta9)
DataTable.at['Mns', '9'] = M9NS
# Mach number after shock
M10 = Mach2(k, M9NS)/np.sin(beta9)
DataTable.at['M', '10'] = M10
# Temp and pressure after shock
P10 = P2oP1(k, M5NS)*P9
T10 = T2oT1(k, M5NS)*T9
DataTable.at['P', '10'] = P10
DataTable.at['T', '10'] = T10
# total temp and pressure after shock
p010 = (Pt20Pt1(k, M9NS)/PoPt(k, M9))*P9
T010 = T1/ToTt(k, M9)
DataTable.at['P0', '10'] = p010
DataTable.at['T0', '10'] = T010
# angle shock makes with lower wall
PHI9 = beta9
DataTable.at['Phi', '9'] = np.rad2deg(PHI9)
# Plot shock
x10 = x9 + (0.9*throat2_height)/np.tan(PHI9)
plt.plot([x9, x10], [-y1, 0], linestyle='dashed')
# Delta stays the same
DataTable.at['Delta', '10'] = 0

plt.hlines(0, 0, x10, color='k')
plt.hlines((-y0 + h2), x1, x10, color='k')



print(DataTable)


########## Plot the full diffuser ##################
plt.figure()
y0 = ynozz[-1]
x0 = 0
x1 = x0 + L2
# Plot the converging diffuser section
plt.plot([x0, x1], [-y0, -y0 + h2], color='k')
plt.plot([x0, x1], [y0, y0 - h2], color='k')


#plot the first oblique shock
xA = x0 + LS23
plt.plot([x0, xA], [-y0, yA], linestyle='dashed')
plt.plot([x0, xA], [y0, yA], linestyle='dashed')

# Plot the second oblique shock
x3 = xA + (0.5*throat2_height)/np.tan(PHI2)
plt.plot([xA, x3], [0, -y1], linestyle='dashed')
plt.plot([xA, x3], [0, y1], linestyle='dashed')

# Plot the third oblique shock
x4 = x3 + (0.5*throat2_height)/np.tan(PHI3)
plt.plot([x3, x4], [-y1, yA], linestyle='dashed')
plt.plot([x3, x4], [y1, yA], linestyle='dashed')

# Plot the fourth oblique shock
x5 = x4 + (0.5*throat2_height)/np.tan(PHI4)
plt.plot([x4, x5], [yA, -y1], linestyle='dashed')
plt.plot([x4, x5], [yA, y1], linestyle='dashed')

# Plot the fifth oblique shock
x6 = x5 + (0.5*throat2_height)/np.tan(PHI5)
plt.plot([x5, x6], [-y1, yA], linestyle='dashed')
plt.plot([x5, x6], [y1, yA], linestyle='dashed')

# Plot the sixth oblique shock
x7 = x6 + (0.5*throat2_height)/np.tan(PHI6)
plt.plot([x6, x7], [yA, -y1], linestyle='dashed')
plt.plot([x6, x7], [yA, y1], linestyle='dashed')

# Plot the seventh oblique shock
x8 = x7 + (0.5*throat2_height)/np.tan(PHI7)
plt.plot([x7, x8], [-y1, yA], linestyle='dashed')
plt.plot([x7, x8], [y1, yA], linestyle='dashed')

# Plot the eigth oblique shock
x9 = x8 + (0.5*throat2_height)/np.tan(PHI8)
plt.plot([x8, x9], [yA, -y1], linestyle='dashed')
plt.plot([x8, x9], [yA, y1], linestyle='dashed')

# Plot the Ninth oblique shock
x10 = x9 + (0.5*throat2_height)/np.tan(PHI9)
plt.plot([x9, x10], [-y1, yA], linestyle='dashed')
plt.plot([x9, x10], [y1, yA], linestyle='dashed')

# Plot the constant area diffuser section
plt.hlines((-y0 + h2), x1, x10, color='k')
plt.hlines((y0 - h2), x1, x10, color='k')

print(x10)
# plt.title('Mach 3 Diffuser (incomplete)')


plt.show()
print()
