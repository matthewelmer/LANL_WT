import numpy as np
from GENozzle import GENozzle
import matplotlib.pyplot as plt

# Define Constant Values
k = 1.4
Me = 3
R_air = 287.058  # J/kgK
delt = np.deg2rad(10)


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



# Generate Mach 3 Nozzle using GE nozzle class
mesh_size = 75
radius_of_expansion = 1.4
Nozzle = GENozzle(k, Me, mesh_size, radius_of_expansion)
# Test section height and width
test_section_h_w = 58*0.0254        # m
# Test section length
test_section_length = 90*0.0254
# get nozzle scaled throat height
throat_height = Nozzle.get_scaled_throat_height(test_section_h_w)
# Get nozzle points scaled to exit height
xnozz, ynozz = Nozzle.scale_wall_points(test_section_h_w)


# Plot the nozzle from the GENozzle Class
plt.figure()
plt.axis('equal')
plt.grid()
plt.title('Mach 3 Wind Tunnel')
# plot the nozzle upper and lower contours
plt.plot(xnozz, ynozz)
plt.plot(xnozz, -ynozz)
#plot the test section walls
plt.hlines(ynozz[-1], xnozz[-1], xnozz[-1]+test_section_length)
plt.hlines(-ynozz[-1], xnozz[-1], xnozz[-1]+test_section_length)
# Save endpoints of test section walls
y0 = ynozz[-1]
x0 = xnozz[-1]+test_section_length



# Assume the following test section conditions
# Region 1
p1 = 3.5  # kPa
T1 = 222.2   # K
M = 3
rho1 = 0.054879 # kg/m3
a1 = np.sqrt(k*R_air*T1)
v = Me*a1
p01 = (p1*1000 + 0.5*rho1*v**2)/1000
# Normal Shock at Mach 3 to get rough estimate for throat2 height
NS_relation = Pt20Pt1(1.4, M)
# use total pressure ratio for a mach 3 normal shock and calculate the height of throat 2
throat2_height = throat_height/Pt20Pt1(1.4, M) # m
# Calculate the height from the 'floor' to the bottom of the second throat
h2 = 0.5*test_section_h_w - 0.5*throat2_height
# Use that length and the wedge angle to find the length of the wedge section
L2 = h2/np.tan(delt)
# Plot the diffuser wedges
y1 = y0-h2
x1 = x0 + L2
# Plot the converging diffuser sections
plt.plot([x0, x0 + L2], [-y0, -y0 + h2])
plt.plot([x0, x0 + L2], [y0, y0 - h2])
# plt.vlines(x0 + L2, -0.5*throat2_height, 0)
# Calculate the first shockwaves
beta1 = beta(delt, Me, 0)

# Calculate the new properties across the first oblique shock, lower side
M1NS = Me*np.sin(beta1)
M2NS = Mach2(k, M1NS)
p2 = P2oP1(k, M1NS)*p1
T2 = T2oT1(k, M1NS)*T1
p02_over_p01 = Pt20Pt1(k, M1NS)
p1_over_p01 = PoPt(k, Me)
T1_over_T01 = ToTt(k, Me)
p02 = (p02_over_p01/p1_over_p01)*p1
T02 = T1/T1_over_T01
M2 = M2NS/np.sin(beta1-delt)

# New properties across the first oblique shock, top side
M3NS = Mach2(k, M1NS)
p3 = P2oP1(k, M1NS)*p1
T3 = T2oT1(k, M1NS)*T1
p03 = (Pt20Pt1(k, M1NS)/PoPt(k, Me))*p1
T03 = T1/ToTt(k, Me)
M3 = M2NS/np.sin(beta1-delt)

# Calculate the angle of the first oblique shock
LS23 = (0.5*test_section_h_w)/np.tan(beta1)
# find the point at which the shocks interact
yA = 0
xA = x0 + LS23
# Plot the initial oblique shocks
plt.plot([x0, xA], [-y0, yA], linestyle='dashed')
plt.plot([x0, xA], [y0, yA], linestyle='dashed')


# find the angles of the refracted oblique shocks
# Calculate the properties across the refracted shocks
beta4 = beta(delt, M2, 0)
beta4refract = -(delt-beta4)
print(np.rad2deg(beta4))
M2NS = M2*np.sin(beta4)
print(M2NS)
M4NS = Mach2(k, M2NS)
print(M4NS)
p4 = P2oP1(k, M2NS)*p2
T4 = T2oT1(k, M2NS)*T2
p04 = (Pt20Pt1(k, M2NS)/PoPt(k, M2))*p2
T04 = T2/ToTt(k, M2)
M4 = M4NS/np.sin(beta4-delt)

x2 = (y1 + xA*np.tan(beta4refract))/np.tan(beta4refract)
# Plot the refracted oblique shocks
plt.plot([xA, x2], [yA, y1], linestyle='dashed')
plt.plot([xA, x2], [yA, -y1], linestyle='dashed')
# plot the second throat lines
plt.hlines(y1, x1, x2)
plt.hlines(-y1, x1, x2)




plt.show()
print()

















test_section_height = 58
print()
Nozzle = GENozzle(k, Me, 50, 1.4)
throat_height = Nozzle.get_scaled_throat_height(test_section_height)
p02_over_p01 = 0.3283
throat2height = throat_height/p02_over_p01
AratioM3 = AoAt(k, Me)
L = ((test_section_height/2) - (throat2height/2))/np.tan(np.deg2rad(10))

print()
