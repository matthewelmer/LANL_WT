import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, linspace, cos, sin, tan
from scipy.optimize import fsolve


def PMfunc(Mg, nu, G1, G2):
    '''Prantdl-Meyer equation as a function to be used in fsolve to solve for Mach #'''
    
    return np.sqrt(G1/G2)*np.arctan(np.sqrt((G2/G1)*(Mg**2 - 1))) - np.arctan(np.sqrt(Mg**2 - 1)) - nu


def expansion_points(r, theta, n):
    '''Given a radius, a full angle, and a n number of points,
        return n number of points along the circle from 270 deg to 270 + theta'''
    # r in unit vector
    # theta in radians
    # n in number of starting points
    rangeAngles = linspace(3 * pi / 2, 3 * pi / 2 + theta, n)
    xlist = np.zeros(n)
    ylist = np.zeros(n)
    for i in range(len(rangeAngles)):
        xlist[i] = r * cos(rangeAngles[i])
        ylist[i] = r * sin(rangeAngles[i]) + (1 + r)
    
    return xlist, ylist


def PMF(G, M, nu, mu):
    '''Given either Mach number, nu(prantdl meyer angle) or mu(mach angle), solve for and return the other two'''
    # Given gamma and either Mach, nu, or mu, calcatate the other two
    # returns angles in radians
    Gp = G + 1
    Gm = G - 1
    
    if type(nu) == np.float64:
        nu = np.array([nu])
    # For known M
    if M != 0:
        nu = np.subtract(np.multiply(sqrt(Gp / Gm), np.arctan(sqrt(np.multiply((Gm / Gp), np.subtract(np.power(M, 2),
                                                                                                      1))))),
                         np.arctan(sqrt(np.subtract(np.power(M, 2), 1))))
        mu = np.arcsin(np.divide(1.0, M))
    
    # For known nu
    elif np.linalg.norm(nu) != 0:
        # Find M
        M = np.zeros_like(nu)
        for i in range(len(nu)):
            # for j in range(len(nu[:, 0])):
            M[i] = fsolve(PMfunc, np.array([1]), args=(nu[i], Gp, Gm))
        mu = np.arcsin(np.divide(1.0, M))
    
    # For known mu
    elif mu != 0:
        M = np.divide(1, np.sin(mu))
        nu = np.subtract(np.multiply(sqrt(Gp / Gm), np.arctan(sqrt(np.multiply((Gm / Gp), np.subtract(np.power(M, 2),
                                                                                                      1))))),
                         np.arctan(sqrt(np.subtract(np.power(M, 2), 1))))
    
    return M, nu, mu


# Set values
# Gamma for working fluid
G = 1.4
# Goal Exit Mach Number
# Me = int(input('Enter Exit Mach Number, integer: '))
Me = 3.1
# Mesh Size
# n = int(input('Enter mesh size, integer: '))
n = 10
# Expansion Section circle radius
r = 0.4

plt.figure()

# intialize datapoint matricies
Km = np.zeros([n, n])  # K- vlaues (Constant along right running characteristic lines)
Kp = np.zeros([n, n])  # K- vlaues (Constant along left running characteristic lines)
Theta = np.zeros([n, n])  # Flow angles relative to the horizontal
Mu = np.zeros([n, n])  # Mach angles
M = np.zeros([n, n])  # Mach Numbers
x = np.zeros([n, n])  # x-coordinates
y = np.zeros([n, n])  # y-coordinates

# Find Numax using Prantdl meyer funcition
no1, B, no2 = PMF(G, Me, 0, 0)
NuMax = B / 2
# Find Starting points along expansion section
x_list, y_list = expansion_points(r, NuMax, n)
# Define the Theta Array with initial equally spread angles
dT = NuMax / n
Theta[:, 0] = np.arange(dT, NuMax + dT, dT)
# Set the first column of Nu equal to theta, then find characteristic constants for each first line
Nu = np.copy(Theta)
Km[:, 0] = np.copy(Theta[:, 0] + Nu[:, 0])
Kp[:, 0] = np.copy(Theta[:, 0] - Nu[:, 0])
# Use the prantdl meyer function to solve for mach and mu
M[:, 0], Nu[:, 0], Mu[:, 0] = PMF(G, 0, Nu[:, 0], 0)

# Find the first reflection line and the intersection of each initial line with it
y[0, 0] = 0  # along centerline
x[0, 0] = np.copy(x_list[0] - y_list[0] / tan(Theta[0, 0] - Mu[0, 0]))  # intersection of slope and centerline along x
# plot the first line
plt.plot([x_list[0], x[0, 0]], [y_list[0], y[0, 0]])
# find the rest of the starting lines, from starting point n to intersection of first refelction line
for i in range(1, n):
    # Find slope of the first line
    s1 = np.copy(tan(Theta[i, 0] - Mu[i, 0]))
    # Slope of the second line
    s2 = np.copy(tan((Theta[i - 1, 0] + Mu[i - 1, 0] + Theta[i, 0] + Mu[i, 0]) / 2))
    # x & y coordinates of intersection of slope1 and 2
    x[i, 0] = np.copy(((y[i - 1, 0] - x[i - 1, 0] * s2) - (y_list[i] - x_list[i] * s1)) / (s1 - s2))
    y[i, 0] = np.copy(y[i - 1, 0] + (x[i, 0] - x[i - 1, 0]) * s2)
    # plot
    plt.plot([x_list[i], x[i, 0]], [y_list[i], y[i, 0]])

# plt.show()
# Find flow properties for the characteristic web
for j in range(1, n):
    for i in range(n - j):
        # Set characteristic constant equal to the one that runs into it
        Km[i, j] = np.copy(Km[i + 1, j - 1])
        
        # if we are on the first row, we are along centerline
        if i == 0:
            # bc centerline, theta = 0
            Theta[i, j] = 0
            # characteristic constants are equal to the previous
            Kp[i, j] = np.copy(-Km[i, j])
            # Nu is equal to the characteristic constant
            Nu[i, j] = np.copy(Km[i, j])
            # Use prantdl meyer to solve for mach number and mach angle given nu
            M[i, j], Nu[i, j], Mu[i, j] = PMF(G, 0, Nu[i, j], 0)
            # find the slope of the line
            s1 = np.copy(tan((Theta[i + 1, j - 1] - Mu[i + 1, j - 1] + Theta[i, j] - Mu[i, j]) / 2))
            # and the next point along that line
            x[i, j] = np.copy(x[i + 1, j - 1] - y[i + 1, j - 1] / s1)
            y[i, j] = 0
        
        # otherwise, use the previous values to solve for the next ones.
        else:
            # charateristic constant is equal to the previous
            Kp[i, j] = np.copy(Kp[i - 1, j])
            # theta is in the middle of the two constants before it
            Theta[i, j] = np.copy((Km[i, j] + Kp[i, j]) / 2)
            # nu is found from the two characteristic constants before it
            Nu[i, j] = np.copy((Km[i, j] - Kp[i, j]) / 2)
            # find mach number and mach angle from nu and the prantdl meyer function
            M[i, j], Nu[i, j], Mu[i, j] = PMF(G, 0, Nu[i, j], 0)
            # slopes of the two lines coming into the point
            s1 = np.copy(tan((Theta[i + 1, j - 1] - Mu[i + 1, j - 1] + Theta[i, j] - Mu[i, j]) / 2))
            s2 = np.copy(tan((Theta[i - 1, j] + Mu[i - 1, j] + Theta[i, j] + Mu[i, j]) / 2))
            # use the slopes to find the x and y locations of the new point
            x[i, j] = np.copy(((y[i - 1, j] - x[i - 1, j] * s2) - (y[i + 1, j - 1] - x[i + 1, j - 1] * s1)) / (s1 - s2))
            y[i, j] = np.copy(y[i - 1, j] + (x[i, j] - x[i - 1, j]) * s2)

# Find the Wall points information
# set wall point arrays
xwall = np.zeros([n])
ywall = np.zeros([n])

# set values for the last expansion section point
x0 = x_list[-1]
y0 = y_list[-1]

# find constants for next point
walls = tan(NuMax)
webs = tan(np.copy(Theta[n - 1, 0] + Mu[n - 1, 0]))

# find the first point in the straightening section
xwall[0] = np.copy(((y[n - 1, 0] - x[n - 1, 0] * webs) - (y0 - x0 * walls)) / (walls - webs))
ywall[0] = np.copy(y0 + (xwall[0] - x0) * walls)
# find all the wall points in the straightening section
for j in range(1, n):
    # find the slope of the wall and the slope of the web
    walls = tan(np.copy((Theta[n - j, j - 1] + Theta[n - j - 1, j]) / 2))
    webs = tan(np.copy(Theta[n - j - 1, j] + Mu[n - j - 1, j]))
    # put the points x and y at the intersection
    xwall[j] = np.copy(
        (y[n - j - 1, j] - ywall[j - 1] - x[n - j - 1, j] * webs + xwall[j - 1] * walls) / (walls - webs))
    ywall[j] = np.copy((xwall[j] - x[n - j - 1, j]) * webs + y[n - j - 1, j])


# plot walls
plt.plot(np.concatenate((x_list, xwall)), np.concatenate((y_list, ywall)))

# plot the characteristic lines that connect to walls
for i in range(n):
    plt.plot([x[n - 1 - i, i], xwall[i]], [y[n - 1 - i, i], ywall[i]])

# plot the left running lines
for i in range(n - 1):
    plt.plot(x[0:n - i, i], y[0:n - i, i])

# plot the right running lines
for c in range(n):
    for r in range(1, n - c):
        plt.plot([x[c, r], x[c + 1, r - 1]], [y[c, r], y[c + 1, r - 1]])

# figure settings
plt.xlim([0, np.amax(xwall)])
plt.axis('equal')
plt.title(('Method of Characteristics, M=' + str(Me) + ', Mesh Size, n=' + str(n) + ', and gamma=' + str(G)))

plt.show()
print()
