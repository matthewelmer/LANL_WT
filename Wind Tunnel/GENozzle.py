import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, linspace, cos, sin, tan
from scipy.optimize import newton


class GENozzle:
    """This Class defines a Nozzle geometry, given a Ratio of Specific Heats (Gamma) for the working fluid,
    an Exit Mach Number, a Mesh size, and a radius for the expansion circle"""
    def __init__(self, G, Me, n, r):
        # Initialize ratio specific heats, Mach Exit, Mesh size, radius of expansion circle
        self.G = G
        self.Me = Me
        self.n = n
        self.r = r
    
        # datapoint matricies
        self.km = np.zeros([n, n])  # K- values (Constant along right running characteristic lines)
        self.kp = np.zeros([n, n])  # K- values (Constant along left running characteristic lines)
        self.theta = np.zeros([n, n])   # Flow angles relative to the horizontal
        self.Nu = None      # prandtl meyer angle
        self.mu = np.zeros([n, n])   # Mach angles
        self.mach = np.zeros([n, n])   # Mach Numbers
        self.x = np.zeros([n, n])   # x-coordinates
        self.y = np.zeros([n, n])   # y-coordinates
        # expansion circle arrays
        self.x_list = None
        self.y_list = None
        # wall points
        self.x_wall = None
        self.y_wall = None
        # other values
        self.NuMax = None
        self.dT = None
        # Perform calculations to find nozzle contour
        self.calculate()
        
    def calculate(self):
        # Find Numax using Prantdl meyer funcition
        no1, b, no2 = self.PMF(self.G, self.Me, 0, 0)
        self.NuMax = b / 2
        # Find Starting points along expansion section
        self.x_list, self.y_list = self.expansion_points(self.r, self.NuMax, self.n)
        # Define the theta Array with initial equally spread angles
        self.dT = self.NuMax / self.n
        self.theta[:, 0] = np.arange(self.dT, self.NuMax + self.dT, self.dT)
        # Set the first column of Nu equal to theta, then find characteristic constants for each first line
        self.Nu = np.copy(self.theta)
        self.km[:, 0] = np.copy(self.theta[:, 0] + self.Nu[:, 0])
        self.kp[:, 0] = np.copy(self.theta[:, 0] - self.Nu[:, 0])
        # Use the prantdl meyer function to solve for mach and mu
        self.mach[:, 0], self.Nu[:, 0], self.mu[:, 0] = self.PMF(self.G, 0, self.Nu[:, 0], 0)
    
        # Find the first reflection line and the intersection of each initial line with it
        self.y[0, 0] = 0  # along centerline
        self.x[0, 0] = np.copy(
            self.x_list[0] - self.y_list[0] / tan(self.theta[0, 0] - self.mu[0, 0]))  # intersection of slope and centerline along x
        
        # find the rest of the starting lines, from starting point n to intersection of first refelction line
        for i in range(1, self.n):
            # Find slope of the first line
            s1 = np.copy(tan(self.theta[i, 0] - self.mu[i, 0]))
            # Slope of the second line
            s2 = np.copy(tan((self.theta[i - 1, 0] + self.mu[i - 1, 0] + self.theta[i, 0] + self.mu[i, 0]) / 2))
            # x & y coordinates of intersection of slope1 and 2
            self.x[i, 0] = np.copy(((self.y[i - 1, 0] - self.x[i - 1, 0] * s2) - (self.y_list[i] - self.x_list[i] * s1)) / (s1 - s2))
            self.y[i, 0] = np.copy(self.y[i - 1, 0] + (self.x[i, 0] - self.x[i - 1, 0]) * s2)

            # Find flow properties for the characteristic web
        for j in range(1, self.n):
            for i in range(self.n - j):
                # Set characteristic constant equal to the one that runs into it
                self.km[i, j] = np.copy(self.km[i + 1, j - 1])
            
                # if we are on the first row, we are along centerline
                if i == 0:
                    # bc centerline, theta = 0
                    self.theta[i, j] = 0
                    # characteristic constants are equal to the previous
                    self.kp[i, j] = np.copy(-self.km[i, j])
                    # Nu is equal to the characteristic constant
                    self.Nu[i, j] = np.copy(self.km[i, j])
                    # Use prantdl meyer to solve for mach number and mach angle given nu
                    self.mach[i, j], self.Nu[i, j], self.mu[i, j] = self.PMF(self.G, 0, self.Nu[i, j], 0)
                    # find the slope of the line
                    s1 = np.copy(tan((self.theta[i + 1, j - 1] - self.mu[i + 1, j - 1] + self.theta[i, j] - self.mu[i, j]) / 2))
                    # and the next point along that line
                    self.x[i, j] = np.copy(self.x[i + 1, j - 1] - self.y[i + 1, j - 1] / s1)
                    self.y[i, j] = 0
            
                # otherwise, use the previous values to solve for the next ones.
                else:
                    # charateristic constant is equal to the previous
                    self.kp[i, j] = np.copy(self.kp[i - 1, j])
                    # theta is in the middle of the two constants before it
                    self.theta[i, j] = np.copy((self.km[i, j] + self.kp[i, j]) / 2)
                    # nu is found from the two characteristic constants before it
                    self.Nu[i, j] = np.copy((self.km[i, j] - self.kp[i, j]) / 2)
                    # find mach number and mach angle from nu and the prantdl meyer function
                    self.mach[i, j], self.Nu[i, j], self.mu[i, j] = self.PMF(self.G, 0, self.Nu[i, j], 0)
                    # slopes of the two lines coming into the point
                    s1 = np.copy(tan((self.theta[i + 1, j - 1] - self.mu[i + 1, j - 1] + self.theta[i, j] - self.mu[i, j]) / 2))
                    s2 = np.copy(tan((self.theta[i - 1, j] + self.mu[i - 1, j] + self.theta[i, j] + self.mu[i, j]) / 2))
                    # use the slopes to find the x and y locations of the new point
                    self.x[i, j] = np.copy(
                        ((self.y[i - 1, j] - self.x[i - 1, j] * s2) - (self.y[i + 1, j - 1] - self.x[i + 1, j - 1] * s1)) / (s1 - s2))
                    self.y[i, j] = np.copy(self.y[i - 1, j] + (self.x[i, j] - self.x[i - 1, j]) * s2)
    
        # Find the Wall points information
        # set wall point arrays
        self.x_wall = np.zeros([self.n])
        self.y_wall = np.zeros([self.n])
    
        # set values for the last expansion section point
        x0 = self.x_list[-1]
        y0 = self.y_list[-1]
    
        # find constants for next point
        walls = tan(self.NuMax)
        webs = tan(np.copy(self.theta[self.n - 1, 0] + self.mu[self.n - 1, 0]))
    
        # find the first point in the straightening section
        self.x_wall[0] = np.copy(((self.y[self.n - 1, 0] - self.x[self.n - 1, 0] * webs) - (y0 - x0 * walls)) / (walls - webs))
        self.y_wall[0] = np.copy(y0 + (self.x_wall[0] - x0) * walls)
        # find all the wall points in the straightening section
        for j in range(1, self.n):
            # find the slope of the wall and the slope of the web
            walls = tan(np.copy((self.theta[self.n - j, j - 1] + self.theta[self.n - j - 1, j]) / 2))
            webs = tan(np.copy(self.theta[self.n - j - 1, j] + self.mu[self.n - j - 1, j]))
            # put the points x and y at the intersection
            self.x_wall[j] = np.copy(
                (self.y[self.n - j - 1, j] - self.y_wall[j - 1] - self.x[self.n - j - 1, j] * webs + self.x_wall[j - 1] * walls) / (walls - webs))
            self.y_wall[j] = np.copy((self.x_wall[j] - self.x[self.n - j - 1, j]) * webs + self.y[self.n - j - 1, j])
    
    # Create function to return nozzle length
    def get_length(self):
        return np.amax(self.x_wall)
    
    # Create function to return nozzle height
    def get_height(self):
        return np.amax(self.y_wall)
    
    # Create function to return wall points
    def wall_points(self):
        xedge = np.concatenate((self.x_list, self.x_wall))
        yedge = np.concatenate((self.y_list, self.y_wall))
        return xedge, yedge
    
    def scale_wall_points(self, exit_height):
        x_edge, y_edge = self.wall_points()
        
        ymx1 = exit_height/2
        scale_factor = ymx1/y_edge[-1]
        ymin1 = y_edge[0]*scale_factor
        xmx1 = x_edge[-1]*scale_factor
        xmin1 = x_edge[0]*scale_factor
        x_edge_1 = np.interp(x_edge, (x_edge.min(), x_edge.max()), (xmin1, xmx1))
        y_edge_1 = np.interp(y_edge, (y_edge.min(), y_edge.max()), (ymin1, ymx1))
        return x_edge_1, y_edge_1

    def get_scaled_length(self, exit_height):
        x_edge, y_edge = self.scale_wall_points(exit_height)
        return np.amax(x_edge)

    def get_scaled_height(self, exit_height):
        x_edge, y_edge = self.scale_wall_points(exit_height)
        return np.amax(y_edge)*2
    
    def get_scaled_throat_height(self, exit_height):
        x_edge, y_edge = self.scale_wall_points(exit_height)
        return np.amin(y_edge)*2
    
    def plot_scaled(self, exit_height):
        x, y = self.scale_wall_points(exit_height)
        plt.figure()
        plt.plot(x, y, c='k')
        plt.plot(x, -y, c='k')
        plt.xlim([0, np.amax(self.x_wall)])
        plt.axis('equal')
        plt.title(('Method of Characteristics, M=' + str(self.Me) + ', Mesh Size, n=' + str(
            self.n) + ', gamma=' + str(self.G), 'and Exit Height =', str(exit_height)))
        plt.grid()
        plt.show()
        
    def plotter(self, full=False):
        plt.figure()
    
        # plot the first lines out of the expansion section
        for i in range(self.n):
            plt.plot([self.x_list[i], self.x[i, 0]], [self.y_list[i], self.y[i, 0]])
            if full:
                plt.plot([self.x_list[i], self.x[i, 0]], [-self.y_list[i], -self.y[i, 0]])
    
        # plot the characteristic lines that connect to walls
        for i in range(self.n):
            plt.plot([self.x[self.n - 1 - i, i], self.x_wall[i]], [self.y[self.n - 1 - i, i], self.y_wall[i]])
            if full:
                plt.plot([self.x[self.n - 1 - i, i], self.x_wall[i]], [-self.y[self.n - 1 - i, i], -self.y_wall[i]])
    
        # plot the left running lines
        for i in range(self.n - 1):
            plt.plot(self.x[0:self.n - i, i], self.y[0:self.n - i, i])
            if full:
                plt.plot(self.x[0:self.n - i, i], -self.y[0:self.n - i, i])
    
        # plot the right running lines
        for c in range(self.n):
            for r in range(1, self.n - c):
                plt.plot([self.x[c, r], self.x[c + 1, r - 1]], [self.y[c, r], self.y[c + 1, r - 1]])
                if full:
                    plt.plot([self.x[c, r], self.x[c + 1, r - 1]], [-self.y[c, r], -self.y[c + 1, r - 1]])
    
        # plot walls
        plt.plot(np.concatenate((self.x_list, self.x_wall)), np.concatenate((self.y_list, self.y_wall)), c='k')
        if full:
            plt.plot(np.concatenate((self.x_list, self.x_wall)), np.concatenate((-self.y_list, -self.y_wall)), c='k')
        # figure settings
        plt.xlim([0, np.amax(self.x_wall)])
        plt.axis('equal')
        plt.title(('Method of Characteristics, M=' + str(self.Me) + ', Mesh Size, n=' + str(self.n) + ', and gamma=' + str(self.G)))
        plt.show()


    def PMfunc(self, Mg, nu, G1, G2):
        '''Prantdl-Meyer equation as a function to be used in fsolve to solve for Mach #'''
    
        return np.sqrt(G1 / G2) * np.arctan(np.sqrt((G2 / G1) * (Mg ** 2 - 1))) - np.arctan(np.sqrt(Mg ** 2 - 1)) - nu


    def expansion_points(self, r, theta, n):
        '''Given a radius, a full angle, and a n number of points,
            return n number of points along the circle from 270 deg to 270 + theta'''
        # r in unit vector
        # theta in radians
        # n in number of starting points
        rangeAngles = linspace(3 * pi / 2, 3 * pi / 2 + theta, n)
        xlist = np.zeros(n)
        ylist = np.zeros(n)
        for i in range(len(rangeAngles)):
            if i == 0:
                xlist[i] = 0
                ylist[i] = r * sin(rangeAngles[i]) + (1 + r)
            else:
                xlist[i] = r * cos(rangeAngles[i])
                ylist[i] = r * sin(rangeAngles[i]) + (1 + r)
        return xlist, ylist


    def PMF(self, G, M, nu, mu):
        '''Given either Mach number, nu(prantdl meyer angle) or mu(mach angle), solve for and return the other two'''
        # Given gamma and either Mach, nu, or mu, calcatate the other two
        # returns angles in radians
        Gp = G + 1
        Gm = G - 1
    
        if type(nu) == np.float64:
            nu = np.array([nu])
        # For known M
        if M != 0:
            nu = np.subtract(
                np.multiply(sqrt(Gp / Gm), np.arctan(sqrt(np.multiply((Gm / Gp), np.subtract(np.power(M, 2),
                                                                                             1))))),
                np.arctan(sqrt(np.subtract(np.power(M, 2), 1))))
            mu = np.arcsin(np.divide(1.0, M))
    
        # For known nu
        elif np.linalg.norm(nu) != 0:
            # Find M
            M = np.zeros_like(nu)
            for i in range(len(nu)):
                # for j in range(len(nu[:, 0])):
                M[i] = newton(self.PMfunc, 1.2, args=(nu[i], Gp, Gm))
            mu = np.arcsin(np.divide(1.0, M))
    
        # For known mu
        elif mu != 0:
            M = np.divide(1, np.sin(mu))
            nu = np.subtract(
                np.multiply(sqrt(Gp / Gm), np.arctan(sqrt(np.multiply((Gm / Gp), np.subtract(np.power(M, 2),
                                                                                             1))))),
                np.arctan(sqrt(np.subtract(np.power(M, 2), 1))))
    
        return M, nu, mu


