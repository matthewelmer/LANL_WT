from GENozzle import GENozzle
import numpy as np
import matplotlib.pyplot as plt

'''This is an example script to show the functions of the GENozzle class'''
gamma = 1.4
Mach_exit = 3
mesh_size = 20
radius_of_expansion = 0.4
test_section_height = 58

# input a ratio of specific heats, exit mach number, mesh size, and radius of expansion section
Nozzle = GENozzle(gamma, Mach_exit, mesh_size, radius_of_expansion)

# Nozzle Length
unitlength = Nozzle.get_length()

# Nozzle Height
unitheight = Nozzle.get_height()

# Get coordinate points for nozzle wall contour
x0, y0 = Nozzle.wall_points()

# Get nozzle points scaled to exit height
x1, y1 = Nozzle.scale_wall_points(test_section_height)

# get nozzle length scaled to exit length
length = Nozzle.get_scaled_length(test_section_height)

# get nozzle height scaled to exit height
height = Nozzle.get_scaled_height(test_section_height)

# Get nozzle throat height scaled to exit height
throat_height = Nozzle.get_scaled_throat_height(test_section_height)

# plot the nozzle, half image
#Nozzle.plotter()

# plot the nozzle, double sided
#Nozzle.plotter(full=True)
# plot the nozzle wall contour scaled to the exit height
#Nozzle.plot_scaled(test_section_height)

# Create an array of nozzles
q = 10
nozzle_array = np.zeros([q, 2], dtype=GENozzle)

for i in range(len(nozzle_array)):
    for j in range(len(nozzle_array[0])):
        if j == 0:
            nozzle_array[i, j] = GENozzle(1.4, Mach_exit, (i+1)*q, 0.4)
        if j == 1:
            nozzle_array[i, j] = GENozzle(1.66, Mach_exit, (i+1)*q, 0.4)
    

# Check vs Area Ratio
AoverA_t = lambda M, k: (((k+1)*0.5)**(-(k+1)/(2*(k-1))))*((1 + ((k-1)*0.5)*M**2)**((k+1)/(2*(k-1)))/M)

# Air, mach 3
heightAir = nozzle_array[-1, 0].get_scaled_height(test_section_height)
throatAir = nozzle_array[-1, 0].get_scaled_throat_height(test_section_height)
print('For Air, Area ratio, A/A_t =', AoverA_t(3, 1.4), 'MoC area ratio A/A_t =', heightAir/throatAir)

# Helium, mach 3
heightHe = nozzle_array[-1, 1].get_scaled_height(test_section_height)
throatHe = nozzle_array[-1, 1].get_scaled_throat_height(test_section_height)
print('For Helium, Area ratio, A/A_t =', AoverA_t(3, 1.66), 'MoC area ratio A/A_t =', heightHe/throatHe)
print('Accurate to at least 3 decimals')


print()

