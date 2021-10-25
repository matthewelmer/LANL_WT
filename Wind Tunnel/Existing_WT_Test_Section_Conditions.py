import matplotlib.pyplot as plt
import numpy as np

KPA_TO_PSIA = 0.14503773800722
PSIA_TO_KPA = 1 / KPA_TO_PSIA
RANKINE_TO_KELVIN = 0.55555555556
KELVIN_TO_RANKINE = 1 / RANKINE_TO_KELVIN

# These values from: https://www1.grc.nasa.gov/facilities/htf/
mach_arr = np.array([5, 6, 7])
static_low_pres_arr = np.array([0.118, 0.071, 0.071])  # psia
static_high_pres_arr = np.array([0.74, 0.61, 0.33])  # psia
static_low_temp_arr = np.array([384, 390, 412])  # Rankine
static_high_temp_arr = np.array([428, 451, 451])  # Rankine


static_low_pres_arr = static_low_pres_arr * PSIA_TO_KPA  # kPa
static_high_pres_arr = static_high_pres_arr * PSIA_TO_KPA  # kPa
static_low_temp_arr = static_low_temp_arr * RANKINE_TO_KELVIN  # Kelvin
static_high_temp_arr = static_high_temp_arr * RANKINE_TO_KELVIN  # Kelvin

fig = plt.figure("Existing Data")

p1 = fig.add_subplot(1, 2, 1)
p1.plot(mach_arr, static_low_pres_arr, label="lower value")
p1.plot(mach_arr, static_high_pres_arr, label="upper value")
p1.set(title="Pressure", xlabel="Mach #", ylabel="Pressure (kPa)")
plt.legend()
plt.tight_layout()

p2 = fig.add_subplot(1, 2, 2)
p2.plot(mach_arr, static_low_temp_arr, label="lower value")
p2.plot(mach_arr, static_high_temp_arr, label="upper value")
p2.set(title="Temperature", xlabel="Mach #", ylabel="Temperature (K)")
plt.legend()
plt.tight_layout()

plt.show()
