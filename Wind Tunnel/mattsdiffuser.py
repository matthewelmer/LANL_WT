import numpy as np
import matplotlib.pyplot as plt
from cubic_roots import solve_cubic

# constants
gamma = 1.4
R_air = 287.058  # J / kg K
RAD_TO_DEG = 180 / np.pi
DEG_TO_RAD = 1/RAD_TO_DEG
M_TO_FT = 3.2808
FT_TO_M = 1 / M_TO_FT
KPA_TO_PA = 1000
PA_TO_KPA = 1 / KPA_TO_PA

def betafunc(mach, delta, gamma=1.4):
    b = -(mach**2 + 2) / mach**2 - gamma * np.sin(delta)**2
    c = (2 * mach**2 + 1) / mach**4 + ((gamma + 1)**2 / 4 + (gamma - 1) / mach**2) * np.sin(delta)**2
    d = -np.cos(delta)**2 / mach**4
    roots = solve_cubic(1, b, c, d)
    # rootscpy = np.copy(roots)
    roots = np.delete(roots, roots.argmin())
    roots = np.delete(roots, roots.argmax())

    sinbeta = np.sqrt(roots[0].real)
    if sinbeta >= 1:
        if sinbeta < 1.005:
            print(f"betafunc returned arcsin(1) instead of undefined arcsin({sinbeta})")
            return np.arcsin(1)
        # else:
        #     for i in range(len(rootscpy)):
        #         if abs(rootscpy[i].real - 1) < 1e-6:
        #             print(f"betafunc returned arcsin(1) instead of undefined arcsin({roots[0]})")
        #             return np.arcsin(1)
    
    return np.arcsin(sinbeta)

def deltafunc(beta, mach, gamma=1.4):
    return np.arctan(2*np.cot(beta)*((mach**2*(np.sin(beta)**2) - 1)/(mach**2*(gamma + 2*np.cos(2*beta) + 2))))


# Isentropic expansion ratio functions
def ToTt(M, gma=1.4):
    """static temperature over total temperature"""
    return (1 + ((gma-1)/2)*M**2)**-1


def PoPt(M, gma=1.4):
    """static pressure over total pressure"""
    return ToTt(M, gma)**(gma/(gma-1))


def rhoorhot(M, gma=1.4):
    """static density over total density"""
    return ToTt(M, gma)**(1/(gma-1))


def AoAt(M, gma=1.4):
    """area over throat area"""
    return 1 / M * ((gma+1) / 2)**(-(gma + 1)/(2 * (gma-1))) * (1 + ((gma-1) / 2) * M**2)**((gma+1)/(2 * (gma-1)))


def P2oP1(M, gma=1.4):
    """pressure after shock over pressure before shock"""
    return (2*gma*M**2 - (gma - 1))/(gma + 1)


def rho2orho1(M, gma=1.4):
    """density after shock over density before shock"""
    return ((gma+1)*M**2)/((gma - 1)*M**2 + 2)


def T2oT1(M, gma=1.4):
    """static temperature after shock over static temperature before shock"""
    return ((2*gma*M**2 - (gma - 1))*((gma - 1)*M**2 + 2))/((gma+1)**2*M**2)


def M2(M, gma=1.4):
    """mach number after shock"""
    return ((gma-1)*M**2 + 2)/(2*gma*M**2 - (gma-1))


def Pt2oPt1(M, gma=1.4):
    """total pressure after shock over total pressure before shock"""
    return (((gma+1) * M**2)/((gma-1) * M**2 + 2))**(gma / (gma-1)) * ((gma+1) / (2 * gma * M**2 - (gma-1)))**(1/(gma-1))


Mi = (3, 5, 7, 10)  # Mach number
Ti = (225, 154, 86, 50)  # K
p_i = (3.5, 3.0, 1.5, 0.5)  # kPa
rhoi = (0.0542, 0.0679, 0.0608, 0.0348)  # kg/m^3
expansion_radius = (1.4, 1.067, 0.733, 0.4)  # circle radius of expansion section
delta = (7.1 * DEG_TO_RAD, 11.03 * DEG_TO_RAD, 12.6 * DEG_TO_RAD, 13.7285 * DEG_TO_RAD)  # tune these

dfig = plt.figure("Diffusers", constrained_layout=True)
dfig.supxlabel("Longitudinal Direction (m)")
dfig.supylabel("Vertical Direction (m)")
m3p = dfig.add_subplot(2, 2, 1)
m5p = dfig.add_subplot(2, 2, 2)
m7p = dfig.add_subplot(2, 2, 3)
m10p = dfig.add_subplot(2, 2, 4)
subplots = [m3p, m5p, m7p, m10p]

test_section_h_w = 28 * FT_TO_M  # height and width
test_section_length = 2.286  # m
for i in range(len(Mi)):
    subplots[i].set(
        title=f"Mach {Mi[i]}",
        xlim=(0, 20),
        ylim=(test_section_h_w/2 - 7.5, test_section_h_w/2 + 7.5)
    )
    p01 = p_i[i] * 1/PoPt(Mi[i])

    p_arr = np.array([p_i[i]])
    T_arr = np.array([Ti[i]])
    p0_arr = np.array([p01])
    beta_arr = np.array([betafunc(Mi[i], delta[i])])
    M_arr = np.array([Mi[i]])

    throat_height = test_section_h_w**2 * 1/AoAt(Mi[i]) / test_section_h_w
    throat2_height = throat_height/Pt2oPt1(Mi[i])  # approximate second throat height w/ normal shock relation to avoid choking flow
    wedge_height = (test_section_h_w - throat2_height) / 2
    wedge_length = wedge_height / np.tan(delta[i])
    subplots[i].plot([0, wedge_length], [0, wedge_height], color="k")
    subplots[i].plot([0, wedge_length], [test_section_h_w, test_section_h_w - wedge_height], color="k")

    j = 0
    iter_max = 15
    m1sinbeta = 69.420
    while M_arr[-1] > 1 and j < iter_max:
        m1 = M_arr[-1]
        m1sinbeta = m1 * np.sin(beta_arr[-1])
        if m1sinbeta <= 1 + 1e-6:
            break
        p2 = P2oP1(m1sinbeta) * p_arr[-1]
        t2 = T2oT1(m1sinbeta) * T_arr[-1]
        p02 = Pt2oPt1(m1sinbeta) * p0_arr[-1]
        if len(M_arr) <= 2:  # only do the delta for the first two
            m2 = M2(m1sinbeta) / np.sin(beta_arr[-1] - delta[i])
            if np.isnan(m2):
                break
            beta2 = betafunc(m2, delta[i])
        else:
            m2 = M2(m1sinbeta) / np.sin(beta_arr[-1])
            if np.isnan(m2):
                break
            beta2 = betafunc(m2, 0)
        
        p_arr = np.append(p_arr, p2)
        T_arr = np.append(T_arr, t2)
        p0_arr = np.append(p0_arr, p02)
        M_arr = np.append(M_arr, m2)
        beta_arr = np.append(beta_arr, beta2)
        j += 1

    length = test_section_h_w / (2 * np.tan(beta_arr[0]))
    firstshock_length = length
    subplots[i].plot([0, length], [0, test_section_h_w/2], color=f"C0")
    subplots[i].plot([0, length], [test_section_h_w, test_section_h_w/2], color=f"C0")
    # print(f"shock len tally: |", end="")
    for j in range(1, len(beta_arr) - 1):
        # print("|", end="")
        old_length = length
        length += throat2_height / (2 * np.tan(beta_arr[j]))
        if j%2: # if j is odd
            subplots[i].plot([old_length, length], [test_section_h_w/2, test_section_h_w/2 - throat2_height/2], color=f"C{j}")
            subplots[i].plot([old_length, length], [test_section_h_w/2, test_section_h_w/2 + throat2_height/2], color=f"C{j}")
        else:
            subplots[i].plot([old_length, length], [test_section_h_w/2 - throat2_height/2, test_section_h_w/2], color=f"C{j}")
            subplots[i].plot([old_length, length], [test_section_h_w/2 + throat2_height/2, test_section_h_w/2], color=f"C{j}")
    if M_arr[-1] > 1:
        old_length = length
        length += throat2_height / (2 * np.tan(beta_arr[-1]))
        if len(M_arr)%2:  # if there is an odd number of mach numbers, the last one comes from the outside in
            subplots[i].plot([old_length, length], [test_section_h_w/2 - throat2_height/2, test_section_h_w/2], color=f"C{len(beta_arr) - 1}")
            subplots[i].plot([old_length, length], [test_section_h_w/2 + throat2_height/2, test_section_h_w/2], color=f"C{len(beta_arr) - 1}")
        else:  # if there is an even number, the last one comes from the centerline outwards
            subplots[i].plot([old_length, length], [test_section_h_w/2, test_section_h_w/2 - throat2_height/2], color=f"C{len(beta_arr) - 1}")
            subplots[i].plot([old_length, length], [test_section_h_w/2, test_section_h_w/2 + throat2_height/2], color=f"C{len(beta_arr) - 1}")

    # print()
    print(f"Mach {Mi[i]} Diffuser:")
    print(f"M: {M_arr}")
    print(f"T: {T_arr}")
    print(f"p0: {p0_arr}")
    print(f"p: {p_arr}")
    print(f"beta: {beta_arr * RAD_TO_DEG}")
    print(f"total diffuser length: {length} meters")
    normalshock_pressure_ratio = Pt2oPt1(Mi[i])
    print(f"target stagnation pressure ratio: {normalshock_pressure_ratio}")
    stagnation_pressure_ratio = p0_arr[-1] / p0_arr[0]
    print(f"stagnation pressure ratio: {stagnation_pressure_ratio}")
    subplots[i].plot([wedge_length, length], [wedge_height, wedge_height], color="k")
    subplots[i].plot([wedge_length, length], [test_section_h_w - wedge_height, test_section_h_w - wedge_height], color="k")
    subplots[i].text(subplots[i].get_xlim()[0], subplots[i].get_ylim()[0] + 0.5, f"$\\frac{{p_{{0, 2}}}}{{p_{{0, 1}}}}$ (ideal) = {stagnation_pressure_ratio:.3f}")
    subplots[i].text((subplots[i].get_xlim()[0] + subplots[i].get_xlim()[1]) / 2, subplots[i].get_ylim()[0] + 0.5, f"$\\frac{{p_{{0, 2}}}}{{p_{{0, 1}}}}$ (normal shock) = {normalshock_pressure_ratio:.3f}", horizontalalignment="center")
    subplots[i].text(subplots[i].get_xlim()[1], subplots[i].get_ylim()[0] + 0.5, f"$\delta$ = {delta[i] * RAD_TO_DEG}$^\circ$", horizontalalignment="right")
    
    print()

plt.show()
