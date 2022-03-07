import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#   EVERYTHING IS IN METRIC, FORCES ARE IN NEWTONS
#   48 inches = 1.219 meters
Cp = (2/3)*1.219
Cmount = 1.219
moment_arm = abs(Cp - Cmount)

# calculating/ initializing some variables
M = [3., 5., 7., 10.]
T = [225, 154, 86, 50]
mu = [.00001467, .00001054, .00000592, .00000321]
gamma = 1.4
R = 286
a = []
for temp in T:
    spdsound = np.sqrt(gamma*R*temp)
    a.append(spdsound)

rho_inf = [.054201, .067876, .060773, .034843]
V = []
for i in range(len(M)):
    v = M[i]*a[i]
    V.append(v)
#print(V)
q_inf = []
for j in range(len(rho_inf)):
    q = .5*rho_inf[j]*V[j]**2
    q_inf.append(q)
    #print(q)
Re = []
for k in range(len(V)):
    re = rho_inf[k]*V[k]*1.2192/mu[k]
    Re.append(re)
#print(Re)

#   Because the tb is symmetric the lift and drag forces will be the same in magnitude for +AoA or -AoA
AoAcalc = np.linspace(np.deg2rad(0), np.deg2rad(45), 100)
AoA = np.linspace(0, 45, 100)

S_plan = .247741
thetav = np.deg2rad(8/48)

pitch = np.linspace(0, 45, 100)
yaw = np.linspace(0, 45, 100)
L = []
D = []
tau = []
u_tau = []
fl_thickness = []
nu = []
cf_vals = []
for q in q_inf:
    n = q_inf.index(q)
    # skin friction coeff changes from case to case based on Re, so it will just get redeclared here
    cf = .455/(np.log10(Re[n]))**2.58
    cf_vals.append(cf)
    #print(cf)

    # tracking some variables for first layer thickness
    nu_case = mu[n]/rho_inf[n]
    tau_w = .5*cf*rho_inf[n]*V[n]**2
    utau = np.sqrt(tau_w/rho_inf[n])
    y = nu_case/utau
    tau.append(tau_w)
    u_tau.append(utau)
    fl_thickness.append(y)
    nu.append(nu_case)

    Cl = []
    Cd = []
    CD = []
    #   Cl and Cd are based on the geometry of the TB, the equations were derived via modified newtonian theory
    #   do not alter these
    for alpha in AoAcalc:
        Ca = 2 * (np.sin(thetav) ** 2) + (np.sin(alpha) ** 2) * (1 - 3 * (np.sin(thetav) ** 2))
        Cn = (np.cos(thetav) ** 2) * np.sin(2 * alpha)
        cl = Cn * np.cos(alpha) - Ca * np.sin(alpha)

        #   correction to normal force coefficient
        Cn = (np.cos(thetav) ** 2) * np.sin(alpha)
        cd = Cn * np.sin(alpha) + Ca * np.cos(alpha)
        cD = cd + cf
        Cl.append(cl)
        Cd.append(cd)
        CD.append(cD)
    # print(Cl)
    # print(Cd)
    plt.plot(AoA, Cl, label='Cl')
    plt.plot(AoA, CD, label='Cd')
    plt.xlabel('Alpha (degrees)')
    plt.title('Mach {}'.format(M[n]))
    plt.legend()
    plt.show()

    templistL = []
    for i in range(len(pitch)):
        changing_yawL = []
        for j in range(len(yaw)):
            S_calc = S_plan * np.cos(AoAcalc[j])
            l = Cl[i] * q * S_calc
            changing_yawL.append(l)
        templistL.append(changing_yawL)
    L.append(templistL)

    templistD = []
    for i in range(len(pitch)):
        changing_yawD = []
        for j in range(len(yaw)):
            d = (Cd[i]*np.cos(AoAcalc[j]) + Cd[j]*(np.sin(AoAcalc[j])) + cf) * q * S_plan
            changing_yawD.append(d)
        templistD.append(changing_yawD)
    D.append(templistD)

for i in range(len(D)):
    plt.xlabel('Yaw Angle (Degrees)')
    plt.ylabel('Drag (Newtons)')
    plt.title('Drag with changing yaw angle for all pitch angles M = {}'.format(M[i]))
    for j in range(len(D[i])):
        #plt.plot(yaw, D[i][j])
        pass

    #plt.show()

for i in range(len(L)):
    tempmaxlift = 0
    tempmaxpitchl = 0
    tempmaxyawl = 0
    tempmaxdrag = 0
    tempmaxpitchd = 0
    tempmaxyawd = 0
    plt.xlabel('Yaw angle (degrees)')
    plt.ylabel('Lift (Newtons)')
    plt.title('Lift with changing yaw angle for all pitch angles M = {}'.format(M[i]))
    for j in range(len(L[i])):
        #plt.plot(yaw, L[i][j])
        for k in range(len(L[i][j])):
            if L[i][j][k] > tempmaxlift:
                tempmaxlift = L[i][j][k]
                tempmaxpitchl = pitch[j]
                tempmaxyawl = yaw[k]
            if D[i][j][k] > tempmaxdrag:
                tempmaxdrag = D[i][j][k]
                tempmaxpitchd = pitch[j]
                tempmaxyawd = yaw[k]
    print('\nM = {} :'.format(M[i]))
    print('\nMax drag = {} at pitch angle = {} degrees and yaw angle = {} degrees'.format(tempmaxdrag, tempmaxpitchd, tempmaxyawd))
    print('\nMax lift = {} at pitch angle = {} degrees and yaw angle = {} degrees'.format(tempmaxlift, tempmaxpitchl, tempmaxyawl))
    #plt.show()

N_max = 0
for i in range(len(L)):
    for j in range(len(L[i])):
        Nlist = []
        for k in range(len(L[i][j])):
            N = L[i][j][k]*np.cos(AoAcalc[j])*np.cos(AoAcalc)[k] + D[i][j][k]*(np.sin(AoAcalc[j])*np.cos(AoAcalc[k]) + np.sin(AoAcalc[k]))
            Nlist.append(N)
            if abs(N) > N_max:
                N_max = abs(N)
                aoa_max_moment_pitch = pitch[j]
                aoa_max_moment_yaw = yaw[k]
                M_max_moment = M[i]
        #plt.plot(yaw, Nlist)
    #plt.show()
Overall_max_moment_all_M = N_max * moment_arm
#print(moment_arm)
print('\nThe maximum moment that the test body mount must withstand across all Mach configurations is {} \nat Mach {} a pitch angle of {} and yaw angle of {}'.format(Overall_max_moment_all_M, M_max_moment, aoa_max_moment_pitch, aoa_max_moment_yaw))
print('\nReynolds Number:', Re, '\nTau:', tau, '\nShear Rate:', u_tau, '\nFirst Layer Thickness:', fl_thickness, '\nKinematic Viscosity:', nu, '\nSkin Friction Coefficient:', cf_vals)
