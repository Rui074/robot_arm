import numpy as np
import modern_robotics as mr
from arm_data import robot_info
import matplotlib.pyplot as plt

link_names = robot_info['link_names']
joint_names = robot_info['joint_names']
joint_constraints = robot_info['joint_constraints']
M = robot_info['M']
Slist = robot_info['Slist']
Mlist = robot_info['Mlist']
Glist = robot_info['Glist']


g = np.array([0, 0, -9.81])

theta0 = np.zeros(6)
theta_target = np.array([0.0, np.pi/2, 0.0, 0.0, 0.0, 0.0])
delta_theta = theta_target-theta0

t_log = np. linspace(0, 10, 100)

omegalist, alphalist = [], []
for t in t_log:
    s = mr. QuinticTimeScaling(Tf = 10, t = t)
    #find omega, alpha
    t_nor = t / 10
    omegalist.append((30*t_nor**2 - 60*t_nor**3 + 30*t_nor**4)*delta_theta/10)
    alphalist.append((60*t_nor**1 - 180*t_nor**2 + 120*t_nor**3)*delta_theta/10**2)
    
trajectory = mr.JointTrajectory(theta0, theta_target, Tf = 10, N = 100, method = 5)
trajlist, tau_M, tau_c, tau_g = [], [], [], []
for traj, omega, alpha in zip(trajectory, omegalist, alphalist):
    trajlist.append(traj)
    #M
    matrix_M = mr.MassMatrix(traj, Mlist, Glist, Slist)
    tau_M.append(matrix_M @ alpha)
    #c
    matrix_c = mr.VelQuadraticForces(traj, omega, Mlist, Glist, Slist)
    tau_c.append(matrix_c)
    #g
    matrix_g = mr. GravityForces(traj, g, Mlist, Glist, Slist)
    tau_g.append(matrix_g)

trajlist = np. array(trajlist) 
tau_M = np. array(tau_M)
tau_c = np. array(tau_c)
tau_g = np. array(tau_g)


plt. figure(figsize = (10, 8))
plt.subplot(4, 1, 1)
for j in range(6):
    plt.plot(t_log, trajlist[:, j], label=f'joint{j+1}')
plt.xlabel('time (s)'); plt.ylabel('traj')
plt.grid(True); plt.legend()
#tau_M
plt.subplot(4, 1, 2)
for j in range(6):
    plt.plot(t_log, tau_M[:, j], label=f'joint{j+1}')
plt.xlabel('time (s)'); plt.ylabel('tau_M')
plt.grid(True); plt.legend()
#tau_c
plt.subplot(4, 1, 3)
for j in range(6):
    plt.plot(t_log, tau_c[:, j], label=f'joint{j+1}')
plt.xlabel('time (s)'); plt.ylabel('tau_c')
plt.grid(True); plt.legend()
#tau_g
plt.subplot(4, 1, 4)
for j in range(6):
    plt.plot(t_log, tau_g[:, j], label=f'joint{j+1}')
plt.xlabel('time (s)'); plt.ylabel('tau_g')
plt.grid(True); plt.legend()


plt.tight_layout()
plt.show()
