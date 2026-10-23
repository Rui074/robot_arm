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
theta_target = np.array([0.7, 0.4, -0.5, 0.6, 0.5, -0.4])
delta_theta = theta_target-theta0

t_log = np. linspace(0, 10, 101)

omegalist, alphalist = [], []
for t in t_log:
    s = mr. QuinticTimeScaling(Tf = 10, t = t)
    #find omega, alpha
    t_nor = t / 10
    omegalist.append((30*t_nor**2 - 60*t_nor**3 + 30*t_nor**4)*delta_theta/10)
    alphalist.append((60*t_nor**1 - 180*t_nor**2 + 120*t_nor**3)*delta_theta/10**2)
    
trajectory = mr.JointTrajectory(theta0, theta_target, Tf = 10, N = 101, method = 5)
trajlist, taulist = [], []
for traj, omega, alpha in zip(trajectory, omegalist, alphalist):
    trajlist.append(traj)
    #M
    matrix_M = mr.MassMatrix(traj, Mlist, Glist, Slist)
    #c
    matrix_c = mr.VelQuadraticForces(traj, omega, Mlist, Glist, Slist)
    #g
    matrix_g = mr. GravityForces(traj, g, Mlist, Glist, Slist)
    
    taulist.append(matrix_M @ alpha + matrix_c + matrix_g)

trajlist = np. array(trajlist) 
taulist = np. array(taulist)



plt. figure(figsize = (10, 8))
plt.subplot(2, 1, 1)
for j in range(6):
    plt.plot(t_log, trajlist[:, j], label=f'joint{j+1}')
plt.xlabel('time (s)'); plt.ylabel('traj')
plt.grid(True); plt.legend()
#tau
plt.subplot(2, 1, 2)
for j in range(6):
    plt.plot(t_log, taulist[:, j], label=f'joint{j+1}')
plt.xlabel('time (s)'); plt.ylabel('tau')
plt.grid(True); plt.legend()


plt.tight_layout()
plt.show()
