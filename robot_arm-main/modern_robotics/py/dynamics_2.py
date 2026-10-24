import numpy as np
from scipy.signal import savgol_filter
import modern_robotics as mr
from arm_data import robot_info
import matplotlib.pyplot as plt

def ik_with_DLS(Slist, M, target, theta_list_guess, joint_constraints, eomg, ev, lambda_max, w_threshold):
    guess = theta_list_guess.copy()
    
    thetalist_sol, success = mr.IKinSpace(Slist, M, target, guess, eomg=eomg, ev=ev)
    
    if success:
        diff = thetalist_sol - theta_list_guess
        diff_cont = np.arctan2(np.sin(diff), np.cos(diff))
        thetalist_sol = theta_list_guess + diff_cont
        
        in_limit = all(angle[0] <= theta <= angle[1] for theta, angle in zip(thetalist_sol, joint_constraints))
        if in_limit:
            return thetalist_sol, True

    theta = theta_list_guess.copy()
    for _ in range(100):
        Tsb = mr.FKinSpace(M, Slist, theta)
        Vs = mr.Adjoint(Tsb) @ mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Tsb) @ target))
        werr, verr = np.linalg.norm(Vs[:3]), np.linalg.norm(Vs[3:])
        
        if werr <= eomg and verr <= ev:
            diff = theta - theta_list_guess
            diff_cont = np.arctan2(np.sin(diff), np.cos(diff))
            theta_sol = theta_list_guess + diff_cont
            
            in_limit = all(angle[0] <= theta <= angle[1] for theta, angle in zip(thetalist_sol, joint_constraints))
            if in_limit:
                return theta_sol, True
            break

        J = mr.JacobianSpace(Slist, theta)
        JJT = J @ J.T
        w = np.sqrt(max(0, np.linalg.det(JJT)))
        damp = lambda_max * (1 - w / w_threshold)**2 if w < w_threshold else 0.0

        delta_theta = J.T @ np.linalg.solve(JJT + (damp**2 + 1e-6) * np.eye(6), Vs)
        theta = theta + delta_theta

    return theta, False

link_names = robot_info['link_names']
joint_names = robot_info['joint_names']
joint_constraints = robot_info['joint_constraints']
M = robot_info['M']
Slist = robot_info['Slist']
Blist = robot_info['Blist']
Mlist = robot_info['Mlist']
Glist = robot_info['Glist']
g = np.array([0, 0, -9.81])
theta0 = np.zeros(6)
theta_target = np.array([np.pi/2, -np.pi/4, np.pi/6, np.pi/3, np.pi/2, 0])
delta_theta = theta_target-theta0

T_start = mr.FKinSpace(M, Slist, theta0)
T_end = mr.FKinSpace(M, Slist, theta_target)

T_total = 10
N = 201
dt = T_total/(N-1)
method = 5
trajectory = mr.ScrewTrajectory(T_start, T_end, T_total, N, method)

thetalist_guess = theta0.copy()
thetalist = []
for traj in trajectory:
    thetalist_sol, ik_success = ik_with_DLS(Slist, M, traj, thetalist_guess, joint_constraints, eomg=0.01, ev=0.003, lambda_max=0.01, w_threshold=0.01)
    if ik_success:
        thetalist.append(thetalist_sol)
        thetalist_guess = thetalist_sol.copy()
        all_success = True
    else:
        print("target cannot reach")
        all_success = False
        break
    
if all_success:
    thetalist = np.array(thetalist)
    thetalist_arr = np.unwrap(thetalist, axis=0)
    thetalist_smooth = savgol_filter(thetalist_arr, window_length=9, polyorder=2, axis=0)
    omegalist = np.gradient(thetalist_smooth, dt, axis=0)
    alphalist = np.gradient(omegalist, dt, axis=0)
    
    taulist = []
    for theta, omega, alpha in zip(thetalist_smooth, omegalist, alphalist):
        tau = mr.InverseDynamics(theta, omega, alpha, g, np.zeros(6), Mlist, Glist, Slist)
        taulist.append(tau) 
    
    taulist = np.array(taulist)
    t_log = np.linspace(0, T_total, N)
    plt. figure(figsize = (10, 8))
    plt.subplot(2, 1, 1)
    for j in range(6):
        plt.plot(t_log, thetalist_smooth[:, j], label=f'joint{j+1}')
    plt.xlabel('time (s)'); plt.ylabel('theta')
    plt.grid(True); plt.legend()
    
    plt.subplot(2, 1, 2)
    for j in range(6):
        plt.plot(t_log, taulist[:, j], label=f'joint{j+1}')
    plt.xlabel('time (s)'); plt.ylabel('tau')
    plt.grid(True); plt.legend()
    
    plt.tight_layout()
    plt.show()