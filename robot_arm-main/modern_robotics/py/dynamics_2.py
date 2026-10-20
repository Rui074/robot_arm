import numpy as np
import modern_robotics as mr
from arm_data import robot_info
import matplotlib.pyplot as plt

def ik_with_DLS(Slist, M, target, theta_list_guess, joint_constraints, eomg, ev, max_attempts, iters, lambda_max, w_threshold):

    guess = theta_list_guess.copy()
    #ik standard
    for _ in range(max_attempts):
        thetalist_sol, success = mr.IKinSpace(Slist, M, target, guess, eomg=eomg, ev=ev)
        thetalist_sol = np.arctan2(np.sin(thetalist_sol), np.cos(thetalist_sol))
        
        if success and all(angle[0] <= theta <= angle[1] for theta, angle in zip(thetalist_sol, joint_constraints)):
            return thetalist_sol, True
        else:
            guess = theta_list_guess + np.random.uniform(-0.1, 0.1, size=6)

    #DLS method
    print("standard IKinSpace fails, use DLS method")
    for _ in range(100):
        theta = theta_list_guess + np.random.uniform(-1, 1, size=6)
        for _ in range(iters):
            Tsb = mr.FKinSpace(M, Slist, theta)
            Vs = mr.Adjoint(Tsb) @ mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Tsb) @ target))  # 修正: 轉回 space frame
            werr, verr = np.linalg.norm(Vs[:3]), np.linalg.norm(Vs[3:])
            if werr <= eomg and verr <= ev: 
                theta_sol = np.arctan2(np.sin(theta), np.cos(theta))
                if all(angle[0] <= theta <= angle[1] for theta, angle in zip(theta_sol, joint_constraints)):
                    return theta_sol, True
                break
    
            J = mr.JacobianSpace(Slist, theta)
            JJT = J @ J.T
            w = np.sqrt(max(0, np.linalg.det(JJT)))          
            damp = lambda_max * (1 - w / w_threshold) ** 2 if w < w_threshold else 0.0
    
            delta_theta = J.T @ np.linalg.solve(JJT + damp**2 * np.eye(6), Vs)  
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

T_total = 5
N = 100
dt = T_total/(N-1)
method = 5
trajectory = mr.ScrewTrajectory(T_start, T_end, T_total, N, method)

thetalist_guess = theta0.copy()
thetalist = []
for traj in trajectory:
    thetalist_sol, ik_success = ik_with_DLS(Slist, M, traj, thetalist_guess, joint_constraints, eomg=0.01, ev=0.003, max_attempts=50, iters=300, lambda_max=0.01, w_threshold=0.01)
    if ik_success:
        thetalist.append(thetalist_sol)
        all_success = True
    else:
        print("target cannot reach")
        all_success = False
        break
    
if all_success:
    thetalist = np.array(thetalist)
    omegalist = np.gradient(thetalist, dt, axis=0)
    alphalist = np.gradient(omegalist, dt, axis=0)
    
    taulist = []
    for theta, omega, alpha in zip(thetalist, omegalist, alphalist):
        tau = mr.InverseDynamics(theta, omega, alpha, g, np.zeros(6), Mlist, Glist, Slist)
        taulist.append(tau) 
    
    taulist = np.array(taulist)
    t_log = np.linspace(0, T_total, N)
    plt. figure(figsize = (10, 8))
    plt.subplot(2, 1, 1)
    for j in range(6):
        plt.plot(t_log, thetalist[:, j], label=f'joint{j+1}')
    plt.xlabel('time (s)'); plt.ylabel('theta')
    plt.grid(True); plt.legend()
    
    plt.subplot(2, 1, 2)
    for j in range(6):
        plt.plot(t_log, taulist[:, j], label=f'joint{j+1}')
    plt.xlabel('time (s)'); plt.ylabel('tau')
    plt.grid(True); plt.legend()
    
    plt.tight_layout()
    plt.show()