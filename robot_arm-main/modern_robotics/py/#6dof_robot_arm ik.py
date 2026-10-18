#6dof_robot_arm ik

import numpy as np
import modern_robotics as mr

M = np.array([[1, 0, 0, -0.18495], [0, 1, 0, 0.150], [0, 0, 1, 0.98655], [0, 0, 0, 1]])
Slist = np.array([[ 0, 0, 1, 0, 0, 0], 
                  [ 1, 0, 0, 0, 0.07455, 0], 
                  [ 1, 0, 0, 0, 0.49965, 0], 
                  [ 1, 0, 0, 0, 0.8918, 0], 
                  [ 0, 0, 1, 0, 0.110, 0], 
                  [ 1, 0, 0, 0, 0.98655, 0]]).T

thetalist = np.array([0, -np.pi*2/3, 0, np.pi/4, 0, -np.pi/4]) 
T = mr.FKinSpace(M, Slist, thetalist)

print("The transformation matrix T is:", T)
target = T  # 將目標位姿設為剛計算出的正確位姿

thetalist_guess = np.array([0.0, 0.3, 0.0, 0.2, 0.0, 0.2])

thetalist_sol, success = mr.IKinSpace(Slist, M, target, thetalist_guess, eomg=0.01, ev=0.001)

if success:
    # 1. Convert the solution angles from radians to degrees
    thetalist_sol = np.arctan2(np.sin(thetalist_sol), np.cos(thetalist_sol))
    angles_deg = np.rad2deg(thetalist_sol)
    angles_deg = np.where(np.abs(angles_deg) < 1e-5, 0, angles_deg) # 小於 1e-5 直接歸零
    
    print("The solution for the joint angles is:", np.round(angles_deg, 2))
else:
    print("The inverse kinematics solution did not converge.")
