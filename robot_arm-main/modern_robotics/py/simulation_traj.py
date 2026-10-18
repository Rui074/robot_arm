import numpy as np
import modern_robotics as mr
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from diagnose_ik_failure import diagnose_ik_failure

# 1. 初始化 ZMQ 連線
client = RemoteAPIClient()
sim = client.require('sim')

# 2. 開啟同步模式 (極度重要：防止 Play 按下瞬間的物理爆發)
client.setStepping(True)


joint_names = [
    '/UR5/joint',
    '/UR5/link/joint',
    '/UR5/link/joint/link/joint',
    '/UR5/link/joint/link/joint/link/joint',
    '/UR5/link/joint/link/joint/link/joint/link/joint',
    '/UR5/link/joint/link/joint/link/joint/link/joint/link/joint'
]

joint_constraints = [
    [-np.pi, np.pi],   #joint 1
    [-np.pi/2, np.pi/2],   #joint 2
    [-np.pi, np.pi],   #joint 3
    [-np.pi, np.pi],   #joint 4
    [-np.pi, np.pi],   #joint 5
    [-np.pi, np.pi]    #joint 6
]

joint_handles = [sim.getObject(name) for name in joint_names]



def ik_with_constraints(Slist, M, target, theta_list_guess, eomg=0.01, ev=0.001):
    max_attempts = 50
    attempt = 0
    guess = theta_list_guess.copy()
    
    while attempt < max_attempts:
        last_guess_tried = guess.copy()
        thetalist_sol, success = mr.IKinSpace(Slist, M, target, theta_list_guess, eomg=0.01, ev=0.001)
        thetalist_sol = np.arctan2(np.sin(thetalist_sol), np.cos(thetalist_sol))
        
        if success and all(angle[0] <= theta <= angle[1] for theta, angle in zip(thetalist_sol, joint_constraints)):
            ik_success = True
            break
        else:
            theta_list_guess = theta_list_guess + np.random.uniform(-0.1, 0.1, size=6)
            attempt += 1

    else :
        diagnose_ik_failure(Slist, M, target, last_guess_tried, joint_constraints)
        ik_success = False
        
    return thetalist_sol, ik_success



# 4. 開始模擬
try:
    sim.startSimulation()
    
    M = np.array([[1, 0, 0, -0.18495], [0, 1, 0, 0.150], [0, 0, 1, 0.98655], [0, 0, 0, 1]])
    Slist = np.array([[ 0, 0, 1, 0, 0, 0], 
                  [ 1, 0, 0, 0, 0.07455, 0], 
                  [ 1, 0, 0, 0, 0.49965, 0], 
                  [ 1, 0, 0, 0, 0.8918, 0], 
                  [ 0, 0, 1, 0, 0.110, 0], 
                  [ 1, 0, 0, 0, 0.98655, 0]]).T


    current_angles = [sim.getJointPosition(h) for h in joint_handles]
    
    
    T_start = mr.FKinSpace(M, Slist, current_angles)
    
    theta_end = np.array([np.pi/2, -np.pi/4, np.pi/6, np.pi/3, np.pi/2, 0]) #<<--------------------------------測資
    T_end = mr.FKinSpace(M, Slist, theta_end)
    
    T_total = 5
    N = 100
    method = 5
    traj = mr.ScrewTrajectory(T_start, T_end, T_total, N, method)
    theta_list_guess = current_angles.copy()
    
    for T_target in traj:
        
        thetalist_sol, ik_success = ik_with_constraints(Slist, M, T_target, theta_list_guess, eomg=0.01, ev=0.001)
        
        if ik_success:           
            start_angles = [sim.getJointPosition(h) for h in joint_handles]
            # 寫入角度
            for h, angle in zip(joint_handles, thetalist_sol):
                sim.setJointPosition(h, float(angle))        
            client.step()
            theta_list_guess = thetalist_sol
            all_success = True

        else:
            all_success = False
            break
    
    if all_success:
        angles_deg = np.rad2deg(thetalist_sol)
        angles_deg = np.where(np.abs(angles_deg) < 1e-5, 0, angles_deg)
            
        print("手臂完好無損地移動到位姿！", "最終關節角度:", np.round(angles_deg, 2))
    

finally:
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    client.setStepping(False)