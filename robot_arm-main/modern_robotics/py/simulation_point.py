import numpy as np
import modern_robotics as mr
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

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

def ik_with_constraints(Slist, M, target, init_theta, eomg=0.01, ev=0.001):
    max_attempts = 20
    attempt = 0
    

    while attempt < max_attempts:
        thetalist_sol, success = mr.IKinSpace(Slist, M, target, init_theta, eomg=0.01, ev=0.001)
        thetalist_sol = np.arctan2(np.sin(thetalist_sol), np.cos(thetalist_sol))
        
        if success and all(angle[0] <= theta <= angle[1] for theta, angle in zip(thetalist_sol, joint_constraints)):
            ik_success = True
            break
        else:
            init_theta = thetalist_guess + np.random.uniform(-0.05, 0.05, size=6)
            attempt += 1

    else :
        print("IK 解算失敗，請檢查目標位姿是否可達或關節限制是否合理。")
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

    thetalist = np.array([np.pi, -np.pi/2, np.pi/4, np.pi/3, 0, 0]) 
    T = mr.FKinSpace(M, Slist, thetalist)

    print("The transformation matrix T is:", T)
    
    target = T  # 將目標位姿設為剛計算出的正確位姿
    thetalist_guess = np.array([0.5, 0.3, 0.2, 0.2, 0.0, 0.0])
    init_theta = thetalist_guess.copy()
    
    thetalist_sol, ik_success = ik_with_constraints(Slist, M, target, init_theta, eomg=0.01, ev=0.001)

                
    if ik_success:           
        print("IK 解算成功，開始同步控制...")
        start_angles = [sim.getJointPosition(h) for h in joint_handles]
        
        # 進行 100 步的平滑過渡
        for step in range(1, 101):
            interp = np.array(start_angles) + (thetalist_sol - np.array(start_angles)) * (step / 100.0)
            # 寫入角度
            for h, angle in zip(joint_handles, interp):
                sim.setJointPosition(h, float(angle))
            
                # 推進一格模擬時間 (與 CoppeliaSim 物理引擎同步)
            client.step()
        
        angles_deg = np.rad2deg(thetalist_sol)
        angles_deg = np.where(np.abs(angles_deg) < 1e-5, 0, angles_deg)
        
        print("手臂完好無損地移動到位姿！", "最終關節角度:", np.round(angles_deg, 2))
        time.sleep(2)
    

finally:
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    client.setStepping(False)
    
    
    
        
