import numpy as np
import modern_robotics as mr
from simulation_ik_with_DLS import ik_with_DLS
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from diagnose_ik_failure import diagnose_ik_failure
import time


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
    [-np.pi/2, np.pi/2],   #joint 3
    [-np.pi, np.pi],   #joint 4
    [-np.pi, np.pi],   #joint 5
    [-np.pi, np.pi]    #joint 6
]

M = np.array([[0, 0, -1, -0.18495],
              [0, 1,  0,  0.0   ],
              [1, 0,  0,  0.98655],
              [0, 0,  0,  1     ]])

Slist = np.array([[ 0, 0, 1, 0,  0,        0],
                  [-1, 0, 0, 0, -0.07455,  0],
                  [-1, 0, 0, 0, -0.49965,  0],
                  [-1, 0, 0, 0, -0.8918,   0],
                  [ 0, 0, 1, 0,  0.110,    0],
                  [-1, 0, 0, 0, -0.98655,  0]]).T


current_theta = np.array([0.1, 0.15, -0.1, 0.1, 0.1, 0.05])

def check_pose(sim, joint_handles, base_name, tip_name):
    tip = sim.getObject(tip_name)
    base = sim.getObject(base_name)

    def read_tip():
        m = sim.getObjectMatrix(tip, base)
        return np.array([m[0:4], m[4:8], m[8:12], [0, 0, 0, 1]])

    d = 0.3
    for i, h in enumerate(joint_handles):
        theta = np.zeros(6); theta[i] = d
        sim.setJointPosition(h, d)
        T_sim = read_tip()
        T_fk = mr.FKinSpace(M, Slist, theta)
        pos_err = np.linalg.norm(T_sim[:3, 3] - T_fk[:3, 3])
        rot_err = np.linalg.norm(T_sim[:3, :3] - T_fk[:3, :3])
        print(f"joint {i+1}: 位置誤差={pos_err:.5f}  旋轉誤差={rot_err:.5f}")
        sim.setJointPosition(h, 0.0)


def get_target_from_sim(sim, dummy_name, base_name):
    # 將字串路徑轉換為 CoppeliaSim 的整數 Handle
    dummy_handle = sim.getObject(dummy_name)
    base_handle = sim.getObject(base_name)
    
    # 取得相對於 base 的 12 元素矩陣
    matrix_12 = sim.getObjectMatrix(dummy_handle, base_handle)
    
    # 轉換成 numpy 4x4 矩陣
    T = np.array([
        [matrix_12[0], matrix_12[1], matrix_12[2], matrix_12[3]],
        [matrix_12[4], matrix_12[5], matrix_12[6], matrix_12[7]],
        [matrix_12[8], matrix_12[9], matrix_12[10], matrix_12[11]],
        [0, 0, 0, 1]
    ])
    return T

def try_waypoint_with_T(T, guess):
    sol, ok = ik_with_DLS(Slist, M, T, guess, joint_constraints,
                                   eomg=0.05, ev=0.0012, max_attempts=50, iters=300,
                                   lambda_max=0.01, w_threshold=0.01)
    if ok:
        return sol  
    else:
        diagnose_ik_failure(Slist, M, T, guess, joint_constraints, restarts=20, iters=300)
        return None

    
def main():
    try:
        client = RemoteAPIClient()
        sim = client.require('sim')
        client.setStepping(True)
        joint_handles = [sim.getObject(name) for name in joint_names]
        
        
        sim.startSimulation()
        
         
        base_name = '/UR5'
        tip_name = '/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint/link/connection'
        

        s = check_pose(sim, joint_handles, base_name, tip_name)
        T1 = get_target_from_sim(sim, '/UR5/prep_grasp', base_name)
        T2 = get_target_from_sim(sim, '/UR5/grasp', base_name)
        T3 = get_target_from_sim(sim, '/UR5/lift', base_name)
        T4 = get_target_from_sim(sim, '/UR5/prep_place', base_name)
        T5 = get_target_from_sim(sim, '/UR5/place', base_name)
        
        T_list = [T1, T2, T3, T4, T5]
        
        guess = current_theta.copy()
        
        result_1 = try_waypoint_with_T(T1, guess)
        if result_1 is None: print("pre-grasp 失敗"); return
        
        result_2 = try_waypoint_with_T(T2, result_1)
        if result_2 is None: print("grasp 失敗"); return
        
        result_3 = try_waypoint_with_T(T3, result_2)
        if result_3 is None: print("lift 失敗"); return
        
        result_4 = try_waypoint_with_T(T4, result_3)
        if result_4 is None: print("pre-place 失敗"); return
        
        result_5 = try_waypoint_with_T(T5, result_4)
        if result_5 is None: print("place 失敗"); return
        
        R_list = [result_1, result_2, result_3, result_4, result_5]
        
        Tf = 3
        N = 50
        method = 5 
        for i in range(5):
            traj = mr.JointTrajectory(guess, R_list[i], Tf, N, method)
            for target in traj:
                for h, angle in zip(joint_handles, target):
                    sim.setJointPosition(h, float(angle))        
                client.step()
            guess = R_list[i]
            angles_deg = np.rad2deg(R_list[i])
            angles_deg = np.where(np.abs(angles_deg) < 1e-5, 0, angles_deg)
            print("手臂完好無損地移動到位姿！", "最終關節角度:", np.round(angles_deg, 2))
          
    finally:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
        client.setStepping(False)
        
if __name__ =="__main__":
    main()
