import numpy as np
import modern_robotics as mr
from simulation_ik_with_DLS import ik_with_DLS
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from diagnose_ik_failure import diagnose_ik_failure
import time

link_length = {'link_2': 0.07884,
                'link_3':0.4251,
                'link_4':0.39215,
                'link_56':0.09538}

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

M = np.array([[1, 0, 0, -0.18495], [0, 1, 0, 0.150], [0, 0, 1, 0.98655], [0, 0, 0, 1]])

Slist = np.array([[ 0, 0, 1, 0, 0, 0], 
                  [ 1, 0, 0, 0, 0.07455, 0], 
                  [ 1, 0, 0, 0, 0.49965, 0], 
                  [ 1, 0, 0, 0, 0.8918, 0], 
                  [ 0, 0, 1, 0, 0.110, 0], 
                  [ 1, 0, 0, 0, 0.98655, 0]]).T


current_theta = np.array([0.1, 0.15, -0.1, 0.1, 0.1, 0.05])

def try_waypoint(name, position, R, guess):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = position
    sol, ok = ik_with_DLS(Slist, M, T, guess, joint_constraints,
                           eomg=0.005, ev=0.005, max_attempts=50, iters=300,
                           lambda_max=0.01, w_threshold=0.01)
    if ok:
        return T, sol  
    else:
        diagnose_ik_failure(Slist, M, T, guess, joint_constraints, restarts=20, iters=300)
        return None, None
    
def main():
    try:
        client = RemoteAPIClient()
        sim = client.require('sim')
        client.setStepping(True)
        joint_handles = [sim.getObject(name) for name in joint_names]
        
        sim.startSimulation()
        R_down_1 = np.array([[0,0,1],[1,0,0],[0,1,0]])
        R_down_2 = np.array([[0,0,1],[0,0,-1],[-1,0,0]])
        guess = current_theta.copy()
        T1, result_1 = try_waypoint("pre-grasp", [-0.1064, 0.49377, 0.4338], R_down_1, guess)
        if result_1 is None:
            print("pre-grasp 失敗，停止")
            return 
        
        T2, result_2 = try_waypoint("grasp",     [-0.1064, 0.680133, 0.0362892], R_down_1, result_1)
        if result_2 is None:
            print("grasp 失敗，停止")
            return
        
        T3, result_3 = try_waypoint("lift",      [-0.1064, 0.49377, 0.4338], R_down_1, result_2)
        if result_3 is None:
            print("lift 失敗，停止")
            return
        
        T4, result_4= try_waypoint("pre-place", [-0.49377, -0.1064, 0.4338], R_down_2, result_3)
        if result_4 is None:
            print("pre-place 失敗，停止")
            return
        
        T5, result_5= try_waypoint("place",     [-0.680133, -0.1064, 0.0362892], R_down_2, result_4)
        if result_5 is None:
            print("place 失敗，停止")
            return
        
        T_list = [T1, T2, T3, T4, T5]
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