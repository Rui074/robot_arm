import numpy as np
import modern_robotics as mr
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from diagnose_ik_failure import diagnose_ik_failure

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
    [-np.pi, np.pi],   #joint 2
    [-np.pi, np.pi],   #joint 3
    [-np.pi, np.pi],   #joint 4
    [-np.pi, np.pi],   #joint 5
    [-np.pi, np.pi]    #joint 6
    ]



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
    
    print("diagnosing ik failure...")
    
    return theta, False 

def main():
    client = RemoteAPIClient()
    sim = client.require('sim')
    client.setStepping(True)
    
    joint_handles = [sim.getObject(name) for name in joint_names]
    
    try:
        sim.startSimulation()
    
        M = np.array([[1, 0, 0, -0.18495], [0, 1, 0, 0.150], [0, 0, 1, 0.98655], [0, 0, 0, 1]])
        Slist = np.array([[ 0, 0, 1, 0, 0, 0], 
                  [ 1, 0, 0, 0, 0.07455, 0], 
                  [ 1, 0, 0, 0, 0.49965, 0], 
                  [ 1, 0, 0, 0, 0.8918, 0], 
                  [ 0, 0, 1, 0, 0.110, 0], 
                  [ 1, 0, 0, 0, 0.98655, 0]]).T


        current_angles = np.array([sim.getJointPosition(h) for h in joint_handles])
        print("目前場景真實角度:", np.round(np.rad2deg(current_angles), 2))

        T_start = mr.FKinSpace(M, Slist, current_angles)
    
        theta_end = np.array([0, 0, 0, 0, -np.pi/2, 0]) #<<--------------------------------測資
        T_end = mr.FKinSpace(M, Slist, theta_end)
    
        T_total = 5
        N = 100
        method = 5
        traj = mr.ScrewTrajectory(T_start, T_end, T_total, N, method)
        theta_list_guess = current_angles.copy()
    
        for target in traj:
            
            thetalist_sol, ik_success = ik_with_DLS(Slist, M, target, theta_list_guess, joint_constraints, eomg=0.01, ev=0.003, max_attempts=50, iters=300, lambda_max=0.01, w_threshold=0.01)
            
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
        
        
if __name__ =="__main__":
    main()                  
    
        
    
