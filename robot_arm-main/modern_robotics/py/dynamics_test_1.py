import numpy as np
import modern_robotics as mr
from arm_data import robot_info
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import matplotlib.pyplot as plt


dt, T_end, intRes = 0.001, 3.0, 8
N = int(T_end / dt)
theta0 = np.array([0.7, 0.4, -0.5, 0.6, 0.5, -0.4])
dtheta0 = np.zeros(6)


try:
    client = RemoteAPIClient()
    sim = client.require('sim')
    client.setStepping(True)
    joint_handles = [sim.getObject(name) for name in robot_info['joint_names']]
    
    # 模擬開始前:設定初始姿態
    for h, q in zip(joint_handles, theta0):
        sim.setJointPosition(h, float(q))

    sim.startSimulation()
    
    MAX_TORQUE = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
    
    Kp = np.array([200.0, 200.0, 200.0, 40.0, 40.0, 5.0])  
    Kd = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 0.5]) 
    #Ki = np.array([50.0, 50.0, 50.0, 10.0, 10.0, 1.0])
    Ki = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    theta_target = theta0
    dtheta_target = np.zeros(6)

    error_integral = np.zeros(6)

    t_log, theta_log, dtheta_log = [], [], []
    for k in range(N):
        theta  = np.array([sim.getJointPosition(h) for h in joint_handles])
        dtheta = np.array([sim.getJointVelocity(h) for h in joint_handles])
        
        t_log.append(sim.getSimulationTime())
        theta_log.append(theta)
        dtheta_log.append(dtheta)
        
        error = theta_target - theta
        error_integral += error * dt
        error_integral = np.clip(error_integral, -10.0, 10.0)
            
        tau_g = mr.GravityForces(theta, robot_info['g'], robot_info['Mlist'], robot_info['Glist'], robot_info['Slist'])   
        tau_pid = Kp * error + Kd * (dtheta_target - dtheta) + Ki * error_integral
        tau_total = tau_g + tau_pid
        tau_total = np.clip(tau_total, -MAX_TORQUE, MAX_TORQUE)
        

        for h, t in zip(joint_handles, tau_total):
            if t == 0.0:
                sim.setJointTargetVelocity(h, 0.0)
                sim.setJointTargetForce(h, 0.0)
            else:
                sim.setJointTargetForce(h, float(t), True)          
        
        client.step()
    
    theta_log = np.array(theta_log)
    dtheta_log = np.array(dtheta_log)
    

finally:
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    client.setStepping(False)


plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
for j in range(6):
    plt.plot(t_log, theta_log[:, j] - theta_log[0, j], label=f'joint{j+1}')
plt.ylabel('theta - theta0 (rad)')
plt.grid(True); plt.legend()

plt.subplot(2, 1, 2)
for j in range(6):
    plt.plot(t_log, dtheta_log[:, j], label=f'joint{j+1}')
plt.xlabel('time (s)'); plt.ylabel('dtheta (rad/s)')
plt.grid(True); plt.legend()

plt.tight_layout()
plt.show()