import numpy as np
import modern_robotics as mr
from arm_data import robot_info
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import matplotlib.pyplot as plt


client = RemoteAPIClient()
sim = client.require('sim')
client.setStepping(True)
joint_handles = [sim.getObject(name) for name in robot_info['joint_names']]

dt, T_end, intRes = 0.001, 0.3, 8
N = int(T_end / dt)
theta0 = np.array([0.7, 0.4, -0.5, 0.6, 0.5, -0.4])

try:
    for h, q in zip(joint_handles, theta0):
        sim.setJointPosition(h, float(q))
    sim.startSimulation()
    theta_sim = []
    for k in range(N):
        theta_sim.append([sim.getJointPosition(h) for h in joint_handles])
        for h in joint_handles:
            sim.setJointTargetForce(h, 0.0, True)
        client.step()
    theta_sim = np.array(theta_sim)
    
finally:
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    client.setStepping(False)
    

thetamat, _ = mr.ForwardDynamicsTrajectory(
    theta0, np.zeros(6), np.zeros((N, 6)), robot_info['g'], np.zeros((N, 6)),
    robot_info['Mlist'], robot_info['Glist'], robot_info['Slist'], dt, 8)
    
t = np.arange(N) * dt
err = theta_sim - thetamat
print("各關節最大差距 (rad):", np.round(np.max(np.abs(err), axis=0), 5))

fig, axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
for j, ax in enumerate(axs.flat):
    ax.plot(t, thetamat[:, j] - theta0[j], label='mr')
    ax.plot(t, theta_sim[:, j] - theta0[j], '--', label='CoppeliaSim')
    ax.set_title(f'joint{j+1}'); ax.grid(True)
axs[0, 0].legend()
plt.tight_layout(); plt.show()