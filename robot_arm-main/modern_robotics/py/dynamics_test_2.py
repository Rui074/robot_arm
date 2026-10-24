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

g = robot_info['g']
Mlist, Glist, Slist = robot_info['Mlist'], robot_info['Glist'], robot_info['Slist']

dt = 0.001                      # 需與 CoppeliaSim 模擬步長一致
MAX_TORQUE = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
Kp = np.array([300.0, 300.0, 200.0, 50.0, 30.0, 10.0])
Kd = np.array([30.0, 30.0, 20.0, 5.0, 3.0, 1.0])

poses = {
    # "A": np.array([0.7, 0.4, -0.5, 0.6, 0.5, -0.4]),
    # "B": np.array([np.pi/2, -np.pi/4, np.pi/6, np.pi/3, np.pi/2, 0.0]),
    "C": np.array([0.0, np.pi/2, 0.0, 0.0, 0.0, 0.0]),   # 手臂水平，重力力矩最大
}


def run(pose, controller, T):
    """重設到 pose,跑 T 秒,controller(theta, omega) 回傳力矩。回傳 theta, omega, tau 的紀錄。"""
    n = int(T / dt)
    th_log, om_log, tau_log = [], [], []
    try:
        for h, q in zip(joint_handles, pose):
            sim.setJointPosition(h, float(q))
        sim.startSimulation()
        for _ in range(n):
            theta = np.array([sim.getJointPosition(h) for h in joint_handles])
            omega = np.array([sim.getJointVelocity(h) for h in joint_handles])
            tau = np.clip(controller(theta, omega), -MAX_TORQUE, MAX_TORQUE)
            for h, t in zip(joint_handles, tau):
                sim.setJointTargetForce(h, float(t), True)
            th_log.append(theta); om_log.append(omega); tau_log.append(tau)
            client.step()
    finally:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
    return np.array(th_log), np.array(om_log), np.array(tau_log)


np.set_printoptions(precision=3, suppress=True, linewidth=150)

# try:
#     for name, pose in poses.items():
#         print(f"\n===== 姿態 {name}: {pose} =====")

#         # ---- 測試 A：PD 撐住，讀靜止時的力矩 ----
#         def pd_hold(theta, omega):
#             return Kp * (pose - theta) + Kd * (0 - omega)

#         th, om, tau = run(pose, pd_hold, T=3.0)
#         last = slice(-500, None)                       # 最後 0.5 秒
#         tau_sim = tau[last].mean(axis=0)
#         tau_mr = mr.GravityForces(pose, g, Mlist, Glist, Slist)
#         print("[A] 最後0.5s 最大|速度| (應接近0):", np.abs(om[last]).max(axis=0))
#         print("[A] sim 靜止力矩 :", tau_sim)
#         print("[A] mr 重力力矩  :", tau_mr)
#         print("[A] 差 (sim-mr)  :", tau_sim - tau_mr)
        
        
# finally:
#     client.setStepping(False)
    
client.setStepping(True)
    
angles = np.linspace(-np.pi, np.pi, 13)
try:
    for a in angles:
        pose = np.zeros(6); pose[1] = a

        def pd_hold(theta, omega, pose=pose):
                    return Kp * (pose - theta) + Kd * (0 - omega)

        th, om, tau = run(pose, pd_hold, T=2.0)
        tau_sim = tau[-300:].mean(axis=0)
        tau_mr = mr.GravityForces(pose, g, Mlist, Glist, Slist)
        print(f"θ2設定={a:6.2f} 實際={th[-1,1]:6.2f} | "
            f"sim τ2,τ3={tau_sim[1]:8.2f},{tau_sim[2]:8.2f} | "
            f"mr τ2,τ3={tau_mr[1]:8.2f},{tau_mr[2]:8.2f}")
finally:
    client.setStepping(False)