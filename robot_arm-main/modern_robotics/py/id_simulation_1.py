import time
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import modern_robotics as mr
from arm_data import robot_info
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def ik_with_DLS(Slist, M, target, theta_list_guess, joint_constraints,
                eomg, ev, lambda_max, w_threshold):
    guess = theta_list_guess.copy()

    thetalist_sol, success = mr.IKinSpace(Slist, M, target, guess, eomg=eomg, ev=ev)

    if success:
        diff = thetalist_sol - theta_list_guess
        diff_cont = np.arctan2(np.sin(diff), np.cos(diff))
        thetalist_sol = theta_list_guess + diff_cont

        in_limit = all(lo <= th <= hi
                       for th, (lo, hi) in zip(thetalist_sol, joint_constraints))
        if in_limit:
            return thetalist_sol, True

    theta = theta_list_guess.copy()
    for _ in range(100):
        Tsb = mr.FKinSpace(M, Slist, theta)
        Vs = mr.Adjoint(Tsb) @ mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Tsb) @ target))
        werr, verr = np.linalg.norm(Vs[:3]), np.linalg.norm(Vs[3:])

        if werr <= eomg and verr <= ev:
            diff = theta - theta_list_guess
            diff_cont = np.arctan2(np.sin(diff), np.cos(diff))
            theta_sol = theta_list_guess + diff_cont

            in_limit = all(lo <= th <= hi
                           for th, (lo, hi) in zip(theta_sol, joint_constraints))
            if in_limit:
                return theta_sol, True
            break

        J = mr.JacobianSpace(Slist, theta)
        JJT = J @ J.T
        w = np.sqrt(max(0, np.linalg.det(JJT)))
        damp = lambda_max * (1 - w / w_threshold) ** 2 if w < w_threshold else 0.0

        delta_theta = J.T @ np.linalg.solve(JJT + (damp ** 2 + 1e-6) * np.eye(6), Vs)
        theta = theta + delta_theta

    return theta, False


# ------------------------------------------------------------------
# 機器人參數
# ------------------------------------------------------------------
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
theta_target = np.array([np.pi / 2, -np.pi / 4, np.pi / 6, np.pi / 3, np.pi / 2, 0])

MAX_TORQUE = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
Kp = np.array([30.0, 100.0, 70.0, 15.0, 5.0, 1.0])
Kd = np.array([1.5, 10.0, 7.0, 1.0, 0.5, 0.1])
Ki = np.array([0.0, 4.0, 2.0, 0.0, 0.0, 0.0])

# ------------------------------------------------------------------
# 軌跡規劃 (IK 仍然只算 N 個點)
# ------------------------------------------------------------------
T_total = 10
N = 201
method = 5
HOLD_T = 2.0        # 軌跡結束後多靜止幾秒觀察穩態誤差，設 0 關閉
dt = 0.01           # 控制 / 模擬步長，需與 CoppeliaSim 一致

T_start = mr.FKinSpace(M, Slist, theta0)
T_end = mr.FKinSpace(M, Slist, theta_target)
trajectory = mr.ScrewTrajectory(T_start, T_end, T_total, N, method)

thetalist_guess = theta0.copy()
thetalist = []
all_success = True
for traj in trajectory:
    thetalist_sol, ik_success = ik_with_DLS(
        Slist, M, traj, thetalist_guess, joint_constraints,
        eomg=0.01, ev=0.003, lambda_max=0.01, w_threshold=0.01)
    if ik_success:
        thetalist.append(thetalist_sol)
        thetalist_guess = thetalist_sol.copy()
    else:
        print("target cannot reach")
        all_success = False
        break

if not all_success:
    raise SystemExit

thetalist = np.array(thetalist)
thetalist_arr = np.unwrap(thetalist, axis=0)
thetalist_smooth = savgol_filter(thetalist_arr, window_length=9, polyorder=2, axis=0)

# ------------------------------------------------------------------
# 用三次樣條在細時間軸上解析求 theta / omega / alpha (取代 np.gradient)
# ------------------------------------------------------------------
t_traj = np.linspace(0, T_total, N)
cs = CubicSpline(t_traj, thetalist_smooth, axis=0, bc_type='clamped')  # 頭尾速度 = 0

t_ctrl = np.arange(0, T_total + dt / 2, dt)
theta_ref = cs(t_ctrl)
omega_ref = cs(t_ctrl, 1)
alpha_ref = cs(t_ctrl, 2)

if HOLD_T > 0:
    n_hold = int(round(HOLD_T / dt))
    t_ctrl = np.concatenate([t_ctrl, t_ctrl[-1] + dt * np.arange(1, n_hold + 1)])
    theta_ref = np.vstack([theta_ref, np.tile(theta_ref[-1], (n_hold, 1))])
    omega_ref = np.vstack([omega_ref, np.zeros((n_hold, 6))])
    alpha_ref = np.vstack([alpha_ref, np.zeros((n_hold, 6))])

print("max |alpha_ref| (rad/s^2):", np.round(np.abs(alpha_ref).max(axis=0), 3))

tau_ff_list = np.array([
    mr.InverseDynamics(th, om, al, g, np.zeros(6), Mlist, Glist, Slist)
    for th, om, al in zip(theta_ref, omega_ref, alpha_ref)
])

# ------------------------------------------------------------------
# CoppeliaSim 模擬
# ------------------------------------------------------------------
theta_actual_list, t_actual_list = [], []
tau_ff_log, tau_pid_log, tau_total_log = [], [], []

client = RemoteAPIClient()
sim = client.require('sim')
try:
    client.setStepping(True)
    joint_handles = [sim.getObject(name) for name in robot_info['joint_names']]

    sim.setFloatParam(sim.floatparam_simulation_time_step, dt)
    print("sim dt =", sim.getSimulationTimeStep())

    for h, q in zip(joint_handles, theta0):
        sim.setJointPosition(h, float(q))

    sim.startSimulation()

    I_error = np.zeros(6)   # 放在迴圈外，才會真的累積
    for th_r, om_r, tau_ff in zip(theta_ref, omega_ref, tau_ff_list):
        th_a = np.array([sim.getJointPosition(h) for h in joint_handles])
        om_a = np.array([sim.getJointVelocity(h) for h in joint_handles])
        t_a = sim.getSimulationTime()

        P_error = th_r - th_a
        I_error = np.clip(I_error + P_error * dt, -10.0, 10.0)
        D_error = om_r - om_a
        tau_pid = Kp * P_error + Kd * D_error + Ki * I_error
        tau_total = np.clip(tau_ff + tau_pid, -MAX_TORQUE, MAX_TORQUE)

        for h, t in zip(joint_handles, tau_total):
            sim.setJointTargetForce(h, float(t), True)

        theta_actual_list.append(th_a)
        t_actual_list.append(t_a)
        tau_ff_log.append(tau_ff)
        tau_pid_log.append(tau_pid)
        tau_total_log.append(tau_total)
        client.step()

finally:
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    client.setStepping(False)

# ------------------------------------------------------------------
# 分析與繪圖
# ------------------------------------------------------------------
theta_actual = np.array(theta_actual_list)
t_actual = np.array(t_actual_list)
tau_ff_arr = np.array(tau_ff_log)
tau_pid_arr = np.array(tau_pid_log)
tau_total_arr = np.array(tau_total_log)
err = theta_ref - theta_actual

np.set_printoptions(precision=4, suppress=True, linewidth=150)
dts = np.diff(t_actual)
print("sim 時間步長 min/max:", dts.min(), dts.max(), "(應全部等於", dt, ")")
print("最大 |誤差| (rad):", np.abs(err).max(axis=0))
print("最後誤差 (rad)   :", err[-1])
print("PID 力矩 RMS (Nm):", np.sqrt((tau_pid_arr ** 2).mean(axis=0)))
print("前饋力矩 RMS (Nm):", np.sqrt((tau_ff_arr ** 2).mean(axis=0)))

fig, axs = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for j in range(6):
    axs[0].plot(t_actual, theta_actual[:, j], color=colors[j], label=f'joint{j+1} act.')
    axs[0].plot(t_ctrl, theta_ref[:, j], '--', color='gray')
    axs[1].plot(t_actual, err[:, j], color=colors[j], label=f'joint{j+1} err.')
    axs[2].plot(t_actual, tau_ff_arr[:, j], color=colors[j], label=f'joint{j+1} FF')
    axs[3].plot(t_actual, tau_pid_arr[:, j], color=colors[j], label=f'joint{j+1} PID')
axs[0].set_title('Actual (solid) vs reference (dashed)'); axs[0].set_ylabel('Angle (rad)')
axs[1].set_title('theta error (ref - actual)'); axs[1].set_ylabel('Error (rad)')
axs[2].set_title('Feedforward torque'); axs[2].set_ylabel('Torque (Nm)')
axs[3].set_title('PID torque'); axs[3].set_ylabel('Torque (Nm)')
axs[3].set_xlabel('Sim Time (s)')
for ax in axs:
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=7, ncol=2)
plt.tight_layout()
plt.show()