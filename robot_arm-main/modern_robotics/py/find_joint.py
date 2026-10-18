import numpy as np
import modern_robotics as mr
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from arm_data import robot_info

client = RemoteAPIClient()
sim = client.require('sim')



def read_scene(sim, link_names, joint_names, Slist):
    jh = [sim.getObject(n) for n in joint_names]
    lh = [sim.getObject(n) for n in link_names]

    # 1) 手臂必須在零位
    q = np.array([sim.getJointPosition(h) for h in jh])
    assert np.max(np.abs(q)) < 1e-6, f"手臂不在零位: {np.round(q, 4)}"

    # 2) 基座座標系原點(世界座標): x,y 取 joint1,z 由 joint2 的高度反推
    pj = np.array([sim.getObjectPosition(h, sim.handle_world) for h in jh])
    base = np.array([pj[0][0], pj[0][1], pj[1][2] + Slist[4, 1]])

    # 3) 驗證 Slist 跟場景幾何一致
    rel = pj - base
    for j in (1, 2, 3, 5):
        assert abs(rel[j][1]) < 2e-3 and abs(rel[j][2] + Slist[4, j]) < 2e-3, \
            f"關節{j+1} 位置跟 Slist 對不上: {np.round(rel[j], 4)}"
    assert abs(rel[4][0] + Slist[4, 4]) < 2e-3, f"關節5 的 x 位置對不上: {np.round(rel[4], 4)}"

    # 4) 質量、質心、慣量(都轉成世界座標方向,並扣掉基座原點)
    masses, coms, inertias = [], [], []
    for h in lh:
        I_flat, com_T = sim.getShapeInertia(h)
        I_loc = np.array(I_flat).reshape(3, 3)
        com_loc = np.array(com_T)[[3, 7, 11]]
        Tw = np.array(sim.getObjectMatrix(h, sim.handle_world)).reshape(3, 4)
        R, p = Tw[:, :3], Tw[:, 3]
        masses.append(sim.getShapeMass(h))
        coms.append(R @ com_loc + p - base)
        inertias.append(R @ I_loc @ R.T)
    return masses, np.array(coms), inertias


def build_Mlist_Glist(masses, coms_base, inertias_base, M):
    Mlist, prev = [], np.zeros(3)
    for c in coms_base:
        Mi = np.eye(4); Mi[:3, 3] = c - prev; Mlist.append(Mi); prev = c
    Ml = np.eye(4); Ml[:3, :3] = M[:3, :3]; Ml[:3, 3] = M[:3, 3] - prev
    Mlist.append(Ml)
    Glist = []
    for m, I in zip(masses, inertias_base):
        G = np.zeros((6, 6)); G[:3, :3] = I; G[3:, 3:] = m * np.eye(3)
        Glist.append(G)
    return Mlist, Glist

masses, coms_base, inertias_base = read_scene(sim, robot_info['link_names'], robot_info['joint_names'], robot_info['Slist'])

Mlist, Glist = build_Mlist_Glist(masses, coms_base, inertias_base, robot_info['M'])

np.savez('robot_dynamics_data.npz',
                Mlist=np.array(Mlist),
                Glist=np.array(Glist))




    
    
    