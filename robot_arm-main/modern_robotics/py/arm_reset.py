from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
from arm_data import robot_info

client = RemoteAPIClient()
sim = client.require('sim')

joint_names = [
    '/UR5/joint',
    '/UR5/link/joint',
    '/UR5/link/joint/link/joint',
    '/UR5/link/joint/link/joint/link/joint',
    '/UR5/link/joint/link/joint/link/joint/link/joint',
    '/UR5/link/joint/link/joint/link/joint/link/joint/link/joint'
]
joint_handles = [sim.getObject(name) for name in joint_names]

# sim.setJointPosition(joint_handles[5], 0)

for h in joint_handles:
    sim.setJointPosition(h, 0.0)
    

    

