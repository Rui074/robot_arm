import numpy as np
import modern_robotics as mr


link_names = [
    '/UR5/link',
    '/UR5/link/joint/link',
    '/UR5/link/joint/link/joint/link',
    '/UR5/link/joint/link/joint/link/joint/link',
    '/UR5/link/joint/link/joint/link/joint/link/joint/link',
    '/UR5/link/joint/link/joint/link/joint/link/joint/link/joint/link'
]

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


data = np.load('robot_dynamics_data.npz')
Mlist = list(data['Mlist'])
Glist = list(data['Glist'])

robot_info = {
    'link_names':link_names,
    'joint_names':joint_names,
    'n':6,
    'joint_constraints':joint_constraints,
    'M':M,
    'Slist':Slist,
    'Mlist':Mlist,
    'Glist':Glist,
    'g': np.array([0, 0, -9.81])
}