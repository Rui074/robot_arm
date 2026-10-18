#6dof_robot_arm fk

import numpy as np
import modern_robotics as mr



M = np.array([[1, 0, 0, -0.18495], [0, 1, 0, 0.150], [0, 0, 1, 0.98655], [0, 0, 0, 1]])

Slist = np.array([[ 0, 0, 1, 0, 0, 0], 
                  [ 1, 0, 0, 0, 0.07455, 0], 
                  [ 1, 0, 0, 0, 0.49965, 0], 
                  [ 1, 0, 0, 0, 0.8918, 0], 
                  [ 0, 0, 1, 0, 0.110, 0], 
                  [ 1, 0, 0, 0, 0.98655, 0]]).T

thetalist = np.array([0, np.pi*2/3, 0, np.pi/4, 0, np.pi/4])  # Joint angles in radians  #only need to change the angles to get different end_effector positions

# Calculate the forward kinematics using the product of exponentials formula
T_1 = mr.FKinSpace(M, Slist, thetalist)


def sanitize_T(T):
    """ 強制修復 4x4 齊次變換矩陣，使其旋轉部分嚴格滿足 SO(3) 正交性 """
    R = T[:3, :3]
    p = T[:3, 3]
    
    # 利用 SVD (奇異值分解) 重構完美的正交矩陣
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    
    # 確保行列式 det(R) = +1 (排除反射矩陣)
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
        
    T_clean = np.eye(4)
    T_clean[:3, :3] = R_clean
    T_clean[:3, 3] = p
    return T_clean

T = sanitize_T(T_1)
print("The transformation matrix T is:", T)







                 
                 

