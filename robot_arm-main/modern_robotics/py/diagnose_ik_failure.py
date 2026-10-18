import numpy as np
import modern_robotics as mr

def diagnose_ik_failure(Slist, M, target, guess, joint_constraints, restarts=20, iters=300):
    cond = np.linalg.cond(mr.JacobianSpace(Slist, guess))

    def ik_dls_adaptive(theta0, target, iters=300, eomg=0.01, ev=0.005,
                     w_threshold=0.01, lambda_max=0.01):
        theta = theta0.copy()
        for _ in range(iters):
            Tsb = mr.FKinSpace(M, Slist, theta)
            Vs = mr.Adjoint(Tsb) @ mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Tsb) @ target))  # 修正: 轉回 space frame
            werr, verr = np.linalg.norm(Vs[:3]), np.linalg.norm(Vs[3:])
            if werr <= eomg and verr <= ev:      # 提早收斂就跳出，不用硬跑滿 iters
                break

            J = mr.JacobianSpace(Slist, theta)
            JJT = J @ J.T
            w = np.sqrt(max(0, np.linalg.det(JJT)))          # manipulability measure
            damp = lambda_max * (1 - w / w_threshold) ** 2 if w < w_threshold else 0.0

            delta_theta = J.T @ np.linalg.solve(JJT + damp**2 * np.eye(6), Vs)  # solve比inv數值上更穩
            theta = theta + delta_theta

        return theta, werr, verr
    

    best_verr = np.inf
    for _ in range(restarts):                       # 窮舉重啟,不設限制,純粹問「到底能不能到」
        theta0 = guess + np.random.uniform(-1, 1, size=6)
        theta_sol, werr, verr = ik_dls_adaptive(theta0, target)
        if verr < best_verr:
            best_verr, best_werr, best_theta = verr, werr, theta_sol

    wrapped = np.arctan2(np.sin(best_theta), np.cos(best_theta))
    in_limits = all(a[0] <= th <= a[1] for th, a in zip(wrapped, joint_constraints))

    print(f"cond(J) = {cond:.1f}")
    print(f"窮舉硬解後最佳誤差: 位置 {best_verr*1000:.2f} mm, 姿態 {best_werr:.5f} rad")
    print(f"最佳解是否在關節限制內: {in_limits}")

    if cond > 5000:
        print("=> 判定: 真奇異/病態，改路徑繞開，調參數沒用")
    elif best_verr > 0.01:
        print("=> 判定: 大機率不可達，改目標點或改路徑")
    elif not in_limits:
        print("=> 判定: 可達但違反關節限制，去調限位或路徑，不是解算法問題")
    else:
        print("=> 判定: 可達且不奇異，換 damped least squares 或加大疊代數就能解")

    return wrapped





