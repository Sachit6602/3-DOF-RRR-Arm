import numpy as np

class Robot_Pos:

    def __init__(self, link_lengths):
        self.link_lengths = np.array(link_lengths, dtype=float)
        self.n = len(self.link_lengths)

    #Forward Kinematics to compute end-effector position
    def forward_kin(self, q):
        L1, L2, L3 = self.link_lengths
        q1, q2, q3 = q
        c1 = np.cos(q1)
        s1 = np.sin(q1)
        c12 = np.cos(q1 + q2)
        s12 = np.sin(q1 + q2)
        c123 = np.cos(q1 + q2 + q3)
        s123 = np.sin(q1 + q2 + q3)

        x = L1 * c1 + L2 * c12 + L3 * c123
        y = L1 * s1 + L2 * s12 + L3 * s123
        return np.array([x, y])

    #Compute Jacobian matrix(2x3) at joint
    def jacobian(self, q):
        L1, L2, L3 = self.link_lengths
        q1, q2, q3 = q
        s1 = np.sin(q1)
        c1 = np.cos(q1)
        s12 = np.sin(q1 + q2)
        c12 = np.cos(q1 + q2)
        s123 = np.sin(q1 + q2 + q3)
        c123 = np.cos(q1 + q2 + q3)


        j11 = -L1 * s1 - L2 * s12 - L3 * s123
        j12 = -L2 * s12 - L3 * s123
        j13 = -L3 * s123
        j21 =  L1 * c1 + L2 * c12 + L3 * c123
        j22 =  L2 * c12 + L3 * c123
        j23 =  L3 * c123

        J = np.array([[j11, j12, j13],
                      [j21, j22, j23]])
        return J

    #Get joint positions
    def joint_positions(self, q):
        pts = [(0.0, 0.0)]
        L = self.link_lengths
        angles = np.cumsum(q)
        for i in range(self.n):
            xi = pts[-1][0] + L[i] * np.cos(angles[i])
            yi = pts[-1][1] + L[i] * np.sin(angles[i])
            pts.append((xi, yi))
        return pts
    
#Compute pseudoinverse to improve numerical stability
def pseudoinverse(J, damp=1e-6):
    JT = J.T
    return np.linalg.inv(JT @ J + damp * np.eye(J.shape[1])) @ JT

class Controller:
    def __init__(self, robot, Kp=10.0):
        self.robot = robot
        self.Kp = np.atleast_1d(Kp).astype(float)

    #Compute joint velocities for one step
    def step(self, q, x_target, dt):
        
        x_cur = self.robot.forward_kin(q)
        error = x_target - x_cur

        
        v_cmd = self.Kp * error 

        J = self.robot.jacobian(q)
        J_psinv = pseudoinverse(J)
        return J_psinv @ v_cmd

