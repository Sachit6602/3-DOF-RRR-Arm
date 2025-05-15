# main.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from robot_pos import Robot_Pos, Controller, pseudoinverse

#Can change these if needed
L1, L2, L3 = 1.0, 1.0, 1.0 #link lengths
total_time = 10.0
control_dt = 0.001
Kp_gain = 10.0
K_rep = 0.2

def run_simulation(robot, controller, target_func,
                   obstacle_center, obstacle_radius,
                   K_rep, total_time=10.0, control_dt=0.001):

    steps = int(total_time / control_dt)
    t_hist = np.arange(steps) * control_dt

    q_hist = np.zeros((steps, robot.n))
    x_target_hist = np.zeros((steps, 2))

    q = np.zeros(robot.n)
    for i, t in enumerate(t_hist):
        x_cur = robot.forward_kin(q)
        x_tar = target_func(t)
        x_target_hist[i] = x_tar

        error = x_tar - x_cur
        v_cmd = controller.Kp * error

        diff = x_cur - obstacle_center
        dist = np.linalg.norm(diff)
        if dist < obstacle_radius:
            #repulsion
            mag = K_rep * (1.0/dist - 1.0/obstacle_radius) / (dist**2)
            v_cmd += mag * (diff / dist)

        #compute joint velocity 
        J = robot.jacobian(q)
        dq = pseudoinverse(J) @ v_cmd


        q = q + dq * control_dt
        q_hist[i] = q

    return t_hist, q_hist, x_target_hist

def main():

    #obstacle placement
    obs_center = np.array([0.5*L1, 0.2*L1])
    obs_radius = L1/6.0

    robot = Robot_Pos([L1, L2, L3])
    controller = Controller(robot, Kp=Kp_gain)

    def target_traj(t):
        R = 0.5
        omega = 2*np.pi*0.2
        return np.array([R*np.cos(omega*t), R*np.sin(omega*t)])

    t_hist, q_hist, x_target_hist = run_simulation(
        robot, controller, target_traj,
        obs_center, obs_radius, K_rep,
        total_time=total_time,
        control_dt=control_dt
    )

    fig, ax = plt.subplots()
    ax.set_aspect('equal','box')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("Optional Task")

    # draw obstacle
    from matplotlib.patches import Circle
    circ = Circle(obs_center, obs_radius, fill=False, color='red', lw=2)
    ax.add_patch(circ)

    line, = ax.plot([], [], lw=3, marker='o')
    joints = ax.scatter([], [], s=50, c='k')
    target = ax.scatter([], [], s=100, c='g')

    def animate(i):
        q     = q_hist[i]
        x_tar = x_target_hist[i]
        pts   = robot.joint_positions(q)
        xs, ys= zip(*pts)
        line.set_data(xs, ys)
        joints.set_offsets(pts)
        target.set_offsets([x_tar])
        return line, joints, target

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(
        fig, animate,
        frames=len(t_hist),
        interval=control_dt*1000,
        blit=False
    )
    plt.show()

if __name__ == "__main__":
    main()
