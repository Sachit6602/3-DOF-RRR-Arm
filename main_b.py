import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from robot_pos import Robot_Pos, Controller

#Can change these if needed
L1, L2, L3 = 1.0, 1.0, 1.0 #link lengths
f = 0.5                    # ball oscillation frequency 
update_rate = 5.0         # target data update rate
control_rate = 50.0      # robot control loop frequency
total_time = 100           # how long to simulate 

#Compute joint states and Target positions over time
def simulate(robot, controller, L):
    dt_ctrl = 1.0 / control_rate
    dt_update = 1.0 / update_rate
    omega = 2*np.pi*f

    steps = int(total_time / dt_ctrl)
    t_ctrl = np.arange(steps) * dt_ctrl

    #joint angles
    q_hist = np.zeros((steps, robot.n))
    #green ball position
    x_cpos = np.zeros((steps, 2))
    #last updated ball position
    x_lpos = np.zeros((steps, 2))

    next_upd = 0.0
    last_meas = np.array([2*L, 0.0])
    q = np.zeros(robot.n)

    for i, t in enumerate(t_ctrl):
        #green ball position
        y_true = L * np.sin(omega * t)
        xt = np.array([2*L, y_true])
        x_cpos[i] = xt

        #updating at given rate
        if t >= next_upd:
            last_meas = xt.copy()
            next_upd += dt_update
        x_lpos[i] = last_meas

        #updating robot position
        dq = controller.step(q, last_meas, dt_ctrl)
        q = q + dq * dt_ctrl
        q_hist[i] = q

    return t_ctrl, q_hist, x_cpos, x_lpos

def main():

    robot = Robot_Pos([L1, L2, L3])
    controller = Controller(robot, Kp=50.0)
    t_ctrl, q_hist, x_cpos, x_lpos = simulate(robot, controller, 1.0)

    #fixed FPS for better animation and understanding
    display_fps = 60.0
    total_frames = int(total_time * display_fps)
    interval_ms = 1000.0 / display_fps

    #animation
    fig, ax = plt.subplots()
    ax.set_aspect('equal','box')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Task B")

    link_line, = ax.plot([], [], lw=3, marker='o')
    joints, = ax.plot([], [], 'ko', ms=6)
    t_ball = ax.scatter([], [], s=80, c='green', marker='o', label='Green Ball')
    ax.legend(loc='upper right')

    start_time = None

    def animate(frame):
        nonlocal start_time
        if start_time is None:
            start_time = time.perf_counter()

        #run in real-time
        elapsed = time.perf_counter() - start_time
        if elapsed > total_time:
            elapsed = total_time
        idx = int((elapsed / total_time) * len(t_ctrl))
        if idx >= len(t_ctrl):
            idx = len(t_ctrl) - 1

        # update robot
        pts = robot.joint_positions(q_hist[idx])
        xs, ys = zip(*pts)
        link_line.set_data(xs, ys)
        joints.set_data(xs, ys)

        # update ball
        t_ball.set_offsets([ x_cpos[idx] ])


        return link_line, joints, t_ball

    #storing in variable to avoid garbage dumping the animation
    anim = FuncAnimation(
        fig, animate,
        frames=total_frames,
        interval=interval_ms,
        blit=False,
        repeat=False
    )

    plt.show()

if __name__=="__main__":
    main()
