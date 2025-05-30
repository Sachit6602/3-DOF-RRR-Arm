# RRR Robot End-Effector Control Assessment

## Overview

This archive contains four Python scripts and a `requirements.txt` to demonstrate an end-effector control for a planar RRR arm:

- **`main_a.py`** – Task A
- **`main_b.py`** – Task B
- **`main_op.py`** – Optional Task
- **`robot_pos.py`** – shared kinematics (forward kinematics, Jacobian, pseudoinverse) and Controller class  
- **`requirements.txt`** – Python dependencies

---

## Dependencies & System Requirements

    Python: 3.7 or newer

    Packages (see requirements.txt):
    - numpy
    - matplotlib

    Platform: cross-platform (Windows, Linux, macOS)

---

## Setup & Installation

1. **Unzip** the provided archive and `cd` into its folder.

2. **Create** and activate a Python 3 virtual environment:

   **macOS / Linux**  

   cd h_assessment/

   python3 -m venv venv

   source venv/bin/activate

   pip install -r requirements.txt

   **Windows** 

   cd h_assessment/

   python -m venv venv

   venv\Scripts\Activate

   pip install -r requirements.txt

---

## Running the Scripts

     #To run Task A
     python main_a.py

     #To run Task B
     python main_b.py

     #To run Optional Task
     python main_op.py

---

## Approach
# 1. Robot Kinematics

I modeled the planar RRR arm with three revolute joints of lengths  
**L₁, L₂, L₃**.

**Forward kinematics** computes the end-effector position **(x, y)** from joint angles  
**q = [q₁, q₂, q₃]**.

The **Jacobian** **J(q) ∈ ℝ²×³** maps small changes in joint angles to Cartesian velocities:

**ẋ = J(q) · q̇**

I computed a **damped pseudoinverse J⁺** so that even near singular configurations the inversion remains stable.



# 2. Resolved-Rate P-Controller

At each control timestep:

- **Read** the current end-effector position **x_cur** via forward kinematics.
- **Compute** the Cartesian error to the (held) target **x_targ**:  
  **e = x_targ − x_cur**
- **Form** a velocity command in Cartesian space with proportional gain **Kₚ**:  
  **v_cmd = Kₚ · e**
- **Convert** this to joint-rate commands **q̇** by:  
  **q̇ = J⁺ · v_cmd**
- **Integrate** over the control timestep **Δt**:  
  **q ← q + q̇ · Δt**
  
# 3. Frequency - Change

- The green target moves continuously at frequency **𝑓**.
- The **target_rate** and **control_rate** can be changed, which I mentioned inside the code to extent the experimentation

# 4. Visualization
 - I animated the arm and the target using Matplotlib’s **FuncAnimation**.
 - The **obstacle** is drawn as a circle, and the **target** as a moving green marker.

 
---

## Observation

**Task A (30 Hz target, 1 kHz control)**

The green target moves smoothly along its path , updating its position 30 times per second. Because the control rate (1000 Hz) is much higher than the target update rate (30 Hz), the arm holds on each new target position, smoothly converging toward it, then seamlessly transitions when the next target position arrives. In animation you see a very smooth chase, with minimal lag.

**Task B (5 Hz target, 50 Hz control)**

Lowering the target update rate to 5 Hz means only 5 new positions per second. Between those, the controller holds the last target position. Reducing the control rate to 50 Hz means the loop runs every 20 ms (rather than every 1 ms), so each corrective step is larger. What I observed is that the green target still moves at the same speed along its path (since its frequency 𝑓 is same), but its target position jumps only 5 times per second—so the arm sees a stair-step reference. The arm itself reacts in 50 Hz increments, making its motion visibly more jerky and slower to converge on each new target position. 

---

## Code Documentation
- I have separated the functions and made the variables as global variables to tweak any changes
- Feel free to explore, modify parameters, and extend the controllers or kinematics for your own experiments!
 

   
