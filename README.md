
# 3-DOF Robot Manipulator Simulation Library

Python library for modeling a three-link robotic manipulator with three degrees of freedom (one prismatic and two revolute joints). Developed as part of a research internship at ITMO University.

## Purpose

This project is designed for:
- **Educational purposes** — studying robot kinematics and dynamics without access to physical hardware
- **Preliminary analysis** — testing control algorithms and trajectory planning before deployment on real systems
- **Verification** — comparing new algorithms with reference solutions

## Mathematical Model

### Manipulator Structure

The manipulator (Model 2 from [Alushin et al., 2009]) has three degrees of freedom:
1. **Link 1**: Prismatic joint along the vertical Z-axis (height adjustment, coordinate $z$)
2. **Link 2**: Revolute joint around the vertical Z-axis (angle $\psi$)
3. **Link 3**: Revolute joint around the vertical Z-axis (angle $\xi$)

The manipulator operates in two stages: first, the required height is set, then the planar positioning problem is solved in the horizontal plane.

### Kinematics

#### Forward Kinematics

Using Denavit-Hartenberg parameters, the end-effector coordinates are:

$$
\boxed{
\begin{cases}
x = L_1\cos\psi + L_2\cos(\psi+\xi) \\
y = L_1\sin\psi + L_2\sin(\psi+\xi) \\
z = z_0
\end{cases}
}
\tag{1}
$$

where $L_1$ and $L_2$ are link lengths, $\psi$ and $\xi$ are joint angles, and $z_0$ is the fixed height.

**DH-parameters of the three-link manipulator:**

| Link $i$ | $a_i$ | $\alpha_i$ | $d_i$ | $\theta_i$ | Type |
|----------|-------|------------|-------|------------|------|
| 1 | 0 | 0 | $z$ | 0 | Prismatic |
| 2 | $L_1$ | 0 | 0 | $\psi$ | Revolute |
| 3 | $L_2$ | 0 | 0 | $\xi$ | Revolute |

The homogeneous transformation matrices for each link:

$$
A_1^0 = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & z \\
0 & 0 & 0 & 1
\end{bmatrix}, \quad
A_2^1 = \begin{bmatrix}
\cos\psi & -\sin\psi & 0 & L_1\cos\psi \\
\sin\psi & \cos\psi & 0 & L_1\sin\psi \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \quad
A_3^2 = \begin{bmatrix}
\cos\xi & -\sin\xi & 0 & L_2\cos\xi \\
\sin\xi & \cos\xi & 0 & L_2\sin\xi \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

The resulting transformation matrix:

$$
T_3^0 = A_1^0 \cdot A_2^1 \cdot A_3^2 = \begin{bmatrix}
\cos(\psi+\xi) & -\sin(\psi+\xi) & 0 & L_1\cos\psi + L_2\cos(\psi+\xi) \\
\sin(\psi+\xi) & \cos(\psi+\xi) & 0 & L_1\sin\psi + L_2\sin(\psi+\xi) \\
0 & 0 & 1 & z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

#### Inverse Kinematics

Analytical solution for joint angles given target coordinates $(x, y, z)$:

$$
\boxed{
\begin{cases}
z_0 = z \\
\psi = \text{atan2}(y, x) - \arccos\left(\dfrac{x^2 + y^2 + L_1^2 - L_2^2}{2L_1\sqrt{x^2 + y^2}}\right) \\
\xi = \arccos\left(\dfrac{x^2 + y^2 - L_1^2 - L_2^2}{2L_1L_2}\right)
\end{cases}
}
\tag{2}
$$

**Geometric derivation:**

Let $r = \sqrt{x^2 + y^2}$ be the distance from the base to the target projection on the $Oxy$ plane.

From the law of cosines for the triangle formed by links $L_1$, $L_2$, and distance $r$:

$$
r^2 = L_1^2 + L_2^2 + 2L_1L_2\cos\xi
$$

Thus:

$$
\xi = \arccos\left(\frac{r^2 - L_1^2 - L_2^2}{2L_1L_2}\right)
$$

For $\psi$, using the angle $\alpha = \text{atan2}(y, x)$ and $\beta = \arccos\left(\frac{r^2 + L_1^2 - L_2^2}{2rL_1}\right)$:

$$
\psi = \alpha - \beta
$$

**Reachability condition:**

$$
\boxed{|L_1 - L_2| \leq r \leq L_1 + L_2}
\tag{3}
$$

If this condition is not satisfied, the target point is outside the workspace and no solution exists.

### Dynamics

Equations of motion derived using the Lagrange-Euler formulation:

$$
\boxed{M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = \tau}
\tag{4}
$$

where:
- $q = [z, \psi, \xi]^T$ — generalized coordinates
- $\dot{q} = [\dot{z}, \dot{\psi}, \dot{\xi}]^T$ — generalized velocities
- $\ddot{q} = [\ddot{z}, \ddot{\psi}, \ddot{\xi}]^T$ — generalized accelerations
- $\tau = [F_z, M_\psi, M_\xi]^T$ — generalized forces

**Kinetic energy:**

$$
T = \frac{1}{2} m_1 \dot{z}^2 + \frac{1}{2} I_{z,2} \dot{\psi}^2 + \frac{1}{2} I_{z,3} (\dot{\psi} + \dot{\xi})^2
$$

**Potential energy:**

$$
P = m_1 g \frac{z}{2} + (m_2 + m_3) g z_0
$$

**Inertia matrix $M(q)$:**

$$
M(q) = \begin{bmatrix}
m_1 & 0 & 0 \\
0 & I_{z2} + I_{z3} & I_{z3} \\
0 & I_{z3} & I_{z3}
\end{bmatrix}
\tag{5}
$$

**Coriolis and centrifugal matrix $C(q, \dot{q})$:**

$$
C(q, \dot{q}) = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & -I_{z3}\dot{\xi} \\
0 & I_{z3}\dot{\psi} & 0
\end{bmatrix}
\tag{6}
$$

**Gravity vector $G(q)$:**

$$
G(q) = \begin{bmatrix}
m_1 g / 2 + (m_2 + m_3) g \\
0 \\
0
\end{bmatrix}
\tag{7}
$$

**Complete system of dynamic equations:**

$$
\begin{cases}
m_1 \ddot{z} + \dfrac{m_1 g}{2} + (m_2 + m_3) g = F_z \\
(I_{z2} + I_{z3}) \ddot{\psi} + I_{z3} \ddot{\xi} = M_\psi \\
I_{z3} \ddot{\psi} + I_{z3} \ddot{\xi} = M_\xi
\end{cases}
$$

## Model Constraints

| Constraint Type | Description | Formula |
|----------------|-------------|---------|
| **Reachability** | Target distance within workspace | $\|L_1 - L_2\| \leq r \leq L_1 + L_2$, $r = \sqrt{x^2 + y^2}$ |
| **Joint Angles** | Mechanical limits | $\psi_{\min} \leq \psi \leq \psi_{\max}$, $\xi \in [-\pi, \pi]$ |
| **Height** | Vertical range | $z_{\min} \leq z \leq z_{\max}$ |
| **Velocities** | Maximum angular/linear speeds | $\|\dot{\psi}\| \leq \dot{\psi}_{\max}$, $\|\dot{\xi}\| \leq \dot{\xi}_{\max}$, $\|\dot{z}\| \leq \dot{z}_{\max}$ |
| **Accelerations** | Maximum accelerations | $\|\ddot{\psi}\| \leq \ddot{\psi}_{\max}$, $\|\ddot{\xi}\| \leq \ddot{\xi}_{\max}$, $\|\ddot{z}\| \leq \ddot{z}_{\max}$ |
| **Forces/Torques** | Actuator limits | $\|F_z\| \leq F_{z,\max}$, $\|M_\psi\| \leq M_{\psi,\max}$, $\|M_\xi\| \leq M_{\xi,\max}$ |

### Constraint Explanations

1. **Reachability**: Determines whether the manipulator can physically reach the target point. This condition follows from the triangle inequality for the manipulator links. If violated, the target is outside the workspace.

2. **Joint Angles**: Mechanical constraints on joint rotation angles. $\xi \in [-\pi, \pi]$ corresponds to full rotation of link 3 around the vertical axis. $\psi$ limits are determined by the manipulator design.

3. **Height**: Constraints on vertical movement of link 1, determined by the manipulator design (stand length, actuator stroke, etc.).

4. **Velocities and Accelerations**: Dynamic constraints related to actuator characteristics (maximum motor speed, maximum torque) and allowable structural loads.

5. **Forces/Torques**: Actuator saturation limits that must be respected for physical realizability.

## Features

- Forward kinematics computation using DH-parameters
- Analytical inverse kinematics with reachability verification
- Dynamics modeling based on Lagrange equations
- 3D visualization of the manipulator
- Top view with workspace boundaries
- Trajectory animation
- Constraint checking (velocities, accelerations, torques)
- Workspace plotting

## Project Structure

```
3dof-robot-manipulator-simulation/
├── src/
│   ├── kinematics.py      # Forward and inverse kinematics
│   ├── dynamics.py        # Lagrange dynamics equations
│   ├── robot.py           # Main Robot class
│   └── visualization.py   # 3D, top view and animation
├── examples/
│   └── demo.py            # Complete demonstration
├── docs/
│   └── report.pdf         # Full research report
├── requirements.txt       # Python dependencies
├── .gitignore            # Ignored files
├── LICENSE               # MIT License
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GeraDrobierro/3dof-robot-manipulator-simulation.git
cd 3dof-robot-manipulator-simulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.robot import Robot
from src.visualization import plot_both_views
import matplotlib.pyplot as plt

# Create robot with parameters
robot = Robot(
    L1=1.0, L2=0.8,
    m1=1.0, m2=0.5, m3=0.3,
    I_z2=0.1, I_z3=0.05,
    z_min=0.0, z_max=1.5,
    psi_min=-3.14, psi_max=3.14,
    g=9.81
)

# Set target position
result = robot.inverse_kinematics(x=1.2, y=0.8, z=0.5)

if result['success']:
    print(f"Joint angles: psi={result['psi']:.3f}, xi={result['xi']:.3f}")
    
    # Compute required torques
    dynamics = robot.compute_dynamics(ddz=0.2, ddpsi=0.5, ddxi=0.3)
    print(f"Required force: F_z={dynamics['F_z']:.3f} N")
    print(f"Required torque M_psi={dynamics['M_psi']:.3f} N·m")
    print(f"Required torque M_xi={dynamics['M_xi']:.3f} N·m")
    
    # Visualize
    plot_both_views(robot)
    plt.show()
```

## Implementation Details

### Kinematics Module (kinematics.py)
- `forward_kinematics(L1, L2, psi, xi)` — computes end-effector coordinates using equation (1)
- `inverse_kinematics_2d(L1, L2, x, y)` — analytical inverse kinematics solution using equation (2)
- `get_joint_positions(L1, L2, z, psi, xi)` — coordinates of all joints
- `check_reachability(L1, L2, x, y)` — verifies if target is reachable using condition (3)
- `get_workspace_boundary(L1, L2, n_points)` — returns workspace boundaries

### Dynamics Module (dynamics.py)
- `inertia_matrix(m1, m2, m3, I_z2, I_z3)` — computes inertia matrix $M(q)$ from equation (5)
- `coriolis_matrix(I_z3, dpsi, dxi)` — computes Coriolis matrix $C(q, \dot{q})$ from equation (6)
- `gravity_vector(m1, m2, m3, g)` — computes gravity vector $G(q)$ from equation (7)
- `compute_torques(...)` — full dynamics equation $M\ddot{q} + C\dot{q} + G = \tau$ from (4)

### Robot Class (robot.py)
- `set_state(z, psi, xi, dz, dpsi, dxi)` — sets current state
- `forward_kinematics()` — computes end-effector position
- `inverse_kinematics(x, y, z)` — solves inverse kinematics with constraints
- `compute_dynamics(ddz, ddpsi, ddxi)` — computes required torques
- `plan_trajectory(target_z, target_psi, target_xi, n_steps)` — generates trajectory using linear interpolation
- `check_constraints()` — verifies all model constraints

### Visualization Module (visualization.py)
- `draw_3d(robot)` — 3D model with joint labels and projection
- `draw_top_view(robot)` — overhead view with workspace boundaries
- `plot_both_views(robot)` — combined 3D and top view
- `animate_manipulator(robot, trajectory)` — trajectory animation
- `plot_workspace(robot)` — workspace visualization

## Testing Results

### Kinematics
- Positioning accuracy: $< 10^{-15}$ m (machine precision)
- Verified against analytical solutions for multiple test cases
- Reachability condition correctly identifies invalid targets
- Forward kinematics validated using the DH-parameter approach

**Sample test results:**

| $\psi$ (rad) | $\xi$ (rad) | $x$ (computed) | $y$ (computed) | Error |
|--------------|-------------|----------------|----------------|-------|
| 0.200 | 0.100 | 1.744 | 0.435 | $< 10^{-15}$ |
| 0.025 | 1.292 | 1.200 | 0.800 | $< 10^{-15}$ |
| 0.000 | 0.000 | 1.800 | 0.000 | $< 10^{-15}$ |
| 1.571 | 0.785 | -0.566 | 1.414 | $< 10^{-15}$ |

**Inverse kinematics verification:**

| Target | Found Angles | Obtained End-effector | Error |
|--------|--------------|----------------------|-------|
| (1.200, 0.800, 0.500) | $\psi = 0.025$, $\xi = 1.292$ | (1.200, 0.800, 0.500) | $2.22 \cdot 10^{-16}$ m |

### Dynamics
- Static case: torques match gravitational forces
- Dynamic case: verified with constant acceleration tests
- Constraint checking correctly detects violations

**Static case verification:**
For $m_1 = 1.0$, $m_2 = 0.5$, $m_3 = 0.3$, $g = 9.81$:

$$
F_z = \frac{m_1 g}{2} + (m_2 + m_3)g = 4.905 + 7.848 = 12.753 \text{ N}
$$

**Dynamic case verification:**
For $\ddot{\psi} = 0.5$ rad/s², $\ddot{\xi} = 0.3$ rad/s²:

$$
M_\psi = (I_{z2} + I_{z3})\ddot{\psi} + I_{z3}\ddot{\xi} = 0.15 \cdot 0.5 + 0.05 \cdot 0.3 = 0.090 \text{ N·m}
$$
$$
M_\xi = I_{z3}\ddot{\psi} + I_{z3}\ddot{\xi} = 0.05 \cdot 0.5 + 0.05 \cdot 0.3 = 0.040 \text{ N·m}
$$

### Comparison with Reference Solutions

| Criterion | This Solution | Analytical Solution [Corke, 2011] |
|-----------|---------------|-----------------------------------|
| Manipulator Type | 3 DOF (prismatic + 2 revolute) | 2 DOF (2 revolute) |
| Kinematics Method | DH-parameters | DH-parameters |
| Dynamics Method | Lagrange equations | Lagrange equations |
| Implementation | Python + NumPy + Matplotlib | MATLAB® Robotics Toolbox |
| Visualization | 3D + top view + animation | 2D graphics only |
| Constraint Checking | Complete (velocities, accelerations, torques) | Partial (reachability only) |
| Positioning Accuracy | $< 10^{-15}$ m | $< 10^{-15}$ m |

## Documentation

The complete research report includes:
- Mathematical derivation of DH-parameters
- Analytical solutions for forward and inverse kinematics
- Lagrange dynamics derivation with matrix forms
- Trajectory planning methodology
- Testing and verification results
- Complete source code listings


## Future Work

- **Optimal trajectory planning** using variational calculus and differential geometry — finding trajectories that minimize energy consumption, time, or actuator loads
- **Continuum (soft) manipulator modeling** with differential geometry — describing manipulator shape through curvature and torsion, using variational calculus to find optimal configurations
- **Integration with closed-loop control systems** (PID, MPC) — implementing feedback control with real-time data
- **Friction modeling and actuator dynamics** — accounting for joint friction, link elasticity, and actuator dynamics
- **Workspace analysis** — detailed study of reachable configurations and singularities

## Author

**David Sverdlov**  
Student, Faculty of Robotics and Artificial Intelligence  
ITMO University, St. Petersburg, Russia

## References

1. Corke P. Robotics, Vision and Control: Fundamental Algorithms in MATLAB®. Springer-Verlag Berlin Heidelberg, 2011. (Springer Tracts in Advanced Robotics ; Vol. 73). ISBN 978-3-642-20143-1. DOI: 10.1007/978-3-642-20144-8.

2. Sobh T.M., Dekhil M., Henderson T.C., Sabbavarapu A. Modeling of a Three-Link Robot Manipulator. University of Bridgeport, 1994. URL: https://masters.donntu.ru/2012/etf/pavlov/library/translation.htm

3. Luo Y., Jing M., Ji T., Sun F., Liu H. A Robust Tube-Based Smooth-MPC for Robot Manipulator Planning. arXiv:2103.09693, 2021. URL: https://arxiv.org/abs/2103.09693

4. Alushin Yu.A., Rachek V.M., Verzhansky P.M. Kinematic and Dynamic Analysis of Typical Three-Link Manipulators. Problems of Mechanical Engineering and Machine Reliability, 2009. No. 3, pp. 474–483.

5. Klimov O.A. Modeling of Mechanical Properties of a Mechatronic Device with Three Degrees of Mobility. Omsk State Technical University, 2012. 24 p.
