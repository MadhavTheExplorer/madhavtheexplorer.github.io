---
layout: post
title:  "Discretizing Systems"
date:   2025-07-27 13:40:34 +0530
categories: controls
tags: controls
image: /projects/blog_setup/assets/img/gui_img.png
comments: true
summary: "After having a continuous state space representation of the system, sometimes it is important to convert it to discrete time representation."
---

{% include toc.html %}

## Introduction
Following a beautifully demonstrated method to discretize a system with proof as written in an [article](https://www.researchgate.net/profile/Mohamed_Mourad_Lafifi/post/How_to_discretize_a_continuous_system_model_using_a_zero-order_hold_method/attachment/5dfb23053843b0938396fcbd/AS%3A837720276484098%401576739486363/download/S9+_+Controller+Discretization.pdf) from research gate. Briefly summarizing relevant points from that article here.

Consider a continuous domain system

$$ \dot{\mathbf{x}} = \mathbf{Ax} + \mathbf{Bu}, \quad \mathbf{y}= \mathbf{Cx}$$

Some notation that I will follow:  
1. Bold capital letters are matrices
2. Bold small letters are vectors
3. Small letters are scalars

Its discrete time variation with zero order hold (ZOH) will be

$$ \mathbf{x}_{k+1} = \mathbf{A}_d\mathbf{x}_k + \mathbf{B}_d\mathbf{u}_k, \quad \mathbf{y}_k= \mathbf{C}_d\mathbf{x}_k$$

where subscript \\(k\\) denotes \\(k^{th}\\) time step and 

$$
\begin{align}
    \mathbf{A}_d &= e^{\mathbf{A}\Delta t} \\
    \mathbf{B}_d &= \int_0^{\Delta t} e^{\mathbf{A}\tau}\mathrm{d}\tau \mathbf{B} \\
    \mathbf{C}_d &= \mathbf{C}
\end{align}
$$

## Background Info
### Matrix Exponential
As mentioned on Wikipedia about [Matrix exponential](https://en.wikipedia.org/wiki/Matrix_exponential), the formula to calculate is given by

$$ e^{\mathbf{X}} = \sum_{k=0}^{\infty}{\frac{1}{k!}\mathbf{X}^k} $$

which is nothing but Taylor series expansion of exponential function, similar to a scalar exponential.  

Calculation of this matrix exponential becomes easier if the matrix in question is diagonalizable, i.e. if there exists an invertible matrix \\(\mathbf{P}\\) and a diagonal matrix \\(\mathbf{D}\\) such that \\(\mathbf{X}\\) can be written as

$$ \mathbf{X} = \mathbf{PDP}^{-1} $$

\\(\mathbf{P}\\) is often called [Change of Basis](https://en.wikipedia.org/wiki/Change_of_basis#Endomorphisms) matrix, encountered in the definition of [similar matrices](https://en.wikipedia.org/wiki/Similar_matrices). The implication of this is profound.

$$  \begin{align}
        e^{\mathbf{X}}  &= \sum_{k=0}^{\infty}{\frac{1}{k!}\mathbf{X}^k} \\
                        &= \sum_{k=0}^{\infty}{\frac{1}{k!}\left(\mathbf{PDP}^{-1}\right)^k} \\
                        &= \mathbf{P}\left[\sum_{k=0}^{\infty}{\frac{1}{k!}\mathbf{D}^k}\right]\mathbf{P}^{-1} \\
                        &= \mathbf{P} e^{\mathbf{D}} \mathbf{P}^{-1}
    \end{align}    
$$

and exponential of a diagonal matrix is the matrix of exponential of its diagonal components.

## Examples
### Example 1
#### Problem
Consider a simple block of mass \\(m\\) on a frictionless surface with an input force \\(F\\) along the surface. The equation of motion is \\(F = ma\\). Or

$$ \ddot{x} = \frac{F}{m} $$

#### Formulation
The state space equation becomes

$$ \begin{align}
    \frac{\mathrm{d}}{\mathrm{d}t}
    \begin{bmatrix}
    x \\
    \dot{x}
    \end{bmatrix}
    &=
    \begin{bmatrix}
        0 & 1 \\
        0 & 0
    \end{bmatrix}
    \begin{bmatrix}
    x \\
    \dot{x}
    \end{bmatrix}
    +
    \begin{bmatrix}
    0 \\
    \frac{1}{m}
    \end{bmatrix}
    F
    \\
   \dot{\mathbf{x}} &= \mathbf{Ax} + \mathbf{Bu}
\end{align} $$

#### Solution
Here we can observe that 

$$ \begin{align}
    \mathbf{A}^k &= \mathbf{0} \quad \forall k=2,3,4 \cdots \\
    e^{\mathbf{A}\Delta t}  &= \sum_{k=0}^{\infty}{\frac{1}{k!}\left(\mathbf{A}\Delta t\right)^k} \\
            \mathbf{A}_d    &= \mathbf{I} + \mathbf{A}\Delta t + \mathbf{0} \\
                            &= \begin{bmatrix}
                                1 & \Delta t \\
                                0 & 1
                                \end{bmatrix}
\end{align} $$

Now to find \\(\mathbf{B}_d\\)

$$ \begin{align}
    \mathbf{B}_d    &= \int_0^{\Delta t} e^{\mathbf{A}\tau}\mathrm{d}\tau \mathbf{B} \\
                    &= \begin{bmatrix}
                        \int_0^{\Delta t} 1 \mathrm{d}\tau & \int_0^{\Delta t} \Delta \tau \mathrm{d}\tau \\
                        0 & \int_0^{\Delta t} 1 \mathrm{d}\tau
                        \end{bmatrix} 
                        \begin{bmatrix}
                        0 \\
                        \frac{1}{m}
                        \end{bmatrix}\\
                    &= \begin{bmatrix}
                        \Delta t & \frac{(\Delta t)^2}{2} \\
                        0 & \Delta t
                        \end{bmatrix}
                        \begin{bmatrix}
                        0 \\
                        \frac{1}{m}
                        \end{bmatrix} \\
                    &= \begin{bmatrix}
                        \frac{(\Delta t)^2}{2m} \\
                        \frac{\Delta t}{m}
                        \end{bmatrix}
\end{align} $$

Here let us assume the measurable elements of the system \\(\mathbf{y}\\) are all its states. So the matrix \\(\mathbf{C}\\) is identity \\(\mathbf{I}\\), i.e.

$$ \begin{align}
    \mathbf{y}  &= \mathbf{Cx} \\
                &= \mathbf{Ix} \\
    \mathbf{y}  &= \mathbf{x}
\end{align} $$

So \\(\mathbf{C}_d=\mathbf{C} = \mathbf{I}\\). And

$$ \mathbf{x}_{k+1} = \mathbf{A}_d\mathbf{x}_k + \mathbf{B}_d\mathbf{u}_k, \quad \mathbf{y}_k = \mathbf{C}_d\mathbf{x}_k $$

#### Verification
The theoretical discretization can be verified by comparing with SciPy's `cont2discrete` function. The following Python code demonstrates this comparison, using the value \\(m=1\\):

```python
from scipy.signal import cont2discrete, lti
import numpy as np

# Define the continuous-time system for Example 1 (mass on frictionless surface)
# Assume m = 1 for simplicity
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1]])  # 1/m = 1
C = np.array([[1, 0], [0, 1]])  # Full state observation
D = np.array([[0], [0]])

# Create state-space system
l_system = lti(A, B, C, D)

dt = 1  # Sampling time

method = 'zoh'
d_system = cont2discrete((A, B, C, D), dt, method=method)
print("ZOH Discretization:")
print(f"A_d = \n{d_system[0]}")
print(f"B_d = \n{d_system[1]}")
```

The ZOH method produces:

$$ \mathbf{A}_d = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} \text{ for }\Delta t = 1 $$

$$ \mathbf{B}_d = \begin{bmatrix} 0.5 \\ 1 \end{bmatrix}\text{ for }\Delta t = 1\text{ and }m = 1 $$

### Example 2
#### Problem
Consider a simple spring mass system. Spring having spring constant \\(k\\) and the weight of the attached mass \\(m\\). We can apply a force input \\(F\\) in the positive x direction. The equation of motion is \\(F - kx = ma\\). Or

$$ \ddot{x} = \frac{F}{m}-\frac{k}{m}x $$

#### Formulation
The state space equation becomes

$$ \begin{align}
    \frac{\mathrm{d}}{\mathrm{d}t}
    \begin{bmatrix}
    x \\
    \dot{x}
    \end{bmatrix}
    &=
    \begin{bmatrix}
        0 & 1 \\
        -\frac{k}{m} & 0
    \end{bmatrix}
    \begin{bmatrix}
    x \\
    \dot{x}
    \end{bmatrix}
    +
    \begin{bmatrix}
    0 \\
    \frac{1}{m}
    \end{bmatrix}
    F
    \\
   \dot{\mathbf{x}} &= \mathbf{Ax} + \mathbf{Bu}
\end{align} $$

#### Solution 1
Here we can observe that 

$$ \begin{align}
    \mathbf{A}^2    &= \begin{bmatrix}
                        0 & 1 \\
                        -\frac{k}{m} & 0
                        \end{bmatrix}\begin{bmatrix}
                        0 & 1 \\
                        -\frac{k}{m} & 0
                        \end{bmatrix} \\
                    &=  -\frac{k}{m}\begin{bmatrix}
                        1 & 0 \\
                        0 & 1
                        \end{bmatrix} \\
                    &= -\frac{k}{m}\mathbf{I} 
\end{align} $$

So we can find \\(\mathbf{A}_d\\) by simply applying the expanded form of the matrix exponential.

$$ \begin{align}
    e^{\mathbf{A}\Delta t}  &= \sum_{i=0}^{\infty}{\frac{1}{i!}\left(\mathbf{A}\Delta t\right)^i} \\
            \mathbf{A}_d    &= \mathbf{I} + \Delta t\mathbf{A} + \frac{(\Delta t)^2}{2!}\left(\frac{-k}{m}\right)\mathbf{I} + \frac{(\Delta t)^3}{3!}\left(\frac{-k}{m}\right)\mathbf{A} + \frac{(\Delta t)^4}{4!}\left(\frac{-k}{m}\right)^2\mathbf{I} + \frac{(\Delta t)^5}{5!}\left(\frac{-k}{m}\right)^2\mathbf{A} + \frac{(\Delta t)^6}{6!}\left(\frac{-k}{m}\right)^3\mathbf{I} + \cdots 
\end{align} $$

Writing down its individual elements, keeping in mind [Taylor series expansion](https://en.wikipedia.org/wiki/Taylor_series#Trigonometric_functions) of cos(x) and sin(x)

$$ \begin{align}
            [\mathbf{A}_d]_{11} &= 1 + 0 + \frac{(\Delta t)^2}{2!}\left(\frac{-k}{m}\right) + 0 + \frac{(\Delta t)^4}{4!}\left(\frac{-k}{m}\right)^2 + 0 + \cdots \\
                                &= \sum_{n=0}^{\infty}{\frac{(\Delta t)^{2n}}{(2n)!}\left(\frac{-k}{m}\right)^n} \\
                                &= \sum_{n=0}^{\infty}{\frac{(-1)^{n}}{(2n)!}\left((\Delta t)\sqrt{\frac{k}{m}}\right)^{2n}}, \quad \text{manipulating} \\
                                &= \cos\left((\Delta t)\sqrt{\frac{k}{m}}\right), \quad \text{Taylor series expansion of cos(x)} \\
            [\mathbf{A}_d]_{12} &= 0+1+0+\frac{(\Delta t)^3}{3!}\left(\frac{-k}{m}\right)+0+ +\frac{(\Delta t)^5}{5!}\left(\frac{-k}{m}\right)^2+0+\cdots \\
                                &= \sum_{n=0}^{\infty}{\frac{(\Delta t)^{2n+1}}{(2n+1)!}\left(\frac{-k}{m}\right)^n} \\
                                &= \frac{\sum_{n=0}^{\infty}{\frac{(-1)^{n}}{(2n+1)!}\left((\Delta t)\sqrt{\frac{k}{m}}\right)^{2n+1}}}{\sqrt{\frac{k}{m}}}, \quad \text{manipulating} \\
                                &= \sqrt{\frac{m}{k}}\sin\left((\Delta t)\sqrt{\frac{k}{m}}\right), \quad \text{Taylor series expansion of sin(x)} \\
            [\mathbf{A}_d]_{21} &= 0+\left(\frac{-k}{m}\right)+0+\frac{(\Delta t)^3}{3!}\left(\frac{-k}{m}\right)^2+0+ +\frac{(\Delta t)^5}{5!}\left(\frac{-k}{m}\right)^3+0+\cdots \\
                                &= \left(\frac{-k}{m}\right)[\mathbf{A}_d]_{12} \\
            [\mathbf{A}_d]_{22} &= [\mathbf{A}_d]_{11}
\end{align} $$

Summarising the complete \\(\mathbf{A}_d\\) matrix

$$ \mathbf{A}_d = \begin{bmatrix}
                    \cos\left((\Delta t)\sqrt{\frac{k}{m}}\right) & \sqrt{\frac{m}{k}}\sin\left((\Delta t)\sqrt{\frac{k}{m}}\right) \\
                    -\sqrt{\frac{k}{m}}\sin\left((\Delta t)\sqrt{\frac{k}{m}}\right) & \cos\left((\Delta t)\sqrt{\frac{k}{m}}\right)
                    \end {bmatrix}
$$                    

Now to find \\(\mathbf{B}_d\\), keep in mind the integrals of cos and sin

$$ \begin{align}
    \int_{0}^{\Delta t}\cos\left(\tau \sqrt{\frac{k}{m}}\right)\mathrm{d}\tau &= \sqrt{\frac{m}{k}}\sin\left(\Delta t \sqrt{\frac{k}{m}}\right) \\
    \int_{0}^{\Delta t}\sin\left(\tau \sqrt{\frac{k}{m}}\right)\mathrm{d}\tau &= \sqrt{\frac{m}{k}}\left(1-\cos\left(\Delta t \sqrt{\frac{k}{m}}\right)\right)
\end{align} $$

So the equation for \\(\mathbf{B}_d\\) becomes

$$ \begin{align}
    \mathbf{B}_d    &= \int_0^{\Delta t} e^{\mathbf{A}\tau}\mathrm{d}\tau \mathbf{B} \\
                    &= \begin{bmatrix}
                    \int_0^{\Delta t}{\cos\left(\tau\sqrt{\frac{k}{m}}\right)\mathrm{d}\tau} & \sqrt{\frac{m}{k}}\int_0^{\Delta t}{\sin\left(\tau\sqrt{\frac{k}{m}}\right)\mathrm{d}\tau} \\
                    -\sqrt{\frac{k}{m}}\int_0^{\Delta t}{\sin\left(\tau\sqrt{\frac{k}{m}}\right)\mathrm{d}\tau} & \int_0^{\Delta t}{\cos\left(\tau\sqrt{\frac{k}{m}}\right)\mathrm{d}\tau}
                    \end {bmatrix} 
                    \begin{bmatrix}
                    0 \\
                    \frac{1}{m}
                    \end{bmatrix}\\
                    &= \begin{bmatrix}
                    \sqrt{\frac{m}{k}}\sin\left(\Delta t \sqrt{\frac{k}{m}}\right) & \frac{m}{k}\left(1-\cos\left(\Delta t \sqrt{\frac{k}{m}}\right)\right) \\
                    -\left(1-\cos\left(\Delta t \sqrt{\frac{k}{m}}\right)\right) & \sqrt{\frac{m}{k}}\sin\left(\Delta t \sqrt{\frac{k}{m}}\right)
                    \end {bmatrix} 
                    \begin{bmatrix}
                    0 \\
                    \frac{1}{m}
                    \end{bmatrix}\\
                    &= \begin{bmatrix}
                    \frac{1}{k}\left(1-\cos\left(\Delta t \sqrt{\frac{k}{m}}\right)\right) \\
                    \sqrt{\frac{1}{mk}}\sin\left(\Delta t \sqrt{\frac{k}{m}}\right)
                    \end{bmatrix}
\end{align} $$

Here again, let us assume the measurable elements of the system \\(\mathbf{y}\\) are all its states. So the matrix \\(\mathbf{C}\\) is identity \\(\mathbf{I}\\), i.e.

$$ \begin{align}
    \mathbf{y}  &= \mathbf{Cx} \\
                &= \mathbf{Ix} \\
    \mathbf{y}  &= \mathbf{x}
\end{align} $$

So \\(\mathbf{C}_d=\mathbf{C} = \mathbf{I}\\). And

$$ \mathbf{x}_{k+1} = \mathbf{A}_d\mathbf{x}_k + \mathbf{B}_d\mathbf{u}_k, \quad \mathbf{y}_k = \mathbf{C}_d\mathbf{x}_k $$

#### Solution 2 (Eigenvalue Decomposition Method)
An alternative approach uses eigenvalue decomposition of the \\(\mathbf{A}\\) matrix. For the spring-mass system:

$$ \mathbf{A} = \begin{bmatrix} 0 & 1 \\ -\frac{k}{m} & 0 \end{bmatrix} $$

First, we find the eigenvalues by solving \\(\det(\mathbf{A} - \lambda\mathbf{I}) = 0\\):

$$ \det\begin{bmatrix} -\lambda & 1 \\ -\frac{k}{m} & -\lambda \end{bmatrix} = \lambda^2 + \frac{k}{m} = 0 $$

The eigenvalues are \\(\lambda_1 = j\sqrt{\frac{k}{m}}\\) and \\(\lambda_2 = -j\sqrt{\frac{k}{m}}\\).

The corresponding eigenvectors are:
- For \\(\lambda_1 = j\sqrt{\frac{k}{m}}\\): 

$$ \mathbf{v}_1 = \begin{bmatrix} 1 \\ j\sqrt{\frac{k}{m}} \end{bmatrix} $$

- For \\(\lambda_2 = -j\sqrt{\frac{k}{m}}\\): 

$$\mathbf{v}_2 = \begin{bmatrix} 1 \\ -j\sqrt{\frac{k}{m}} \end{bmatrix} $$

The matrix \\(\mathbf{P}\\) of eigenvectors and diagonal matrix \\(\mathbf{D}\\) are:

$$ \mathbf{P} = \begin{bmatrix} 1 & 1 \\ j\sqrt{\frac{k}{m}} & -j\sqrt{\frac{k}{m}} \end{bmatrix}, \quad \mathbf{D} = \begin{bmatrix} j\sqrt{\frac{k}{m}} & 0 \\ 0 & -j\sqrt{\frac{k}{m}} \end{bmatrix} $$

Then \\(\mathbf{A}_d = \mathbf{P}e^{\mathbf{D}\Delta t}\mathbf{P}^{-1}\\), where:

$$ \mathbf{P}^{-1} = \frac{1}{2} \begin{bmatrix} 1 & -j\sqrt{\frac{m}{k}} \\
                                        1 & j\sqrt{\frac{m}{k}}
                        \end{bmatrix}, \quad
e^{\mathbf{D}\Delta t} = \begin{bmatrix} e^{j\sqrt{\frac{k}{m}}\Delta t} & 0 \\ 0 & e^{-j\sqrt{\frac{k}{m}}\Delta t} \end{bmatrix} $$

Using Euler's formula \\(e^{j\theta} = \cos(\theta) + j\sin(\theta)\\), this simplifies to the same result as Solution 1. Well do it yourself or just believe me.

#### Verification 
The analytical solution can be verified using SciPy:

```python
from scipy.signal import cont2discrete, lti
import numpy as np

# Define the continuous-time system for Example 2 (spring-mass system)
# Let k/m = 1 for simplicity (natural frequency = 1 rad/s)
A = np.array([[0, 1], [-1, 0]])
B = np.array([[0], [1]])  # 1/m = 1
C = np.array([[1, 0], [0, 1]])  # Full state observation
D = np.array([[0], [0]])

# Create state-space system
l_system = lti(A, B, C, D)

dt = 0.1  # Sampling time

method = 'zoh'
d_system = cont2discrete((A, B, C, D), dt, method=method)
print("ZOH Discretization for Spring-Mass System:")
print(f"A_d = \n{d_system[0]}")
print(f"B_d = \n{d_system[1]}")
```

For \\(k = 1\\), \\(m = 1\\) and \\(\Delta t = 0.1\\), the ZOH method should produce matrices that match our analytical expressions with \\(\cos(0.1) \approx 0.995\\) and \\(\sin(0.1) \approx 0.0998\\).

## General Method

For practical implementation, SciPy provides robust numerical methods for system discretization that work for any linear time-invariant system. The general approach is the one used in the verification of the mentioned examples.

### Comparison of Different Discretization Methods

Different discretization methods have varying properties:

1. **Zero-Order Hold (ZOH)**: Most common, assumes input is constant between sampling instants
2. **First-Order Hold (FOH)**: Assumes linear interpolation of input between samples
3. **Bilinear (Tustin)**: Frequency domain method, preserves stability
4. **Euler**: Simple forward difference approximation
5. **Backward Difference**: Backward Euler method
6. **Impulse**: For impulse-invariant transformation

```python
from scipy.signal import cont2discrete, lti, dstep
import numpy as np
import matplotlib.pyplot as plt

# Define the continuous-time system
A = np.array([[0, 1], [-1, 0]])
B = np.array([[0], [1]])
C = np.array([[1, 0], [0, 1]])
D = np.array([[0], [0]])

# Create state-space system properly
l_system = lti(A, B, C, D)
# Change to Transfer Function to use step function
# input = 0 specifies which index of input to use in case of multi-input systems
t, x = l_system.to_tf(input=0).step(T=np.linspace(0, 10, 100)) 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

dt = 0.1

# Plot position on the first subplot
ax1.plot(t, x[:,0], label='Continuous', linewidth=3)
for method in ['zoh', 'bilinear', 'euler', 'backward_diff', 'foh', 'impulse']:
   d_system = cont2discrete((A, B, C, D), dt, method=method)
   if method == 'zoh':
      print(d_system)
   s, x_d = dstep(d_system)
   ax1.step(s, np.squeeze(x_d)[:,0], label=method, where='post')
ax1.axis([t[0], t[-1], np.min(x[:,0]), np.max(x[:,0])])
ax1.legend(loc='best')
ax1.set_title('Position Output')
ax1.set_xlabel('Time')
ax1.set_ylabel('Position')

# Plot velocity on the second subplot
ax2.plot(t, x[:,1], label='Continuous Velocity', linewidth=3)
for method in ['zoh', 'bilinear', 'euler', 'backward_diff', 'foh', 'impulse']:
   d_system = cont2discrete((A, B, C, D), dt, method=method)
   s, x_d = dstep(d_system)
   ax2.step(s, np.squeeze(x_d)[:,1], label=method, where='post')
ax2.axis([t[0], t[-1], np.min(x[:,1]), np.max(x[:,1])])
ax2.legend(loc='best')
ax2.set_title('Velocity Output')
ax2.set_xlabel('Time')
ax2.set_ylabel('Velocity')
fig.tight_layout()
plt.show()
```

Below is a visualization of the discretized spring-mass system response:

![Discretized Spring-Mass System Response](assets/img/discretize_spring_mass_system.png)

Here we see that the `euler` method explodes marginally stable system mentioned in example 2. And `backward_diff` method dampens the system.  

This general approach works for any linear time-invariant system and provides a robust foundation for practical control system implementation. The choice of discretization method depends on the specific requirements of the application, such as frequency response preservation, computational efficiency, and numerical stability.