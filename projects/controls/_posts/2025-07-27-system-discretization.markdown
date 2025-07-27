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
                        \end{bmatrix} \\
                    &= \begin{bmatrix}
                        \Delta t & \frac{(\Delta t)^2}{2} \\
                        0 & \Delta t
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
    e^{\mathbf{A}\Delta t}  &= \sum_{k=0}^{\infty}{\frac{1}{k!}\left(\mathbf{A}\Delta t\right)^k} \\
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
                    \end {bmatrix} \\
                    &= \begin{bmatrix}
                    \sqrt{\frac{m}{k}}\sin\left(\Delta t \sqrt{\frac{k}{m}}\right) & \frac{m}{k}\left(1-\cos\left(\Delta t \sqrt{\frac{k}{m}}\right)\right) \\
                    -\left(1-\cos\left(\Delta t \sqrt{\frac{k}{m}}\right)\right) & \sqrt{\frac{m}{k}}\sin\left(\Delta t \sqrt{\frac{k}{m}}\right)
                    \end {bmatrix} 
\end{align} $$

Here again, let us assume the measurable elements of the system \\(\mathbf{y}\\) are all its states. So the matrix \\(\mathbf{C}\\) is identity \\(\mathbf{I}\\), i.e.

$$ \begin{align}
    \mathbf{y}  &= \mathbf{Cx} \\
                &= \mathbf{Ix} \\
    \mathbf{y}  &= \mathbf{x}
\end{align} $$

So \\(\mathbf{C}_d=\mathbf{C} = \mathbf{I}\\). And

$$ \mathbf{x}_{k+1} = \mathbf{A}_d\mathbf{x}_k + \mathbf{B}_d\mathbf{u}_k, \quad \mathbf{y}_k = \mathbf{C}_d\mathbf{x}_k $$

#### Solution 2
Now a second solution to the same problem. Which can be algorithmically more sophisticated to implement, based on diagonalization. Here we find eigenvalues and corresponding eigenvectors to find the diagonalized form and then calculate the matrices.

#### Verification

## General Method