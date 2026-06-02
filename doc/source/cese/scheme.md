# Schemes

Numerical computation, facilitated by the advancement of digital computers,
enables all kinds of simulation.  We use it to solve non-linear hyperbolic
partial differential equations (PDEs), which come from conservation laws
{cite:p}`lax_hyperbolic_1973`.

The conservation element and solution element (CESE) method solves conservation
laws, which can be written in the following form in one-dimensional space

```{math}
:label: e:cese:1d_pde

\frac{\partial u}{\partial t}
+ \frac{\partial f(u)}{\partial x} = 0
.
```

$u$ is the dependent solution variable and $f(u)$ is a function.  $(x, t)$ is
the independent variables defining the two axes of two-dimensional Euclidean
space.  Let $\mathbf{h} \defeq (f(u),u)$ and rewrite Eq. {eq}`e:cese:1d_pde` to
$\nabla\cdot\mathbf{h} = 0$ with the divergence operator.  Assuming Eq.
{eq}`e:cese:1d_pde` applies everywhere in the control volume $V$, we write

$$
\int_V\nabla\cdot\mathbf{h}\dif v = 0
.
$$

By using the divergence theorem, the above differential equation is cast into
an integral equation over the control surface $S(V)$ surrounding $V$

```{math}
:label: e:cese:1d_integral_form

\oint_{S(V)}\mathbf{h}\cdot\dif\hat{\sigma} = 0
.
```

$\dif\hat{\sigma}$ is the infinitesimal surface vector.  Eq.
{eq}`e:cese:1d_integral_form` does not require the point-wise divergence free
condition.

The CESE method solves the integral equation, instead of the differential
equation, by enforcing the space-time flux conservation as shown in Eq.
{eq}`e:cese:1d_integral_form` {cite:p}`chang_method_1995`.  The solution
variable and its partial derivative in space are independent but solved
together.  It uses a compact stencil that defines two entities: the
conservation element (CE) and solution element (SE).  Space-time invariants are
used to minimize numerical dissipation, but the characteristics-based methods
are not used for obtaining solution.  Ad hoc treatments are avoided as much as
possible.  {cite:t}`chang_new_1991` and {cite:t}`chang_method_1995`.

```{eval-rst}
.. pstake:: schematic/cce.tex
   :align: center

   A compounded conservation element (CCE), the area enclosed by the
   red dots of :math:`\square\mathrm{BCEF}`, contains two basic
   conservation elements (BCEs), the area enclosed by the blue dots of
   :math:`\square\mathrm{ABCD}` and :math:`\square\mathrm{ADEF}`.
```

The CEs discretize the space-time for the integral equation to be solved (Eq.
{eq}`e:cese:1d_integral_form`).  $\mathrm{CE}(j,n)$ denotes a single CE
associated with the grid point $(x_j, t^n)$.  In the CE, the conservation of
$\mathbf{h}$ is approximated as

```{math}
:label: e:conserv_of_approx_h

\oint_{S(\mathrm{CE})}\mathbf{h}^*\cdot \dif\hat{\sigma} = 0
```

where $\mathbf{h}^*$, which will be defined later with SEs, denotes the
approximation of $\mathbf{h}$.  A CE defined like that is a compounded
conservation element (CCE), consisting of two adjacent basic conservation
elements (BCEs) $\mathrm{CE}_-$ and $\mathrm{CE}_+$.  Eq.
{eq}`e:conserv_of_approx_h` holds in both CCEs and BCEs, i.e.,

$$
\oint_{S(\mathrm{CE}_\pm)}\mathbf{h}^*\cdot \dif\hat{\sigma} = 0 .
$$

$S(\mathrm{CE}_{\pm})$ is the bounding surface surrounding
$\mathrm{CE}_{\pm}$.

```{eval-rst}
.. pstake:: schematic/cese_marching.tex
   :align: center

   Time-marching the solution by using the cross-shaped solution elements.  The
   red dotted crosses mark the SEs at :math:`t=t^0`.  The blue dotted crosses
   mark the SEs at :math:`t=t^{1/2}`.  The orange dotted crosses mark the SEs
   at :math:`t=t^1`.  The bigger dots at the horizontal middle points of the
   crosses are the solution points.  The arrows show how the solution variable
   and its spatial derivative, that are defined at the solution points, at the
   previous half time step propagate to those at the next half time step.  The
   solution points at the boundary, :math:`x=x_0` and :math:`x=x_4`, need to be
   updated by boundary-condition treatments, rather than the CESE method
   scheme.
```

The SEs determine $\mathbf{h}^*$.  There is more than one way to define SEs,
while an effective and consistent approach is shown above.  Let
$\mathrm{SE}(j,n)$ denote the SE associated with the grid point $(j,n)$, which
is the cross-shaped mark enclosed by the dotted line.  The solution variable
approximation is written as

$$
u^*(x,t;j,n) = u_j^n + (u_x)_j^n(x-x_j) + (u_t)_j^n(t-t^n) .
$$

The grid point $(x_j, t^n)$ is used as the solution point.  $u_j^n$,
$(u_x)_j^n$, and $(u_t)_j^n$ hold constant in $\mathrm{SE}(j,n)$.  It should be
noted that every CE is surrounded by SEs.  Fluxes evaluated through the CE
boundary depends only on the approximation within SEs.  To proceed, write

$$
\frac{\partial u^*(x,t;j,n)}{\partial x} = (u_x)_j^n, \quad
\frac{\partial u^*(x,t;j,n)}{\partial t} = (u_t)_j^n .
$$

Substitute the approximated solution variable $u^*$ back to the original
differential equation (Eq. {eq}`e:cese:1d_pde`) and obtain the relation between
$(u_x)_j^n$ and $(u_t)_j^n$ as

$$
              (u_t)_j^n + (f_u)_j^n(u_x)_j^n = 0
\;\Rightarrow\; (u_t)_j^n = -(f_u)_j^n(u_x)_j^n .
$$

The approximated solution variable $u^*$ is then rewritten as

$$
u^*(x,t;j,n) = u_j^n
+ (u_x)_j^n\left[(x-x_j) - (f_u)_j^n(t-t^n)\right] .
$$

Similarly, the approximated function

$$
f^*(x,t;j,n) = f_j^n + (f_x)_j^n(x-x_j) + (f_t)_j^n(t-t^n)
$$

is rewritten as

$$
f^*(x,t;j,n) = f_j^n + (f_u)_j^n (u_x)_j^n \left[
  (x-x_j) - (f_u)_j^n(t-t^n)
\right] .
$$

## The $c$-Scheme

To this point, we are ready to write the time-marching formulae for the
$c$-scheme.  It is the most simple time-marching scheme for the CESE method.
The formulae include updating the solution variable and the spatial derivative.
The part for the solution variable $u$ will be obtained by enforcing the
space-time flux conservation over the CCE.  It should be noted that the height
of the CEs and SEs is the half time step $\Delta t/2$, not the full time step
$\Delta t$.

```{eval-rst}
.. pstake:: schematic/se_flux.tex
   :align: center

   Space-time flux at the boundary of
   :math:`\mathrm{CE}(j, n+\frac{1}{2})` defined by
   :math:`\mathrm{SE}(j-\frac{1}{2}, n)` (red),
   :math:`\mathrm{SE}(j+\frac{1}{2}, n)` (blue), and
   :math:`\mathrm{SE}(j, n+\frac{1}{2})` (orange).  :math:`x_j`
   denotes the grid point of the :math:`j`-th SE,
   :math:`x_j^{\pm}` the right and left end points, :math:`x_j^s`
   the solution point.
   :math:`\Delta x_j^+ \overset{\text{def}}{=} x_j^+ - x_j` and
   :math:`\Delta x_j^- \overset{\text{def}}{=} x_j - x_j^-` are the
   length of the right and left arms of the :math:`j`-th SE.
```

```{eval-rst}
.. pstake:: schematic/nonuni_se.tex
   :align: center

   The SE definition for a non-uniform one-dimensional grid.  The cross-shaped
   marks are the SEs, and the solid dots are the associated solution points.
```

The solution point must be at the center of the SE to make first-order
approximation consistent.  In a non-uniform grid, the solution points may not
collocate with grid points.  The approximation formulae should be changed to

$$
\begin{aligned}
u^*(x,t;j,n) &= u_j^n + (u_x)_j^n \left[
  (x-x_j^s) - (f_u)_j^n(t-t^n)
\right] , \\
f^*(x,t;j,n) &= f_j^n + (f_u)_j^n (u_x)_j^n \left[
  (x-x_j^s) - (f_u)_j^n(t-t^n)
\right] .
\end{aligned}
$$

The formula for the solution variable $u$ is found to be

```{math}
:label: e:formula:u

u_j^{n+\frac{1}{2}}
  = \frac{1}{\Delta x_j}\left\{
    (u^*)_{j-\frac{1}{2},+}^n \Delta x_{j-\frac{1}{2}}^+
  + (u^*)_{j+\frac{1}{2},-}^n \Delta x_{j+\frac{1}{2}}^-
  + \frac{\Delta t}{2} \left[
      (f^*)_{j-\frac{1}{2}}^{n,+}
    - (f^*)_{j+\frac{1}{2}}^{n,+}
    \right]
    \right\}
```

where

$$
\begin{aligned}
(u^*)_{j\mp\frac{1}{2},\pm}^n
 &= u_{j\mp\frac{1}{2}}^n
  + (u_x)_{j\mp\frac{1}{2}}^n
    \left( x_{j\mp\frac{1}{2}}
         \pm \frac{1}{2} \Delta x_{j\mp\frac{1}{2}}^{\pm}
         - x_{j\mp\frac{1}{2}}^s \right), \\
(f^*)_{j\mp\frac{1}{2}}^{n,\pm}
 &= f_{j\mp\frac{1}{2}}^n
  + (f_u)_{j\mp\frac{1}{2}}^n(u_x)_{j\mp\frac{1}{2}}^n
    \left[x_{j\mp\frac{1}{2}} - x_{j\mp\frac{1}{2}}^s
        - (f_u)_{j\mp\frac{1}{2}}^n \frac{\Delta t}{4}
    \right] .
\end{aligned}
$$

The first-order derivative $u_x$ needs another formula.  The one of
the $c$-scheme is

```{math}
:label: e:formula:ux:c

(u_x)_j^{n+\frac{1}{2}} = \frac{
  (u')_{j+\frac{1}{2}}^n - (u')_{j-\frac{1}{2}}^n
}{\Delta x_j}
```

where

$$
(u')_{j\pm\frac{1}{2}}^n \defeq
    u_{j\pm\frac{1}{2}}^n
  + (u_x)_{j\pm\frac{1}{2}}^n \left[
      x_{j\pm\frac{1}{2}} - x_{j\pm\frac{1}{2}}^s
    - (f_u)_{j\pm\frac{1}{2}}^n \frac{\Delta t}{2} \right]
$$

are the Taylor expansion to $t^{n+\frac{1}{2}}$ with respect to
$\mathrm{SE}(j\pm\frac{1}{2}, n)$.  Eqs. {eq}`e:formula:u` and
{eq}`e:formula:ux:c` together form the $c$-scheme.

## Weighting Function for Discontinuity

A weighting function should be introduced to treat discontinuity in space.
Eqs. {eq}`e:formula:u` and {eq}`e:formula:ux:c` are called the $c$-scheme
because Eq. {eq}`e:formula:ux:c` uses central-differencing to approximate
$(u_x)_j^{n+\frac{1}{2}}$.  It is second-order accurate but doesn't give
correct result when the solution variable $u$ is discontinuous between
$x_{j-\frac{1}{2}}$ and $x_{j+\frac{1}{2}}$.  To define the weighting function,
let

$$
(u_{x\pm})_j^{n+\frac{1}{2}} \defeq
  \frac{
    (u')_{j\pm\frac{1}{2}}^n - u_j^{n+\frac{1}{2}}
  }{x_{j\pm\frac{1}{2}} - x_j}
$$

be the spatial differences in the two intervals $[x_j, x_{j+\frac{1}{2}}]$ and
$[x_{j-\frac{1}{2}}, x_j]$, respectively.  The weighted average of the spatial
difference can then be calculated with

$$
(u_x^w)_j^{n+\frac{1}{2}} =
W\left(
  (u_{x-})_j^{n+\frac{1}{2}}, (u_{x+})_j^{n+\frac{1}{2}}
\right)
$$

where $W$ is the weighting function.  When there is discontinuity in $u$
between $x_{j-\frac{1}{2}}$ and $x_{j+\frac{1}{2}}$, the weighting function
should detect it and approximate the spatial differencing properly.  A choice
of the weighting function is

```{math}
:label: e:formula:wa

W_{\alpha} =
  \frac{|u_{x+}|^{\alpha}u_{x-}
      + |u_{x-}|^{\alpha}u_{x+}}
       {|u_{x-}|^{\alpha} + |u_{x+}|^{\alpha}},
```

where $\alpha \in \mathbb{R}$ is a constant parameter and $|u_{x-}|^{\alpha} +
|u_{x+}|^{\alpha} > 0$.  $\alpha$ is often picked as a positive integer, e.g.,
2, for saving computation cycles.  For non-linear equations or discontinuous
initial conditions, a weighting function, e.g., Eq. {eq}`e:formula:wa`, is
necessary to keep the solution from diverging.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
