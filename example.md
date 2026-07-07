6
2
0
2

n
u
J

1
1

]

Y
S
.
s
s
e
e
[

1
v
5
6
4
3
1
.
6
0
6
2
:
v
i
X
r
a

Embodied Opinion Dynamics
for Safety-Critical Motion Control
in Dynamic Environments

Zhiqi Tang ∗ Yu Xing ∗∗

∗ Department of Electrical and Electronic Engineering, The University
of Manchester, UK (e-mail: zhiqi.tang@manchester.ac.uk)
∗∗ The Faculty of Computer Science, RWTH Aachen University,
Aachen, Germany (e-mail: yu.xing@rwth-aachen.de)

Abstract: This paper proposes a novel adaptive control framework that embeds nonlinear
opinion dynamics within the dynamical sensorimotor layers of an automated vehicle governed
by second-order nonholonomic bicycle kinematics. The framework enables an ego vehicle to
perform adaptive decision-making and achieve safe motion control under interaction uncertainty
with non-cooperative neighboring agents. We consider a representative case study in which
an ego vehicle autonomously attempts to merge into a lane occupied by human-driven or
automated vehicles whose intentions are unknown. Within the proposed framework, the ego
vehicle adaptively selects and executes merging versus non-merging behaviors in response to
changing environmental conditions. Formal safety guarantees, as well as equilibrium and stability
analyses of the closed-loop system, are provided. Numerical simulations further demonstrate the
effectiveness of the proposed approach.

Keywords: Nonlinear Opinion Dynamics, Reactive Collision Avoidance, Adaptive Control,
Automated Vehicle

1. INTRODUCTION

Dynamical systems have provided valuable insights into
how decision-making emerges in various biological con-
texts, including human and honeybee swarms (Gray et al.,
2018). These decision-making mechanisms are often value-
based, such that the agent makes a decision when a speci-
fied variable crosses a static or dynamic threshold. Recent
developments in nonlinear opinion dynamics (Bizyaeva
et al., 2022) have shown advantages in its flexible, tunable
sensitivity, allowing the model to be either sensitive or
robust to input. However, these models generally neglect
the agent’s control over its own physical dynamics. It
remains an open question how to better systematize the
advantages of these decision-making mechanisms in closed-
loop control of physical systems.

Inspired by the embodied decision-making of biological
organisms whose sensory input, actions, and cognitive
processes are interconnected (Lepora and Pezzulo, 2015),
Reverdy and Koditschek (2018) and Reverdy et al. (2021)
propose a motivation dynamics framework (a dynamical
systems framework) for a single robot to plan and exe-
cute recurrent coverage tasks. In the framework, value-
based decision-making mechanisms are embedded directly
within the dynamical sensorimotor layers of autonomous
physical systems. Similarly, Amorim et al. (2024) propose
a threshold-based decision-making framework for agents
dynamically choosing between two spatial tasks that ac-
counts for the agent’s physical state. A key advantage of
using a continuous motivation state in these approaches
is the ability to encode a much richer set of behaviors

than what is possible using a discrete switching variable.
This richer behavioral repertoire has been also shown to
facilitate deadlock resolution in robot navigation tasks. For
instance, Cathcart et al. (2023) propose a proactive social-
navigation framework in which a robot forms an “opinion’’
regarding how and in which direction to pass a human
mover. Likewise, Qi et al. (2025) exploit nonlinear opinion
dynamics within a safety controller to achieve blocking
resolution without communication or predefined rules.

All of the works mentioned above consider only elementary
robot dynamics, such as single-integrator models. More-
over, except for Reverdy and Koditschek (2018), these
approaches do not provide stability analysis and con-
vergence guarantees for the physical state in the closed-
loop system. Consequently, it remains an open problem
to develop a general motivation-dynamics framework that
provides formal performance guarantees for robots with
more complex and realistic dynamics operating in dynamic
environments.

Inspired by the approach proposed by Reverdy and
Koditschek (2018), this paper introduces a novel adap-
tive control framework that integrates nonlinear opinion
dynamics within the dynamical sensorimotor layers of an
automated vehicle governed by second-order kinematics of
a nonholonomic bicycle model. The proposed framework
is tailored for dynamic environments populated by non-
cooperative neighboring agents, enabling the vehicle to
perform adaptive decision-making and achieve safe motion
control under interaction uncertainty.

Specifically, we consider a representative case study of
a single ego vehicle autonomously merging into a lane
occupied by a group of neighboring vehicles when the op-
portunity arises, while maintaining safety at all times. The
primary challenge arises from that these neighboring ve-
hicles may consist of both human-driven and autonomous
vehicles that cannot coordinate with or communicate with
the ego vehicle. As a result, their intentions remain un-
known, and they may or may not yield suﬀicient space for
the ego vehicle to merge. Within the proposed framework,
illustrated in Fig. 1, the opinion state of the ego vehicle
represents its preference for executing a merging maneuver
into the neighboring lane. This opinion is modeled as
a dynamical system driven by the ego vehicle’s physical
state relative to surrounding traﬀic. Conversely, the ego
vehicle’s motion controller is informed by this opinion
state, ensuring that the vehicle’s desired behavior emerges
from the coupling between decision dynamics and physical
dynamics.

Fig. 1. The proposed control framework.

In summary, the main contributions of the paper are as
follows:

(1) We propose a novel control framework that couples
nonlinear opinion dynamics with the second-order
nonholonomic bicycle dynamics of an ego vehicle, en-
abling the adaptive selection and execution of merg-
ing versus non-merging behaviors when interacting
with non-cooperative neighboring agents.

(2) We provide formal safety guarantees and equilibrium
analysis by leveraging the constructive formulation of
dissipative barrier feedback for safety-critical control.
(3) We establish the stability of the overall closed-loop
system by exploiting the time-scale separation be-
tween the opinion dynamics and the vehicle dynamics.

The remainder of this paper is organized as follows. In
Section 2, we introduce the vehicle model and formulate
the problem addressed in this work. Section 3 presents the
detailed design of the proposed framework along with the
associated theoretical results. In Section 4, we demonstrate
the effectiveness of the method through numerical evalua-
tions. Finally, Section 5 provides concluding remarks.

2. VEHICLE MODEL AND PROBLEM
FORMULATION

This section introduces the ego vehicle model and formally
states the main problem addressed in this paper.

Fig. 2. Kinematic bicycle model for 2-dimensional vehicle

motion.

2.1 Vehicle model

The ego vehicle is modeled using the second-order kine-
matics of a nonholonomic bicycle, as follows:

2

6
6
6
6
6
4

3

7
7
7
5 =

2

6
6
6
4

˙xr
˙yr
˙θ
˙vr
˙δ

vr cos(θ)
vr sin(θ)
vr tan(δ)
L
0
0

3

7
7
7
7
7
5

+

2

6
6
6
4

3

7
7
7
5

0 0
0 0
0 0
1 0
0 1

(cid:20)

(cid:21)

a
ω

(1)

where (xr, yr) and θ indicate the rear axle center position
and orientation of the ego vehicle in the common global
frame, vr is the speed measured at the rear wheel, and
finally δ , L are the steering angle and wheelbase of vehicle,
respectively, as shown in Fig. 2. The control inputs of the
system are the longitudinal acceleration a and the angular
rate of the steering wheel ω.

(cid:26)

In this paper, the focus is on the control design for the
center of the front axle of the vehicle, modeled as a double
integrator:

˙p = v
˙v = u
where p ∈ R2 and v ∈ R2 are the position and velocity of
of the front axle, and u ∈ R2 is the control input to be
designed. As shown in Fig. 2, the position of the front axle
can be presented using the rear axle position
(cid:21)

(2)

(cid:21)

(cid:20)

(cid:20)

p =

xh
yh

=

xr + L cos(θ)
yr + L sin(θ)

.

(3)

Take the second time derivatives of the above equation,
the control input (a, ω) of the kinematic bicycle model (1)
can be transferred from the control design u of the double
integrator model similarly to Chen et al. (2024), as follows
cos(θ) − sin(θ) tan(δ) −vr sin(θ) sec2(δ)
sin(θ) + cos(θ) tan(δ) vr cos(θ) sec2(δ)
{z
A

a
ω

−1

=

(cid:21)

(cid:20)

(cid:21)

(cid:20)

}
1

3

(4)

− v2
r
L
v2
r
L

sin(θ) tan(δ) − v2
r
L
cos(θ) tan(δ) − v2
r
L

cos(θ) tan2(δ)

7
5

C
A

sin(θ) tan2(δ)

|
2

0

B
@u −

6
4

This expression is valid as long as det A ̸= 0. Direct
calculation gives det A = vr sec2(δ), so a solution exists
as long as δ ̸= π
2 and vr ̸= 0. Physical constraints of the
vehicle ensure δ < π
2 . To guarantee invertibility of A when
|vr| < ϵv (with ϵv a small positive constant), vr in A is
replaced by sign(vr) ϵv.

ሶ𝜇=𝑓𝜇(𝜇,𝑥,𝑡)𝜖ሶ𝑧=𝑓𝑧(𝑧,𝜇)ሶ𝑥=𝑓(𝑥,𝑢𝑥,𝑥∗𝑧,𝑡)Bifurcation parameterOpinion dynamicsLow-level safe motion controlHigh-level decision making𝜇𝑧𝑥sensorsEnvironmental informationxy(𝑥𝑖ℎ,𝑦𝑖ℎ)ehMoatkimdealasod2.2 Problem formulation

In this paper, we consider the problem of an ego vehicle
that intends to autonomously merge into a lane occupied
by a group of neighboring vehicles whenever an oppor-
tunity arises, while maintaining safety at all times. The
challenge arises from the fact that the intentions of the
neighboring vehicles are unknown. These vehicles may or
may not yield space for the ego vehicle to merge.

Given the complexity of the problem, it is assumed that
the ego vehicle has already identified the leader vehicle
to follow in the platoon of neighboring vehicles, in prepa-
ration for the merge. As illustrated in Fig. 3, Vehicle 1
is designated as the leader, while Vehicle 2 is the vehicle
immediately following Vehicle 1 in the same lane before the
ego vehicle merges. Additionally, the following assumption
is made regarding Vehicles 1 and 2:
Assumption 1. For each neighboring vehicle j ∈ {1, 2}, the
position pj, velocity vj, and acceleration uj are bounded
for all time. Each vehicle drives in the same lane and
maintains a safe distance from other vehicles in that lane.

Fig. 3. Illustration of the ego vehicle (green) and neigh-

boring vehicles (blue).

We begin by describing the proposed framework in Fig. 1,
after which we present a formal problem formulation. The
first step involves designing an opinion that represents the
ego vehicle’s preference for lane-changing. This opinion is
denoted by z ∈ R with dynamics

(5)
˙z = fz(z, µ),
where fz(·) is a nonlinear function to be designed, and µ ∈
R is a bifurcation parameter that varies the equilibrium of
z. Note that a change in the equilibrium corresponds to a
continuous switch in the vehicle’s opinion.

As the opinion z should naturally be influenced by the
physical state of the ego vehicle x = [p⊤ v⊤]⊤ as well as
the surrounding environment (i.e., the states of Vehicles
1 and 2), the dynamics of the bifurcation parameter are
designed as

˙µ = fµ(µ, x, t),

(6)

where t denotes time.

To inform the vehicle dynamics with the opinion, the
opinion is used to coordinate the desired state of the ego
vehicle, resulting in closed-loop dynamics of the form

˙x = f

(7)
where u(·) ∈ R2 is the safe motion controller to be
designed.

x, u(x, x∗(z), t)

,

(cid:0)

(cid:1)

Based on the above framework, the central problem ad-
dressed in this paper is the design of (i) suitable opinion
dynamics (5), (ii) bifurcation parameter dynamics (6),
and (iii) a safe motion controller u(·), such that the ego

Fig. 4. Bifurcation diagram of the system (8). The Solid
line represents stable equilibria, whereas the dotted
line represents unstable ones.

vehicle remains safe at all time while successfully executing
a merging maneuver between Vehicles 1 and 2 when there
is a chance.

3. PROPOSED FRAMEWORK DESIGN

In this section, the detailed design of the proposed frame-
work is presented.

Recall (5), let z = 0 represents the opinion to stay in the
original lane, and z > ϵ1 > 0 represents the opinion to
change lanes. The opinion dynamics of z is designed as 1

˙z =

1
ϵ

(µz − z2),

(8)

a = 0 and z∗

where ϵ > 0.
System (8) has two equilibrium points, z∗
b = µ
when µ ̸= 0, and a unique equilibrium point z∗ = 0 when
µ = 0 (see Fig. 4). The stability properties are summarized
in the following lemma for completeness.
Lemma 1. Consider system (8) with ϵ > 0. If z(0) ≥ 0,
then z(t) ≥ 0 for all t ≥ 0.
When µ ̸= 0, the system has two equilibrium points, z∗
and z∗

a = 0

b = µ.
• If µ < 0, z∗

z∗
b is unstable.
• If µ > 0, z∗

asymptotically stable.

a is locally asymptotically stable, whereas

a is unstable, whereas z∗

b

is locally

Proof. For z(0) ≥ 0, the right-hand side of (8),
˙z =
ϵ z(µ − z), ensures z(t) ≥ 0 for all t ≥ 0. Equilibria satisfy
1
b = µ when µ ̸= 0, and z∗ = 0
˙z = 0, giving z∗
ϵ (µ − 2z∗),
dz ˙z|z∗ = 1
when µ = 0. Linearization yields d
which implies:

a = 0 and z∗

• z∗
• z∗

a = 0 is stable if µ < 0 and unstable if µ > 0.
b = µ is stable if µ > 0 and unstable if µ < 0.

Next, the bifurcation parameter (6) is designed with the
following dynamics:
(cid:16)

˙µ = −kµµ + tanh

− k ρ⊤g1 ρ⊤g2 ( ¯d21 − 2r)

(cid:1)(cid:17)

+ ϵ1

,

(cid:0) ˙¯d21
¯d21

(9)
where ρ ∈ S1 denotes the constant longitudinal direction of
the road, gj = p−pj
∥p−pj ∥ for j ∈ {1, 2} is the unit vector from
the ego vehicle to neighboring Vehicle j, ¯d21 = ∥p2−p1∥−r
1 Note that this system with ϵ = 1 corresponds to the normal form
of a transcritical bifurcation (Perko, 2013, Section 4.2, Example 2).

7-1-0.500.51z$-1-0.500.51StableUnstable21v21 with
∥p2−p1∥ and v21 = v2 − v1, kµ and k are positive

with r > 0 representing a safe margin,
¯g21 = p2−p1
gains and ϵ1 is a positive scalar.

˙¯d21 = ¯g⊤

This design ensures that the opinion z will deviate from its
neutral state only if the following conditions are satisfied
simultaneously:

(1) The ego vehicle is in the correct relative position to

merge, i.e., ρ⊤g1 ρ⊤g2 < 0.

(2) There is suﬀicient space for merging, i.e., ¯d21 > 2r.
(3) The divergent flow (ratio of relative velocity and
relative distance between Vehicles 1 and 2) satisfies
˙¯d21
¯d21

> −ϵ1.

Finally, the method of dissipative barrier feedback (Tang
et al., 2023) is employed for safe control design:

u = un + uc,

where the nominal tracking controller is defined as

un = −kd(e1 − e∗

1(z)) − kvν1 + u1,

with

e1 = p − p1,

ν1 = v − v1,

(10)

(11)

and

e∗
1 = ρrρ + (1 − w(z))ηrη
(12)
denoting the desired relative position coordinated with
the opinion z, where the weight is defined as w(z) =
tanh(kwz). Here, ρ ∈ S1 is the longitudinal direction of
the road, η ∈ S1 is the lateral direction, rρ and rη are
constant offsets, and u1 is the acceleration of the leader
vehicle.

To ensure safety, the dissipative barrier feedback is de-
signed as

uc = −

X

kogj

j∈{1,2}

˙dj
dj

,

(13)

where

dj = ∥p − pj∥ − r,

˙dj = g⊤

j νj.

The following lemma establishes formal safety guarantees,
as well as equilibrium and stability analyses of the closed-
loop system under the proposed framework.
Lemma 2. Consider two neighboring vehicles satisfying
Assumption 1, and an ego vehicle deciding whether to
merge between them. The ego vehicle operates under the
safe controller (10), the opinion dynamics (8), and the
bifurcation dynamics (9). Suppose the initial position p(0)
velocity v(0) are bounded and safe according to Lemma 3
˙dj
(0) bounded for j ∈ {1, 2}).
in the Appendix (dj(0) > 0,
dj
Assume also that the initial opinion satisfies z(0) > 0, and
that the gains kp, kv, ko, kw, kµ, k, and ϵ are positive and
bounded. Then:
(1) The ego vehicle remains safe for all t ≥ 0,

i.e.,

dj(t) > 0 and

˙dj
dj

(t) bounded for j ∈ {1, 2}.

(2) There exists a suﬀiciently large gain kw > 0 and
a constant ϵ∗ > 0 such that, for all 0 < ϵ <
ϵ∗, the equilibrium points (e1, ν1) = (e∗
1(z∗), 0) are
asymptotically stable, provided that v21(t) converges
to zero.

converge to zero, the closed-loop system (18) is input-to-
state stable with respect to v21.

Proof. Proof of item 1):
Recall that ˙dj = g⊤
that:

j νj, j ∈ {1, 2}, and hence one verifies

− kv

˙dj − kog⊤

j gl

˙dl
dl

+ αj, l ∈ {1, 2}, l ̸= j, (14)

¨dj = −ko

˙dj
dj

where

α1 = −g⊤

1 kp(e1 − e∗

α2 = −g⊤

2 (kp(e1 − e∗

1(z)) +

∥πg1 ν1∥2
d1 + r
1(z)) + u2 − u1 + v21) +

,

∥πg2 ν2∥2
d2 + r

with v21 = v2 − v1 and πy = I − yy⊤ the projection
operator for S1. We prove that dj remains positive using
contradiction, in two cases.

i) Assume that one of d1 or d2 approaches zero at a finite
time T , while the other remains positive on [0, T ].

Integrating (14) from 0 to T gives:

ko(ln dj(T ) − ln dj(0))
= − ˙dj(T ) + ˙dj(0) − kv(dj(T ) − dj(0))
˙dl
dl

(αj − kog⊤

j gl

)dτ

+

Z

T

0

(15)

The left-hand side of the equation tends to negative
infinity, however the right-hand side of the equation is
either bounded or tends to positive infinity. This is because
˙dj is either bounded or negative
as dj approaches zero,
infinity; e1, e∗
̸= j) remain
is also bounded; u2 and v12 are bounded
positive, hence
by Assumption 1, and αj is either bounded or tends to
positive inifinity. This yields a contradiction.

1(z), and dl (l ∈ {1, 2}, l
˙dl
dl

˙d1
d1

1 g2)(

¨d1 + ¨d2 = −ko(1 + g⊤

ii) Assume now that d1 and d2 approach zero simultane-
ously. Summing ¨d1 and ¨d2 yields:
˙d2
d2

) − kv( ˙d1 + ˙d2) + α1 + α2
(16)
Using the fact that |g⊤
1 g2| ≤ 1 and Under Assumption 1,
if −1 ≤ g⊤
1 g2 ≤ −1 + ϵ2 (for some small ϵ2 > 0), the ego
vehicle has already merged, so d1 and d2 cannot both go to
zero simultaneously. Hence, we only consider g⊤
1 g2 > −1 +
ϵ2 on [0, T ].

+

Integrating (16), it yields:

X

β

ko(ln dj(T ) − ln dj(0))

j∈{1,2}
X

(cid:0)

=

j∈{1,2}

− ˙dj(T ) + ˙dj(0) − kvdj(T )

(17)

Z

T

(cid:1)

+ kvdj(0) +

αjdτ

0
where β ∈ [min(1 − g⊤
scalar, such that :
R

T

0 (1 + g⊤

1 g2)(

˙d1
d1

+

˙d2
d2

)dτ = β

1 g2), max(1 − g⊤

1 g2)] is a positive

P

j∈{1,2}(ln dj(T ) − ln dj(0)).

Remark 1. Note that v21 is not required to converge to
zero, as assumed in Assumption 1. When v21 does not

Using the same contradiction argument as for (15), we con-
clude that d1 and d2 cannot approach zero simultaneously.

Proof of item 2):
Consider the dynamics of (e1, ν1):
8
><

˙e1 =ν1

˙ν1 = − kp(e1 − e∗

1(z)) − kvν1 − ko

>:

X

j∈{1,2}

gjg⊤
j νj
dj

.

(18)

Using Lemma 1, substitute z∗ ∈ {0, µ} into (18), it yields:

8
><

˙e1 =ν1

8
><

˙e1 =ν1

˙ν1 = − kp(e1 − e∗

1(z∗)) − kvν1 − ko

>:

X

j∈{1,2}

gjg⊤
j νj
dj

(19)
Since ν2 = ν1 − v21, the system (19) is a cascaded system
perturbed by v21. The unforced subsystem is:

˙ν1 = − kp(e1 − e∗

1(z∗)) − kvν1 − ko

>:

X

j∈{1,2}

gjg⊤
j ν1
dj

Consider the following Lyapunov function candidate

L =

kp
2

∥e1 − e∗

1(z∗)∥2 +

1
2

∥ν1∥2

(20)

(21)

Recall (12) and using the fact that kw is large enough, one
has w(z∗
) = 0 and w(z∗
) ≈ 1 and hence one can consider
ia
ib
e∗
i1(z∗
) = ρrρ + ηrη and e∗
i1(z∗
) = ρrρ as constant. In this
ib
ia
setting, one verifies that:
X

˙L = −kv∥ν1∥2 − ko

|g⊤

j ν1|2/dj ≤ 0

(22)

j∈{1,2}

which implies boundedness of (e1, ν1), as long as dj > 0.
Boundedness of ˙ν1, ˙gj, and ˙dj ensures that ¨L is bounded.
Using Barbalet’s lemma, one concludes that the equilib-
rium points (e1, ν1) = (e1(z∗), 0) of the unforced system
(20) are asymptotically stable. If v21 → 0, the same
conclusion holds for the cascaded system (19).
Finally, define ξ = [e⊤
perturbation form

1 ]⊤ and write the singular

1 ν⊤

˙ξ = fe(t, ξ, z, ϵ)
ϵ ˙z = fz(t, z, µ, ϵ)
˙µ = fµ(t, xe, ϵ).

(23)

Using Lemma 1, the stability of (19), and the boundedness
of fe, fz, an argument analogous to (Kokotović et al., 1986,
Section 7.5, Theorem 5.1) implies that, for suﬀiciently
small ϵ > 0, the equilibrium

(e1, ν1) = (e∗

1(z∗), 0)

of the full system (18) is locally asymptotically stable on
each stable opinion branch, provided v21 → 0. This conclu-
sion is applied branch-wise away from the bifurcation point
µ = 0, where the stable equilibrium of the fast opinion
dynamics is locally asymptotically stable.

4. NUMERICAL RESULTS

This section presents a numerical simulation of the pro-
posed framework. To demonstrate the advantages of the
method, we consider an aggressive driving behavior for
Vehicle 2, in which it intermittently changes its intention

Fig. 5. Evolution of opinion state z and the position of ego
vehicle projected on the lateral direction of the road
η⊤p. The red line is the lateral position of the lane
containing Vehicle 1 and 2.

between yielding and not yielding to the ego vehicle. Oper-
ationally, this corresponds to Vehicle 2 switching between
deceleration and acceleration. The gains and parameters
are chosen as kp = 0.7, kv = 2, ko = 1, kw = 40, kµ =
5, k = 20, ϵ = 0.05, r = 0.6, ϵ1 = 0.5. The animation
illustrating the interaction among Vehicle 1, Vehicle 2, and
the ego vehicle under the proposed control framework is
available at: https://bit.ly/3Kh6tHf. Figure 5 shows the
evolution of the opinion state z, together with the ego
vehicle’s lateral position along the road. From the video
and Fig. 5, we observe that when t < 10, the opinion state
transitions continuously from a small positive value toward
zero in response to the aggressive behavior of Vehicle 2,
which alternates between yielding and not yielding to the
ego vehicle. After t = 20, the ego vehicle merges between
Vehicle 1 and 2. Owing to the time-scale separation in
the design, the physical state evolves smoothly and does
not exhibit oscillatory behavior. Finally, the comparison
between Fig. 6 and Fig. 7 demonstrates the effectiveness
of the proposed framework in achieving reactive collision
avoidance in a dynamic environment.

5. CONCLUSIONS

This work presented an adaptive embodied decision-
making framework that integrates nonlinear opinion dy-
namics with the safety-critical control of a nonholo-
nomic autonomous vehicle. By coupling decision states
with physical dynamics, the proposed method enables
an ego vehicle to react to interaction uncertainty and
adjust its behavior accordingly while maintaining formal
safety guarantees. Analytical results established equilib-
rium properties and closed-loop stability, and numerical
simulations demonstrated effective behavior adaptation
and safe motion execution in dynamic non-cooperative
environments. Future work will extend this framework to
multi-vehicle coordination and more complex interaction
scenarios.

0510152025303500.050.10.150.20510152025303500.511.5Conference on Intelligent Robots and Systems (IROS),
4052–4058. IEEE.

Chen, X., Tang, Z., Johansson, K.H., and Mårtensson,
J. (2024). Safe platooning and merging control using
constructive barrier feedback. European Journal of
Control, 80, 101060.

Gray, R., Franci, A., Srivastava, V., and Leonard, N.E.
(2018). Multiagent decision-making dynamics inspired
by honeybees. IEEE Transactions on Control of Net-
work Systems, 5(2), 793–806.

Kokotović, P.V., Khalil, H.K., and O’Reilly, J. (1986).
Singular Perturbation Methods in Control: Analysis and
Design. Academic Press, London.

Lepora, N.F. and Pezzulo, G. (2015). Embodied choice:
how action influences perceptual decision making. PLoS
Computational Biology, 11(4), e1004110.

Perko, L. (2013). Differential Equations and Dynamical

Systems. Springer Science & Business Media.

Qi, S., Tang, Z., Sun, Z., and Haesaert, S. (2025). Inte-
grating opinion dynamics into safety control for decen-
tralized airplane encounter resolution. arXiv preprint
arXiv:2508.00156.

Reverdy, P. and Koditschek, D.E. (2018). A dynamical
system for prioritizing and coordinating motivations.
SIAM Journal on Applied Dynamical Systems, 17(2),
1683–1715.

Reverdy, P.B., Vasilopoulos, V., and Koditschek, D.E.
(2021). Motivation dynamics for autonomous com-
IEEE Transactions on
position of navigation tasks.
Robotics.

Tang, Z., Cunha, R., Hamel, T., and Silvestre, C. (2023).
Reactive collision avoidance for leader-follower forma-
tion control of 2nd-order systems. In IEEE Conference
on Decision and Control (CDC). IEEE.

APPENDIX

Lemma 3. Given the dynamics
˙d
d
with ko a positive gain and α(t) a continuous and bounded
function. Then for any initial condition satisfying d(0) > 0
and ϕ(0) =

˙d(0)
d(0) bounded, the following assertions hold:

¨d = −ko

− α(t)

(.1)

(1) d remains positive, ∀t ≥ 0.
(2) d converges to zero as t → ∞ if and only if

limt→∞

R ⊤
0 α(τ )dτ → +∞.

(3) If d converges to zero, then ˙d is bounded and con-
verges to zero, and ϕ(t) remains bounded, ∀t ≥ 0.
Furthermore, if α(t) converges to a positive constant
α0 > ϵ > 0, then ˙d
and hence ¨d converges to
d
zero.

→ − α0
ko

This lemma shows that as long as the initial distance d(0)
is positive and ϕ(0) is bounded, then d(t) will never cross
zero, and ϕ(t) remains bounded. The proof of this Lemma
can be found in Tang et al. (2023).

Fig. 6. Evolution of the safety distance d1 and d2 under
the proposed framework with collision avoidance term
uc. d1 and d2 remain positive.

Fig. 7. Evolution of the safety distance d1 and d2 under the
proposed framework without collision avoidance term
uc (ko = 0). The safety constraint is violated (d2 is
negative) at some point between t = 5 and t = 10.

DECLARATION OF GENERATIVE AI AND
AI-ASSISTED TECHNOLOGIES IN THE WRITING
PROCESS

During the preparation of this work, the authors used
ChatGPT in order to improve the language of some
paragraphs. After using this tool/service, the authors
reviewed and edited the content as needed and take full
responsibility for the content of the publication.

REFERENCES

Amorim, G., Santos, M., Park, S., Franci, A., and Leonard,
N.E. (2024). Threshold decision-making dynamics adap-
tive to physical constraints and changing environment.
In 2024 European Control Conference (ECC), 1908–
1913. IEEE.

Bizyaeva, A., Franci, A., and Leonard, N.E. (2022). Non-
linear opinion dynamics with tunable sensitivity. IEEE
Transactions on Automatic Control, 68(3), 1415–1430.
Cathcart, C., Santos, M., Park, S., and Leonard, N.E.
(2023).
Proactive opinion-driven robot navigation
around human movers. In 2023 IEEE/RSJ International

0510152025300.511.505101520253000.510510152025300.511.5051015202530-0.500.51