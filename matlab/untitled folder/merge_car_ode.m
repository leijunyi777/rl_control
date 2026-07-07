function ydot = merge_car_ode(t,y,r,r_rho,lh,la,L)
p3=y(1:2);
v3=y(3:4);
z=y(5);
mu=y(6);
theta=wrapToPi(y(9));
vr=y(10);
delta=y(11);

%% external state
ye=external_position_car(t,lh);

p1=ye(:,1);
p2=ye(:,2);
v1=ye(:,3);
v2=ye(:,4);

%% variables
rho=[0; 1];eta=[1; 0];

e21=p2-p1; 
e31=p3-p1; 
e32=p3-p2;
v31=v3-v1;
v32=v3-v2;
v21=v2-v1;


d21=norm(e21)-r;
d31=norm(e31)-r;
d32=norm(e32)-r;

g31=e31/norm(e31);
g32=e32/norm(e32);
g21=e21/norm(e21);

dotd21=g21'*v21;

phi31=g31'*v31/d31;
phi32=g32'*v32/d32;
phi21=g21'*v21/d21;
% phi31=v31/d31;
% phi32=v32/d32;
%% opinion dynamics


k_mu=5;
k=20;
kw=40;
epsilon=0.05;
epsilon2=0.5;

 
mudot=-k_mu*mu+tanh(-k*rho'*g31*rho'*g32*(d21-2*r)*(phi21+epsilon2));
tanh(-k*rho'*g31*rho'*g32*(d21-2*r)*(phi21+epsilon2))
%mu=tanh(-k*rho'*g31*rho'*g32*(d21-2*r)*(phi21+epsilon2));
%mu=tanh(-k*rho'*g31*rho'*g32*(d21-2*r));
zdot=1/epsilon*(-z^2+mu*z);


%% low level control for agent 3
w=tanh(kw*z);
e31d=rho*r_rho+eta*(w*lh+(1-w)*la);

kp=0.7;kv=2;ko=1;
u3=-kp*(e31-e31d)-kv*v31-ko*g31*phi31-ko*g32*phi32;

%u3=-0.2*rho*rho'*(e31-e31d+v31)-1.5*eta*eta'*(e31-e31d+v31)-g31*phi31-g32*phi32;

%% Transfer to rear position
A=[cos(theta)-sin(theta)*tan(delta) -vr*sin(theta)*sec(delta)^2;sin(theta)+cos(theta)*tan(delta) vr*cos(theta)*sec(delta)^2];
B=-vr^2/L*[sin(theta)*tan(delta)+cos(theta)*tan(delta)^2;-cos(theta)*tan(delta)+sin(theta)*tan(delta)^2];
ur=A^(-1)*(u3-B);

C=[vr*cos(theta);vr*sin(theta);vr*tan(theta)/L;0;0];
D=[0 0;0 0;0 0;1 0;0 1];
dotxr=C+D*ur;

eps_delta = 1e-3;
if abs(delta) < eps_delta
    dotxr(3) = vr/L * delta;   % 用 tan(delta) ≈ delta
else
    dotxr(3) = vr/L * tan(delta);
end

%% output 
ydot = [v3;u3;zdot;mudot;dotxr];