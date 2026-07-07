function ydot = merge251114_ode(t,y,r,r_rho,lh,la)
p3=y(1:2);
v3=y(3:4);
z=y(5);
mu=y(6);



%% external state
ye=external_position(t,lh);

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



k=20;
epsilon=0.03;
epsilon2=0.5;

 
mudot=-5*mu+tanh(-k*rho'*g31*rho'*g32*(d21-2*r)*(phi21+epsilon2));
tanh(-k*rho'*g31*rho'*g32*(d21-2*r)*(phi21+epsilon2))
%mu=tanh(-k*rho'*g31*rho'*g32*(d21-2*r)*(phi21+epsilon2));
%mu=tanh(-k*rho'*g31*rho'*g32*(d21-2*r));
zdot=1/epsilon*(-z^2+mu*z);


%% low level control for agent 3
w=tanh(40*z);
e31d=rho*r_rho+eta*(w*lh+(1-w)*la);


u3=-(e31-e31d)-v31-g31*phi31-g32*phi32;
%% output 
ydot = [v3;u3;zdot;mudot];