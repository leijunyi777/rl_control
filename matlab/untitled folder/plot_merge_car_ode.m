close all
clear all 
COLOURS=get(gca,'colororder');
blue=COLOURS(1,:);
red=COLOURS(2,:);
yellow=COLOURS(3,:);
purple=COLOURS(4,:);
green=COLOURS(5,:);
light_blue=COLOURS(6,:);
dark_red=COLOURS(7,:);

c(1,:)=red;
c(2,:)=yellow;
c(3,:)=light_blue;
c(4,:)=[0 0 1];
c(5,:)=[1 0 0];
%% parameters
r=0.6;r_rho=-1;

lh=0;la=1.3;

L=0.4; %Lenth of the car
w=L/2; % width of the car

d=2;
n=1;% number of automated car
ne=2; % number of human driven car
%% Initial condition
pr0=[la;-0.9];
theta0=pi/2;
p30=pr0+L*[cos(theta0);sin(theta0)];

vr0=1;
delta0=0;
v30=vr0*[cos(theta0)-sin(theta0)*tan(delta0);sin(theta0)+cos(theta0)*tan(delta0)];
%v30=[0;1];

xr0=[pr0;theta0;vr0;delta0];
z0=0.01;
mu0=0;

%% time step
tmax=35;
tf = linspace(0, tmax, 300);
s=size(tf);

%% External signal 
for i=1:s(2)
    yd(i:i+1,:)=external_position_car(tf(i),lh);

    p1(i,:)=yd(i:i+1,1);
    p2(i,:)=yd(i:i+1,2);
end

%% Solving ODE 
[T,sol] = ode45(@(t, y) merge_car_ode(t,y,r,r_rho,lh,la,L), tf, [p30;v30;z0;mu0;xr0]);

for i=1:s(2)
    p(i,:)=sol(i,1:n*d);  
  
    z(i,:)=sol(i,n*2*d+1);
mu(i,:)=sol(i,n*2*d+1);

    pr(i,:)=sol(i,n*2*d+3:n*2*d+4);
    theta(i,:)=wrapToPi(sol(i,n*2*d+5));

    ps(i,:)=[p1(i,:) p2(i,:) p(i,:)];

    p_center(i,:)=pr(i,:)+L/2*[cos(theta(i)) sin(theta(i))];

    d31(i,:)=norm(p(i,:)-p1(i,:))-r;
    d32(i,:)=norm(p(i,:)-p2(i,:))-r;
end 




%% Plot


% for i=1:s(2)
%     clf
%     
%     plot(T, z(:,1), 'LineWidth', 1,'Color', 'b');
%     hold on
%     plot(tf(i), z(i,1),'r*');
%     hold on
%     axis equal
%     xlabel('$t[s]$','Interpreter', 'latex'); ylabel('$z$','Interpreter', 'latex');
% grid on
% %axis([ 0 tmax -.1 1.1]);
% set(gca,'FontSize',9)
% 
%     drawnow
% end

subplot(3,1,1)
plot(T, z(:,1), 'LineWidth', 1,'Color', 'b');
    hold on
    xlabel('$t[s]$','Interpreter', 'latex'); ylabel('$z$','Interpreter', 'latex');
grid on
%axis([ 0 tmax -.1 1.1]);
set(gca,'FontSize',14)

subplot(3,1,2)
plot(T, mu(:,1), 'LineWidth', 1,'Color', 'b');
    hold on
    xlabel('$t[s]$','Interpreter', 'latex'); ylabel('$\mu$','Interpreter', 'latex');
grid on
%axis([ 0 tmax -.1 1.1]);
set(gca,'FontSize',14)

subplot(3,1,3)
lane=zeros(s(2),1);
plot(T, p(:,1), 'LineWidth', 1,'Color', 'b');
hold on
plot(T, lane(:,1), 'LineWidth', 1, 'Color','r');
hold on
xlabel('$t[s]$','Interpreter', 'latex'); ylabel('$\eta^\top p$','Interpreter', 'latex');
grid on
axis([ 0 tmax -.1 1.5]);
set(gca,'FontSize',14)

% subplot(2,1,1)
% plot(T, d31(:,1), 'LineWidth', 1,'Color', 'k');
%     hold on
%     xlabel('$t[s]$','Interpreter', 'latex'); ylabel('$d_1$','Interpreter', 'latex');
% grid on
% %axis([ 0 tmax -.1 1.1]);
% set(gca,'FontSize',14)
% 
% subplot(2,1,2)
% 
% %plot(T, p(:,1), 'LineWidth', 1,'Color', 'b');
% plot(T, d32(:,1), 'LineWidth', 1,'Color', 'k');
% hold on
% xlabel('$t[s]$','Interpreter', 'latex'); ylabel('$d_2$','Interpreter', 'latex');
% grid on
% set(gca,'FontSize',14)

%% Animation
% Step 2: Rectangle corners in local coordinates
corners_local = [ L/2,  w/2;
                  L/2, -w/2;
                 -L/2, -w/2;
                 -L/2,  w/2 ]';



figure


for l=1:s(2)
    clf


    % Step 3: Rotation matrix
    R = [cos(theta(l)) -sin(theta(l));
     sin(theta(l))  cos(theta(l))];
% Rotate and translate corners
corners_world = R * corners_local + p_center(l,:)';
% Close the rectangle
corners_world = [corners_world corners_world(:,1)];
plot(corners_world(1,:), corners_world(2,:), 'r-', 'LineWidth', 2);
hold on

for i=1:ne
plot(ps(:,d*(i-1)+1),ps(:,d*(i-1)+2),'Color', 'k')
plot(ps(:,d*(i-1)+1)+la,ps(:,d*(i-1)+2),'Color', 'k')

    hold on
    rectangle('Position', [ps(l,d*(i-1)+1)-w/2 ps(l,d*(i-1)+2)-L w L], 'EdgeColor', 'b', 'LineWidth', 2);
    %grid on
 %axis ([-5 5 0 31])
 
  axis equal
 % xlim([-4 4]);
end

drawnow
end
