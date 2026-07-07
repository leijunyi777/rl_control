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
r=0.2;r_rho=-1;

lh=0;la=1;

z0=0.01;
mu0=0;

d=2;

p30=[1;-2];


v30=[0;1];
n=1;% number of automated car
ne=2; % number of human driven car


tmax=30;
tf = linspace(0, tmax, 300);
s=size(tf);

%% External signal 
for i=1:s(2)
    yd(i:i+1,:)=external_position(tf(i),lh);

    p1(i,:)=yd(i:i+1,1);
    p2(i,:)=yd(i:i+1,2);
end

%% Solving ODE 
[T,sol] = ode45(@(t, y) merge251114_ode(t,y,r,r_rho,lh,la), tf, [p30;v30;z0;mu0]);

for i=1:s(2)
    p(i,:)=sol(i,1:n*d);  
  
    z(i,:)=sol(i,n*2*d+1);

    ps(i,:)=[p1(i,:) p2(i,:) p(i,:)];
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

subplot(2,1,1)
plot(T, z(:,1), 'LineWidth', 1,'Color', 'b');
    hold on
    xlabel('$t[s]$','Interpreter', 'latex'); ylabel('$z$','Interpreter', 'latex');
grid on
%axis([ 0 tmax -.1 1.1]);
set(gca,'FontSize',9)

subplot(2,1,2)

plot(T, p(:,1), 'LineWidth', 1,'Color', 'b');
grid on


figure
th = 0:pi/50:2*pi;



for l=1:s(2)
    clf
for i=1:n+ne
    xunit(i,:)= r /2* cos(th)+ps(l,d*(i-1)+1);
    yunit(i,:) = r /2* sin(th)+ps(l,d*(i-1)+2);

plot(ps(:,d*(i-1)+1),ps(:,d*(i-1)+2),'Color', c(i,:))

    hold on
    a(i)=plot(xunit(i,:), yunit(i,:),'Color', c(i,:));
    grid on
 %axis ([-2.25 4 -1.75 1.75])
 axis equal
end

drawnow
end
