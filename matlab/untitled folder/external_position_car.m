function [y] = external_position_car(t,lh)
lr=t+1;
p1=[lh; lr];
v1=[0;1];

t0=4*pi+3/2*pi;
if t<=t0
p2=[lh;lr+0.5*sin(t)-1.5];
v2=[0;0.5*cos(t)+1];
else
    p2=[lh;lr-2];
    v2=[0;1];
end


y=[p1,p2,v1,v2];
end
