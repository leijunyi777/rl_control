function [y] = external_position(t,lh)
p1=[lh; t+1];
v1=[0;1];

t0=4*pi+3/2*pi;
if t<=t0
p2=[lh;0.4*sin(t)+t];
v2=[0;0.4*cos(t)+1];
else
    p2=[lh;t+0.4*sin(t0)];
    v2=[0;1];
end


y=[p1,p2,v1,v2];
end

