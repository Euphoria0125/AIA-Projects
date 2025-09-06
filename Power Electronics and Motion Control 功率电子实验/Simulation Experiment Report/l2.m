%% 仿真实验 2
clear all; 
close all; 
clc
m = [	0.1	0.2	0.3	0.4	0.5	0.6	0.7	0.8	0.9	1.0];
u0 = [	11.986	20.862	30.406	40.054	51.823	59.982	70.112	78.214	88.948	98.070];
figure(1)
plot(m,u0,'k+','LineWidth',1);
hold on
f = polyfit(m,u0,1);
x = 0.1:0.1:1;
fy = f(1)*x + f(2);
plot(x,fy,'r','LineWidth',1);
title("u_0=f(m)");
xlabel("m");
ylabel("u_0");
legend("原始测量�?,"拟合曲线");
