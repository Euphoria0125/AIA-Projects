%% 仿真实验 1
clear all; 
close all; 
clc
fr = [5.0	10.0	15.0	20.0	25.0	30.0	35.0	40.0	45.0	50.0	55.0	60.0];
f0 = [5.00	10.10	14.97	19.69	25.00	29.25	34.48	40.84	45.21	49.54	56.07	59.82];
figure(1)
plot(fr,f0,'k+','LineWidth',1);
hold on
f = polyfit(fr,f0,1);
x = 0:2:60;
fy = f(1)*x + f(2);
plot(x,fy,'r','LineWidth',1);
title("f_0=f(f_r)");
xlabel("f_r");
ylabel("f_0");
legend("原始测量�?,"拟合曲线");
