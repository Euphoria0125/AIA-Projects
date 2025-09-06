%% 仿真实验 3
clear all; 
close all; 
clc
udc = [	100.0	110.0	120.0	130.0	140.0	150.0	160.0	170.0	180.0	190.0	200.0];
u0 = [50.913 55.808	60.427	66.083	70.712	75.837	80.905	84.317	91.287	95.304	99.496];
figure(1)
plot(udc,u0,'k+','LineWidth',1);
hold on
f = polyfit(udc,u0,1);
x = 100:2:200;
fy = f(1)*x + f(2);
plot(x,fy,'r','LineWidth',1);
title("u_0=f(m)");
xlabel("u_D_C");
ylabel("u_0");
legend("原始测量�?,"拟合曲线");
