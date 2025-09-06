clear all; 
close all; 
clc 

%% �������ݼ�
n=200;
N=(n+n)*0.8;
m1 = [-5;0]; % ��ֵ����
%m1 = [1;0];
m2 = [0;5];
%m2 = [0;1];
s = [1 0; 0 1]; % Э�������
Data_X1 = mvnrnd(m1,s, n);  % ����n����Ԫ��̬�ֲ������
Data_X2 = mvnrnd(m2,s, n);
x0=ones(1,n);   %������������
X1=[x0;Data_X1.'];    
Y1=ones(1,n);     %"+1"��������ı��
X2=[x0;Data_X2.'];     
Y2=-ones(1,n);     %"-1"��������ı��

trainData=[X1(1:3,1:0.8*n),X2(1:3,1:0.8*n)];      %��������ѡȡ80%ѵ������
trainLabel=[Y1(1,1:0.8*n),Y2(1,1:0.8*n)];         %�����������ѡȡѵ���������
testData=[X1(1:3,0.8*n+1:n),X2(1:3,0.8*n+1:n)];       %��������ѡȡ20%��������
testLabel=[Y1(1,0.8*n+1:n),Y2(1,0.8*n+1:n)];          %�����������ѡȡ�����������

xlswrite('X1.xlsx',Data_X1);
xlswrite('X2.xlsx',Data_X2);

%% Primal-SVM
z=zeros(1,2);
Q=[0 z;z.' eye(2)];
p=zeros(1,2+1);
A1=zeros(3,(n+n)*0.8);   %(n+n)*0.8Ϊѵ����������
for i=1:3
    A1(i,:)=trainData(i,:).*trainLabel;
end
A=-1*A1';     % ���ι滮����Լ��������<=
c=(zeros(1,2*n*0.8)-1).'; 
u=quadprog(Q,p,A,c);
figure(1);
plot(X1(2,:),X1(3,:),'rx',X2(2,:),X2(3,:),'b*');  
X=-5:0.2:5;
Y=-1*(u(2,:)*X+u(1,:))/u(3,:);   %��������
hold on;
plot(X,Y,'m-','LineWidth',2);
title('Primal-SVM');
legend('��+1����','��-1����','������');

%������ȷ��
num=0;
train_num=size(trainData,2);
for i=1:train_num
    if(sign(u'*trainData(:,i)) ~= trainLabel(:,i))
        num=num+1;
    end
end
fprintf("Primal-SVMѵ������ȷ��Ϊ%f\n",1-num/train_num);
num=0;
test_num=size(testData,2);
for i=1:test_num
    if(sign(u'*testData(:,i)) ~= testLabel(:,i))
        num=num+1;
    end
end
fprintf("Primal-SVM���Լ���ȷ��Ϊ%f\n",1-num/test_num);

%% Dual-SVM
trainData_D=trainData(2:3,:);      
trainLabel_D=trainLabel(1,:);       
testData_D=testData(2:3,:);      
testLabel_D=testLabel(1,:);    

Q=(trainData_D.'*trainData_D).*(trainLabel_D.'*trainLabel_D);
p=zeros(1,N)-1;
a1=eye(N);
a2=trainLabel;
a3=trainLabel*-1;
A=[a1;a2;a3];
A=-1*A;  
c=zeros(N+2,1);
u=quadprog(Q,p,A,c);
w=trainData_D*(trainLabel_D.'.*u);
index=find(u > 1e-4);
b=trainLabel_D(1,index(1))-w'*trainData_D(:,index(1)); 
supporting_vector=trainData_D(:,index);
figure(2);
plot(X1(2,:),X1(3,:),'rx',X2(2,:),X2(3,:),'b*');  
hold on;
X=-5:0.2:5;
Y=-1*(w(1,:)*X+b)/w(2,:);
plot(X,Y,'m-','LineWidth',2);
hold on;
plot(supporting_vector(1,:),supporting_vector(2,:),'go','LineWidth', 2,'markersize', 10);
title('Dual-SVM');
legend("��+1����","��-1����","������","֧������");

%������ȷ��
w=[b;w];
num=0;
train_num=size(trainData,2);
for i=1:train_num
    if(sign(w'*trainData(:,i)) ~= trainLabel(:,i))
        num=num+1;
    end
end
fprintf("Dual-SVMѵ������ȷ��Ϊ%f\n",1-num/train_num);
num=0;
test_num=size(testData,2);
for i=1:test_num
    if(sign(w'*testData(:,i)) ~= testLabel(:,i))
        num=num+1;
    end
end
fprintf("Dual-SVM���Լ���ȷ��Ϊ%f\n",1-num/test_num);

%% Kernel-SVM���Ĵζ���ʽ�˺�����
trainData_K=trainData(2:3,:);      
trainLabel_K=trainLabel(1,:);       
testData_K=testData(2:3,:);      
testLabel_K=testLabel(1,:);  

N=size(trainData_K,2);
Q=zeros(N,N);
for i=1:N
    for j=1:N
        Q(i,j)=trainLabel_K(1,i)*trainLabel_K(1,j)*(0.5+0.5*trainData_K(:,i)'*trainData_K(:,j)).^4; 
    end
end
p=zeros(1,N)-1;
a1=eye(N);
a2=trainLabel_K;
a3=trainLabel_K*-1;
A=[a1;a2;a3];
A=-1*A; 
c=zeros(N+2,1);
u=quadprog(Q,p,A,c);
index=find(u > 1e-4);
svLabel=trainLabel_K(1,index)';  
svData=trainData_K(:,index);     
xx = linspace(min(trainData_K(1,:)),max(trainData_K(1,:)),50);   
yy = linspace(min(trainData_K(2,:)),max(trainData_K(2,:)),50);   
[X,Y] = meshgrid(xx,yy);   %��������
Z=zeros(50,50);
alpha=u(index);   %֧��������ϵ��
k=sum((svData-svData(:,1)).^2,1);
b=svLabel(1)-(alpha.*svLabel)'*(0.5+0.5*svData'*svData(:,1)).^4; 
for i=1:50
    for j=1:50
        Z(i,j)=(alpha.*svLabel)'*(0.5+0.5*svData'*[X(i,j);Y(i,j)]).^4+b;
    end
end
figure(3);
plot(X1(2,:),X1(3,:),'rx',X2(2,:),X2(3,:),'b*');  
hold on;
plot(svData(1,:),svData(2,:),'go','LineWidth', 2,'markersize', 10);   % ֧������
hold on;
contour(X,Y,Z,'LineWidth', 1);
title('Kernel-SVM���Ĵζ���ʽ�˺�����');
legend("��+1����","��-1����","֧������","������");

%������ȷ��
num=0;
train_num=size(trainData_K,2);
for i=1:train_num
    if(sign((alpha.*svLabel)'*(0.5+0.5*svData'*trainData_K(:,i)).^4+b) ~= trainLabel_K(:,i))
        num=num+1;
    end
end
fprintf("Kernel-SVM���Ĵζ���ʽ�˺���)ѵ������ȷ��Ϊ%f\n",1-num/train_num);
num=0;
test_num=size(testData_K,2);
for i=1:test_num
    if(sign((alpha.*svLabel).'*(0.5+0.5*svData'*testData_K(:,i)).^4+b) ~= testLabel_K(:,i))
        num=num+1;
    end
end
fprintf("Kernel-SVM���Ĵζ���ʽ�˺���)���Լ���ȷ��Ϊ%f\n",1-num/test_num);

%% Kernel-SVM����˹�˺�����
N=size(trainData_K,2);
Q=zeros(N,N);
for i=1:N
    for j=1:N
        Q(i,j)=trainLabel_K(1,i)*trainLabel_K(1,j)*exp(-sum((trainData_K(:,i)-trainData_K(:,j)).^2,1)).';
    end
end
p=zeros(1,N)-1;
a1=eye(N);
a2=trainLabel_K;
a3=trainLabel_K*-1;
A=[a1;a2;a3];
A=-1*A; 
c=zeros(N+2,1);
u=quadprog(Q,p,A,c);
index=find(u > 1e-4);
svLabel=trainLabel_K(1,index)';  
svData=trainData_K(:,index);     
xx = linspace(min(trainData_K(1,:)),max(trainData_K(1,:)),50);   
yy = linspace(min(trainData_K(2,:)),max(trainData_K(2,:)),50);   
[X,Y] = meshgrid(xx,yy);   %��������
Z=zeros(50,50);
alpha=u(index);   %֧��������ϵ��
k=sum((svData-svData(:,1)).^2,1);
b=svLabel(1)-(alpha.*svLabel)'*exp(-sum((svData-svData(:,1)).^2,1)).';
for i=1:50
    for j=1:50
        Z(i,j)=(alpha.*svLabel).'*exp(-sum((svData-[X(i,j);Y(i,j)]).^2,1)).'+b;
    end
end
figure(4);
plot(X1(2,:),X1(3,:),'rx',X2(2,:),X2(3,:),'b*');  
hold on;
plot(svData(1,:),svData(2,:),'go','LineWidth', 2,'markersize', 10);   % ֧������
hold on;
contour(X,Y,Z,'LineWidth', 1);
title('Kernel-SVM����˹�˺�����');
legend("+1����","��-1����","֧������","������");

%������ȷ��
num=0;
train_num=size(trainData_K,2);
for i=1:train_num
    if(sign((alpha.*svLabel)'*(0.5+0.5*svData'*trainData_K(:,i)).^4+b) ~= trainLabel_K(:,i))
        num=num+1;
    end
end
fprintf("Kernel-SVM����˹�˺�����ѵ������ȷ��Ϊ%f\n",1-num/train_num);
num=0;
test_num=size(testData_K,2);
for i=1:test_num
    if(sign((alpha.*svLabel)'*(0.5+0.5*svData'*testData_K(:,i)).^4+b) ~= testLabel_K(:,i))
        num=num+1;
    end
end
fprintf("Kernel-SVM����˹�˺��������Լ���ȷ��Ϊ%f\n",1-num/test_num);



