clear all; 
close all; 
clc 

%% ����
trainImages=loadMNISTImages('train-images-idx3-ubyte'); %28*28*60000
testImages=loadMNISTImages('t10k-images-idx3-ubyte');  %28*28*10000
trainLabels=loadMNISTLabels('train-labels-idx1-ubyte');
trainLabels(trainLabels==0)=10;
testLabels=loadMNISTLabels('t10k-labels-idx1-ubyte');
testLabels(testLabels==0)=10;

A=zeros(1,60000)+1;
B=zeros(1,10000)+1;
trainData=[A;trainImages].';
testData=[B;testImages].';
trainLabel=zeros(60000,10);
testLabel=zeros(10000,10);

for i=1:60000
    switch(trainLabels(i))
        case 1
            trainLabel(i,:)=[1 0 0 0 0 0 0 0 0 0];
        case 2
            trainLabel(i,:)=[0 1 0 0 0 0 0 0 0 0];
        case 3
            trainLabel(i,:)=[0 0 1 0 0 0 0 0 0 0];
        case 4
            trainLabel(i,:)=[0 0 0 1 0 0 0 0 0 0];   
        case 5
            trainLabel(i,:)=[0 0 0 0 1 0 0 0 0 0];
        case 6
            trainLabel(i,:)=[0 0 0 0 0 1 0 0 0 0];
        case 7
            trainLabel(i,:)=[0 0 0 0 0 0 1 0 0 0];
        case 8
            trainLabel(i,:)=[0 0 0 0 0 0 0 1 0 0];
        case 9
            trainLabel(i,:)=[0 0 0 0 0 0 0 0 1 0];
        case 10
            trainLabel(i,:)=[0 0 0 0 0 0 0 0 0 1];
    end
end
for i=1:10000
    switch(testLabels(i))
        case 1
            testLabel(i,:)=[1 0 0 0 0 0 0 0 0 0];
        case 2
            testLabel(i,:)=[0 1 0 0 0 0 0 0 0 0];
        case 3
            testLabel(i,:)=[0 0 1 0 0 0 0 0 0 0];
        case 4
            testLabel(i,:)=[0 0 0 1 0 0 0 0 0 0];  
        case 5
            testLabel(i,:)=[0 0 0 0 1 0 0 0 0 0];
        case 6
            testLabel(i,:)=[0 0 0 0 0 1 0 0 0 0];
        case 7
            testLabel(i,:)=[0 0 0 0 0 0 1 0 0 0];
        case 8
            testLabel(i,:)=[0 0 0 0 0 0 0 1 0 0];
        case 9
            testLabel(i,:)=[0 0 0 0 0 0 0 0 1 0];
        case 10
            testLabel(i,:)=[0 0 0 0 0 0 0 0 0 1];
    end
end

%% ���ݷ���
w=0.1*randn(10,785);  %��ֵΪ0����׼��Ϊ0.01����̬�ֲ������������
eta=0.0009;
batchsize=256;
s=zeros(batchsize,10);
epochnum=10;
loss=zeros(1,epochnum);
acc=zeros(2,epochnum);
y=trainLabel;
for epoch=1:epochnum   
    for i=1:floor(size(trainData,1)/batchsize)
        s=trainData((i-1)*batchsize+1:i*batchsize,:)*w';
        y1=exp(s)./sum(exp(s),2);
        grad=(y1-y((i-1)*batchsize+1:i*batchsize,:))'*trainData((i-1)*batchsize+1:i*batchsize,:);
        w=w-eta*grad;
    end
    s2=trainData*w'; 
    y2=exp(s2)./sum(exp(s2),2);
    Lin=y.*log(y2)*-1;
    loss(epoch)=1/60000*sum(Lin(:));
    %ѵ����׼ȷ��
    trainnum=0;
    [~,index1]=max(y2,[],2);
    [~,index2]=max(y,[],2);
    trainnum=length(find(index1==index2));
    acc(1,epoch)=trainnum/60000;
    %���Լ�׼ȷ��
    testnum=0;
    s3=testData*w';  
    y3=exp(s3)./sum(exp(s3),2);
    [~,index1]=max(y3,[],2);
    [~,index2]=max(testLabel,[],2);
    testnum=length(find(index1==index2));
    acc(2,epoch)=testnum/10000;
end
fprintf('���Լ����ྫ��Ϊ%f\n',acc(2,10));
X=1:1:10;
figure(1)
plot(X,loss,'b-','LineWidth',1);
title('loss');
xlabel('epoch');
ylabel('loss');
legend("��ʧ��������");
figure(2)
plot(X,acc(1,:),'b-','LineWidth',1);
legend("ѵ�����ľ�����ȷ��");
title('trainAccuracy');
xlabel('epoch');
ylabel('accuracy');
figure(3)
plot(X,acc(2,:),'b-','LineWidth',1);
title('testAccuracy');
xlabel('epoch');
ylabel('accuracy');
legend("���Լ��ķ��ྫȷ��");


%% 10������ȡ������
randnum=round(rand(1,10)*10000);
testSample=testData(randnum,:);
s=testSample*w';  
yhat=exp(s)./sum(exp(s),2);
[~,index1]=max(yhat,[],2);
[~,index2]=max(testLabel(randnum,:),[],2);
for i=1:10
    fprintf("��%d����д����ʵ��ֵΪ%d,softmaxԤ��ֵΪ%d\n",i,index2(i),index1(i));
end


