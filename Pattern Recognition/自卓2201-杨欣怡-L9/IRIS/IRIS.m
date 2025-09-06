clear all; 
close all; 
clc 

%% IRIS���ݼ�����
iris = xlsread('iris.csv');
x1=iris(:,2);
x2=iris(:,3);
x3=iris(:,4);
x4=iris(:,5)；
Data=[x1 x2 x3 x4];
[m,n]=size(Data);
for i=1:m/3
    Data(i,5:7)=[1 0 0];
end
for i=m/3+1:2*m/3
    Data(i,5:7)=[0 1 0];
end
for i=2*m/3+1:m
    Data(i,5:7)=[0 0 1];
end

%% ovo
w0=[1;1;1];
w=zeros(3,4);
w=[w0,w];
k=1;
num=randperm(50,30);
for i=1:2
    for j=i+1:3
        trainData=[Data((i-1)*50+num,:);Data((j-1)*50+num,:)];
        w(k,:)=pocket(w(k,:),trainData);
        k=k+1;
    end
end

trainData=[Data(num,:);Data(50+num,:);Data(100+num,:)];
testData=Data;
testData(ismember(testData,trainData,'rows')==1,:)=[];  %�Ӿ�����ɾ���Ӿ���
trainA=zeros(size(trainData,1),1)+1;
trainB=zeros(size(trainData,1),3);
trainData=[trainA,trainData,trainB];
testA=zeros(size(testData,1),1)+1;
testB=zeros(size(testData,1),3);
testData=[testA,testData,testB];

for i=1:size(trainData,1)
    for j=1:3
        if(trainData(i,1:5)*w(j,:)'>0)
            switch(j)
                case 1
                    trainData(i,9)=trainData(i,9)+1;   %��1��
                case 2
                    trainData(i,9)=trainData(i,9)+1;   %��1��
                case 3
                    trainData(i,10)=trainData(i,10)+1;  %��2��
            end
        else
            switch(j)
                case 1
                    trainData(i,10)=trainData(i,10)+1;   %��2��
                case 2
                    trainData(i,11)=trainData(i,11)+1;   %��3��
                case 3
                    trainData(i,11)=trainData(i,11)+1;  %��3��
            end
        end
    end
end
correct=0;
for i=1:size(trainData,1)
    [m,l]=max(trainData(i,9:11),[],2);
    switch(l)
        case 1
            if trainData(i,6:8)==[1 0 0]
                correct=correct+1;
            end
        case 2
            if trainData(i,6:8)==[0 1 0]
                correct=correct+1;
            end
        case 3
            if trainData(i,6:8)==[0 0 1]
                correct=correct+1;
            end
    end
end
fprintf("ovoѵ����������ȷ��Ϊ%f\n",correct/size(trainData,1));
for i=1:size(testData,1)
    for j=1:3
        if(testData(i,1:5)*w(j,:)'>0)
            switch(j)
                case 1
                    testData(i,9)=testData(i,9)+1;   %��1��
                case 2
                    testData(i,9)=testData(i,9)+1;   %��1��
                case 3
                    testData(i,10)=testData(i,10)+1;  %��2��
            end
        else
            switch(j)
                case 1
                    testData(i,10)=testData(i,10)+1;   %��2��
                case 2
                    testData(i,11)=testData(i,11)+1;   %��3��
                case 3
                    testData(i,11)=testData(i,11)+1;  %��3��
            end
        end
    end
end
correct=0;
for i=1:size(testData,1)
    [m,l]=max(testData(i,9:11),[],2);
    switch(l)
        case 1
            if testData(i,6:8)==[1 0 0]
                correct=correct+1;
            end
        case 2
            if testData(i,6:8)==[0 1 0]
                correct=correct+1;
            end
        case 3
            if testData(i,6:8)==[0 0 1]
                correct=correct+1;
            end
    end
end
fprintf("ovo���Լ�������ȷ��Ϊ%f\n",correct/size(testData,1));

%% Softmax
eta=1e-5;  
w=zeros(3,5);
num=randperm(50,30);
trainData=[Data(num,:);Data(50+num,:);Data(100+num,:)];  
trainData=trainData(randperm(size(trainData,1)),:);    %�����ݴ���������
A=zeros(size(trainData,1),1)+1;  
trainData=[A,trainData];
for i=1:size(trainData,1)
        if(trainData(i,6:8)==[1 0 0])
            trainData(i,6)=1;
        elseif(trainData(i,6:8)==[0 1 0])
            trainData(i,6)=2;
        else
            trainData(i,6)=3;
        end
end
s=zeros(1,3);
for iterations=1:10000   
    for i=1:size(trainData,1)
        s=trainData(i,1:5)*w';
        s=exp(s)./sum(exp(s));
        if(trainData(i,6)==1)
        w(1,:)=w(1,:)-eta*(s(1)-1)*trainData(i,1:5);
        w(2,:)=w(2,:)-eta*s(2)*trainData(i,1:5);
        w(3,:)=w(3,:)-eta*s(3)*trainData(i,1:5);
        elseif(trainData(i,6)==2)
        w(1,:)=w(1,:)-eta*s(1)*trainData(i,1:5);
        w(2,:)=w(2,:)-eta*(s(2)-1)*trainData(i,1:5);
        w(3,:)=w(3,:)-eta*s(3)*trainData(i,1:5);
        else
        w(1,:)=w(1,:)-eta*s(1)*trainData(i,1:5);
        w(2,:)=w(2,:)-eta*s(2)*trainData(i,1:5);
        w(3,:)=w(3,:)-eta*(s(3)-1)*trainData(i,1:5);
        end
    end
end
correct=0;
for i=1:size(trainData,1)
    s=trainData(i,1:5)*w';
    s=exp(s)./sum(exp(s));
    [m,n]=max(s);  %ȡ�����ֵ���±�
    switch(n)
        case 1
            if(trainData(i,6)==1)
                correct=correct+1;
            end
        case 2
            if(trainData(i,6)==2)
            correct=correct+1;
            end
        case 3
            if(trainData(i,6)==3)
            correct=correct+1; 
            end
    end
end
fprintf("Softmaxѵ����������ȷ��Ϊ%f\n",correct/size(trainData,1));

trainData=[Data(num,:);Data(50+num,:);Data(100+num,:)];
testData=Data;
testData(ismember(testData,trainData,'rows')==1,:)=[];  %�Ӿ�����ɾ���Ӿ���
A=zeros(size(testData,1),1)+1;
testData=[A,testData];
correct=0;
for i=1:size(testData,1)
    s=testData(i,1:5)*w';
    s=exp(s)/sum(exp(s));
    [m,n]=max(s);  %ȡ�����ֵ���±�
    switch(n)
        case 1
            if(testData(i,6:8)==[1 0 0])
                correct=correct+1;
            end
        case 2
            if(testData(i,6:8)==[0 1 0])
            correct=correct+1;
            end
        case 3
            if(testData(i,6:8)==[0 0 1])
            correct=correct+1; 
            end
    end
end
fprintf("Softmax���Լ�������ȷ��Ϊ%f\n",correct/size(testData,1));




















