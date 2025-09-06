function w = pocket(w,Data)
for i=1:size(Data,1)  % ����ǩ������ѭ����
    if(i<=30)
        Data(i,5)=1;
    else
        Data(i,5)=-1;
    end
end
A=zeros(size(Data,1),1)+1;
Data=[A,Data];      %�������һ��1
Data=Data(:,1:6);   %������ǩ
h=zeros(size(Data,1),1);  
times=0;
pocket=w;
while 1     %��Ȩ��W
    for i=1:size(Data,1) 
        h(i)=sign(Data(i,1:5)*w');   %���㺯�������ֵ1����-1
        if h(i)==0
            h(i)=-1;
        end
        if h(i)~=Data(i,6)   %���������������ʵֵ�����
            w=w+Data(i,1:5)*Data(i,6);  %��Ȩ�ظ���
            if(Errornum(w,Data(:,1:5),Data(:,6))<Errornum(pocket,Data(:,1:5),Data(:,6)))
                pocket=w;
            end
        end
    end
    times=times+1;
    if h==Data(:,6)   %ѭ��ֱ�������еĺ����������ʵֵ��ͬ
        error=Errornum(pocket,Data(:,1:5),Data(:,6));
        w=pocket;
        break;
    end
    if times>1000
        error=Errornum(pocket,Data(:,1:5),Data(:,6));
        w=pocket;
        break;
    end
end
end