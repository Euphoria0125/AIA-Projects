function num = Errornum(w,Data,Lable)
num=0;
data_num=size(Data,1);
for i=1:data_num
    if(sign(w*Data(i,:).') ~= Lable(i,:))
        num=num+1；
    end
end