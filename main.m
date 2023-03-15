function [outputArg1,outputArg2] = main(inputArg1,inputArg2)

tic
train_x=load('Normalization_breastw.txt');
iteration=100;
MyBasicGAN(train_x,iteration);


temp_X=load('fake_data.txt');
RandFromFakedata=100;
[m,n]=size(temp_X);
randX=temp_X(randperm(m),:);
X=randX(1:RandFromFakedata,:);
Label=load('Label_breastw.txt');
ADLabels=load('Label_breastw.txt');
Abnormal_number=239;
Real_Data=load('Normalization_breastw.txt')';
AE_DXS_Leaky_ReLU__XZ(X',Label,ADLabels,Abnormal_number, Real_Data);
toc
end

