function [outputArg1,outputArg2] = AE_DXS_ReLU__XZ(X,Label,ADLabels,Abnormal_number, Real_Data)

y=X;

iteration=100;
LearningRate=0.0001;
[m,n]=size(X);


Layer2_hiddensize=2;
Layer3_hiddensize=m;
Layer2_w=rand(Layer2_hiddensize,m);
Layer3_w=rand(Layer3_hiddensize,Layer2_hiddensize);

Layer2_b=rand(Layer2_hiddensize,1);

Layer3_b=rand(Layer3_hiddensize,1);
Layer2_output=rand(Layer2_hiddensize,1);
Layer3_output=rand(Layer3_hiddensize,1);
Layer2_e=rand(Layer2_hiddensize,1);
Layer3_e=rand(Layer3_hiddensize,1);

% restor_Layer2_w=Layer2_w;
% restor_Layer3_w=Layer3_w;
% restor_Layer2_b=Layer2_b;
% restor_Layer3_b=Layer3_b;
tic
for t=1:iteration
    for i=1:n
%         Layer2_output(:,i) = ReLU( Layer2_w * X(:,i) - Layer2_b);
          Layer2_output(:,i) = Leaky_ReLU( Layer2_w * X(:,i) - Layer2_b);
    end
    total_Layer2_output=sum(Layer2_output,2)/n;
    for i=1:n
%         Layer3_output(:,i) = ReLU( Layer3_w * Layer2_output(:,i) - Layer3_b);
          Layer3_output(:,i) = Leaky_ReLU( Layer3_w * Layer2_output(:,i) - Layer3_b);
    end
    total_Layer3_output=sum(Layer3_output,2)/n;

    for i=1:n
        if (Layer3_w * Layer2_output(:,i) - Layer3_b)<0

              temp_Layer3_e(:,i)=(y(:,i)-Layer3_output(:,i))*0.25;
        else
            temp_Layer3_e(:,i)=(y(:,i)-Layer3_output(:,i));
        end          
    end
    Layer3_e=sum(temp_Layer3_e,2)/n;
    for i=1:n
        if Layer2_w * X(:,i) - Layer2_b<0

              temp_Layer2_e(:,i)=Layer3_w' * Layer3_e * 0.25;
        else
            temp_Layer2_e(:,i)=Layer3_w' * Layer3_e;
        end
    end
    Layer2_e=sum(temp_Layer2_e,2)/n;
    
    Layer3_w = Layer3_w + LearningRate  * Layer3_e * total_Layer2_output';
    testX=sum(X,2)/n;
    Layer2_w = Layer2_w + LearningRate * Layer2_e * testX';
    Layer3_b = Layer3_b - LearningRate * Layer3_e;
    Layer2_b = Layer2_b - LearningRate * Layer2_e;
    
    Loss=(y-Layer3_output).*(y-Layer3_output);
    EverySample_Loss=sum(Loss,1)/m;
    TotalLoss(t,:)=sum(EverySample_Loss)/n;
 


    if t==iteration
        L1= Leaky_ReLU( Layer2_w * Real_Data - Layer2_b); 
        L2= Leaky_ReLU( Layer3_w * L1 - Layer3_b);
        error=Real_Data-L2;
        mse=sum(error.*error)';
    end

end
toc
disp(['Runtime: ',num2str(toc)]);

[OF_value,index_number]=sort(mse);
testPlot=Layer2_output';
auc = Measure_AUC(mse, ADLabels);
disp(auc)
[m,n]=size(Real_Data);
ODA_AbnormalObject_Number=index_number(n-Abnormal_number+1:end,:);
ODA_NormalObject_Number=index_number(1:n-Abnormal_number,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Measure  %%%%%%%%%%%%%%%%%%%%%%%%

[Real_NormalObject_Number,Real_Normal]=find(Label==0);
[Real_AbnormalObject_Number,Real_Abnormal]=find(Label==1);


TP=length(intersect(Real_AbnormalObject_Number,ODA_AbnormalObject_Number));
FP=length(Real_AbnormalObject_Number)-TP;
TN=length(intersect(Real_NormalObject_Number,ODA_NormalObject_Number));
FN=length(Real_NormalObject_Number)-TN;


ACC=(TP+TN)/(TP+TN+FP+FN);
fprintf('ACC= %8.5f\n',ACC*100)

DR=TP/(TP+FN);
fprintf('DR= %8.5f\n',DR*100)

P=TP/(TP+FP);
fprintf('P= %8.5f\n',P*100)

FAR=FP/(TN+FP);
fprintf('FAR= %8.5f\n',FAR*100)


 
 end

