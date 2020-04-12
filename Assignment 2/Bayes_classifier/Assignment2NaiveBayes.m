%Assignment2NaiveBayes.m
% Johnny C. Li jcl2222@columbia.edu
% Codes below implements the Naive Bayes classification.
% This is the single round implementation. Perform classification on 9/10
% of the total data set using 1/10 as test set for ONE run.
%

%DATA PROCESSING
%Load X and y data from file
x_data=csvread('X.csv',0,0);
y_data=csvread('y.csv',0,0);

%Randomly shuffle the rows of both data
data_size=size(x_data,1);
new_index=randperm(data_size, data_size).'; %Create permutation
x_data_shuffled = [];
y_data_shuffled = [];
%For loop to shuffle the data based on generated indices.
for i=1:data_size
x_data_shuffled=cat(1,x_data_shuffled,x_data(new_index(i),:));
y_data_shuffled=cat(1,y_data_shuffled,y_data(new_index(i),:));
end

%Below should be placed into the for loop once completed
%-------------------------------------------------------------------------
%PARTITIONING
t=1;
interval = data_size/10;
%Cuts out the test set
x_test_set=x_data_shuffled(1+(t-1)*interval:1+(t-1)*interval+interval-1,:);
y_test_set=y_data_shuffled(1+(t-1)*interval:1+(t-1)*interval+interval-1,:);
%Cut of the training set
if t==1
        x_train_set=x_data_shuffled(1+(t-1)*interval+interval:end,:);
        y_train_set=y_data_shuffled(1+(t-1)*interval+interval:end,:);
    elseif t==10
        x_train_set=x_data_shuffled(1:end-interval,:);
        y_train_set=y_data_shuffled(1:end-interval,:);
    else
        x_train_set=cat(1,x_data_shuffled(1:(t-1)*interval,:),x_data_shuffled(1+(t-1)*interval+interval:end,:));
        y_train_set=cat(1,y_data_shuffled(1:(t-1)*interval,:),y_data_shuffled(1+(t-1)*interval+interval:end,:));
end

%TRAINING
%C0 is mail C1 is spam. pC0 is probability of Class C0 vice versa.
pC0=sum(y_train_set(:) == 0)/size(y_train_set,1);
pC1=sum(y_train_set(:) == 1)/size(y_train_set,1);

%Total occurance of features for the two classes. 
poC0f=0;
poC1f=0;
sumall=0;
for i=1:size(y_train_set,1)
    sumall=sumall+sum(x_train_set(i,:),'all');
    if(y_train_set(i)==0)
        poC0f=poC0f+sum(x_train_set(i,:),'all');
    else
        poC1f=poC1f+sum(x_train_set(i,:),'all');
    end
end

pozero=[];
poone=[];
%Find the pxgC0 and pxgC1
for i=1:size(x_data,2)
    countzero=0;
    countone=0;
    for j=1:size(x_train_set,1)
        if(y_train_set(j)==0)
            countzero=countzero+x_train_set(j,i);
        else
            countone=countone+x_train_set(j,i);
        end
    end
    
    pozero=[pozero (countzero+1)/(poC0f+size(x_train_set,2))];
    poone=[poone (countone+1)/(poC1f+size(x_train_set,2))];
end

%TESTING
predictedy=[];
for i=1:size(x_test_set,1)
    c0prod=[];
    c1prod=[];
    for j=1:size(x_test_set,2)
        c0prod=[c0prod pozero(j)^x_test_set(i,j)];
        c1prod=[c1prod poone(j)^x_test_set(i,j)];
    end
    c0res=prod(c0prod,'all')*pC0;
    c1res=prod(c1prod,'all')*pC1;    
    if(c0res>c1res)
        y=0;
    else
        y=1;
    end
    predictedy = cat(1,predictedy,y);
end

%CHECK FOR MISMATCH
mismatchcount=0;
matchcount=0;
for i=1:size(y_test_set,1)
    if y_test_set(i)~=predictedy(i)
        mismatchcount=mismatchcount+1;
    else
        matchcount=matchcount+1;
    end
end
errorpercent = mismatchcount/size(y_test_set,1)*100;

%-------------------------------------------------------------------------

%Probability of x_i given C0
%poxgC0=
%Probability of x_i given C1
%poxgC1=

% for t=1:10
% %Splits data into train and test set.
% interval = data_size/10;
% x_test_set=x_data_shuffled(1+(t-1)*interval:1+(t-1)*interval+interval-1,:);
% y_test_set=y_data_shuffled(1+(t-1)*interval:1+(t-1)*interval+interval-1,:);
%     if t==1
%             x_train_set=x_data_shuffled(1+(t-1)*interval+interval:end,:);
%             y_train_set=y_data_shuffled(1+(t-1)*interval+interval:end,:);
%         elseif t==10
%             x_train_set=x_data_shuffled(1:end-interval,:);
%             y_train_set=y_data_shuffled(1:end-interval,:);
%         else
%             x_train_set=cat(1,x_data_shuffled(1:(t-1)*interval,:),x_data_shuffled(1+(t-1)*interval+interval:end,:));
%             y_train_set=cat(1,y_data_shuffled(1:(t-1)*interval,:),y_data_shuffled(1+(t-1)*interval+interval:end,:));
%     end
% 
% end