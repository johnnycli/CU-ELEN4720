%Assignment2NaiveBayesFor.m
%
% Johnny C. Li jcl2222@columbia.edu
% Codes below implements the Naive Bayes classification.
% The code takes 4600 data rows and shuffle them randomly. The code then
% divide the data into 10 sets and each run one of the sets (sequentially)
% is kept as test set while the rest of the 9 sets are used as training.
%
% Training is based of y=arg max_y p(Class)Prod_i=1 to n(p(xi|Class)).
% The probabilities p(xi|Ck) is done by 
% ((Count of xi given Ck)+1)/(total cases for Ck|feature size|)
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

%Predefine vectors to store end results from each run.
teny0yp0=[];
teny0yp1=[];
teny1yp0=[];
teny1yp1=[];

%Predefine for stem
stemdata0=[];
stemdata1=[];

for t=1:10
%PARTITIONING
interval = data_size/10;
%Cuts out the test set
x_test_set=x_data_shuffled(1+(t-1)*interval:1+(t-1)*interval+interval-1,:);
y_test_set=y_data_shuffled(1+(t-1)*interval:1+(t-1)*interval+interval-1,:);
%Combine the rest of the 9 groups into one matrix as training set.
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
%Below takes the y training truth to compute the p(C0) and p(C1).
pC0=sum(y_train_set(:) == 0)/size(y_train_set,1);
pC1=sum(y_train_set(:) == 1)/size(y_train_set,1);

%Total occurance of features for the two classes. 
%poC0f is used to calculate the total feature frequency of all xi given
%classes C0 and C1.
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

%Find the pxgC0 and pxgC1
%This is to find the total count of xi given classes C0 and C1.
pozero=[];
poone=[];
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
    %Format the results into an array.
    pozero=[pozero (countzero+1)/(poC0f+size(x_train_set,2))];
    poone=[poone (countone+1)/(poC1f+size(x_train_set,2))];
end

    stemdata0=cat(1,stemdata0,pozero);
    stemdata1=cat(1,stemdata1,poone);

%TESTING
%Use the reserved test data. Compared the predicted y and true y.
predictedy=[];
for i=1:size(x_test_set,1)
    c0prod=[];
    c1prod=[];
    for j=1:size(x_test_set,2)
        %Here the probability is taken to be (trained prob)^frequency
        c0prod=[c0prod pozero(j)^x_test_set(i,j)];
        c1prod=[c1prod poone(j)^x_test_set(i,j)];
    end
    %Final evaluation of the y=arg max_y condition mentioned above.
    c0res=prod(c0prod,'all')*pC0;
    c1res=prod(c1prod,'all')*pC1;
    %Comparing the two results and assign class prediction.
    if(c0res>c1res)
        y=0;
    else
        y=1;
    end
    predictedy = cat(1,predictedy,y);
end

%CHECK FOR MISMATCH
%Define all four possible cases of y=0,1 and y'=0,1 combination.
y0yp0=0;
y0yp1=0;
y1yp0=0;
y1yp1=0;
for i=1:size(y_test_set,1)
    if y_test_set(i)==0 && predictedy(i)==0
        y0yp0=y0yp0+1;
    elseif y_test_set(i)==0 && predictedy(i)==1
        y0yp1=y0yp1+1;
    elseif y_test_set(i)==1 && predictedy(i)==0
        y1yp0=y1yp0+1;
    elseif y_test_set(i)==1 && predictedy(i)==1
        y1yp1=y1yp1+1;
    end
end
%Format result into array for checking purposes.
teny0yp0=[teny0yp0 y0yp0];
teny0yp1=[teny0yp1 y0yp1];
teny1yp0=[teny1yp0 y1yp0];
teny1yp1=[teny1yp1 y1yp1];

end

%Format the final result into a table and compute the accuracy.
table=[sum(teny0yp0,'all'),sum(teny0yp1,'all');sum(teny1yp0,'all'),sum(teny1yp1,'all')];
sumcheck=sum(teny0yp0,'all')+sum(teny0yp1,'all')+sum(teny1yp0,'all')+sum(teny1yp1,'all');
accuracy =  trace(table)/4600;
%Plot stem
figure
plotmatrix=[transpose(1:1:size(x_data,2)) transpose(mean(stemdata0,1)) transpose(mean(stemdata1,1))];
hold on
title('Stem Plot of Features given Class')
xlabel('Feature number') 
ylabel('Probability of feature given class') 
stem(plotmatrix(:,1),plotmatrix(:,2));
stem(plotmatrix(:,1),plotmatrix(:,3));
legend({'Class 0 - Ham','Class 1 - Spam'},'Location','northeast')
hold off