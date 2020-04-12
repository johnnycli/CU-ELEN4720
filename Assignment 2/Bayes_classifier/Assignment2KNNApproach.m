%Assignment2KNNApproach.m
%
% Johnny C. Li jcl2222@columbia.edu
% Codes below implements the k-NN classification.
% The code takes 4600 data rows and shuffle them randomly. The code then
% divide the data into 10 sets and each run one of the sets (sequentially)
% is kept as test set while the rest of the 9 sets are used as training.
%
% There isn't much on training. The test data point is compared against the
% train data set to find the nearest k neighbors and perform a soft vote on
% the final classification.
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

accforten=[];
for t=1:10
    accperrun=[];
    
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
    %Compute all distances in L1.
    
    %Predefine matrix for k=1~20 and 460 tests
    yresmatrix=[];
    
    %Start the 460 loop
    for i=1:size(x_test_set,1)
        %Define array to store distance
        x_dist=[];
        %Start the 4140 loop to calculate distance between 4140 train
        %points and the single test point.
        for j=1:size(x_train_set,1)
            x_dist=cat(1,x_dist,sum(abs(x_test_set(i,:)-x_train_set(j,:))));
        end
        %Sort the distance from min to max. Including the index.
        [sorted_x, index] = sort(x_dist(:,1),'ascend');
        
        %Start the 20 loop to go through 20 cases of nearest neighbors.
        yrestot=[];
        for k=1:20
            %Int for soft vote count
            zvote=0;
            ovote=0;
            yres=[];
            for n=1:k
                if(y_train_set(index(n),1)==0)
                    zvote=zvote+1;
                else
                    ovote=ovote+1;
                end
            end
            %Assign class based on vote.
            if(zvote>ovote)
                yres=cat(1,yres,0);
            else
                yres=cat(1,yres,1);
            end
            %Concat k cases by column.
            yrestot=cat(2,yrestot,yres);
        end
        %Concat 460 test cases by row.
        yresmatrix=cat(1,yresmatrix,yrestot);
    end
    
    %Calculate accuracy
    for k=1:20
        corrcount=0;
        for r=1:size(y_test_set,1)
            if(yresmatrix(r,k)==y_test_set(r))
                corrcount=corrcount+1;
            end
        end
        accperrun=cat(2,accperrun,corrcount/size(y_test_set,1));
    end
    accforten=cat(1,accforten,accperrun);
end

%Plot results
x=(1:1:20).';
figure
hold on
title('k-nn Accuracy')
xlabel('k') 
ylabel('Accuracy') 
plot(x,mean(accforten,1).')
hold off
