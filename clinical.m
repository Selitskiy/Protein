
%% Clear everything 
clearvars -global;
clear all; close all; clc;

addpath('~/ANNLib/');
addpath('~/Protein/');


%% Load data
dataDir = '~/Downloads/';
dataFile = 'clinical.csv';

dataFullFile = strcat(dataDir,dataFile);

data = readtable(dataFullFile);

[r, c] = size(data);

cNum = zeros([c 1]);
cFl = zeros([c 1]);

%% Reformat data - decide


%for i = 1:c

%Outcome (categorical response)
i = 2;
cNum(i) = 1; %uniN;

%Surv.months (keep numerical)
i = 3;
cNum(i) = 1;
cFl(i) = 1;

%Age (keep numerical)
i = 4;
cNum(i) = 1;
cFl(i) = 1;

% Grade (break into 1 numerical plus 1 flag unknown)
i = 5;
cNum(i) = 2;%uniN;

%Num.primaries (keep numerical)
i = 6;
cNum(i) = 1;
cFl(i) = 1;

% T (one-hot, 'UNK' no encoding)
i = 7;
uni = unique(data(:,i));
[uniN, ~] = size(uni);
cNum(i) = uniN-1;

% N (break into 1 numerical plus 1 flag unknown/NaN)
i = 8;
tmp = data{:,i};
tmp(isnan(tmp)) = 9;
uni = unique(tmp);
[uniN, ~] = size(uni);
cNum(i) = 2; %uniN;

% M (break into 1 numerical plus 1 flag unknown)
i = 9;
tmp = data{:,i};
tmp(strcmp(tmp,'NULL')) = {'9'};
uni = unique(tmp);
[uniN, ~] = size(uni);
cNum(i) = 2; %uniN;

%Radiation (convert numerical 0->1, 5->0)
i = 10;
cNum(i) = 1;

% Stage (one-hot)
i = 11;
uni = unique(data(:,i));
[uniN, ~] = size(uni);
cNum(i) = uniN;

% Primary site (one-hot)
i = 12;
uni = unique(data(:,i));
[uniN, ~] = size(uni);
cNum(i) = uniN;

% Hystology (one-hot)
i = 13;
uni = unique(data(:,i));
[uniN, ~] = size(uni);
cNum(i) = uniN;

% Tumor size (keep 1 numerical plus 1 flag unknown/NaN)
i = 14;
cNum(i) = 2;
cFl(i) = 1;

%Num mutated genes (keep numerical)
i = 15;
cNum(i) = 1;
cFl(i) = 1;

%Num mutations (keep numerical)
i = 16;
cNum(i) = 1;
cFl(i) = 1;

%% Reformat data - execute

sNum = sum(cNum);
dataNum = zeros(r, sNum);

%Outcome (categorical numerical 0/1 for Alive/Dead)
i = 2;
iSt = 1+sum(cNum(1:i-1));
dataNum(:, iSt) = grp2idx(categorical(data{:,i}))-1;

%Surv.months (keep numerical)
i = 3;
iSt = 1+sum(cNum(1:i-1));
dataNum(:, iSt) = double(data{:,i});
dataNum(:, iSt) = normalize(dataNum(:,iSt), "range");

%Age (keep numerical)
i = 4;
iSt = 1+sum(cNum(1:i-1));
dataNum(:, iSt) = double(data{:,i});
dataNum(:, iSt) = normalize(dataNum(:,iSt), "range");

% Grade (break into 1 numerical plus 1 flag known(1)/unknown(0))
i = 5;
iSt = 1+sum(cNum(1:i-1));
dataNum(data{:,i} ~=9, iSt) = double(data{data{:,i} ~=9,i});
dataNum(:, iSt) = normalize(dataNum(:,iSt), "range");
dataNum(data{:,i} ~=9, iSt+1) = 1;

%Num.primaries (keep numerical)
i = 6;
iSt = 1+sum(cNum(1:i-1));
dataNum(:, iSt) = double(data{:,i});
dataNum(:, iSt) = normalize(dataNum(:,iSt), "range");

% T (one-hot, 'UNK' no encoding)
i = 7;
iSt = 1+sum(cNum(1:i-1));
uni = unique(data(:,i));
[uniN, ~] = size(uni);
uniIdx = grp2idx(categorical(uni{:,1}));
for j = 1:uniN
    if ~strcmp(uni{j,1}, 'UNK')
        dataNum(strcmp(data{:,i}, string(uni{j,1})), iSt) = 1;
        iSt = iSt+1;
    end
end

% N (break into 1 numerical plus 1 flag unknown/NaN)
i = 8;
iSt = 1+sum(cNum(1:i-1));
dataNum(~isnan(data{:,i}), iSt) = double(data{~isnan(data{:,i}),i});
dataNum(:, iSt) = normalize(dataNum(:,iSt), "range");
dataNum(~isnan(data{:,i}), iSt+1) = 1;

% M (break into 1 numerical plus 1 flag unknown)
i = 9;
iSt = 1+sum(cNum(1:i-1));
dataNum(~strcmp(data{:,i}, 'NULL'), iSt) = double(string(data{~strcmp(data{:,i}, 'NULL'),i}));
dataNum(:, iSt) = normalize(dataNum(:,iSt), "range");
dataNum(~strcmp(data{:,i}, 'NULL'), iSt+1) = 1;

%Radiation (convert numerical 0->1, 5->0)
i = 10;
iSt = 1+sum(cNum(1:i-1));
dataNum(data{:,i}==0, iSt) = 1; 

% Stage (one-hot)
i = 11;
iSt = 1+sum(cNum(1:i-1));
uni = unique(data(:,i));
[uniN, ~] = size(uni);
uniIdx = grp2idx(categorical(uni{:,1}));
for j = 1:uniN
        dataNum(strcmp(data{:,i}, string(uni{j,1})), iSt) = 1;
        iSt = iSt+1;
end

% Primary site (one-hot)
i = 12;
iSt = 1+sum(cNum(1:i-1));
uni = unique(data(:,i));
[uniN, ~] = size(uni);
%uniIdx = grp2idx(categorical(uni{:,1}));
for j = 1:uniN
        dataNum(strcmp(data{:,i}, string(uni{j,1})), iSt) = 1;
        iSt = iSt+1;
end

% Hystology (one-hot)
i = 13;
iSt = 1+sum(cNum(1:i-1));
uni = unique(data(:,i));
[uniN, ~] = size(uni);
uniIdx = grp2idx(categorical(uni{:,1}));
for j = 1:uniN
        dataNum(strcmp(data{:,i}, string(uni{j,1})), iSt) = 1;
        iSt = iSt+1;
end

% Tumor size (keep 1 numerical plus 1 flag unknown/NaN)
i = 14;
iSt = 1+sum(cNum(1:i-1));
dataNum(~isnan(data{:,i}), iSt) = double(data{~isnan(data{:,i}),i});
dataNum(:, iSt) = normalize(dataNum(:,iSt), "range");
dataNum(~isnan(data{:,i}), iSt+1) = 1;

%Num mutated genes (keep numerical)
i = 15;
iSt = 1+sum(cNum(1:i-1));
dataNum(:, iSt) = double(data{:,i});
dataNum(:, iSt) = normalize(dataNum(:,iSt), "range");

%Num mutations (keep numerical)
i = 16;
iSt = 1+sum(cNum(1:i-1));
dataNum(:, iSt) = double(data{:,i});
dataNum(:, iSt) = normalize(dataNum(:,iSt), "range");


%% Select test set

tPer = 0.25;
% Test set for 12 month survival - select good record for which we know
% patient survived or not 12 month mark
goodTestAIdx = data{:,3}>=12 & strcmp(data{:,2},'Alive');
goodTestDIdx = data{:,3}<=12 & ~strcmp(data{:,2},'Alive');

nTestA = floor(sum(goodTestAIdx) * tPer * r/100);
nTestD = floor(sum(goodTestDIdx) * tPer * r/100);

goodTestA = dataNum(goodTestAIdx, :);
goodTestD = dataNum(goodTestDIdx, :);

[nGoodA,~] = size(goodTestA);
[nGoodD,~] = size(goodTestD);

testAIdx = randsample(nGoodA, nTestA);
testDIdx = randsample(nGoodD, nTestD);

allAIdx2 = [1:nGoodA]';
trainAIdx2 = setdiff(allAIdx2, testAIdx);

allDIdx2 = [1:nGoodD]';
trainDIdx2 = setdiff(allDIdx2, testDIdx);

testA = goodTestA(testAIdx,:);
testD = goodTestD(testDIdx,:);
test = vertcat(testA, testD);

%set test survival month to target 12
test(:,2) = 12/max(data{:,3});


% Train set as everyting not in test set
trainA2 = goodTestA(trainAIdx2,:);
trainD2 = goodTestD(trainDIdx2,:);

%Rebalance Alive data by replicating more TrainA2 with with previous months
%mult = 8;
%[n, ~] = size(trainA2);
%trainA3 = [];
%for i = 1:n
%    trainAd2 = repmat(trainA2(i,:),[mult 1]);
%    maxMonth = trainA2(i,2);
%    dMonth = maxMonth/mult;
%    for ii = 1:mult
%        trainAd2(ii,2)= dMonth * ii;
%    end
%    trainA3 = vertcat(trainA3, trainAd2);
%end




goodTrainAIdx = data{:,3}<12 & strcmp(data{:,2},'Alive');
goodTrainDIdx = data{:,3}>12 & ~strcmp(data{:,2},'Alive');

trainA = dataNum(goodTrainAIdx, :);
trainD = dataNum(goodTrainDIdx, :);


%[n, ~] = size(trainA);
%trainA4 = [];
%for i = 1:n
%    trainAd4 = repmat(trainA(i,:),[mult 1]);
%    maxMonth = trainA(i,2);
%    dMonth = maxMonth/mult;
%    for ii = 1:mult
%        trainAd4(ii,2)= dMonth * ii;
%    end
%    trainA4 = vertcat(trainA4, trainAd4);
%end


train = vertcat(trainA, trainD, trainA2, trainD2); %, trainA3, trainA4);


trX = train(:, 2:end);
trY = categorical(train(:, 1));

%% Model and parameters
ini_rate = 0.001; %0.001
max_epoch = 3000;

m_in = sNum-1;
n_out = 2;
k_bottle = floor((c-1) * 1.);

%cNet = LrReLUNet2Cl(m_in, n_out, ini_rate, max_epoch, .5);

%cNet = LrReLULQPNet2Cl(m_in, n_out, ini_rate, max_epoch, 1);
%cNet = LrReLUSQPNet2Cl(m_in, n_out, ini_rate, max_epoch, 1);
%cNet = LrReLU3SQPNet2Cl(m_in, n_out, ini_rate, max_epoch, 1);
%cNet = LrReLURQPNet2Cl(m_in, n_out, ini_rate, max_epoch, 1);

cNet = BTL3SQPNet2Cl(m_in, k_bottle, n_out, ini_rate, max_epoch, 1);


cNet.mb_size = 64;
cNet = cNet.Create();
                        %% Training
                        fprintf('Training Net\n');
                    

                        % GPU on
                        gpuDevice(1);
                        reset(gpuDevice(1));

                        % Updates weights from previous training with previous slice of no-bind data
                        cNet = cNet.Train(trX, trY);


                        % GPU off
                        delete(gcp('nocreate'));
                        gpuDevice([]);

%% Test
tsX = test(:, 2:end);
tsY = categorical(test(:, 1));


                fprintf('Predicting\n');

                % GPU on
                gpuDevice(1);
                reset(gpuDevice(1));

                [R1, R2, R3] = cNet.Predict(tsX);
    
                % GPU off
                delete(gcp('nocreate'));
                gpuDevice([]);


%% Metrics
TP = sum((tsY(:) == categorical(1)) & (R2(:) == tsY(:)));
FP = sum((tsY(:) ~= categorical(1)) & (R2(:) == categorical(1)));
TN = sum((tsY(:) ~= categorical(1)) & (R2(:) == tsY(:)));
FN = sum((tsY(:) == categorical(1)) & (R2(:) ~= categorical(1)));

%
acc = (TP + TN) ./ (TP + TN + FP + FN);

Pr = TP ./ (TP + FP);
Pr(TP==0) = 0;

Rec = TP ./ (TP + FN);
Sp = TN ./ (TN + FP);
Fo = FP ./ (FP + TN);

F1 = 2*Rec.*Pr./(Rec+Pr);
F1(TP==0) = 0;



fprintf('Accuracy %f, Precision %f, Recall %f, Specificity %f, F1 %f\n', acc, Pr, Rec, Sp, F1);
