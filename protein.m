%% Clear everything 
clearvars -global;
clear all; close all; clc;

% Mem cleanup
%ngpu = gpuDeviceCount();
%for i=1:ngpu
%    reset(gpuDevice(i));
%end


%% General config

% Amino residue frame one-side length
resWindowLen = 7;
resWindowWhole = 2*resWindowLen + 1;
resNum = 26;

% RNA base frame one-side length
baseWindowLen = 7;
baseWindowWhole = 2*baseWindowLen + 1;
baseNum = 4;

mr_in = resNum * resWindowWhole;
mb_in = baseNum * baseWindowWhole;
m_in = mr_in + mb_in;

n_out = 2; % bind or not

bindScaleNo = 1;
noBindScaleNo = 1;%50;

scaleInFiles = 1;%2;

noBindPerc = 70; %70;

% Load tarin data
dataIdxDir = '~/data/Protein/practip-data';
dataTrIdxFile = 'train.lst';


%%
%[X, Y, mTrBind, mTrNoBind, Xcontr, Ycontr, Ncontr] = build_tensors(dataIdxDir, dataTrIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
%    baseWindowLen, baseWindowWhole, baseNum, scaleNo, scaleInFiles);


%%
ini_rate = 0.001; 
max_epoch = 10; %200

%cNet = AnnClasNet2D(m_in, n_out, ini_rate, max_epoch);
cNet = ReluClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = SigClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = TanhClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = RbfClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = TransClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = KgClasNet2D(m_in, n_out, ini_rate, max_epoch);


nTrain = 5; %25
nNets = 5;
[cNet, cNets, X, Y, mTrBind, mTrNoBind, Xcontr, Ycontr, Ncontr, t1, t2, noBindThresh] = train_tensors(cNet, nNets, nTrain, dataIdxDir, dataTrIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
    baseWindowLen, baseWindowWhole, baseNum, bindScaleNo, noBindScaleNo, scaleInFiles, noBindPerc);

%[mWhole, ~] = size(X);
%cNet.mb_size = 2^floor(log2(mWhole)-4);
%cNet = cNet.Create();

% GPU on
%gpuDevice(1);
%reset(gpuDevice(1));

%t1 = clock();
%cNet = cNet.Train(X, Y);
%t2 = clock();

% GPU off
%delete(gcp('nocreate'));
%gpuDevice([]);

%%
dataTsIdxFile = 'test.lst';
scaleNoTs = 0;

% GPU on
%gpuDevice(1);
%reset(gpuDevice(1));

%[X2, Yh2, mTsBind, mTsNoBind, X2contr, Y2contr, N2contr] = build_tensors(dataIdxDir, dataTsIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
%    baseWindowLen, baseWindowWhole, baseNum, scaleNoTs, 1);
%[X2, Y2] = cNet.Predict(X2);

%
[TP, TN, FP, FN, mTsBind, mTsNoBind, meanActTP, meanActFN, meanActTN, meanActFP, sigActTP, sigActFN] = predict_tensors_test(cNet, cNets, nNets, dataIdxDir, dataTsIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
    baseWindowLen, baseWindowWhole, baseNum, scaleNoTs, 1, noBindThresh);

% GPU off
%delete(gcp('nocreate'));
%gpuDevice([]);

%
%[nY2, ~] = size(Y2);
%acc = sum(Yh2 == Y2) / nY2;

%TP = sum(Yh2(1:mTsBind) == Y2(1:mTsBind));
%TN = sum(Yh2(mTsBind+1:end) == Y2(mTsBind+1:end));
%FN = sum(Yh2(1:mTsBind) ~= Y2(1:mTsBind));
%FP = sum(Yh2(mTsBind+1:end) ~= Y2(mTsBind+1:end));

acc = (TP + TN) / (TP + TN + FP + FN);
Pr = TP / (TP + FP);
Rec = TP / (TP + FN);
Sp = TN / (TN + FP);
F1 = 2*Rec*Pr/(Rec+Pr);

%
fprintf('Model %s, mb_size %d, max_epoch %d, ResWindow %d, BaseWindow %d, NoBindRat %d, TrainBindN %d, TrainNoBindN %d, BindScaleNo %d, NoBindScaleNo %d, ScaleInFiles %d\n',...
    cNet.name, cNet.mb_size, max_epoch, resWindowWhole, baseWindowWhole, noBindScaleNo, mTrBind, mTrNoBind, bindScaleNo, noBindScaleNo, scaleInFiles);
fprintf('NNets %d, NTrain %d, NoBindPerc %d, NoBindThresh %f, NoBindTsRat %d, TestBindN %d, TestNoBindN %d\n',...
    nNets, nTrain, noBindPerc, noBindThresh, scaleNoTs, mTsBind, mTsNoBind);
fprintf('Accuracy %f, Precision %f, Recall %f, Specificity %f, F1 %f, TP %d, TN %d, FN %d, FP %d, TrTime %f s\n',...
    acc, Pr, Rec, Sp, F1, TP, TN, FN, FP, etime(t2, t1));