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

nTrain = 5; %25
nNets = 3;


% Load tarin data
dataIdxDir = '~/data/Protein/practip-data';
dataTrIdxFile = 'train.lst';


%%
%[X, Y, mTrBind, mTrNoBind, Xcontr, Ycontr, Ncontr] = build_tensors(dataIdxDir, dataTrIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
%    baseWindowLen, baseWindowWhole, baseNum, scaleNo, scaleInFiles);


%%
ini_rate = 0.001; 
max_epoch = [10, 10, 30]; %200

%cNet = AnnClasNet2D(m_in, n_out, ini_rate, max_epoch);
cNet = ReluClasNet2D(m_in, n_out, ini_rate, max_epoch(1));
%cNet = SigClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = TanhClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = RbfClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = TransClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = KgClasNet2D(m_in, n_out, ini_rate, max_epoch);

nNetTypes = 2;
cNetTypes = cell([nNetTypes, 1]);

cNetTypes{1} = cNet;

cNet2 = KgClasNet2D(m_in, n_out, ini_rate, max_epoch(3));
cNetTypes{2} = cNet2;

%cNet3 = TanhClasNet2D(m_in, n_out, ini_rate, max_epoch(2));
%cNetTypes{3} = cNet3;


[cNets, mTrBind, mTrNoBind, Xcontr, Ycontr, Ncontr, t1, t2, noBindThresh] = train_tensors(cNetTypes, nNets, nTrain, dataIdxDir, dataTrIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
    baseWindowLen, baseWindowWhole, baseNum, bindScaleNo, noBindScaleNo, scaleInFiles, noBindPerc);

[nNetsEns, ~] = size(cNets);



%%
dataTsIdxFile = 'test.lst';
scaleNoTs = 0;


%
%[TP, TN, FP, FN, mTsBind, mTsNoBind, meanActTP, meanActFN, meanActTN, meanActFP, sigActTP, sigActFN] = predict_tensors_test(cNet, cNets, nNets, dataIdxDir, dataTsIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
%    baseWindowLen, baseWindowWhole, baseNum, scaleNoTs, 1, noBindThresh);

%
[TP, TN, FP, FN, mTsBind, mTsNoBind, meanActTP, meanActFN, meanActTN, meanActFP, sigActTP, sigActFN] = predict_tensors_test(cNets, dataIdxDir, dataTsIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
    baseWindowLen, baseWindowWhole, baseNum, scaleNoTs, 1, noBindThresh);


acc = (TP + TN) / (TP + TN + FP + FN);
Pr = TP / (TP + FP);
Rec = TP / (TP + FN);
Sp = TN / (TN + FP);
F1 = 2*Rec*Pr/(Rec+Pr);

%%
fprintf('Model1 %s, mb_size %d, max_epoch %d, ResWindow %d, BaseWindow %d, NoBindRat %d, TrainBindN %d, TrainNoBindN %d, BindScaleNo %d, NoBindScaleNo %d, ScaleInFiles %d\n',...
    cNetTypes{1}.name, cNetTypes{1}.mb_size, max_epoch, resWindowWhole, baseWindowWhole, noBindScaleNo, mTrBind, mTrNoBind, bindScaleNo, noBindScaleNo, scaleInFiles);
fprintf('NNetTypes %d, NNets %d, NTrain %d, NoBindPerc %d, NoBindThresh1 %f, NoBindThresh2 %f, NoBindThresh3 %f, NoBindTsRat %d, TestBindN %d, TestNoBindN %d\n',...
    nNetTypes, nNets, nTrain, noBindPerc, noBindThresh(1), noBindThresh(2), noBindThresh(3), scaleNoTs, mTsBind, mTsNoBind);
fprintf('Accuracy %f, Precision %f, Recall %f, Specificity %f, F1 %f, TP %d, TN %d, FN %d, FP %d, TrTime %f s\n',...
    acc, Pr, Rec, Sp, F1, TP, TN, FN, FP, etime(t2, t1));