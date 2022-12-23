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
resWindowLen = 5;
resWindowWhole = 2*resWindowLen + 1;
resNum = 26;

% RNA base frame one-side length
baseWindowLen = 5;
baseWindowWhole = 2*baseWindowLen + 1;
baseNum = 4;

mr_in = resNum * resWindowWhole;
mb_in = baseNum * baseWindowWhole;
m_in = mr_in + mb_in;

n_out = 2; % bind or not

scaleNo = 1;

% Load tarin data
dataIdxDir = '~/data/Protein/practip-data';
dataTrIdxFile = 'train.lst';


ini_rate = 0.001; 
max_epoch = 50; %200

%%
[X, Y, mTrBind, mTrNoBind] = build_tensors(dataIdxDir, dataTrIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
    baseWindowLen, baseWindowWhole, baseNum, scaleNo);

%%
%cNet = AnnClasNet2D(m_in, n_out, ini_rate, max_epoch);
cNet = ReluClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = SigClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = TanhClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = RbfClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = TransClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = KgClasNet2D(m_in, n_out, ini_rate, max_epoch);

[mWhole, ~] = size(X);
cNet.mb_size = 2^floor(log2(mWhole)-4);

cNet = cNet.Create();

gpuDevice(1);
reset(gpuDevice(1));

t1 = clock();
cNet = cNet.Train(X, Y);
t2 = clock();

%
dataTsIdxFile = 'test.lst';

[X2, Yh2, mTsBind, mTsNoBind] = build_tensors(dataIdxDir, dataTsIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
    baseWindowLen, baseWindowWhole, baseNum, scaleNo);

%
[X2, Y2] = cNet.Predict(X2);

delete(gcp('nocreate'));


%
[nY2, ~] = size(Y2);
acc = sum(Yh2 == Y2) / nY2;

TP = sum(Yh2(1:mTsBind) == Y2(1:mTsBind));
TN = sum(Yh2(mTsBind+1:end) == Y2(mTsBind+1:end));
FN = sum(Yh2(1:mTsBind) ~= Y2(1:mTsBind));
FP = sum(Yh2(mTsBind+1:end) ~= Y2(mTsBind+1:end));

Pr = TP / (TP + FP);
Rec = TP / (TP + FN);
Sp = TN / (TN + FP);
F1 = 2*Rec*Pr/(Rec+Pr);

%
fprintf('Model %s, mb_size %d, max_epoch %d, ResWindow %d, BaseWindow %d, NoBindRat %d, TrainBindN %d, TrainNoBindN %d, TrstBindN %d, TestNoBindN %d\n',...
    cNet.name, cNet.mb_size, max_epoch, resWindowWhole, baseWindowWhole, scaleNo, mTrBind, mTrNoBind, mTsBind, mTsNoBind);
fprintf('Accuracy %f, Precision %f, Recall %f, Specificity %f, F1 %f, TP %d, TN %d, FN %d, FP %d, TrTime %f s\n',...
    acc, Pr, Rec, Sp, F1, TP, TN, FN, FP, etime(t2, t1));