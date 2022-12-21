%% Clear everything 
clearvars -global;
clear all; close all; clc;

% Mem cleanup
ngpu = gpuDeviceCount();
for i=1:ngpu
    reset(gpuDevice(i));
end

%gpuDevice(1);
%reset(gpuDevice(1));


%% General config

% Amino residue frame one-side length
resWindowLen = 3;
resWindowWhole = 2*resWindowLen + 1;
resNum = 26;

% RNA base frame one-side length
baseWindowLen = 3;
baseWindowWhole = 2*baseWindowLen + 1;
baseNum = 4;

m_in = resNum * resWindowWhole + baseNum * baseWindowWhole;
n_out = 2; % bind or not

scaleNo = 1;

% Load tarin data
dataIdxDir = '~/data/Protein/practip-data';
dataTrIdxFile = 'train.lst';


ini_rate = 0.001; 
max_epoch = 50;

%%
[X, Y] = build_tensors(dataIdxDir, dataTrIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
    baseWindowLen, baseWindowWhole, baseNum, scaleNo);

%%
%cNet = AnnClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = ReluClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = SigClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = TanhClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = RbfClasNet2D(m_in, n_out, ini_rate, max_epoch);
cNet = TransClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = KgClasNet2D(m_in, n_out, ini_rate, max_epoch);

[mWhole, ~] = size(X);
cNet.mb_size = 2^floor(log2(mWhole)-4);

cNet = cNet.Create();
cNet = cNet.Train(X, Y);

%
dataTsIdxFile = 'test.lst';

[X2, Yh2] = build_tensors(dataIdxDir, dataTsIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
    baseWindowLen, baseWindowWhole, baseNum, scaleNo);

%
[X2, Y2] = cNet.Predict(X2);

delete(gcp('nocreate'));


%
[nY2, ~] = size(Y2);
acc = sum(Yh2 == Y2) / nY2;

fprintf('Model %s, mb_size %d, ResWindow %d, BaseWindow %d, NoBindRat %d, Accuracy %f\n', cNet.name, cNet.mb_size, resWindowWhole, baseWindowWhole, scaleNo, acc);