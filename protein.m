%% Clear everything 
clearvars -global;
clear all; close all; clc;

% Mem cleanup
ngpu = gpuDeviceCount();
for i=1:ngpu
    reset(gpuDevice(i));
end

%% General config

% Amino residue frame one-side length
resWindowLen = 2;
resWindowWhole = 2*resWindowLen + 1;
resNum = 26;

% RNA base frame one-side length
baseWindowLen = 2;
baseWindowWhole = 2*baseWindowLen + 1;
baseNum = 4;

m_in = resNum * resWindowWhole + baseNum * baseWindowWhole;
n_out = 2; % bind or not

%% Load tarin data
dataIdxDir = '~/data/Protein/practip-data';
dataTrIdxFile = 'train.lst';


dataTrIdxFN = strcat(dataIdxDir,'/',dataTrIdxFile);

trIdxM = readmatrix(dataTrIdxFN, FileType='text', OutputType='string', Delimiter=' ');
[n, ~] = size(trIdxM);


% Count all residue-base pairs
mAll = 0;
for i = 1:n
    dataDatFile = trIdxM(i,5);
    dataDatFN = strcat(dataIdxDir,'/',dataDatFile);
    trDatM = readmatrix(dataDatFN, FileType='text', Delimiter=' ');
    [m, ~] = size(trDatM);

    %
    mAll = mAll + m;

    fprintf('Counting tr+ dat: %d/%d\n', i, n)
end


%% Build train data with binding
trBindM = zeros([mAll, m_in]);
trBindY = categorical(ones([mAll, 1]));

% Fill in all residue-base pairs - positive examples part of training dataset
mCur = 1;
for i = 1:n
    dataDatFile = trIdxM(i,5);
    dataDatFN = strcat(dataIdxDir,'/',dataDatFile);
    trDatM = readmatrix(dataDatFN, FileType='text', Delimiter=' ');
    [m, ~] = size(trDatM);
    
    %
    resFaFile = trIdxM(i,1);
    resFaFN = strcat(dataIdxDir,'/',resFaFile);
    resFaMstr = readmatrix(resFaFN, FileType='text', OutputType='string', Delimiter=' ');
    resFaM = double(char(resFaMstr(2))) - 64;
    [~, res_len] = size(resFaM);

    baseFaFile = trIdxM(i,3);
    baseFaFN = strcat(dataIdxDir,'/',baseFaFile);
    baseFaMstr = readmatrix(baseFaFN, FileType='text', OutputType='string', Delimiter=' ');
    baseFaM = double(char(baseFaMstr(2)));
    cSrch = double('A');
    baseFaM(baseFaM==cSrch) = 1;
    cSrch = double('C');
    baseFaM(baseFaM==cSrch) = 2;
    cSrch = double('G');
    baseFaM(baseFaM==cSrch) = 3;
    cSrch = double('U');
    baseFaM(baseFaM==cSrch) = 4;
    [~, base_len] = size(baseFaM);


    for j = 1:m

        trBindM(mCur,:) = bind_1hot(resFaM, baseFaM, trBindM(mCur,:), resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, trDatM(j,1), trDatM(j,2), mCur);
        mCur = mCur + 1;

    end



    fprintf('Loading tr dat: %d/%d\n', i, n)
end


%% Count all no-bind residue-base pairs
mNone = 0;
for i = 1:n
    dataDatFile = trIdxM(i,5);
    dataDatFN = strcat(dataIdxDir,'/',dataDatFile);
    trDatM = readmatrix(dataDatFN, FileType='text', Delimiter=' ');
    [m, ~] = size(trDatM);

    %
    resFaFile = trIdxM(i,1);
    resFaFN = strcat(dataIdxDir,'/',resFaFile);
    resFaMstr = readmatrix(resFaFN, FileType='text', OutputType='string', Delimiter=' ');
    resFaM = double(char(resFaMstr(2))) - 64;
    [~, res_len] = size(resFaM);

    baseFaFile = trIdxM(i,3);
    baseFaFN = strcat(dataIdxDir,'/',baseFaFile);
    baseFaMstr = readmatrix(baseFaFN, FileType='text', OutputType='string', Delimiter=' ');
    baseFaM = double(char(baseFaMstr(2)));
    cSrch = double('A');
    baseFaM(baseFaM==cSrch) = 1;
    cSrch = double('C');
    baseFaM(baseFaM==cSrch) = 2;
    cSrch = double('G');
    baseFaM(baseFaM==cSrch) = 3;
    cSrch = double('U');
    baseFaM(baseFaM==cSrch) = 4;
    [~, base_len] = size(baseFaM);


    % go through all non-interlaping (to reduce size of the dataset) residues in the given amino-acid
    for r = 1:resWindowWhole:res_len

        % find upper and lower bounds of the bind area, to create non-bind
        % dataset from above and below.
        % Limitation: only one bind area for a given residue window
        % (different positions in dat file refering to the same residue
        % windows are not counted in, so possible multi-labling)
        inResFl = 0;
        inRes = 0;
        upBound = -1;
        lowBound = -1;
        for j = 1:m
            if ~inResFl && (resFaM(r) == trDatM(j, 1))
                inRes = resFaM(r);
                inResFl = 1;
                upBound = trDatM(j, 2) - baseWindowWhole;
            end
            
            if (inResFl) 
               if inRes ~= trDatM(j, 1)
                inResFl = 0;
                break;
               else
                lowBound = trDatM(j, 2) + baseWindowWhole;
               end
            end
        end

        if upBound >= 0
            mNone = mNone + upBound + 1;
        end
        if (lowBound >= 0) && (lowBound <= base_len)
            mNone = mNone + base_len - lowBound + 1;
        end
        % no binds for a given residue
        if (upBound < 0) && (lowBound < 0)
            mNone = mNone + base_len;
        end

    end

    fprintf('Counting tr- dat: %d/%d\n', i, n)
end

trNoBindM = zeros([mNone, m_in]);

%% Loading all no bind residue-base pairs
mCur = 1;
for i = 1:n
    dataDatFile = trIdxM(i,5);
    dataDatFN = strcat(dataIdxDir,'/',dataDatFile);
    trDatM = readmatrix(dataDatFN, FileType='text', Delimiter=' ');
    [m, ~] = size(trDatM);

    %
    resFaFile = trIdxM(i,1);
    resFaFN = strcat(dataIdxDir,'/',resFaFile);
    resFaMstr = readmatrix(resFaFN, FileType='text', OutputType='string', Delimiter=' ');
    resFaM = double(char(resFaMstr(2))) - 64;
    [~, res_len] = size(resFaM);

    baseFaFile = trIdxM(i,3);
    baseFaFN = strcat(dataIdxDir,'/',baseFaFile);
    baseFaMstr = readmatrix(baseFaFN, FileType='text', OutputType='string', Delimiter=' ');
    baseFaM = double(char(baseFaMstr(2)));
    cSrch = double('A');
    baseFaM(baseFaM==cSrch) = 1;
    cSrch = double('C');
    baseFaM(baseFaM==cSrch) = 2;
    cSrch = double('G');
    baseFaM(baseFaM==cSrch) = 3;
    cSrch = double('U');
    baseFaM(baseFaM==cSrch) = 4;
    [~, base_len] = size(baseFaM);


    % go through all non-interlaping (to reduce size of the dataset) residues in the given amino-acid
    for r = 1:resWindowWhole:res_len

        % find upper and lower bounds of the bind area, to create non-bind
        % dataset from above and below.
        % Limitation: only one bind area for a given residue window
        % (different positions in dat file refering to the same residue
        % windows are not counted in, so possible multi-labling)
        inResFl = 0;
        inRes = 0;
        upBound = -1;
        lowBound = -1;
        for j = 1:m
            if ~inResFl && (resFaM(r) == trDatM(j, 1))
                inRes = resFaM(r);
                inResFl = 1;
                upBound = trDatM(j, 2) - baseWindowWhole;
            end
            
            if (inResFl) 
               if inRes ~= trDatM(j, 1)
                inResFl = 0;
                break;
               else
                lowBound = trDatM(j, 2) + baseWindowWhole;
               end
            end
        end

        if upBound >= 0
            for b = 1:upBound
                trNoBindM(mCur,:) = bind_1hot(resFaM, baseFaM, trNoBindM(mCur,:), resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
                mCur = mCur + 1;
            end
        end
        if (lowBound >= 0) && (lowBound <= base_len)
            for b = lowBound:base_len
                trNoBindM(mCur,:) = bind_1hot(resFaM, baseFaM, trNoBindM(mCur,:), resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
                mCur = mCur + 1;
            end
        end
        % no binds for a given residue
        if (upBound < 0) && (lowBound < 0)
            for b = 1:base_len
                trNoBindM(mCur,:) = bind_1hot(resFaM, baseFaM, trNoBindM(mCur,:), resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
                mCur = mCur + 1;
            end
        end

    end

    fprintf('Loading tr- dat: %d/%d\n', i, n)
end

scaleNo = 10;
trNoBindBalM = trNoBindM(randperm(mCur, mAll*scaleNo), :);
clear("trNoBindM");

trNoBindY = categorical(zeros([mAll*scaleNo, 1]));

%%
mWhole = mAll*(scaleNo+1);
trMX = zeros([mWhole, m_in]);
trMX(1:mAll,:) = trBindM;
trMX(mAll+1:end,:) = trNoBindBalM;

trMY = categorical(zeros([mWhole, 1]));
trMY(1:mAll,:) = trBindY;
trMY(mAll+1:end,:) = trNoBindY;

%%

ini_rate = 0.001; 
max_epoch = 25;

cNet = AnnClasNet2D(m_in, n_out, ini_rate, max_epoch);
cNet.mb_size = 2^floor(log2(mWhole)-4);
cNet = cNet.Create();
cNet = cNet.Train(trMX, trMY);

%%
dataTsIdxFile = 'test.lst';

[X2, Yh2] = build_tensors('test', dataIdxDir, dataTsIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
    baseWindowLen, baseWindowWhole, baseNum, scaleNo);

%%
[X2, Y2] = cNet.Predict(X2);

%%
[nY2, ~] = size(Y2);
acc = sum(Yh2 == Y2) / nY2
