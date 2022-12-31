function [cNet, cNets, trMX, trMY, mAll, mAllNo, Xcontr, Ycontr, Ncontr, t1, t2, noBindThresh] = train_tensors(cNet, nNets, nTrain, dataIdxDir, dataTrIdxFile, m_in, ...
    resWindowLen, resWindowWhole, resNum, baseWindowLen, baseWindowWhole, baseNum, bindScaleNo, noBindScaleNo, scaleInFiles, noBindPerc)

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

    fprintf('Counting %s+ dat: %d/%d\n', dataTrIdxFile, i, n)
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


    fprintf('Loading %s+ dat: %d/%d\n', dataTrIdxFile, i, n)
end


%% Count all no-bind residue-base pairs
mNone = 0;
ns = floor(n/scaleInFiles);
for i = 1:ns
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

    fprintf('Counting %s- dat: %d/%d\n', dataTrIdxFile, i, n)
end

trNoBindM = zeros([mNone, m_in]);

%% Loading all no bind residue-base pairs
mCur = 1;
for i = 1:ns
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

    fprintf('Loading %s- dat: %d/%d\n', dataTrIdxFile,i, n)
end

if noBindScaleNo
    mAllNo = floor(mAll*noBindScaleNo);
else
    mAllNo = mCur;
end

if bindScaleNo
    mAllYes = floor(mAll*bindScaleNo);
else
    mAllYes = mAll;
end

%% Repeated retraining with new no-bind folds
cNets = cell([nNets, 1]);
mWhole = mAllYes + mAllNo;
cNet.mb_size = 2^floor(log2(mWhole)-4);

% Save only necessary slice of the non-bind data to save space
trNoBindLimM = trNoBindM(randperm(mCur, mAllNo*(nTrain+1)), :);
clear("trNoBindM");

t1 = clock();
for l = 1:nNets

    cNet = cNet.Create();
    
    trMX = zeros([mWhole, m_in]);
    trMY = categorical(zeros([mWhole, 1]));

    for k = 1:bindScaleNo
        trMX(1+(k-1)*mAll:k*mAll,:) = trBindM;
        trMY(1+(k-1)*mAll:k*mAll,:) = trBindY;
    end


    for k = 1:nTrain
        trNoBindBalM = trNoBindLimM(1+(k-1)*mAllNo:k*mAllNo, :);
        trNoBindY = categorical(zeros([mAllNo, 1]));

        %
        trMX(mAllYes+1:end,:) = trNoBindBalM;
        trMY(mAllYes+1:end,:) = trNoBindY;


        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));

        cNet = cNet.Train(trMX, trMY);
        cNets{l} = cNet;

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]);
    end
end
t2 = clock();


%% Find threshold for given percentle of FP no-bind predictions
noBindThresh = 0;
if noBindPerc
    noBindX = trNoBindLimM(mAllNo*nTrain+1:end, :);
    % GPU on
    gpuDevice(1);
    reset(gpuDevice(1));

    [noBindX, noBindY, noBindA] = cNet.Predict(noBindX);
    
    % GPU off
    delete(gcp('nocreate'));
    gpuDevice([]);

    noBindThresh = prctile(noBindA((noBindY == categorical(1)), 2), noBindPerc);
end

%% Convert input into strings (for sorting, uniqueness and contradiction detection)
Xcontr = []; 
Ycontr = []; 
Ncontr = 0;
%[Xcontr, Ycontr, Ncontr] = find_doubles(trMX, trMY, mAll, mAllNo, resWindowWhole, resNum, baseWindowLen, baseWindowWhole, baseNum);


end