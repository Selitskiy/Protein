function [TP, TN, FP, FN, mBind, mNoBind, meanActTP, meanActFN, meanActTN, meanActFP, sigActTP, sigActFN] = predict_tensors_test(cNet, dataIdxDir, dataTrIdxFile, m_in, ...
    resWindowLen, resWindowWhole, resNum, baseWindowLen, baseWindowWhole, baseNum, scaleNo, scaleInFiles, threshP)

dataTrIdxFN = strcat(dataIdxDir,'/',dataTrIdxFile);

trIdxM = readmatrix(dataTrIdxFN, FileType='text', OutputType='string', Delimiter=' ');
[n, ~] = size(trIdxM);


% Count all residue-base pairs
mBind = 0;
for i = 1:n
    dataDatFile = trIdxM(i,5);
    dataDatFN = strcat(dataIdxDir,'/',dataDatFile);
    trDatM = readmatrix(dataDatFN, FileType='text', Delimiter=' ');
    [m, ~] = size(trDatM);

    %
    mBind = mBind + m;

    fprintf('Counting %s+ dat: %d/%d\n', dataTrIdxFile, i, n)
end


%% Build data with binding
bindX = zeros([mBind, m_in]);
bindYh = categorical(ones([mBind, 1]));

% Fill in all residue-base pairs - positive examples part of training dataset
mCur = 0;
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
        mCur = mCur + 1;
        bindX(mCur,:) = bind_1hot(resFaM, baseFaM, bindX(mCur,:), resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, trDatM(j,1), trDatM(j,2), mCur);
    end


    fprintf('Loading %s+ dat: %d/%d\n', dataTrIdxFile, i, n)
end

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

[bindX, bindY, bindA] = cNet.Predict(bindX);

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);


TPIdx = (bindYh == bindY);
FNIdx = (bindYh ~= bindY);

nTP = sum(TPIdx);
meanActTP = sum(bindA(TPIdx,2))/nTP;
sigActTP = std(bindA(TPIdx,2));

nFN = sum(FNIdx);
meanActFN = sum(bindA(FNIdx,1))/nFN;
sigActFN = std(bindA(FNIdx,1));


TPIdxCond = ((bindYh == bindY) & (bindA(:,2) > threshP));
FNIdxCond = ((bindYh ~= bindY) | ((bindYh == bindY) & (bindA(:,2) <= threshP)));

TP = sum(TPIdxCond);
TN = 0;
FN = sum(FNIdxCond);
FP = 0;

%% Count all no-bind residue-base pairs
mNoBind = 0;
%mCur = 0;
ns = floor(n/scaleInFiles);

nTNold = 0;
sumActTNold = 0;
nFPold = 0;
sumActFPold = 0;

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

    mNone = 0;
    % go through all non-interlaping (to reduce size of the dataset) residues in the given amino-acid
    for r = 1:resWindowWhole:res_len
    %for r = 1:res_len

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

    fprintf('Loading %s- dat: %d/%d\n', dataTrIdxFile, i, n)


    noBindX = zeros([mNone, m_in]);
    noBindYh = categorical(zeros([mNone, 1]));

    % Loading all no-bind residue-base pairs

    mCur = 0;
    % go through all non-interlaping (to reduce size of the dataset) residues in the given amino-acid
    for r = 1:resWindowWhole:res_len
    %for r = 1:res_len

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
                mCur = mCur + 1;
                mNoBind = mNoBind + 1;
                noBindX(mCur,:) = bind_1hot(resFaM, baseFaM, noBindX(mCur,:), resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
            end
        end
        if (lowBound >= 0) && (lowBound <= base_len)
            for b = lowBound:base_len
                mCur = mCur + 1;
                mNoBind = mNoBind + 1;
                noBindX(mCur,:) = bind_1hot(resFaM, baseFaM, noBindX(mCur,:), resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
            end
        end
        % no binds for a given residue
        if (upBound < 0) && (lowBound < 0)
            for b = 1:base_len
                mCur = mCur + 1;
                mNoBind = mNoBind + 1;
                noBindX(mCur,:) = bind_1hot(resFaM, baseFaM, noBindX(mCur,:), resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
            end
        end

    end

    fprintf('Predicting %s- dat: %d/%d\n', dataTrIdxFile,i, n)

    % GPU on
    gpuDevice(1);
    reset(gpuDevice(1));

    [noBindX, noBindY, noBindA] = cNet.Predict(noBindX);
    
    % GPU off
    delete(gcp('nocreate'));
    gpuDevice([]);


    TNIdx = (noBindYh(1:mCur) == noBindY(1:mCur));
    FPIdx = (noBindYh(1:mCur) ~= noBindY(1:mCur));

    nTNcur = sum(TNIdx);
    sumActTNcur = sum(noBindA(TNIdx,1));

    nFPcur = sum(FPIdx);
    sumActFPcur = sum(noBindA(FPIdx,2));

    nTN = nTNold + nTNcur;
    sumActTN = (sumActTNold + sumActTNcur);
    meanActTN = sumActTN / nTN;

    nFP = nFPold + nFPcur;
    sumActFP = (sumActFPold + sumActFPcur);
    meanActFP = sumActFP / nFP;


    nTNold = nTN;
    sumActTNold = sumActTN;
    nFPold = nFP;
    sumActFPold = sumActFP;


    TNIdxCond = ((noBindYh(1:mCur) == noBindY(1:mCur)) | ((noBindYh(1:mCur) ~= noBindY(1:mCur)) & (noBindA(1:mCur,2) <= threshP)));
    FPIdxCond = ((noBindYh(1:mCur) ~= noBindY(1:mCur)) & (noBindA(1:mCur,2) > threshP));

    TP = TP + 0;
    TN = TN + sum(TNIdxCond);
    FN = FN + 0;
    FP = FP + sum(FPIdxCond);

end



end