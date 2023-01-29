function [TP, TN, FP, FN, mBind, mNoBind, meanActTP, meanActFN, meanActTN, meanActFP, sigActTP, sigActFN] = predict_tensors_test(cNets, dataIdxDir, dataTrIdxFile, m_in, ...
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

    fprintf('Counting %s+ dat: %d/%d\n', dataTrIdxFile, i, n);
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


    fprintf('Loading %s+ dat: %d/%d\n', dataTrIdxFile, i, n);
end

[~, nThresh] = size(threshP);
sumThresh = sum(threshP, "all");

%TPIdx = ones([mBind, 1]);
%FNIdx = ones([mBind, 1]);
TPIdxCond = ones([mBind, nThresh]);
%FNIdxCond = ones([mBind, 1]);

[nNets, ~] = size(cNets);


for l = 1:nNets
    fprintf('Predicting bind Net %d\n', l);

    % GPU on
    gpuDevice(1);
    reset(gpuDevice(1));

    cNet = cNets{l};
    [~, bindY, bindA] = cNet.Predict(bindX);

    % GPU off
    delete(gcp('nocreate'));
    gpuDevice([]);


    TPIdx = (bindYh == bindY);
    %FNIdx = (bindYh ~= bindY);
    FNIdx = ~TPIdx;

    nTP = sum(TPIdx);
    meanActTP = sum(bindA(TPIdx,2))/nTP;
    sigActTP = std(bindA(TPIdx,2));

    nFN = sum(FNIdx);
    meanActFN = sum(bindA(FNIdx,1))/nFN;
    sigActFN = std(bindA(FNIdx,1));

    for ll = 1:nThresh

        if sumThresh
            TPIdxCond(:,ll) = TPIdxCond(:,ll) & (bindA(:,2) >= threshP(l,ll));
        else
            TPIdxCond(:,ll) = TPIdxCond(:,ll) & (bindYh == bindY);
        end

        %TPIdxCond(:,ll) = TPIdxCond(:,ll) & ((bindYh == bindY) & (bindA(:,2) >= threshP(l,ll)));
        %TPIdxCond(:,ll) = TPIdxCond(:,ll) & ( ((bindYh == bindY) & (bindA(:,2) >= threshP(l,ll))) | ((bindYh ~= bindY) & (bindA(:,1) < threshP(l,ll))) );
    end


end

FNIdxCond = ~TPIdxCond;

TP = sum(TPIdxCond,1);
TN = zeros([1,nThresh]);
FN = sum(FNIdxCond,1);
FP = zeros([1,nThresh]);


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

    fprintf('Loading %s- dat: %d/%d\n', dataTrIdxFile, i, n);


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

    fprintf('Predicting %s- dat: %d/%d\n', dataTrIdxFile,i, n);


    if scaleNo == 0

        TNIdx = ones([mCur, 1]);
        %FPIdx = ones([mCur, 1]);
        TNIdxCond = ones([mCur, nThresh]);
        FPIdxCond = ones([mCur, nThresh]);

        for l = 1:nNets
            fprintf('Predicting no-bind Net %d\n', l);

            % GPU on
            gpuDevice(1);
            reset(gpuDevice(1));

            cNet = cNets{l};
            [~, noBindY, noBindA] = cNet.Predict(noBindX);
    
            % GPU off
            delete(gcp('nocreate'));
            gpuDevice([]);


            TNIdx = TNIdx & (noBindYh(1:mCur) == noBindY(1:mCur));
            %FPIdx = FPIdx & (noBindYh(1:mCur) ~= noBindY(1:mCur));
            FPIdx = ~TNIdx;

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

            for ll = 1:nThresh

                if sumThresh
                    FPIdxCond(:,ll) = FPIdxCond(:,ll) & (noBindA(1:mCur,2) >= threshP(l,ll));
                else
                    FPIdxCond(:,ll) = FPIdxCond(:,ll) & (noBindYh(1:mCur) ~= noBindY(1:mCur));
                end

                %TNIdxCond(:,ll) = TNIdxCond(:,ll) & (noBindA(1:mCur,1) >= threshP(l,ll));
                %TNIdxCond(:,ll) = TNIdxCond(:,ll) & ((noBindYh(1:mCur) == noBindY(1:mCur)) & (noBindA(1:mCur,1) > threshP(l,ll)));
                %TNIdxCond(:,ll) = TNIdxCond(:,ll) & ( ((noBindYh(1:mCur) == noBindY(1:mCur)) & (noBindA(1:mCur,1) > threshP(l,ll))) | ((noBindYh(1:mCur) ~= noBindY(1:mCur)) & (noBindA(1:mCur,2) <= threshP(l,ll))) );
            end
            
        end

        %FPIdx = ~TNIdx;
        
        TNIdxCond = ~FPIdxCond;
        %FPIdxCond = ~TNIdxCond;

        %if threshP(l) > 0 
            TN = TN + sum(TNIdxCond,1);
            FP = FP + sum(FPIdxCond,1);
        %else
        %    TN = TN + sum(TNIdx);
        %    FP = FP + sum(FPIdx);
        %end

        TP = TP + 0;
        FN = FN + 0;

    end

end


    if scaleNo
        % Save only necessary slice of the non-bind data

    end

end