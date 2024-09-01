function [TP, TN, FP, FN, mBind, mNoBind, meanActTP, meanActFN, meanActTN, meanActFP, sigActTP, sigActFN] = predict_tensors_fold(cNets, dataIdxDir, dataTrIdxFile, m_in, ...
    resWindowLen, resWindowWhole, resNum, baseWindowLen, baseWindowWhole, baseNum, scaleNo, scaleInFiles, threshP, threshVote, maxNoBind)

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

[nNets, nFolds] = size(cNets);

[~, nThresh] = size(threshP);
sumThresh = sum(threshP, "all");

%TPIdx = ones([mBind, 1]);
%FNIdx = ones([mBind, 1]);
TPIdxCond = ones([mBind, nFolds, nThresh]);
TPIdxSum = zeros([mBind, nFolds, nThresh]);
%FNIdxCond = ones([mBind, 1]);

bindY = categorical(zeros([mBind, nFolds]));
bindA = zeros([mBind, 2, nFolds]);



for l = 1:nNets
    for f = 1:nFolds

        fprintf('Predicting bind Net %d %d\n', l, f);

        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));

        cNet = cNets{l,f};
        [~, bindY(:,l,f), bindA(:,:,l,f)] = cNet.Predict(bindX);

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]);


        TPIdx = (bindYh == bindY(:,l,f));
        %FNIdx = (bindYh ~= bindY(:,f));
        FNIdx = ~TPIdx;

        nTP = sum(TPIdx);
        meanActTP = sum(bindA(TPIdx,2,l,f))/nTP;
        sigActTP = std(bindA(TPIdx,2,l,f));

        nFN = sum(FNIdx);
        meanActFN = sum(bindA(FNIdx,1,l,f))/nFN;
        sigActFN = std(bindA(FNIdx,1,l,f));


        for ll = 1:nThresh

            if sumThresh
                % for ROC
                TPIdxCond(:,f,ll) = TPIdxCond(:,f,ll) & (bindA(:,2,f) >= threshP(l,ll));
                TPIdxSum(:,f,ll) = TPIdxSum(:,f,ll) + (bindA(:,2,f) >= threshP(l,ll));
            else
                TPIdxCond(:,f,ll) = TPIdxCond(:,f,ll) & (bindYh == bindY(:,f));
                TPIdxSum(:,f,ll) = TPIdxSum(:,f,ll) + (bindYh == bindY(:,f));
            end

            %TPIdxCond(:,ll) = TPIdxCond(:,ll) & ((bindYh == bindY) & (bindA(:,2) >= threshP(l,ll)));
            %TPIdxCond(:,ll) = TPIdxCond(:,ll) & ( ((bindYh == bindY) & (bindA(:,2) >= threshP(l,ll))) | ((bindYh ~= bindY) & (bindA(:,1) < threshP(l,ll))) );
        end

    end
end

if threshVote
    for ll = 1:nThresh
        TPIdxCond(:,:,ll) = TPIdxSum(:,:,ll) >= threshVote;
    end
end

FNIdxCond = ~TPIdxCond;



TP = sum(TPIdxCond,1);
TN = zeros([1,nFolds,nThresh]);
FN = sum(FNIdxCond,1);
FP = zeros([1,nFolds,nThresh]);


%% Count all no-bind residue-base pairs
mNoBind = 0;
%mCur = 0;
ns = floor(n/scaleInFiles);

nTNold = 0;
sumActTNold = 0;
nFPold = 0;
sumActFPold = 0;

mCurAcc = 0;

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
            mNone = mNone + upBound + 1;

            for b = 1:upBound
                mCur = mCur + 1;
            end
        end
        if (lowBound >= 0) && (lowBound <= base_len)
            mNone = mNone + base_len - lowBound + 1;

            for b = lowBound:base_len
                mCur = mCur + 1;
            end
        end
        % no binds for a given residue
        if (upBound < 0) && (lowBound < 0)
            mNone = mNone + base_len;

            for b = 1:base_len
                mCur = mCur + 1;
            end
        end


    end

    fprintf('Loading %s- dat: %d/%d\n', dataTrIdxFile, i, n);


    %noBindX = zeros([mNone, m_in]);
    %noBindYh = categorical(zeros([mNone, 1]));
    noBindX = zeros([mCur, m_in]);
    noBindYh = categorical(zeros([mCur, 1]));

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


    if mCur
        fprintf('Predicting %s- dat: %d/%d\n', dataTrIdxFile,i, n);
    else
        fprintf('Empty %s- dat: %d/%d\n', dataTrIdxFile,i, n);
        continue;
    end

    if (maxNoBind) && (mCurAcc > maxNoBind)
        break;
    end
    mCurAcc = mCurAcc + mCur;


    if scaleNo == 0

        %TNIdx = ones([mCur, 1]);
        %FPIdx = ones([mCur, 1]);
        %TNIdxCond = ones([mCur, nThresh]);
        FPIdxCond = ones([mCur, nFolds, nThresh]);
        FPIdxSum = zeros([mCur, nFolds, nThresh]);

        noBindY = categorical(zeros([mCur, nFolds]));
        noBindA = zeros([mCur, 2, nFolds]);


        for l = 1:nNets
            for f = 1:nFolds

                fprintf('Predicting no-bind Net %d %d\n', l, f);

                % GPU on
                gpuDevice(1);
                reset(gpuDevice(1));

                cNet = cNets{l,f};
                [~, noBindY(:,f), noBindA(:,:,f)] = cNet.Predict(noBindX);
    
                % GPU off
                delete(gcp('nocreate'));
                gpuDevice([]);

                TNIdx = noBindYh(1:mCur) == noBindY(1:mCur, f);
                %TNIdx = TNIdx & (noBindYh(1:mCur) == noBindY(1:mCur,f));
                %FPIdx = FPIdx & (noBindYh(1:mCur) ~= noBindY(1:mCur,f));
                FPIdx = ~TNIdx;

                nTNcur = sum(TNIdx);
                sumActTNcur = sum(noBindA(TNIdx,1,f));

                nFPcur = sum(FPIdx);
                sumActFPcur = sum(noBindA(FPIdx,2,f));

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
                        FPIdxCond(:,f,ll) = FPIdxCond(:,f,ll) & (noBindA(1:mCur,2,f) >= threshP(l,ll));
                        FPIdxSum(:,f,ll) = FPIdxSum(:,f,ll) + (noBindA(1:mCur,2,f) >= threshP(l,ll));
                    else
                        FPIdxCond(:,f,ll) = FPIdxCond(:,f,ll) & (noBindYh(1:mCur) ~= noBindY(1:mCur,f));
                        FPIdxSum(:,f,ll) = FPIdxSum(:,f,ll) + (noBindYh(1:mCur) ~= noBindY(1:mCur,f));
                    end

                    %TNIdxCond(:,ll) = TNIdxCond(:,ll) & (noBindA(1:mCur,1) >= threshP(l,ll));
                    %TNIdxCond(:,ll) = TNIdxCond(:,ll) & ((noBindYh(1:mCur) == noBindY(1:mCur)) & (noBindA(1:mCur,1) > threshP(l,ll)));
                    %TNIdxCond(:,ll) = TNIdxCond(:,ll) & ( ((noBindYh(1:mCur) == noBindY(1:mCur)) & (noBindA(1:mCur,1) > threshP(l,ll))) | ((noBindYh(1:mCur) ~= noBindY(1:mCur)) & (noBindA(1:mCur,2) <= threshP(l,ll))) );
                end

            end
            
        end



        if threshVote
            for ll = 1:nThresh
                FPIdxCond(:,:,ll) = FPIdxSum(:,:,ll) >= threshVote;
            end
        end
        
        TNIdxCond = ~FPIdxCond;


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