function [cNets, mAllYes, mAllNo, Xcontr, Ycontr, Ncontr, t1, t2, noBindThresh] = train_tensors(cNetTypes, nNets, nTrain, dataIdxDir, dataTrIdxFile, m_in, ...
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

    fprintf('Counting %s+ dat: %d/%d\n', dataTrIdxFile, i, n);
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


    fprintf('Loading %s+ dat: %d/%d\n', dataTrIdxFile, i, n);
end


%% Count all no-bind residue-base pairs
offFolds = 0;
for f = 1:scaleInFiles

mNone = 0;
ns = floor(n/scaleInFiles);
for i = 1:ns
    dataDatFile = trIdxM(offFolds+i, 5);
    dataDatFN = strcat(dataIdxDir,'/',dataDatFile);
    trDatM = readmatrix(dataDatFN, FileType='text', Delimiter=' ');
    [m, ~] = size(trDatM);

    %
    resFaFile = trIdxM(offFolds+i, 1);
    resFaFN = strcat(dataIdxDir,'/',resFaFile);
    resFaMstr = readmatrix(resFaFN, FileType='text', OutputType='string', Delimiter=' ');
    resFaM = double(char(resFaMstr(2))) - 64;
    [~, res_len] = size(resFaM);

    baseFaFile = trIdxM(offFolds+i,3);
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

    fprintf('Counting %s- dat: %d/%d fold%d\n', dataTrIdxFile, offFolds+i, n, f);
end

trNoBindM = zeros([mNone, m_in]);

%% Loading all no bind residue-base pairs
mCur = 1;
for i = 1:ns
    dataDatFile = trIdxM(offFolds+i, 5);
    dataDatFN = strcat(dataIdxDir,'/',dataDatFile);
    trDatM = readmatrix(dataDatFN, FileType='text', Delimiter=' ');
    [m, ~] = size(trDatM);

    %
    resFaFile = trIdxM(offFolds+i, 1);
    resFaFN = strcat(dataIdxDir,'/',resFaFile);
    resFaMstr = readmatrix(resFaFN, FileType='text', OutputType='string', Delimiter=' ');
    resFaM = double(char(resFaMstr(2))) - 64;
    [~, res_len] = size(resFaM);

    baseFaFile = trIdxM(offFolds+i, 3);
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

    fprintf('Loading %s- dat: %d/%d fold %d\n', dataTrIdxFile, offFolds+i, n, f);
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
[nNetTypes, ~] = size(cNetTypes);

cNets = cell([nNets*nNetTypes, 1]);
mWhole = mAllYes + mAllNo;

nTrainMax = floor((mCur-mAllYes)/mAllNo);
if nTrain == 0
    nTrain = nTrainMax;
end

% Save only necessary slice of the non-bind data to save space
for i = 1:nTrain
    dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.', string(resWindowLen), '.', string(baseWindowLen),...
        '.', string(mAllNo), '.', string((f-1)*nTrain + i), '.mat');

    if ~isfile(dataTrNoBindFN)
     %trNoBindBalM = trNoBindM(randperm(mCur, mAllNo), :);
     trNoBindBalM = trNoBindM(randperm(mCur-mAllYes, mAllNo), :);
     save(dataTrNoBindFN, 'trNoBindBalM');

     fprintf('Saving %s- dat: %d fold %d\n', dataTrNoBindFN, i, f);
    end
end
%trNoBindLimM = trNoBindM(randperm(mCur, mAllNo*nTrain), :);

%trNoBindThreshM = trNoBindM(randperm(mCur, mAllYes), :);
trNoBindThreshM = trNoBindM(end-mAllYes:end, :);
clear("trNoBindM");
clear("trNoBindBalM");

offFolds = offFolds + ns;

end %f


noBindThresh = zeros([nNets*nNetTypes, 1]);
            
% GPU on
%gpuDevice(1);
%reset(gpuDevice(1));

t1 = clock();
for j = 1:nNetTypes

    % Sets new current model in heterogenious model list
    cNet = cNetTypes{j};

    cNet.mb_size = 2^floor(log2(mWhole)-4);


    for l = 1:nNets

        % Resets weights
        cNet = cNet.Create();
    
        trMX = zeros([mWhole, m_in]);
        trMY = categorical(zeros([mWhole, 1]));

        for k = 1:bindScaleNo
            trMX(1+(k-1)*mAll:k*mAll,:) = trBindM;
            trMY(1+(k-1)*mAll:k*mAll,:) = trBindY;
        end


        %Saved name
        cNetName = strcat(dataIdxDir,'/prot.', string(cNet.name), '.', string(cNet.mb_size), '.', string(cNet.max_epoch),...
            '.', string(resWindowLen), '.', string(baseWindowLen), '.', string(mAllYes), '.', string(mAllNo),...
            '.', string(l), '.', string(nTrain), '.mat');


        if ~isfile(cNetName)

            for k = 1:nTrain*scaleInFiles
                %trNoBindBalM = trNoBindLimM(1+(k-1)*mAllNo:k*mAllNo, :);
                dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.', string(resWindowLen), '.', string(baseWindowLen),...
                    '.', string(mAllNo), '.', string(k), '.mat');
                load(dataTrNoBindFN, 'trNoBindBalM');

                trNoBindY = categorical(zeros([mAllNo, 1]));

                %
                trMX(mAllYes+1:end,:) = trNoBindBalM;
                trMY(mAllYes+1:end,:) = trNoBindY;

                clear("trNoBindBalM");

                fprintf('Training Net type %d, Net instance %d, Train fold %d\n', j, l, k);

                % GPU on
                gpuDevice(1);
                reset(gpuDevice(1));

                % Updates weights from previous training with previous slice of no-bind data
                cNet = cNet.Train(trMX, trMY);
                %cNets{(j-1)*nNets + l} = cNet;

                % GPU off
                delete(gcp('nocreate'));
                gpuDevice([]);
            end

            save(cNetName, 'cNet');
        
        else

            load(cNetName, 'cNet');
            fprintf('Loading Net type %d, Net instance %d\n', j, l);
        end

        cNets{(j-1)*nNets + l} = cNet;


        %% Find threshold for given percentle of FP no-bind predictions

        if noBindPerc
            fprintf('Predicting no-bind destribition Net type %d, Net instance %d, Train fold %d\n', j, l, k);

            %noBindX = trNoBindLimM(mAllNo*nTrain+1:end, :);
            
            % GPU on
            gpuDevice(1);
            reset(gpuDevice(1));

            [~, noBindY, noBindA] = cNet.Predict(trNoBindThreshM);
    
            % GPU off
            delete(gcp('nocreate'));
            gpuDevice([]);

            curThresh = 0;
            cntFP = floor(mAllYes * (100 - noBindPerc) / 100);
            noBindAs = sort(noBindA(:,2), "descend");
            for k = 1:mAllYes
                curThresh = noBindAs(k);
                if k >= cntFP
                    break;
                end
            end
            noBindThresh((j-1)*nNets + l) = curThresh;

            %noBindThresh((j-1)*nNets + l) = prctile(noBindA((noBindY == categorical(1)), 2), noBindPerc);
        else
            noBindThresh((j-1)*nNets + l) = 0;
        end

    end
end
t2 = clock();
            
% GPU off
%delete(gcp('nocreate'));
%gpuDevice([]);

%% Convert input into strings (for sorting, uniqueness and contradiction detection)
Xcontr = []; 
Ycontr = []; 
Ncontr = 0;
%[Xcontr, Ycontr, Ncontr] = find_doubles(trMX, trMY, mAll, mAllNo, resWindowWhole, resNum, baseWindowLen, baseWindowWhole, baseNum);


end