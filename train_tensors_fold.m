function [cNets, mAllYes, mAllNo, Xcontr, Ycontr, Ncontr, t1, t2, noBindThresh] = train_tensors_fold(cNetTypes, nNets, nTrain, dataIdxDir, dataTrIdxFile, m_in, ...
    resWindowLen, resWindowWhole, resNum, baseWindowLen, baseWindowWhole, baseNum, bindScaleNo, noBindScaleNo, foldInFiles, noBindPerc)

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


    fprintf('Building %s+ dat: %d/%d\n', dataTrIdxFile, i, n);
end

%scaleInFiles = 0;

%% Count all no-bind residue-base pairs

mAllNo = zeros([foldInFiles,1]);
offFolds = 0;
for f = 1:foldInFiles

    mNone = 0;
    ns = floor(n/foldInFiles);
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

        fprintf('Counting %s- dat: %d/%d fold slice %d\n', dataTrIdxFile, offFolds+i, n, f);
    end


    if noBindScaleNo
        mAllNo(f) = floor(mAll*noBindScaleNo);
    else
        mAllNo(f) = mCur-1;
    end

    if bindScaleNo
        mAllYes = floor(mAll*bindScaleNo);
    else
        mAllYes = mAll;
    end

    dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.fs.', string(resWindowLen), '.', string(baseWindowLen),...
                    '.', string(mAllNo(f)), '.', string(f), '.', string(foldInFiles), '.mat');

    if isfile(dataTrNoBindFN)
        fprintf('Loading %s- dat: %d fold slice %d\n', dataTrNoBindFN, i, f);

        %load(dataTrNoBindFN, 'trNoBindM');
    else

    %trNoBindM = zeros([mNone, m_in]);
    trNoBindM = zeros([mCur-1, m_in]);

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

        fprintf('Building %s- dat: %d/%d fold slice %d\n', dataTrIdxFile, offFolds+i, n, f);
    end

    %if noBindScaleNo
    %    mAllNo = floor(mAll*noBindScaleNo);
    %else
    %    mAllNo = mCur;
    %end

    %if bindScaleNo
    %    mAllYes = floor(mAll*bindScaleNo);
    %else
    %    mAllYes = mAll;
    %end

    % Save fold slice of the non-bind data to save space
    dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.fs.', string(resWindowLen), '.', string(baseWindowLen),...
        '.', string(mAllNo(f)), '.', string(f), '.', string(foldInFiles), '.mat');

    if ~isfile(dataTrNoBindFN)
        fprintf('Saving %s- dat: %d fold slice %d\n', dataTrNoBindFN, i, f);

        save(dataTrNoBindFN, 'trNoBindM');
    end

    clear("trNoBindM");

    end


    offFolds = offFolds + ns;
end



%% Repeated retraining with new no-bind folds
[nNetTypes, ~] = size(cNetTypes);

cNets = cell([nNets*nNetTypes, 1]);


%nTrainMax = floor((mCur-mAllYes)/mAllNo);
%if nTrain == 0
%    nTrain = nTrainMax;
%end

% Save only necessary slice of the non-bind data to save space
%for i = 1:nTrain
%    dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.', string(resWindowLen), '.', string(baseWindowLen),...
%        '.', string(mAllNo), '.', string((f-1)*nTrain + i), '.mat');

%    if ~isfile(dataTrNoBindFN)
     %trNoBindBalM = trNoBindM(randperm(mCur, mAllNo), :);
%     trNoBindBalM = trNoBindM(randperm(mCur-mAllYes, mAllNo), :);
%     save(dataTrNoBindFN, 'trNoBindBalM');

%     fprintf('Saving %s- dat: %d fold %d\n', dataTrNoBindFN, i, f);
%    end
%end
%trNoBindLimM = trNoBindM(randperm(mCur, mAllNo*nTrain), :);

%%trNoBindThreshM = trNoBindM(randperm(mCur, mAllYes), :);
%trNoBindThreshM = trNoBindM(end-mAllYes:end, :);
%clear("trNoBindM");
%clear("trNoBindBalM");

%offFolds = offFolds + ns;

%end %f



noBindThresh = zeros([nNets*nNetTypes, 1]);
            
% GPU on
%gpuDevice(1);
%reset(gpuDevice(1));

t1 = clock();
for j = 1:nNetTypes

    % Sets new current model in heterogenious model list
    cNet = cNetTypes{j};

    mWhole = mAllYes + mAllNo(1)*2;
    cNet.mb_size = 2^floor(log2(mWhole)-4);


    for l = 1:nNets

        % Resets weights
        cNet = cNet.Create();
    



        %Saved name
        %cNetName = strcat(dataIdxDir,'/prot.', string(cNet.name), '.', string(cNet.mb_size), '.', string(cNet.max_epoch),...
        %    '.', string(resWindowLen), '.', string(baseWindowLen), '.', string(mAllYes), '.', string(mAllNo),...
        %    '.', string(l), '.', string(nTrain), '.mat');


        %if ~isfile(cNetName)

            for k = 1:foldInFiles
                for m = k+1:foldInFiles

                    mWhole = mAllYes + mAllNo(k) + mAllNo(m);

                    trMX = zeros([mWhole, m_in]);
                    trMY = categorical(zeros([mWhole, 1]));

                    %fprintf('Loading %s+ dat: fold slice %d %d\n', dataTrNoBindFN, k, m);

                    for k = 1:bindScaleNo
                        trMX(1+(k-1)*mAll:k*mAll,:) = trBindM;
                        trMY(1+(k-1)*mAll:k*mAll,:) = trBindY;
                    end


                    fprintf('Loading %s- dat: fold slice %d %d\n', dataTrNoBindFN, k, m);

                    dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.fs.', string(resWindowLen), '.', string(baseWindowLen),...
                    '.', string(mAllNo(k)), '.', string(k), '.', string(foldInFiles), '.mat');

                    load(dataTrNoBindFN, 'trNoBindM');

                    trNoBindY = categorical(zeros([mAllNo(k), 1]));

                    trMX(mAllYes+1:mAllYes+mAllNo(k),:) = trNoBindM;
                    trMY(mAllYes+1:mAllYes+mAllNo(k),:) = trNoBindY;

                    clear("trNoBindM");


                    dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.fs.', string(resWindowLen), '.', string(baseWindowLen),...
                    '.', string(mAllNo(m)), '.', string(m), '.', string(foldInFiles), '.mat');

                    load(dataTrNoBindFN, 'trNoBindM');

                    trNoBindY = categorical(zeros([mAllNo(m), 1]));

                    trMX(mAllYes+mAllNo(k)+1:end,:) = trNoBindM;
                    trMY(mAllYes+mAllNo(k)+1:end,:) = trNoBindY;      

                    clear("trNoBindM");                    
                    clear("trNoBindY");



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
            end

            %save(cNetName, 'cNet');
        
        %else

        %    load(cNetName, 'cNet');
        %    fprintf('Loading Net type %d, Net instance %d\n', j, l);
        %end

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