function [cNets, mAllYes, mAllNo, Xcontr, Ycontr, Ncontr, t1, t2, noBindThresh] = train_tensors_fold(cNetTypes, nNets, nTrain, dataIdxDir, dataTrIdxFile, m_in, ...
    resWindowLen, resWindowWhole, resNum, baseWindowLen, baseWindowWhole, baseNum, bindScaleNo, noBindScaleNo, foldInFiles, noBindPerc, useDB)

dataTrIdxFN = strcat(dataIdxDir,'/',dataTrIdxFile);

trIdxM = readmatrix(dataTrIdxFN, FileType='text', OutputType='string', Delimiter=' ');
[n, ~] = size(trIdxM);

randFold = 1; %1 0; CHANGE
dbLoaded = 1;
dbBuff = 200000;
epochsIni = -1;

if useDB
    conn = apacheCassandra('sielicki','sS543228$','PortNumber',9042);

    query2 = 'USE protein;';
    results = executecql(conn, query2);

    tableName = strcat('noBind_', string(resWindowWhole), '_', string(baseWindowWhole));
    insTpl = strcat("INSERT INTO ", tableName, " (pk");

    %for i = 1:resWindowWhole
    %    insTpl = strcat(insTpl, ", r", string(i));
    %end

    %for i = 1:baseWindowWhole
    %    insTpl = strcat(insTpl, ", b", string(i));
    %end

    insTpl = strcat(insTpl, ", val) ");
    %insTpl = strcat(insTpl, ") ");
end


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
mCurAll = 1;
mCurAllc = 1;
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
                    mCurAllc = mCurAllc + 1;
                end
            end
            if (lowBound >= 0) && (lowBound <= base_len)
                mNone = mNone + base_len - lowBound + 1;

                for b = lowBound:base_len
                    mCur = mCur + 1;
                    mCurAllc = mCurAllc + 1;
                end
            end
            % no binds for a given residue
            if (upBound < 0) && (lowBound < 0)
                mNone = mNone + base_len;

                for b = 1:base_len
                    mCur = mCur + 1;
                    mCurAllc = mCurAllc + 1;
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

    %if isfile(dataTrNoBindFN)
    %    if ~useDB
    %     fprintf('Found %s- dat: %d fold slice %d\n', dataTrNoBindFN, i, f);

         % DEBUG!!! Not needed, will load later
    %     %load(dataTrNoBindFN, 'trNoBindM');
    %    end
    %else

    %trNoBindM = zeros([mNone, m_in]);
    if ~useDB
        trNoBindM = zeros([mCur-1, m_in]);
    else
        sl0 = zeros([1, m_in]);
    end


    if ~dbLoaded

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

                    if ~useDB
                        trNoBindM(mCur,:) = bind_1hot(resFaM, baseFaM, trNoBindM(mCur,:), resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
                    else
                        query4 = strcat(insTpl, " VALUES(");

                        sl = bind_1hot(resFaM, baseFaM, sl0, resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
                        sls = string(sl);
                        stst = strjoin(sls);
                        
                        query4 = strcat(query4, string(mCurAll), ", '", stst, "');");
                        results = executecql(conn, query4);
                        %x = str2num(stst);
                    end
                    
                    mCur = mCur + 1;
                    mCurAll = mCurAll + 1;
                end
            end
            if (lowBound >= 0) && (lowBound <= base_len)
                for b = lowBound:base_len

                    if ~useDB
                        trNoBindM(mCur,:) = bind_1hot(resFaM, baseFaM, trNoBindM(mCur,:), resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
                    else
                        query4 = strcat(insTpl, " VALUES(");

                        sl = bind_1hot(resFaM, baseFaM, sl0, resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
                        sls = string(sl);
                        stst = strjoin(sls);
                        
                        query4 = strcat(query4, string(mCurAll), ", '", stst, "');");
                        results = executecql(conn, query4);
                        %x = str2num(stst);
                    end

                    
                    mCur = mCur + 1;
                    mCurAll = mCurAll + 1;
                end
            end
            % no binds for a given residue
            if (upBound < 0) && (lowBound < 0)
                for b = 1:base_len

                    if ~useDB
                        trNoBindM(mCur,:) = bind_1hot(resFaM, baseFaM, trNoBindM(mCur,:), resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
                    else
                        query4 = strcat(insTpl, " VALUES(");

                        sl = bind_1hot(resFaM, baseFaM, sl0, resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r-1, b-1, mCur);
                        sls = string(sl);
                        stst = strjoin(sls);
                        
                        query4 = strcat(query4, string(mCurAll), ", '", stst, "');");
                        results = executecql(conn, query4);
                        %x = str2num(stst);
                    end


                    mCur = mCur + 1;
                    mCurAll = mCurAll + 1;
                end
            end

        end

        fprintf('Building %s- dat: %d/%d fold slice %d\n', dataTrIdxFile, offFolds+i, n, f);
    end


    if ~useDB
        % Save fold slice of the non-bind data to save space
        dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.fs.', string(resWindowLen), '.', string(baseWindowLen),...
                    '.', string(mAllNo(f)), '.', string(f), '.', string(foldInFiles), '.mat');

        if ~isfile(dataTrNoBindFN)
            fprintf('Saving %s- dat: %d fold slice %d\n', dataTrNoBindFN, i, f);

            save(dataTrNoBindFN, 'trNoBindM');
        end

        clear("trNoBindM");
    end

    end %dbLoaded

    %end % Load/No load data slice


    offFolds = offFolds + ns;
end



%% Repeated retraining with new no-bind folds
[nNetTypes, ~] = size(cNetTypes);

folds = foldInFiles * floor((foldInFiles-1)/2);

if nTrain == 1
    cNets = cell([nNets*nNetTypes, folds]);
else
    cNets = cell([nNets*nNetTypes, 1]);
end



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
%Number of Net types in ensemble
for j = 1:nNetTypes

    % Sets new current model in heterogenious model list
    cNet = cNetTypes{j};

    mWhole = mAllYes + mAllNo(1)*2;
    %cNet.mb_size = 2^floor(log2(mWhole)-4);
    %cNet.mb_size = 2^floor(log2(mWhole)-5); %lrrelu CHANGE
    %cNet.mb_size = 2^floor(log2(mWhole)-6); %lrrelulqp layers
    cNet.mb_size = 2^floor(log2(mWhole)-7); %sin q
    %cNet.mb_size = 2^floor(log2(mWhole)-17);

    %Number of retrains on Big Data (not fitting into memory)
    if nTrain ~= 1
        epochsTarget = 1; %1
        if foldInFiles ~= 1
            nReTrain = floor(cNet.max_epoch/(foldInFiles-1)/epochsTarget);
        else
            nReTrain = floor(((mCurAllc-1) / cNet.mb_size * cNet.max_epoch));
        end
    end



    %Number of Nets of the same type in ensemble
    for l = 1:nNets


        if nTrain == 1
    
            fold = 0;
            for k = 1:foldInFiles
                for m = k+1:foldInFiles

                    fold = fold + 1;

                    %Exact saved name
                    epochsTarget = cNet.max_epoch;
                    cNetName = strcat(dataIdxDir,'/prot.', string(cNet.name), '.', string(cNet.mb_size), '.', string(cNet.max_epoch),...
                            '.', string(resWindowLen), '.', string(baseWindowLen), '.', string(mAllYes), '.', string(mAllNo(k)), ...
                            '.', string(mAllNo(m)), '.', string(l), '.', string(k), '.', string(m), '.mat');
                    if ~isfile(cNetName)


                        %Not exact epochs match
                        cNetWC = strcat(dataIdxDir,'/prot.', string(cNet.name), '.', string(cNet.mb_size), '.*',...
                            '.', string(resWindowLen), '.', string(baseWindowLen), '.', string(mAllYes), '.', string(mAllNo(k)), ...
                            '.', string(mAllNo(m)), '.', string(l), '.', string(k), '.', string(m), '.mat');
                        cNetFNames = dir(cNetWC);
                        %cNetNames = ls(cNetWC);
                        [nNames,~] = size(cNetFNames);

                        dEpoch = -1;
                        if nNames > 0
                            cNetNames = ls(cNetWC);
                            cNetFNameLoad = strcat(cNetFNames(1).folder,'/',cNetFNames(1).name);
                            cNetNameLoad = cNetNames(1,:);
                            tokens = split(cNetNameLoad,'.');
                            epochs = str2num(tokens{4});
                            dEpoch = epochsTarget - epochs;
                        end

                        if dEpoch > 0
                            % Reset epochs
                            fprintf('Loading %s Net type %d, Net instance %d, Train folds %d %d\n', cNetFNameLoad, j, l, k, m);
                            load(cNetFNameLoad, 'cNet');

                            cNet.options = trainingOptions('adam', ...
                                'ExecutionEnvironment','auto',... %'parallel',...
                                'Shuffle', 'every-epoch',...
                                'MiniBatchSize', cNet.mb_size, ...
                                'InitialLearnRate', cNet.ini_rate, ...
                                'MaxEpochs',dEpoch);
                        else
                            % Resets weights
                            cNet = cNet.Create();
                        end
                

                        mWhole = mAllYes + mAllNo(k) + mAllNo(m);

                        trMX = zeros([mWhole, m_in],'single');
                        trMY = categorical(zeros([mWhole, 1]));


                        for i = 1:bindScaleNo
                            trMX(1+(i-1)*mAll:i*mAll,:) = trBindM;
                            trMY(1+(i-1)*mAll:i*mAll,:) = trBindY;
                        end


                        %% No bind data load
                        if ~useDB
                            dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.fs.', string(resWindowLen), '.', string(baseWindowLen),...
                                    '.', string(mAllNo(k)), '.', string(k), '.', string(foldInFiles), '.mat');

                            fprintf('Loading %s- dat: fold slice %d %d\n', dataTrNoBindFN, k, m);


                            load(dataTrNoBindFN, 'trNoBindM');

                            trNoBindY = categorical(zeros([mAllNo(k), 1]));

                            trMX(mAllYes+1:mAllYes+mAllNo(k),:) = single(trNoBindM);
                            trMY(mAllYes+1:mAllYes+mAllNo(k),:) = trNoBindY;

                            clear("trNoBindM");


                            dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.fs.', string(resWindowLen), '.', string(baseWindowLen),...
                                    '.', string(mAllNo(m)), '.', string(m), '.', string(foldInFiles), '.mat');

                            fprintf('Loading %s- dat: fold slice %d %d\n', dataTrNoBindFN, k, m);


                            load(dataTrNoBindFN, 'trNoBindM');

                            trNoBindY = categorical(zeros([mAllNo(m), 1]));

                            trMX(mAllYes+mAllNo(k)+1:end,:) = trNoBindM;
                            trMY(mAllYes+mAllNo(k)+1:end,:) = trNoBindY;      

                            clear("trNoBindM");                    
                            clear("trNoBindY");
                        else
                            selTpl = strcat("SELECT val FROM ", tableName, " WHERE pk IN (");

                            pn = randperm(mCurAllc-1, cNet.mb_size);
                            ps = string(pn);
                            pst = strjoin(ps, ", ");
                            selTpl = strcat(selTpl, pst);

                            query5 = strcat(selTpl, ");");
                            results = executecql(conn, query5);
                        end

                        %% Training
                        fprintf('Training Net type %d, Net instance %d, Train folds %d %d\n', j, l, k, m);
                    

                        % GPU on
                        gpuDevice(1);
                        reset(gpuDevice(1));

                        % Updates weights from previous training with previous slice of no-bind data
                        cNet = cNet.Train(trMX, trMY);


                        % GPU off
                        delete(gcp('nocreate'));
                        gpuDevice([]);


                        cNet.options = trainingOptions('adam', ...
                            'ExecutionEnvironment','auto',... %'parallel',...
                            'Shuffle', 'every-epoch',...
                            'MiniBatchSize', cNet.mb_size, ...
                            'InitialLearnRate', cNet.ini_rate, ...
                            'MaxEpochs', epochsTarget);
                        cNet.max_epoch = epochsTarget;
                        save(cNetName, 'cNet');

                        clear("trMX");                    
                        clear("trMY");
                    else

                        fprintf('Loading %s Net type %d, Net instance %d, Train folds %d %d\n', cNetName, j, l, k, m);
                        load(cNetName, 'cNet');
                    end

                    cNets{(j-1)*nNets + l, fold} = cNet;

                end
            end

        %Retrain for Big Data
        else

            %DEBUG!!!
                            %selTpl = strcat("SELECT val FROM ", tableName, " WHERE pk IN (");

                            %pn = randperm(mCurAllc-1, 200000); %cNet.mb_size);
                            %ps = string(pn);
                            %pst = strjoin(ps, ", ");
                            %selTpl = strcat(selTpl, pst);

                            %query5 = strcat(selTpl, ");");
                            %results = executecql(conn, query5); %10x1






            fold = 1;
            n1 = 1;
            k1 = 1;

            %epochsTarget = 1;
            cNetWC = strcat(dataIdxDir,'/prot.bd.', string(cNet.name), '.', string(cNet.mb_size), '.', string(cNet.max_epoch),...
                            '.', string(resWindowLen), '.', string(baseWindowLen), '.', string(mAllYes), '.', ...
                            string(l), '.*.*.*.mat'); % k.m.n.mat in wildcards

            cNetFNames = dir(cNetWC);
            [nNames,~] = size(cNetFNames);


            epochs = -1;
            if nNames > 0
                cNetNames = ls(cNetWC);
                cNetFNameLoad = strcat(cNetFNames(1).folder,'/',cNetFNames(1).name);
                cNetNameLoad = cNetNames(1,:);
                tokens = split(cNetNameLoad,'.');
                epochs = str2num(tokens{5});
                epochsIni = epochs;
                no_k = str2num(tokens{6});
                no_m = str2num(tokens{7});
                l1 = str2num(tokens{9});
                k1 = str2num(tokens{10});
                m1 = str2num(tokens{11});
                n1 = str2num(tokens{12});

                cNetName = strcat(dataIdxDir,'/prot.bd.', string(cNet.name), '.', string(cNet.mb_size), '.', string(cNet.max_epoch),...
                        '.', string(resWindowLen), '.', string(baseWindowLen), '.', string(mAllYes),...
                        '.', string(l1), '.', string(k1), '.', string(m1), '.', string(n1), '.mat');
                if ~isfile(cNetName)
                    epochs = -1;
                end
            end

            LoopBr = 0;
            inLoopBr = 0;
            if epochs > 0
                % Load partially trained model
                fprintf('Loading %s Net type %d, Net instance %d, Train folds %d %d %d\n', cNetName, j, l, k1, m1, n1);
                load(cNetName, 'cNet');
                cNet.max_epoch = epochs;
                cNets{(j-1)*nNets + l, fold} = cNet;
                m1 = m1 + 1;
                LoopBr = 1;
                inLoopBr = 1;
                if m1 > foldInFiles
                    k1 = k1 + 1;
                    m1 = k1 + 1;
                end
                if k1 > (foldInFiles-1)
                    k1 = 1;
                    m1 = k1 + 1;
                    n1 = n1 + 1;
                end

            else
                % Resets weights
                cNet = cNet.Create();
            end

            cNet.options = trainingOptions('adam', ...
                            'ExecutionEnvironment','auto',... %'parallel',...
                            'Shuffle', 'every-epoch',...
                            'MiniBatchSize', cNet.mb_size, ...
                            'InitialLearnRate', cNet.ini_rate, ...
                            'MaxEpochs', epochsTarget);



            %fold = 0;
            for n = n1:nReTrain
                %fold = 0;
                for k = k1:foldInFiles

                    if (~LoopBr) || (~inLoopBr)
                        m1 = k+1;
                    end
                    for m = m1:foldInFiles

                        if randFold
                            while true
                                kr = floor(rand()*foldInFiles)+1;
                                if kr <=  foldInFiles
                                    break;
                                end
                            end

                            while true
                                mr = floor(rand()*foldInFiles)+1;
                                if (mr <=  foldInFiles) && (mr ~= kr)
                                    break;
                                end
                            end
                        else
                            kr = k;
                            mr = m;
                        end

                        %fold = 1;
                        cNetNameNew = strcat(dataIdxDir,'/prot.bd.', string(cNet.name), '.', string(cNet.mb_size), '.', string(cNet.max_epoch),...
                                    '.', string(resWindowLen), '.', string(baseWindowLen), '.', string(mAllYes), ...
                                    '.', string(l), '.', string(k), '.', string(m), '.', string(n), '.mat');

                        if ~useDB
                            mWhole = mAllYes + mAllNo(kr) + mAllNo(mr);
                        else
                            kBuff = floor(mAllNo(kr) / dbBuff);
                            mAllNoBuffK = kBuff * dbBuff;
                            mBuff = floor(mAllNo(mr) / dbBuff);
                            mAllNoBuffM = mBuff * dbBuff;
                            mWhole = mAllYes + mAllNoBuffK + mAllNoBuffM;
                        end


                        trMX = zeros([mWhole, m_in],'single');
                        trMY = categorical(zeros([mWhole, 1]));


                        for i = 1:bindScaleNo
                            trMX(1+(i-1)*mAll:i*mAll,:) = trBindM;
                            trMY(1+(i-1)*mAll:i*mAll,:) = trBindY;
                        end


                        %% No binnd load
                        if ~useDB
                            dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.fs.', string(resWindowLen), '.', string(baseWindowLen),...
                                        '.', string(mAllNo(kr)), '.', string(kr), '.', string(foldInFiles), '.mat');

                            fprintf('Loading %s- dat: fold slice %d %d\n', dataTrNoBindFN, kr, mr);


                            load(dataTrNoBindFN, 'trNoBindM');

                            trNoBindY = categorical(zeros([mAllNo(kr), 1]));

                            trMX(mAllYes+1:mAllYes+mAllNo(kr),:) = trNoBindM;
                            trMY(mAllYes+1:mAllYes+mAllNo(kr),:) = trNoBindY;

                            clear("trNoBindM");


                            dataTrNoBindFN = strcat(dataIdxDir,'/',dataTrIdxFile, '.nobind.fs.', string(resWindowLen), '.', string(baseWindowLen),...
                                        '.', string(mAllNo(mr)), '.', string(mr), '.', string(foldInFiles), '.mat');

                            fprintf('Loading %s- dat: fold slice %d %d\n', dataTrNoBindFN, kr, mr);


                            load(dataTrNoBindFN, 'trNoBindM');

                            trNoBindY = categorical(zeros([mAllNo(mr), 1]));

                            trMX(mAllYes+mAllNo(kr)+1:end,:) = trNoBindM;
                            trMY(mAllYes+mAllNo(kr)+1:end,:) = trNoBindY;      

                            clear("trNoBindM");                    
                            clear("trNoBindY");
                        else
                               
                            fprintf('Loading DB- dat: fold slice %d %d\n', kr, mr);

                            selTpl = strcat("SELECT val FROM ", tableName, " WHERE pk IN (");

                            for i=1:kBuff

                                pn = randperm(mCurAllc-1, dbBuff);
                                ps = string(pn);
                                pst = strjoin(ps, ", ");
                                query5 = strcat(selTpl, pst);

                                query5 = strcat(query5, ");");
                                results = executecql(conn, query5);

                                
                                fprintf('Loading DB- dat k: fold slice %d %d %d\n', kr, mr, i);

                                x = zeros([dbBuff, m_in]);
                                parfor ii=1:dbBuff
                                    x(ii,:) = str2num(results(ii,1).Variables);
                                    %trMX(mAllYes+ii+(i-1)*dbBuff,:) = str2num(results(ii,1).Variables);
                                end
                                trMX(mAllYes+1+(i-1)*dbBuff:mAllYes+i*dbBuff,:) = x;
                                %trMX(mAllYes+1+(i-1)*dbBuff:mAllYes+i*dbBuff,:) = str2double(split(results.Variables));
                            end

                            trMY(mAllYes+1:mAllYes+mAllNoBuffK,:) = categorical(zeros([mAllNoBuffK, 1]));



                            for i=1:mBuff

                                pn = randperm(mCurAllc-1, dbBuff);
                                ps = string(pn);
                                pst = strjoin(ps, ", ");
                                query5 = strcat(selTpl, pst);

                                query5 = strcat(query5, ");");
                                results = executecql(conn, query5);

                                
                                fprintf('Loading DB- dat m: fold slice %d %d %d\n', kr, mr, i);

                                x = zeros([dbBuff, m_in]);
                                parfor ii=1:dbBuff
                                    x(ii,:) = str2num(results(ii,1).Variables);
                                    %trMX(mAllYes+mAllNoBuffK+ii+(i-1)*dbBuff,:) = str2num(results(ii,1).Variables);
                                end
                                trMX(mAllYes+mAllNoBuffK+1+(i-1)*dbBuff:mAllYes+mAllNoBuffK+i*dbBuff,:) = x;
                                %trMX(mAllYes+mAllNoBuffK+1+(i-1)*dbBuff:mAllYes+mAllNoBuffK+i*dbBuff,:) = str2double(split(results.Variables));
                            end

                            trMY(mAllYes+mAllNoBuffK+1:end,:) = categorical(zeros([mAllNoBuffM, 1]));

                            clear("results");
                            clear("pn");
                            clear("ps");
                            clear("pst");
                            clear("query5");
                            clear("x");

                        end



                        fprintf('Training Net type %d, Net instance %d, Train folds %d %d (%d %d) %d\n', j, l, k, m, kr, mr, n);
                    

                        % GPU on
                        gpuDevice(1);
                        reset(gpuDevice(1));

                        % Updates weights from previous training with previous slice of no-bind data
                        cNet = cNet.Train(trMX, trMY);


                        % GPU off
                        delete(gcp('nocreate'));
                        gpuDevice([]);


                        cNet.options = trainingOptions('adam', ...
                            'ExecutionEnvironment','auto',... %'parallel',...
                            'Shuffle', 'every-epoch',...
                            'MiniBatchSize', cNet.mb_size, ...
                            'InitialLearnRate', cNet.ini_rate, ...
                            'MaxEpochs', epochsTarget);

                        %cNet.max_epoch = epochsTarget;

                        fprintf('Saving %s Net type %d, Net instance %d, Train folds %d %d (%d %d) %d\n', cNetNameNew, j, l, k, m, kr, mr, n);
                        save(cNetNameNew, 'cNet');

                        if exist('cNetName','var')
                            fprintf('Deleting %s Net type %d, Net instance %d, Initial train folds %d %d %d\n', cNetName, j, l, k1, m1, n1);
                            delete(cNetName);
                        end
                        %Save old name
                        cNetName = cNetNameNew;

                        clear("trMX");                    
                        clear("trMY");


                        cNets{(j-1)*nNets + l, fold} = cNet;

                    end

                    if inLoopBr
                        inLoopBr = 0;
                        %m1 = k+1;
                    end

                end

                %Restore regular iteration from the saved checkpoint model
                if LoopBr
                    k1 = 1;
                    LoopBr = 0;
                    continue;
                end

            end

        end



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