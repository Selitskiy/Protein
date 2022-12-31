function [trMX, trMY, mAll, mAllNo, Xcontr, Ycontr, Ncontr] = build_tensors(dataIdxDir, dataTrIdxFile, m_in, ...
    resWindowLen, resWindowWhole, resNum, baseWindowLen, baseWindowWhole, baseNum, scaleNo, scaleInFiles)

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

if scaleNo
    mAllNo = mAll*scaleNo;
else
    mAllNo = mCur;
end
trNoBindBalM = trNoBindM(randperm(mCur, mAllNo), :);
clear("trNoBindM");

trNoBindY = categorical(zeros([mAllNo, 1]));

%%
mWhole = mAll + mAllNo;
trMX = zeros([mWhole, m_in]);
trMX(1:mAll,:) = trBindM;
trMX(mAll+1:end,:) = trNoBindBalM;

trMY = categorical(zeros([mWhole, 1]));
trMY(1:mAll,:) = trBindY;
trMY(mAll+1:end,:) = trNoBindY;


%% Convert input into strings (for sorting, uniqueness and contradiction detection)
mr_in = resNum * resWindowWhole;
mb_in = baseNum * baseWindowWhole;
mTrBind = mAll;
mTrNoBind = mAllNo;
Xstr = repmat( string(repmat(' ', [1 resWindowWhole + resWindowWhole])), [mTrBind+mTrNoBind 1] );
for i = 1:mTrBind+mTrNoBind
    for j = 1:resWindowWhole
        for k = 1:resNum
            enc = trMX(i, (j-1)*resNum+k);
            if enc
                Xstr{i}(j) = char(k + 64);
                break;
            end
        end
    end

     for j = 1:baseWindowWhole
        for k = 1:baseNum
            enc = trMX(i, mr_in+(j-1)*baseNum+k);
            if enc
                if k == 1
                    Xstr{i}(resWindowWhole+j) = 'A';
                elseif k == 2
                    Xstr{i}(resWindowWhole+j) = 'C';
                elseif k == 3
                    Xstr{i}(resWindowWhole+j) = 'G';
                else
                    Xstr{i}(resWindowWhole+j) = 'U';
                end
                break;
            end
        end
    end   
    fprintf('Xstr(%d)=%s\n', i, Xstr(i));
end

%% Sort, unique, contradictory verdicts
[XstrS, I] = sort(Xstr);
YS = trMY(I);
[mTrS, ~] = size(XstrS);

XstrU = unique(XstrS);
[mTrU, ~] = size(XstrU);
YU = zeros([mTrU, 1]);
NU = zeros([mTrU, 1]);

i = 1;
j = i;
XstrCur = XstrS(i);
XstrCurN = 1;
YstrCur = double(YS(j))-1;
for i = 2:mTrU
    while j <= mTrS
        j = j + 1;
        if strcmp(XstrCur, XstrS(j))
            YstrCur = (YstrCur * XstrCurN + double(YS(j))-1) / (XstrCurN + 1);
            XstrCurN = XstrCurN + 1;
        else
            YU(i-1) = YstrCur;
            NU(i-1) = XstrCurN;

            XstrCur = XstrS(i);
            XstrCurN = 1;
            YstrCur = double(YS(j))-1;
            break;
        end
    end
end

Xcontr = XstrU((YU < 1) & (YU > 0));
Ycontr = YU((YU < 1) & (YU > 0));
Ncontr = NU((YU < 1) & (YU > 0));

end