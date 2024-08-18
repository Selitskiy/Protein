
%% Clear everything 
clearvars -global;
clear all; close all; clc;

addpath('~/ANNLib/');
addpath('~/Protein/');

% Mem cleanup
%ngpu = gpuDeviceCount();
%for i=1:ngpu
%    reset(gpuDevice(i));
%end

useDB = 1; %1;



%% General config

% Amino residue frame one-side length
resWindowLen = 23; %13
resWindowWhole = 2*resWindowLen + 1;
resNum = 26;

% RNA base frame one-side length
baseWindowLen = 23; %13
baseWindowWhole = 2*baseWindowLen + 1;
baseNum = 4;


if useDB
    conn = apacheCassandra('sielicki','sS543228$','PortNumber',9042);
    %t = tablenames(conn);
    query1 = "CREATE KEYSPACE IF NOT EXISTS protein WITH replication = {'class': 'SimpleStrategy', 'replication_factor' : 3};";
    results = executecql(conn, query1);

    query2 = 'USE protein;';
    results = executecql(conn, query2);

    tableName = strcat('noBind_', string(resWindowWhole), '_', string(baseWindowWhole));
    query3 = strcat("CREATE TABLE IF NOT EXISTS ", tableName, " (pk int");

    %for i = 1:resWindowWhole
    %    query3 = strcat(query3, ", r", string(i), " int");
    %end

    %for i = 1:baseWindowWhole
    %    query3 = strcat(query3, ", b", string(i), " int");
    %end

    query3 = strcat(query3, ", val ascii");

    query3 = strcat(query3, ", PRIMARY KEY (pk));");
    results = executecql(conn, query3);
    close(conn);
end


mr_in = resNum * resWindowWhole;
mb_in = baseNum * baseWindowWhole;
m_in = mr_in + mb_in;

n_out = 2; % bind or not

bindScaleNo = 10; %1;
noBindScaleNo = 0; %50;

%scaleInFiles=2;
foldInFiles = 6;
folds = foldInFiles * floor((foldInFiles-1)/2);

noBindPerc = 0; %95;

nTrain = 1000; %1 or 1000(more)
nNets = 3; %1; %5;




% Load tarin data
dataIdxDir = '/media/data2/Protein/practip-data';
dataTrIdxFile = 'train.lst';


%%
%[X, Y, mTrBind, mTrNoBind, Xcontr, Ycontr, Ncontr] = build_tensors(dataIdxDir, dataTrIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
%    baseWindowLen, baseWindowWhole, baseNum, scaleNo, scaleInFiles);


%%
ini_rate = 0.001; 
max_epoch = [floor(50), floor(50), floor(150)]; %* 20; %200


%cNet = AnnClasNet2D(m_in, n_out, ini_rate, max_epoch);
cNet = ReluClasNet2D(m_in, n_out, ini_rate, max_epoch(1), .5);
%cNet = Relu1aClasNet2D(m_in, n_out, ini_rate, max_epoch(1), .5);
%cNet = Relu1bClasNet2D(m_in, n_out, ini_rate, max_epoch(1), .5);
%cNet = Relu3aClasNet2D(m_in, n_out, ini_rate, max_epoch(1), .5);
%cNet = Relu3bClasNet2D(m_in, n_out, ini_rate, max_epoch(1), .5);
%cNet = Relu4ClasNet2D(m_in, n_out, ini_rate, max_epoch(1), .5);

%cNet = SigClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = TanhClasNet2D(m_in, n_out, ini_rate, max_epoch(3));
%cNet = RbfClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = TransClasNet2D(m_in, n_out, ini_rate, max_epoch);
%cNet = KgClasNet2D(m_in, n_out, ini_rate, max_epoch(2));

nNetTypes = 1; %3;
cNetTypes = cell([nNetTypes, 1]);

cNetTypes{1} = cNet;

%cNet2 = KgClasNet2D(m_in, n_out, ini_rate, max_epoch(2));
%cNetTypes{2} = cNet2;

%cNet3 = TanhClasNet2D(m_in, n_out, ini_rate, max_epoch(3));
%cNetTypes{2} = cNet3;
%cNetTypes{3} = cNet3;

if nNets*nNetTypes > 1
    threshVal = floor(nNets*nNetTypes/2) + 1;
else
    threshVal = 0;
end


[cNets, mTrBind, mTrNoBind, Xcontr, Ycontr, Ncontr, t1, t2, noBindThresh] = train_tensors_fold(cNetTypes, nNets, nTrain, dataIdxDir, dataTrIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
    baseWindowLen, baseWindowWhole, baseNum, bindScaleNo, noBindScaleNo, foldInFiles, noBindPerc, useDB);

[nNets, ~] = size(cNets);



%%
dataTsIdxFile = 'test.lst';
scaleNoTs = 0;

%%
calcAUC = 0;
AUC = 0;

if calcAUC
    nStep1 = 10;
    nStep2 = 10;
    nStep3 = 10;
    nStep = nStep1 + nStep2 + nStep3 + 1;
    ROCX = zeros([nStep, 1]);
    ROCY = zeros([nStep, 1]);
    ROCT = zeros([nStep, 1]);

    noBindThreshAUC = zeros([nNets, nStep]);
    for l = 1:nNets
        for i = 1:nStep1
            noBindThreshAUC(l,i) = 0.01/nStep1 * (i-1);
        end

        for i = nStep1+1:nStep1+nStep2
            noBindThreshAUC(l,i) = 0.01 + (0.1-0.01)/nStep2 * (i-nStep1-1);
        end

        for i = nStep1+nStep2+1:nStep1+nStep2+nStep3
            noBindThreshAUC(l,i) = 0.11 + (1-0.11)/nStep3 * (i-(nStep1+nStep2)-1);
        end

        noBindThreshAUC(l,nStep) = 1.001;
    end


    maxNoBind = 0; %378556; %0;

    [TP, TN, FP, FN, mTsBind, mTsNoBind, meanActTP, meanActFN, meanActTN, meanActFP, sigActTP, sigActFN] = predict_tensors_fold(cNets, dataIdxDir, dataTsIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
        baseWindowLen, baseWindowWhole, baseNum, scaleNoTs, 1, noBindThreshAUC, threshVal, maxNoBind);

%%
    for i = 1:nStep

        acc = (TP(i) + TN(i)) / (TP(i) + TN(i) + FP(i) + FN(i));
        Pr = TP(i) / (TP(i) + FP(i));
        Rec = TP(i) / (TP(i) + FN(i));
        Sp = TN(i) / (TN(i) + FP(i));
        Fo = FP(i) / (FP(i) + TN(i));
        F1 = 2*Rec*Pr/(Rec+Pr);

        ROCX(i) = 1 - Sp;
        ROCY(i) = Rec;
        ROCT(i) = noBindThreshAUC(1,i);

        if i>1
            AUC = AUC + (ROCY(i-1) + ROCY(i)) * (ROCX(i-1) - ROCX(i))/2;
        end

    end
end

%%
%noBindThresh = zeros([nNets, 1]);
maxNoBind = 0;

[TP, TN, FP, FN, mTsBind, mTsNoBind, meanActTP, meanActFN, meanActTN, meanActFP, sigActTP, sigActFN] = predict_tensors_fold(cNets, dataIdxDir, dataTsIdxFile, m_in, resWindowLen, resWindowWhole, resNum,... 
        baseWindowLen, baseWindowWhole, baseNum, scaleNoTs, 1, noBindThresh, threshVal, maxNoBind);

%%
acc = (TP + TN) ./ (TP + TN + FP + FN);

Pr = TP ./ (TP + FP);
Pr(TP==0) = 0;

Rec = TP ./ (TP + FN);
Sp = TN ./ (TN + FP);
Fo = FP ./ (FP + TN);

F1 = 2*Rec.*Pr./(Rec+Pr);
F1(TP==0) = 0;


TPM = mean(TP);
TNM = mean(TN);
FPM = mean(FP);
FNM = mean(FN);
accM = mean(acc);
PrM = mean(Pr);
RecM = mean(Rec);
SpM = mean(Sp);
F1M = mean(F1);

TPS = std(TP);
TNS = std(TN);
FPS = std(TP);
FNS = std(FN);
accS = std(acc);
PrS = std(Pr);
RecS = std(Rec);
SpS = std(Sp);
F1S = std(F1);

%%
model_name = "";
for i = 1:nNetTypes
    model_name = strcat(model_name, ".", cNetTypes{i}.name);
end
mb_size = cNets{1,1}.mb_size;

[~, nEns] = size(max_epoch);
max_epoch_str = "";
for i = 1:nEns
    max_epoch_str = strcat(max_epoch_str, ".", string(max_epoch(i)));
end

%%
if calcAUC
    fprintf('Thresh SpecC Recall\n');
    for i = 1:nStep
        fprintf('%f %f %f\n', ROCT(i), ROCX(i), ROCY(i));
    end
 
    f = figure();
    lp = plot(ROCX, ROCY, 'LineStyle', '-', 'Color', 'b', 'MarkerSize', 1, 'LineWidth', 1);
    hold on;
    title(strcat("ROC ", model_name, ", mb=", string(mb_size), ", epoch=", max_epoch_str, ", resL=", num2str(resWindowLen), ", baseL=", num2str(baseWindowLen),...
         ", trBindN=", string(mTrBind), ", trNoBindN=", string(mTrNoBind), ", trainN=", string(nTrain), ", netsN=", string(nNets)));
    xlabel('FPR');
    ylabel('TPR');
end

%%
fprintf('Model %s mb_size %d, max_epoch %s ResWindow %d, BaseWindow %d, TrainBindN %d, TrainNoBindN %d, BindScaleNo %d, NoBindScaleNo %d, ScaleInFiles %f\n\n',...
    model_name, mb_size, max_epoch_str, resWindowWhole, baseWindowWhole, mTrBind, mTrNoBind, bindScaleNo, noBindScaleNo, foldInFiles);

fprintf('NNetTypes %d, NNets %d, NTrain %d, NoBindPerc %d, NoBindThresh1 %f, NoBindTsRat %d, TestBindN %d, TestNoBindN %d ThreshVal %d\n\n',...
    nNetTypes, nNets, nTrain, noBindPerc, noBindThresh(1), scaleNoTs, mTsBind, mTsNoBind, threshVal);

fprintf('Accuracy %f+-%f, Precision %f+-%f, Recall %f+-%f, Specificity %f+-%f, F1 %f+-%f, TP %f+-%f, TN %f+-%f, FN %f+-%f, FP %f+-%f, AUC %f, TrTime %f\n',...
    accM, accS, PrM, PrS, RecM, RecS, SpM, SpS, F1M, F1S, TPM, TPS, TNM, TNS, FNM, FNS, FPM, FPS, AUC, etime(t2, t1));