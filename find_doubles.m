function [Xcontr, Ycontr, Ncontr] = find_doubles(X, Y, mTrBind, mTrNoBind, resWindowWhole, resNum, baseWindowLen, baseWindowWhole, baseNum)

% Convert input into strings (for sorting, uniqueness and contradiction detection)
mr_in = resNum * resWindowWhole;
mb_in = baseNum * baseWindowWhole;

Xstr = repmat( string(repmat(' ', [1 resWindowWhole + resWindowWhole])), [mTrBind+mTrNoBind 1] );
for i = 1:mTrBind+mTrNoBind
    for j = 1:resWindowWhole
        for k = 1:resNum
            enc = X(i, (j-1)*resNum+k);
            if enc
                Xstr{i}(j) = char(k + 64);
                break;
            end
        end
    end

     for j = 1:baseWindowWhole
        for k = 1:baseNum
            enc = X(i, mr_in+(j-1)*baseNum+k);
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

% Sort, unique, contradictory verdicts
[XstrS, I] = sort(Xstr);
YS = Y(I);
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