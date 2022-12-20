function trBindMRow = bind_1hot(resFaM, baseFaM, trBindMRow, resWindowLen, resWindowWhole, baseWindowLen, resNum, baseNum, r_centr, b_centr, mCur)
    
    [~, res_len] = size(resFaM);
    [~, base_len] = size(baseFaM);

    % serial number of the center residue of a bind
    %r_centr = trDatM(j,1);
    for w = -resWindowLen:resWindowLen
        % serial number of the residue in a window of a bind
        r_w = r_centr + w;
        if (r_w >= 0) && (r_w < res_len)
            % one-hot encoding offset of the window
            r_off = resNum * (resWindowLen + w);
            % one-hot encoding position in the window
            trBindMRow(r_off + resFaM(1+r_w)) = 1;
        end
    end

    % serial number of the center residue of a bind
    %b_centr = trDatM(j,2);
    for w = -baseWindowLen:baseWindowLen
        % serial number of the residue in a window of a bind
        b_w = b_centr + w;
        if (b_w >= 0) && (b_w < base_len)
            % one-hot encoding offset of the window
            b_off = resNum * resWindowWhole + baseNum * (baseWindowLen + w);
            % one-hot encoding position in the window
            trBindMRow(b_off + baseFaM(1+b_w)) = 1;
        end
    end

end