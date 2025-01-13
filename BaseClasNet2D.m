classdef BaseClasNet2D

    properties
        name = [];

        %x_off
        %x_in
        %t_in
        m_in

        %y_off
        %y_out
        %t_out
        n_out

        k_hid1
        k_hid2
        ini_rate 
        max_epoch
    
        lGraph = [];
        options = [];
        trainedNet = [];
    end

    methods
        function net = BaseClasNet2D(m_in, n_out, ini_rate, max_epoch, mult)

            net.m_in = m_in;

            net.n_out = n_out;

            %mult = 1;
            net.k_hid1 = floor(mult * (net.m_in + 1));
            net.k_hid2 = floor(mult * (2*net.m_in + 1));
            net.ini_rate = ini_rate;
            net.max_epoch = max_epoch;

        end


        %function [X, X2] = ReScaleIn(net, X, X2, Bi, n_sess, t_sess, sess_off, k_ob, k_tob)

        %    for i = 1:n_sess
        %        for j = 1:k_ob
                    % bounds over session
        %            MeanSess = Bi(3,:,i);
        %            StdSess = Bi(4,:,i);

        %            Mxw = reshape( X(1:net.m_in, j, i), [net.x_in, net.t_in])';

        %            Mxw = generic_mean_std_rescale2D(Mxw, MeanSess, StdSess);

        %            Mx = reshape( Mxw', [net.m_in,1] );
        %            X(1:net.m_in, j, i) = Mx(:);
        %        end
        %    end


         %   for i = 1:t_sess-sess_off
                % bounds over session
         %       MeanSess = Bi(3,:,i);
         %       StdSess = Bi(4,:,i);

         %       for j = 1:k_tob
         %           Mxw = reshape( X2(1:net.m_in, j, i), [net.x_in, net.t_in])';

         %           Mxw = generic_mean_std_rescale2D(Mxw, MeanSess, StdSess);

         %           Mx = reshape( Mxw', [net.m_in,1] );
         %           X2(1:net.m_in, j, i) = Mx(:);
         %       end
         %   end

        %end


        %function [Y, Y2, Yhs2] = ReScaleOut(net, Y, Y2, Yhs2, Bo, Bto, n_sess, t_sess, sess_off, k_ob, k_tob)

        %    for i = 1:n_sess
                % bounds over session
        %        MeanSess = Bo(3,:,i);
        %        StdSess = Bo(4,:,i);

        %        for j = 1:k_ob
                    % extract and scale observation sequence
        %            Myw = reshape( Y(:, j, i), [net.y_out, net.t_out])';

        %            Myw = generic_mean_std_rescale2D(Myw, MeanSess, StdSess);

        %            My = reshape( Myw', [net.n_out,1] );
        %            Y(:, j, i) = My(:);
        %        end
        %    end


        %    for i = 1:t_sess-sess_off

        %        for j = 1:k_tob

        %            MeanSesst = Bto(3,:,j,i);
        %            StdSesst = Bto(4,:,j,i);

        %            Myw = reshape( Y2(:, j, i), [net.y_out, net.t_out])';

        %            Myw = generic_mean_std_rescale2D(Myw, MeanSesst, StdSesst);

        %            My = reshape( Myw', [net.n_out,1] );
        %            Y2(:, j, i) = My(:);


        %            Myw = reshape( Yhs2(:, j, i), [net.y_out, net.t_out])';

        %            Myw = generic_mean_std_rescale2D(Myw, MeanSesst, StdSesst);

        %            My = reshape( Myw', [net.n_out,1] );
        %            Yhs2(:, j, i) = My(:);

        %        end
        %    end
        %end

        %function [Em, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = Calc_mape(net, Y2, Yh2)
        %    [Em, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = generic_calc_mape2D(Y2, Yh2, net.n_out); 
        %end

        %function [Er, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = Calc_rmse(net, Y2, Yh2) 
        %    [Er, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = generic_calc_rmse2D(Y2, Yh2, net.n_out);
        %end

        %function [Ec, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = Calc_cont_rmse(net, Y2, Yh2) 
        %    [Ec, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = generic_calc_cont_rmse2D(Y2, Yh2, net.n_out, net.y_out);
        %end


        %function Err_graph(net, M, Em, Er, l_whole_ex, Y2, Sy2, l_whole, l_sess, k_tob, t_sess, sess_off, offset, l_marg, modelName)
        %    generic_err_graph2D(M, Em, Er, l_whole_ex, Y2, Sy2, l_whole, l_sess, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, k_tob, t_sess, sess_off, offset, l_marg, modelName);
        %end

        %function TestIn_graph(net, M, l_whole_ex, X, Y, X2, Y2, Sx, Sy, Sx2, Sy2, l_whole, n_sess, l_sess, k_ob, k_tob, t_sess, sess_off, offset, l_marg, modelName)
        %    generic_test_in_graph2D(M, l_whole_ex, X, Y, X2, Y2, Sx, Sy, Sx2, Sy2, l_whole, n_sess, l_sess, k_ob, net.x_in, net.t_in, net.y_out, net.t_out, k_tob, t_sess, sess_off, offset, l_marg, modelName);
        %end

    end
end