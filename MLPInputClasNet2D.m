classdef MLPInputClasNet2D

    properties
        mb_size = [];
    end

    methods

        function net = MLPInputClasNet2D()

        end


        %function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)
        %    [X, Xc, Xr, Xs, Ys, Y, Bi, Bo, XI, C, Sx, Sy, k_ob] = generic_train_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, n_sess, norm_fli, norm_flo);
        %    net.mb_size = 2^floor(log2(k_ob)-4);
        %end


        %function [X2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
        %    [X2, Xc2, Xr2, Xs2, Ys2, Ysh2, Yshs2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = generic_test_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob);
        %end


        function net = Train(net, X, Y)
            tNet = trainNetwork(X, Y, net.lGraph, net.options);
            net.trainedNet = tNet;
            net.lGraph = tNet.layerGraph;            
        end


        function [X2, Y2] = Predict(net, X2)
            predictedScores = predict(net.trainedNet, X2);
            predictedClasses = classify(net.trainedNet, X2);
            Y2 = predictedClasses;
        end
        
    end

end