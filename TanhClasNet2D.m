classdef TanhClasNet2D < TanhClasLayers2D & BaseClasNet2D & MLPInputClasNet2D

    properties

    end

    methods
        function net = TanhClasNet2D(m_in, n_out, ini_rate, max_epoch)

            net = net@TanhClasLayers2D();
            net = net@BaseClasNet2D(m_in, n_out, ini_rate, max_epoch);
            net = net@MLPInputClasNet2D();

            net.name = "tanh2dc";

        end


        %function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)

        %    [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors@MLPInputNet2D(net, M, l_sess, n_sess, norm_fli, norm_flo);

        %    net = Create(net);

        %end



        function net = Train(net, X, Y)
            fprintf('Training %s Reg net\n', net.name); 

            net = Train@MLPInputClasNet2D(net, X, Y);
        end

        
    end
end