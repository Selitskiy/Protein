classdef BTL3SQPNet2Cl < BTL3SQPLayers2Cl & BaseClasNet2D & MLPInputClasNet2D
    properties
        k_bottle
    end

    methods
        function net = BTL3SQPNet2Cl(m_in, k_bottle, n_out, ini_rate, max_epoch, mult)

            net = net@BTL3SQPLayers2Cl();
            net = net@BaseClasNet2D(m_in, n_out, ini_rate, max_epoch, mult);
            net = net@MLPInputClasNet2D();

            net.k_bottle = k_bottle;

            net.k_hid1 = floor(mult * (net.k_bottle));
            net.k_hid2 = floor(mult * (2*net.k_bottle + 1));

            net.name = strcat("btl3sqp2dc", num2str(mult*100)); 

        end


        function net = Train(net, X, Y)
            fprintf('Training %s Reg net\n', net.name); 

            net = Train@MLPInputClasNet2D(net, X, Y);
        end

        
    end
end