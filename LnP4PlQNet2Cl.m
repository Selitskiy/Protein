classdef LnP4PlQNet2Cl < LnP4PlQLayers2Cl & BaseClasNet2D & MLPInputClasNet2D
    properties

    end

    methods
        function net = LnP4PlQNet2Cl(m_in, n_out, ini_rate, max_epoch, mult)

            net = net@LnP4PlQLayers2Cl();
            net = net@BaseClasNet2D(m_in, n_out, ini_rate, max_epoch, mult);
            net = net@MLPInputClasNet2D();

            net.name = strcat("lnp4plq2dc", num2str(mult*100)); 

        end


        function net = Train(net, X, Y)
            fprintf('Training %s Reg net\n', net.name); 

            net = Train@MLPInputClasNet2D(net, X, Y);
        end

        
    end
end