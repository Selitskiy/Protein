classdef LnP5PlQNet2Cl < LnP5PlQLayers2Cl & BaseClasNet2D & MLPInputClasNet2D
    properties

    end

    methods
        function net = LnP5PlQNet2Cl(m_in, n_out, ini_rate, max_epoch, mult)

            net = net@LnP5PlQLayers2Cl();
            net = net@BaseClasNet2D(m_in, n_out, ini_rate, max_epoch, mult);
            net = net@MLPInputClasNet2D();

            net.name = strcat("lnp5plq2dc", num2str(mult*100)); 

        end


        function net = Train(net, X, Y)
            fprintf('Training %s Reg net\n', net.name); 

            net = Train@MLPInputClasNet2D(net, X, Y);
        end

        
    end
end