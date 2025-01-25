classdef LnPLrReLUQNACNet2Cl < LnPLrReLUQNACLayers2Cl & BaseClasNet2D & MLPInputClasNet2D
    properties

    end

    methods
        function net = LnPLrReLUQNACNet2Cl(m_in, n_out, ini_rate, max_epoch, mult)

            net = net@LnPLrReLUQNACLayers2Cl();
            net = net@BaseClasNet2D(m_in, n_out, ini_rate, max_epoch, mult);
            net = net@MLPInputClasNet2D();

            net.name = strcat("lnplrreluqnac2dc", num2str(mult*100)); 

        end


        function net = Train(net, X, Y)
            fprintf('Training %s Reg net\n', net.name); 

            net = Train@MLPInputClasNet2D(net, X, Y);
        end

        
    end
end