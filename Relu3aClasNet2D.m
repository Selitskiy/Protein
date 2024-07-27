classdef Relu3aClasNet2D < Relu3aClasLayers2D & BaseClasNet2D & MLPInputClasNet2D

    properties

    end

    methods
        function net = Relu3aClasNet2D(m_in, n_out, ini_rate, max_epoch, mult)

            net = net@Relu3aClasLayers2D();
            net = net@BaseClasNet2D(m_in, n_out, ini_rate, max_epoch, mult);
            net = net@MLPInputClasNet2D();

            net.name = strcat("relu3a2dc", num2str(mult*100)); 

        end


        function net = Train(net, X, Y)
            fprintf('Training %s Reg net\n', net.name); 

            net = Train@MLPInputClasNet2D(net, X, Y);
        end

        
    end
end