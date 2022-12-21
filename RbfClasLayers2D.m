classdef RbfClasLayers2D

    properties

    end

    methods
        function net = RbfClasLayers2D()            
        end


        function net = Create(net)


            layers = [
                featureInputLayer(net.m_in)
                GaussianRBFLayer('RBF1', net.m_in, net.k_hid1)
                GaussianRBFLayer('RBF2', net.k_hid1, net.k_hid2)
                fullyConnectedLayer(net.n_out)
                softmaxLayer
                classificationLayer
            ];

            net.lGraph = layerGraph(layers);


            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'Plots', 'training-progress',...
                'MaxEpochs',net.max_epoch);

                %'Plots', 'training-progress',...
        end


    end

end
