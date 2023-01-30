classdef ReluClasLayers2D

    properties

    end

    methods
        function net = ReluClasLayers2D()            
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)
                fullyConnectedLayer(net.k_hid1)
                reluLayer
                dropoutLayer
                fullyConnectedLayer(net.k_hid2)
                reluLayer
                dropoutLayer
                fullyConnectedLayer(net.n_out)
                softmaxLayer
                classificationLayer
            ];

            net.lGraph = layerGraph(layers);


            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',... %'parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);

                %'Plots', 'training-progress',...
        end


    end

end
