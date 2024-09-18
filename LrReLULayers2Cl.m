classdef LrReLULayers2Cl

    properties

    end

    methods
        function net = LrReLULayers2Cl()            
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)

                fullyConnectedLayer(net.k_hid1)
                LrReLULayer('LrReLU1', net.k_hid1, 1)

                fullyConnectedLayer(net.k_hid2)
                LrReLULayer('LrReLU2', net.k_hid2, 1)

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
