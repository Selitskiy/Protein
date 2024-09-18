classdef QLrReLULayers2Cl

    properties

    end

    methods
        function net = QLrReLULayers2Cl()            
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)

                QExpansionLayer("QExpansion", net.k_hid2)

                LrReLULayer("LrReLU1", net.m_in*net.k_hid2, 1)
                QSumLayer("QSum", net.k_hid2)

                LrReLULayer("LrReLU2", net.k_hid2, 1)

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
