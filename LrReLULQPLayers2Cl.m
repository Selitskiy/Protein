classdef LrReLULQPLayers2Cl

    properties

    end

    methods
        function net = LrReLULQPLayers2Cl()            
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)
                %LrReLUSinLayer("In", net.m_in, pi)

                fullyConnectedLayer(net.k_hid2*net.n_out, 'Name', 'Sum phi QP')
                LrReLULayer("Phi Q", net.k_hid2*net.n_out, 1)
                %LrReLUSinLayer("Phi Q", net.k_hid2*net.n_out, 0)

                QPartitionLayer("ProductPage_Part", net.k_hid2, net.n_out)

                QSumLayer("Sum_PhiQ_Pages", net.k_hid2, 1, net.n_out)

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
