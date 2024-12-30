classdef LrReLUQPLayers2Cl

    properties

    end

    methods
        function net = LrReLUQPLayers2Cl()            
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)

                QExpansionLayer("QPage_Ext", net.k_hid2, net.n_out)

                %RBFQPLayer("phiQP_RBF", net.m_in, net.k_hid2, net.n_out)
                SinQPLayer("phiQP_Sin", net.m_in, net.k_hid2, net.n_out)
                %LrReLULQPLayer("phiQP_LinLrReLU", net.m_in, net.k_hid2, net.n_out, 1)

                PSumLayer("SumP_phiQP", net.m_in, net.k_hid2, net.n_out)

                %RBFQPLayer("PhiQP_RBF", net.k_hid2, 1, net.n_out)
                SinQPLayer("PhiQP_Sin", net.k_hid2, 1, net.n_out)
                %LrReLULQPLayer("PhiQP_LinLrReLU", net.k_hid2, 1, net.n_out, 1)

                QSumLayer("SumP_PhiQP", net.k_hid2, 1, net.n_out)

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
