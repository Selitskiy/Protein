classdef LnP3ExpQLayers2Cl

    properties

    end

    methods
        function net = LnP3ExpQLayers2Cl()            
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)

                fullyConnectedLayer(net.k_hid2*net.n_out, 'Name', "Sum phi QP")

                ExpQLayer("Exp Phi Q", net.k_hid2*net.n_out)

                additionLayer(3, 'Name', "Furier Add")

                QPartitionLayer("ProductPage_Part", net.k_hid2, net.n_out)
                QSumLayer("Sum_PhiQ_Pages", net.k_hid2, 1, net.n_out)

                %fullyConnectedLayer(net.n_out, 'Name', "Sum Phi Q No Representation")

                softmaxLayer
                classificationLayer
            ];

            net.lGraph = layerGraph(layers);


            sLayers = [
                ExpQLayer("Exp Phi Q2", net.k_hid2*net.n_out)
            ];
            net.lGraph = addLayers(net.lGraph, sLayers);
            net.lGraph = connectLayers(net.lGraph, "Sum phi QP", "Exp Phi Q2");
            net.lGraph = connectLayers(net.lGraph,  "Exp Phi Q2", "Furier Add/in2");

            sLayers2 = [
                ExpQLayer("Exp Phi Q3", net.k_hid2*net.n_out)
            ];
            net.lGraph = addLayers(net.lGraph, sLayers2);
            net.lGraph = connectLayers(net.lGraph, "Sum phi QP", "Exp Phi Q3");
            net.lGraph = connectLayers(net.lGraph,  "Exp Phi Q3", "Furier Add/in3");



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
