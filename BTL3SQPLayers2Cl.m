classdef BTL3SQPLayers2Cl

    properties

    end

    methods
        function net = BTL3SQPLayers2Cl()            
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)

                %cosPeTransformerLayer(net.m_in, "Pe")

                %fullyConnectedLayer(net.m_in*net.n_out, 'Name', "Product Expand")

                cosPcTransformerLayer(net.m_in, "Pc")

                fullyConnectedLayer(net.k_bottle, 'Name', "Bottleneck")

                %SinQLayer("Sin phi Q", net.k_bottle)


                fullyConnectedLayer(net.k_hid2*net.n_out, 'Name', "Sum phi QP")

                SinQLayer("Sin Phi Q", net.k_hid2*net.n_out)
                additionLayer(3, 'Name', "Furier Add")

                QPartitionLayer("ProductPage_Part", net.k_hid2, net.n_out)

                QSumLayer("Sum_PhiQ_Pages", net.k_hid2, 1, net.n_out)

                softmaxLayer
                classificationLayer
            ];

            net.lGraph = layerGraph(layers);


            sLayers = [
                SinQLayer("Sin Phi Q2", net.k_hid2*net.n_out)
            ];
            net.lGraph = addLayers(net.lGraph, sLayers);
            net.lGraph = connectLayers(net.lGraph, "Sum phi QP", "Sin Phi Q2");
            net.lGraph = connectLayers(net.lGraph,  "Sin Phi Q2", "Furier Add/in2");

            sLayers2 = [
                SinQLayer("Sin Phi Q3", net.k_hid2*net.n_out)
            ];
            net.lGraph = addLayers(net.lGraph, sLayers2);
            net.lGraph = connectLayers(net.lGraph, "Sum phi QP", "Sin Phi Q3");
            net.lGraph = connectLayers(net.lGraph,  "Sin Phi Q3", "Furier Add/in3");



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
