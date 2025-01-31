classdef LnPPlQNACLayers2Cl

    properties

    end

    methods
        function net = LnPPlQNACLayers2Cl()            
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)

                fullyConnectedLayer(net.k_hid2*net.n_out, 'Name', "Sum phi QP")

                PlQLayer("Sin Phi Q", net.k_hid2*net.n_out)

                %QPartitionLayer("ProductPage_Part", net.k_hid2, net.n_out)
                %QSumLayer("Sum_PhiQ_Pages", net.k_hid2, 1, net.n_out)

                fullyConnectedLayer(net.n_out, 'Name', "Sum Phi Q No Axiom of Choice")

                softmaxLayer
                classificationLayer
            ];

            net.lGraph = layerGraph(layers);


            %sLayers = [
            %    SinQLayer("Sin Phi Q2", net.k_hid2*net.n_out)
            %];
            %net.lGraph = addLayers(net.lGraph, sLayers);
            %net.lGraph = connectLayers(net.lGraph, "Sum phi QP", "Sin Phi Q2");
            %net.lGraph = connectLayers(net.lGraph,  "Sin Phi Q2", "Furier Add/in2");

            %sLayers2 = [
            %    SinQLayer("Sin Phi Q3", net.k_hid2*net.n_out)
            %];
            %net.lGraph = addLayers(net.lGraph, sLayers2);
            %net.lGraph = connectLayers(net.lGraph, "Sum phi QP", "Sin Phi Q3");
            %net.lGraph = connectLayers(net.lGraph,  "Sin Phi Q3", "Furier Add/in3");



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
