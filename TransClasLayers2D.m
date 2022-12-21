classdef TransClasLayers2D

    properties

    end

    methods
        function net = TransClasLayers2D()            
        end


        function net = Create(net)

            oLayers = [
                featureInputLayer(net.m_in,'Name','inputFeature')
                additionLayer(2,'Name','fcAgate')
                fullyConnectedLayer(net.k_hid1,'Name','fcHidden')
                additionLayer(2,'Name','fcAgate2')
                fullyConnectedLayer(net.k_hid2,'Name','fcHidden2')
                fullyConnectedLayer(net.n_out,'Name','fcOut')
                softmaxLayer
                classificationLayer
            ];

            cgraph = layerGraph(oLayers);

            tLayers = [
                transformerLayer(net.m_in,'trans')
                fullyConnectedLayer(net.m_in,'Name','fcTrans')
            ];

            cgraph = addLayers(cgraph, tLayers);
    
            cgraph = connectLayers(cgraph, 'inputFeature', 'trans');
            cgraph = connectLayers(cgraph,'fcTrans','fcAgate/in2');


            t2Layers = [
                transformerLayer(net.k_hid1,'trans2')
                fullyConnectedLayer(net.k_hid1,'Name','fcTrans2')
            ];

            cgraph = addLayers(cgraph, t2Layers);
    
            cgraph = connectLayers(cgraph, 'fcHidden', 'trans2');
            cgraph = connectLayers(cgraph,'fcTrans2','fcAgate2/in2');
    
            net.lGraph = cgraph;


            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',... %'parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'Plots', 'training-progress',...
                'MaxEpochs',net.max_epoch);

                %'Plots', 'training-progress',...
        end


    end

end
