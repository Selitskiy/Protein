classdef KgClasLayers2D

    properties

    end

    methods
        function net = KgClasLayers2D()            
        end


        function net = Create(net)

            oLayers = [
        featureInputLayer(net.m_in,'Name','inputFeature')
        additionLayer(2,'Name','fcAgate')
        multiplicationLayer(2,'Name','fcMgate')
        fullyConnectedLayer(net.k_hid1,'Name','fcHidden')
        additionLayer(2,'Name','fcAgate2')
        multiplicationLayer(2,'Name','fcMgate2') 
        fullyConnectedLayer(net.k_hid2,'Name','fcHidden2')
        fullyConnectedLayer(net.n_out,'Name','fcOut')
        softmaxLayer
        classificationLayer
    ];
    cgraph = layerGraph(oLayers);


    % Allow Linear Transform and Sigmoid activation
    aLayers = [
        fullyConnectedLayer(net.m_in,'Name','fcSig')
        sigmoidLayer('Name','sigAllow')
    ];
    aLayers2 = [
        fullyConnectedLayer(net.k_hid1,'Name','fcSig2')
        sigmoidLayer('Name','sigAllow2')
    ];
    cgraph = addLayers(cgraph, aLayers);
    cgraph = addLayers(cgraph, aLayers2);


    % Update Linear Transform and Tanh activation
    dLayers = [
        fullyConnectedLayer(net.m_in,'Name','fcTanh')
        tanhLayer('Name','tanhUpdate')
    ];
    dLayers2 = [
        fullyConnectedLayer(net.k_hid1,'Name','fcTanh2')
        tanhLayer('Name','tanhUpdate2')
    ];
    cgraph = addLayers(cgraph, dLayers);
    cgraph = addLayers(cgraph, dLayers2);


    % Update Cacade Linear Transform and Hadfamard join with Tanh
    nLayers = [
        fullyConnectedLayer(net.m_in,'Name','fcNorm')
        multiplicationLayer(2,'Name','normMgate')
    ];
    nLayers2 = [
        fullyConnectedLayer(net.k_hid1,'Name','fcNorm2')
        multiplicationLayer(2,'Name','normMgate2')
    ];
    cgraph = addLayers(cgraph, nLayers);
    cgraph = addLayers(cgraph, nLayers2);


    % Cascade-conneect Allow path to main trunk via Haddamard product
    cgraph = connectLayers(cgraph, 'inputFeature', 'fcSig');
    cgraph = connectLayers(cgraph,'sigAllow','fcMgate/in2');

    cgraph = connectLayers(cgraph, 'inputFeature', 'fcSig2');
    cgraph = connectLayers(cgraph,'sigAllow2','fcMgate2/in2');


    % Cascade-connect Update path to main trunk via Addition
    cgraph = connectLayers(cgraph, 'inputFeature', 'fcTanh');
    cgraph = connectLayers(cgraph, 'inputFeature', 'fcTanh2');

    cgraph = connectLayers(cgraph, 'inputFeature', 'fcNorm');
    cgraph = connectLayers(cgraph,'tanhUpdate','normMgate/in2');
    cgraph = connectLayers(cgraph,'normMgate','fcAgate/in2');

    cgraph = connectLayers(cgraph, 'inputFeature', 'fcNorm2');
    cgraph = connectLayers(cgraph,'tanhUpdate2','normMgate2/in2');
    cgraph = connectLayers(cgraph,'normMgate2','fcAgate2/in2');

    
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
