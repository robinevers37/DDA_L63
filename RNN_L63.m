function net = RNN_L63 (input,output, trainInd,valInd, testInd)
    %Define RNN network architecture
    numFeatures = 3;
    numResponses = 3;
    numHiddenUnits = 200;
    
    %Divide data set into train, test, validation sets
    inputTrain  = input(:,trainInd);
    inputVal    = input(:,valInd);
    inputTest   = input(:,testInd);
    
    outputTrain = output(:,trainInd);
    outputVal   = output(:,valInd);
    outputTest  = output(:,testInd);
    
	%Create the network
    layers = [ ...
        sequenceInputLayer(numFeatures)
        fullyConnectedLayer(numResponses)
        regressionLayer
        ];

    options	= trainingOptions('sgdm', ...    % Backpropagation through time (BPTT is SGD for recursive networks)
        'MaxEpochs',1000, ...
        'GradientThreshold',1, ...
        'ValidationData',{inputVal,outputVal}, ...
        'ValidationFrequency',30, ...
        'ValidationPatience',5, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0, ...
        'Plots','training-progress');
    
    %train network
    net         = trainNetwork(inputTrain,outputTrain,layers,options);
    net         = predictAndUpdateState(net,inputVal);
end