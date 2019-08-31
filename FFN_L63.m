function net = FFN_L63 (input,output, trainInd,valInd, testInd)

    trainFcn = 'traingd';  % Gradient descent backpropagation
    
	%Create the network
	hiddenLayerSize = 25;
	net             = feedforwardnet(hiddenLayerSize, trainFcn);
    
    %Divide data set into train, test, validation sets
    net.divideFcn               = 'divideind';
    net.divideParam.trainInd    = trainInd;
    net.divideParam.valInd      = valInd;
    net.divideParam.testInd     = testInd;
    
    %choose learning rate
    net.trainParam.lr = 0.0001;
    net.trainParam.max_fail = 100;
    
    [net,~] = train(net,input,output);
end