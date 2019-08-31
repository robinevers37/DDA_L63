function [nn, net] = DDA_L63 (obs, h, Q, R, trainInd,valInd, testInd)
    % the parameter values used in L63.m
    rho     = 28; sigma = 10; beta = 8/3;   

    %Split the obs into train, test, validate sets
    obsTrain  = obs(:,trainInd);
    obsVal    = obs(:,valInd);
    
    %Find DA solution for training set
    obsTrainAndVal  = [obsTrain,obsVal];
    [DATrainAndVal, modelTrainAndVal]   = DA_L63(obsTrainAndVal,h,size(obsTrainAndVal,2),Q,R);
    
    %Train LSTM network (input:model, output:DA)
    net = RNN_L63 (modelTrainAndVal,DATrainAndVal, trainInd,valInd,[]);

    %Split test set into r-1 intervals
    r = 20;
    numTimeStepsTest     = size(testInd,2);
    numTimeStepsInterval = floor(numTimeStepsTest/r);
    
    %Initialisation for the Kalman filter (test set)
    testStart = testInd(1);
    posterior = obs(:,testStart);
    posterior_DDA = obs(:,testStart);                   % First run of KF is on the posterios of the training case
    nn = obs(:,testStart);                              % First column of NN values (x^n,y^n,z^n) 
    P_posterior = .1*eye(length(posterior));            % Initial covariance (0.1,0,0; 0,0.1,0; 0,0,0.1)
    H           = eye(3);                               % Observation function
    
    %Define RNN network architecture for network updates
    numFeatures = 3;
    numResponses = 3;
    %use when using an LSTM layer
    %numHiddenUnits = 200;
    
    %DDA loop
    for i=1:r %for each interval
        start = testStart + (i-1)* numTimeStepsInterval;
        stop  = testStart + i    * numTimeStepsInterval - 1;
        for j=start:stop %for each timestep in the i'th interval
        %1 Calculate model using DDA
            prior                   = EulerL63(posterior_DDA, h, rho, sigma, beta);     % wj+1 = EulerL63(wj^DA)
            [net(i),posterior_DDA]	= predictAndUpdateState(net(i),prior);              % NN prediction
            nn(:,j)                 = posterior_DDA;                                    % add NN predicted value to (x^n, y^n, z^n
        %2 Assimilation
            prior               = EulerL63(posterior, h, rho, sigma, beta);     % wj+1 = EulerL63(wj^DA)
            F = [1-sigma*h,         h*sigma,       0; 
                (rho-prior(3))*h, 1-h,           -h*prior(1);
                h*prior(2),       h*prior(1),  1-beta*h];                   % TLM of num model
            P_prior             = F*P_posterior*F' + Q;                     % Pj+1 = FPjF' + Q
            % Observation update
            K                   = (P_prior*H')/(H*P_prior*H'+R);            % Kj = Pj*H'*inv(H*Pj*H'+R)
            posterior           = prior + K*(obs(:,j) - H*prior);           % wj = wj + Kj(yj-Hwj)
            P_posterior         = P_prior - K*H*P_prior;                    % Pj = (I-KjH)Pj
        %3 Add additional data to training set for NN
            X_train(:,j) = prior;
            Y_train(:,j) = posterior;
        end
    %4 Train NN on added data
        %Learn weights from previous net
        fullyConnectedLayerWeights  = net(i).Layers(2).Weights;
        fullyConnectedLayerBias     = net(i).Layers(2).Bias;
        
    %USE THESE WHEN ADDING AN LSTM LAYER
        %fullyConnectedLayerWeights  = net(i).Layers(3).Weights;
        %fullyConnectedLayerBias     = net(i).Layers(3).Bias;
        %LSTMLayerInputWeights       = net(i).Layers(2).InputWeights;
        %LSTMLayerRecurrentWeights   = net(i).Layers(2).RecurrentWeights;
        %LSTMLayerBias               = net(i).Layers(2).Bias;

        layers2 = [ ...
            sequenceInputLayer(numFeatures)
        %use when using an LSTM layer
        %    lstmLayer(numHiddenUnits, ...
        %    'InputWeights',LSTMLayerInputWeights, ...
        %    'RecurrentWeights',LSTMLayerRecurrentWeights, ...
        %    'Bias',LSTMLayerBias)
            fullyConnectedLayer(numResponses, ...
            'Weights',fullyConnectedLayerWeights, ...
            'Bias',fullyConnectedLayerBias)
            regressionLayer];
        options2 = trainingOptions('sgdm', ...
            'MaxEpochs',100, ...
            'GradientThreshold',1, ...
            'ValidationFrequency',30, ...
            'ValidationPatience',5, ...
            'InitialLearnRate',0.005, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',15, ...
            'LearnRateDropFactor',0.2, ...
            'Verbose',0, ...
            'Plots','training-progress');
        if i<r
            %correct starting point
            posterior_DDA = posterior;   
            %update network
            net(i+1)      = trainNetwork(X_train,Y_train,layers2,options2);
            [net(i+1),~]  = predictAndUpdateState(net(i+1),X_train);    % update network state after predicting Y_train.
        end
    end
end