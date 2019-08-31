function [prediction, net, rmseTest] = main (network_type) %network_type = FFN, RNN, LSTM
%main function for chapter 1
rho     = 28; sigma = 10; beta = 8/3; %the parameter values used in L63
xinit   = [-0.3,-0.7,0.5];  % initial value for (x,y,z) randn(3,1) for random starting position
h       = 0.01;             % time step
T       = 100;              % max time
time    = 1:h:T;            % 1 to 117 with 0.01 time steps
[~,truth] = ode45(@(t,x) L63(x, rho, sigma, beta),time,xinit);	% truth = (x,y,z)
truth   = truth';           % write as column vector

%following time step for final data point is unknown, hence remove from input
input = truth(:,1:end-1);
%the function finds (x,y,z) for time step t+1
output = truth(:,2:end);

trainRatio = 0.95;
valRatio = 0.025;
testRatio = 0.025;

%Number of timesteps
numTimeSteps        = length(time);
%95 percent for training
numTimeStepsTrain   = floor(trainRatio*numTimeSteps);
%2.5 percent for validation
numTimeStepsVal   = floor(valRatio*numTimeSteps);
%2.5 percent for testing
numTimeStepsTest   = floor(testRatio*numTimeSteps);

%Create index sets for train/validate/test sets
Q                           = numTimeStepsTrain + numTimeStepsVal + numTimeStepsTest;
[trainInd,valInd,testInd]   = divideind(Q,1:numTimeStepsTrain,numTimeStepsTrain+1:numTimeStepsTrain+numTimeStepsVal,numTimeStepsTrain+numTimeStepsVal+1:Q);

if network_type == "FFN" 
    %Train FFN
    net_FFN = FFN_L63 (input, output, trainInd, valInd, testInd);
else if network_type == "RNN" 
    %Train RNN
    net_RNN = RNN_L63 (input, output, trainInd, valInd, testInd);
else if network_type == "LSTM"
    %Train LSTM
    net_LSTM = LSTM_L63 (input, output, trainInd, valInd);
    else disp('Unknown network type')
    end
    end
end

%true value on the test set
inputTest    = input (:,testInd);
outputTrue   = output(:,testInd);
%prediction on the test set
if network_type == "FFN" 
    test_predict_FFN            = net_FFN(inputTest);
else if network_type == "RNN"
    [net_RNN, test_predict_RNN] = predictAndUpdateState(net_RNN,inputTest);
else if network_type == "LSTM" 
    [net_LSTM, test_predict_LSTM] = predictAndUpdateState(net_LSTM,inputTest);
    else disp('Unknown network type')
    end
    end
end

if network_type == "FFN" 
    figure;
    plot3(test_predict_FFN	(1,:),test_predict_FFN (2,:),test_predict_FFN (3,:),'k');      %NN prediction on test set
    hold on;
    plot3(outputTrue     (1,:),outputTrue    (2,:),outputTrue    (3,:),'-.r');    %true value of test set
    legend('FFN prediction', 'true position')
else if network_type == "RNN" 
    figure;
    plot3(test_predict_RNN	(1,:),test_predict_RNN (2,:),test_predict_RNN (3,:),'k');      %NN prediction on test set
    hold on;
    plot3(outputTrue     (1,:),outputTrue    (2,:),outputTrue    (3,:),'-.r');    %true value of test set
    legend('RNN prediction', 'true position')
else if network_type == "LSTM"
    figure;
    plot3(test_predict_LSTM	(1,:),test_predict_LSTM (2,:),test_predict_LSTM (3,:),'k');      %NN prediction on test set
    hold on;
    plot3(outputTrue     (1,:),outputTrue    (2,:),outputTrue    (3,:),'-.r');    %true value of test set
    legend('LSTM prediction', 'true position')    
    else disp('Unknown network type')
    end
    end
end

%output the network, prediction adn test MSE
if network_type == "FFN" 
    prediction = test_predict_FFN;
    net = net_FFN;
    rmseTest = sqrt(sum(sum((outputTrue - test_predict_FFN),2).^2)/numTimeStepsTest);
    else if network_type == "RNN" 
        prediction = test_predict_RNN;
        rmseTest = sqrt(sum(sum((outputTrue - test_predict_RNN),2).^2)/numTimeStepsTest);
        net = net_RNN;
        else if network_type == "LSTM" 
            prediction = test_predict_LSTM;
            rmseTest = sqrt(sum(sum((outputTrue - test_predict_LSTM),2).^2)/numTimeStepsTest);
            net = net_LSTM;
            end
        end
end

end