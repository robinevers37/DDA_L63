%main function for chapter 2 & 3
%Initialise the Lorenz system
rho     = 28; sigma = 10; beta = 8/3;   % the parameter values used in L63.m
init    = [-0.3,-0.7,0.5];              % initial value for (x^t,y^t,z^t)
h       = 0.01;                         % time step
T       = 100;                          % max time
time    = 1:h:T;                        % the interval [0,T] with 0.01 time steps
[~,truth] = ode45(@(t,x) L63(x, rho, sigma, beta),time,init);	% truth = (x^t,y^t,z^t)
truth = truth';                         % write as column vector 

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
numTimeSteps                = numTimeStepsTrain + numTimeStepsVal + numTimeStepsTest;
[trainInd,valInd,testInd]   = divideind(numTimeSteps,1:numTimeStepsTrain,numTimeStepsTrain+1:numTimeStepsTrain+numTimeStepsVal,numTimeStepsTrain+numTimeStepsVal+1:numTimeSteps);
%Start index of test set
testStart                   = testInd(1);

%Define observations
obsNoise = 0.3;
truth = truth(:,1:numTimeSteps);
obs = truth + obsNoise^2*randn(size(truth));   % Add Gaussian noise with variance 0.3

% first observation is [-0.2633, -0.8265, 0.5132]
[~,truth2] = ode45(@(t,x) L63(x, rho, sigma, beta),time,obs(:,1));	% truth2 = (x^t,y^t,z^t)
truth2     = truth2';

%Filter to t=15-17
trueTest   = truth (:,1500:1700);% true state at t=15-17
true2Test  = truth2(:,1500:1700);% true state at t=15-17

%Plot model on observation vs truth
figure;
plot3(true2Test(1,:),true2Test(2,:),true2Test(3,:),'Color', [0, 0.5, 0]);
hold on;
plot3(true2Test(1,1),true2Test(2,1),true2Test(3,1),'o', 'Color',[0, 0.5, 0.1]);
plot3(trueTest(1,:),trueTest(2,:),trueTest(3,:),'-.r');
plot3(trueTest(1,1),trueTest(2,1),trueTest(3,1),'or');
legend('Perturbed trajectory', 'At t=15', 'Unperturbed trajectory', 'At t=15')

%Find nummerical model solution with Euler
model = obs(:,1);
for j = 2:numTimeSteps
    model(:,j)      = Euler_L63(model(:,j-1), h, rho, sigma, beta);
end

%Define the covariance matrices
sysNoise = sqrt(0.03);
Q        = sysNoise^2*eye(3);  %System noise covariance
R        = obsNoise^2;         %Observation noise covariance

%Find DA solution with DA_L63
da = DA_L63(obs,h,numTimeSteps,Q,R);

%Filter to t=15-17
modelTest  = model(:,1500:1700);%predicted state at t=15-17
daTest     = da(:,1500:1700);% DA state at t=15-17

%Plot model on observation vs truth
figure;
plot3(trueTest(1,:),trueTest(2,:),trueTest(3,:),'r');
hold on;
plot3(trueTest(1,1),trueTest(2,1),trueTest(3,1),'or');
plot3(true2Test(1,:),true2Test(2,:),true2Test(3,:),'Color', [0, 0.5, 0]);
plot3(true2Test(1,1),true2Test(2,1),true2Test(3,1),'o', 'Color',[0, 0.5, 0.1]);
plot3(modelTest(1,:),modelTest(2,:),modelTest(3,:),'m');
plot3(modelTest(1,1),modelTest(2,1),modelTest(3,1),'om');
plot3(daTest(1,:),daTest(2,:),daTest(3,:),'b');
plot3(daTest(1,1),daTest(2,1),daTest(3,1),'ob');
legend('Unperturbed true Solution', 'At t=15','Perturbed true Solution', 'At t=15','Forecast model solution', 'At t=15', 'DA Solution', 'At t=15')

%MSE
MSEtrue2    = sum(sum((trueTest - true2Test),2).^2)/201;
MSEmodel    = sum(sum((trueTest - modelTest),2).^2)/201;
MSEda       = sum(sum((trueTest - daTest),2).^2)/201;

%Find DDA solution with DDA_L63
[dda,~] = DDA_L63(obs, h, Q, R, trainInd,valInd, testInd);
testEnd = size(dda,2);

%Filter to test set
modelTest  = model(:,(testStart:testEnd));  % predicted state in test set
daTest     = da(:,(testStart:testEnd));     % DA state in test set
ddaTest    = dda(:,((testStart:testEnd):end));    % DDA state in test set
trueTest   = truth(:,(testStart:testEnd));  % true state in test set
true2Test  = truth2(:,(testStart:testEnd)); % true state in test set starting from obs

%Plot model on observation vs truth
figure;
plot3(trueTest(1,:),trueTest(2,:),trueTest(3,:),'r');
hold on;
plot3(trueTest(1,1),trueTest(2,1),trueTest(3,1),'or');
plot3(true2Test(1,:),true2Test(2,:),true2Test(3,:),'Color', [0, 0.5, 0]);
plot3(true2Test(1,1),true2Test(2,1),true2Test(3,1),'o', 'Color',[0, 0.5, 0.1]);
plot3(daTest(1,:),daTest(2,:),daTest(3,:),'b');
plot3(daTest(1,1),daTest(2,1),daTest(3,1),'ob');
plot3(ddaTest(1,:),ddaTest(2,:),ddaTest(3,:),'c');
plot3(ddaTest(1,1),ddaTest(2,1),ddaTest(3,1),'oc');
legend('Unperturbed true Solution', 'At t=T_{test}(1)','Perturbed true Solution', 'At t=T_{test}(1)', 'DA Solution', 'At t=T_{test}(1)', 'DDA Solution', 'At t=T_{test}(1)')

%MSE
MSEtrue2    = sum(sum((trueTest - true2Test),2).^2)/numTimeStepsTest;
MSEmodel    = sum(sum((trueTest - modelTest),2).^2)/numTimeStepsTest;
MSEda       = sum(sum((trueTest - daTest),2).^2)/numTimeStepsTest;
MSEdda      = sum(sum((trueTest - ddaTest),2).^2)/numTimeStepsTest;