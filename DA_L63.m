function [analysis, model_on_DA] = DA_L63(obs,h,numTimeSteps,Q,R)
    % the parameter values used in L63.m
    rho     = 28; sigma = 10; beta = 8/3;   

    %Initialisation for the Kalman filter 
    posterior   = obs(:,1);                     % First run of KF is on the observed initial state
    model_on_DA = posterior;                    % First column of num predicted values (x^M,y^M,z^M)
    analysis    = posterior;                    % First column of DA values (x^da,y^da,z^da)
    P_posterior = .1*eye(length(posterior));    % Initial covariance (0.1,0,0; 0,0.1,0; 0,0,0.1)
    H           = eye(3);                       % Observation function

    %EXTENDED KALMAN FILTER for training set
    for j = 2:numTimeSteps
        % Prediction step
        prior = Euler_L63(posterior, h, rho, sigma, beta);	% wj+1 = Euler_L63(wj^DA)
        model_on_DA(:,j) = prior;                        	% add num predicted value to (x^M,y^M,z^M)
        F = [1-sigma*h,         h*sigma,       0; 
            (rho-prior(3))*h, 1-h,           -h*prior(1);
            h*prior(2),       h*prior(1),  1-beta*h];       % TLM of num model
        P_prior = F*P_posterior*F' + Q;                     % Pj+1 = FPjF' + Q

        % Observation update
        K               = (P_prior*H')/(H*P_prior*H'+R);	% Kj = Pj*H'*inv(H*Pj*H'+R)
        posterior       = prior + K*(obs(:,j) - H*prior);   % wj = wj + Kj(yj-Hwj)
        P_posterior     = P_prior - K*H*P_prior;            % Pj = (I-KjH)Pj
        analysis(:,j)   = posterior;                        % add wj to (x^da,y^da,z^da)
    end
end