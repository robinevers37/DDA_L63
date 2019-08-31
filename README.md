# DDA_L63
Code for the use of deep data assimilation on the Lorenz 63 system

All the code used in this project can be found in this Github depository .
Two main files exist. The file main.m will produce all the results presented in chapter 2, whereas the file runfile.m produces the results from chapter 3 and 4. 

The programs referred to in this code, with their corresponding uses are

L63.m
input: $u=(x_0,y_0,z_0)$, $\rho$, $\sigma$, $\beta$ \\
output: The answers to the Lorenz equations $dx/dt(u), dy/dt(u), dz/dt(u)$ with parameters $\rho$, $\sigma$, $\beta$ 

Euler_L63.m
input: $u=(x_0,y_0,z_0)$, time step, $\rho$, $\sigma$, $\beta$
output: The Euler discretisation per time step of solution to the Lorenz-system for model values $\rho$, $\sigma$, $\beta$ with initial value $u$

FFN_L63.m
input: input (3 features), output (3 responses), trainInd, valInd, testInd
output: A feedforward network trained on the trainInd indexes and validated on the valInd indexes of the input and output sets.

RNN_L63.m
input: input (3 features), output (3 responses), trainInd, valInd, testInd
output: A recurrent network trained on the trainInd indexes and validated on the valInd indexes of the input and output sets.

LSTM_L63.m
input: input (3 features), output (3 responses), trainInd, valInd
output: A recurrent network with LSTM layer trained on the trainInd indexes and validated on the valInd indexes of the input and output sets.

DA_L63.m
input: observations, time step, total number of time steps, covariance matrices $\textbf{Q}$ and $\textbf{R}$
output: The DA predictions and model-forecast per time step, using  \texttt{Euler\_L63.m} as forecasting model
  
DDA_L63.m
input: observations, time step,  trainInd, valInd, testInd, covariance matrices $\textbf{Q}$ and $\textbf{R}$
output: The DDA predictions for the testInd indexes and a recurrent network with LSTM layer trained on the trainInd indexes and validated on the valInd indexes of the input and output sets.
