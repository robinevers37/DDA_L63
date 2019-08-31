%Define Euler discretisation of solution to Lorenz-system for model values rho,sigma,beta
function x = Euler_L63(x,timestep,rho,sigma,beta)
     h = 0.001;
     n = timestep/h;
     for i = 1:n
        x = x + h*L63(x,rho,sigma,beta);
     end
end