%Define Lorenz-system for model values 28,10,8/3
function dx = L63(x, rho, sigma, beta)
    %rho     = 28; sigma = 10; beta = 8/3;
    dx      = zeros(3,size(x,2));
    dx(1,:) = sigma*(x(2,:) - x(1,:));
    dx(2,:) = x(1,:).*(rho - x(3,:)) - x(2,:);
    dx(3,:) = x(1,:).*x(2,:) - beta*x(3,:);
end