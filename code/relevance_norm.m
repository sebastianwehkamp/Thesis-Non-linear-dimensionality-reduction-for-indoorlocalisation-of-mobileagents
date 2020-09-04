function normed_X = relevance_norm(X,gred,med)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if ~exist('gred','var'), gred = 0.1; end
if exist('med','var') 
    x0 = med;
else
    x0 = median(X(:));
end
% max_val = max(X(:));
% min_val = min(X(:));
% x0 = median(X(:));
% disp(['med:' num2str(x0)]);
% disp(['mean:' num2str(mean(X(:)))]);
% %% g function norm
% h = @(ss,beta) (exp(-beta* ((ss + 50)./-50) +beta)-1)/(exp(2*beta)-1);
% 
% normed_X = h(X,2);
% 
% % g = @(cosa,beta) (exp(-beta*cosa+beta)-1)/(exp(2*beta)-1);
% % g(-0.8, 0.5)

%% sigmoid norm
g = @(x) 1 ./ (1 + exp(-gred .* (x-x0)));%0.2

normed_X = g(X);

%% visualise
% x = linspace(-100, 0);
% 
% %y = h(x,0.5);
% % xt = (x + 100)./100;
% y = g(x);
% figure;
% plot(x,y)
% end

