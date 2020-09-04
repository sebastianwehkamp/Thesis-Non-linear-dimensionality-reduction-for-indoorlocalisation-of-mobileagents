meanVal = mean(mean(strengths_norm)');
medianVal = median(median(strengths_norm)');
stdVal = std(std(strengths_norm)');

boxplot(strengths)


x = linspace(-100,0,400);

sigmoid = 1 ./ (1+exp(-0.1*(x+45)) ) ;

figure;

plot(x,sigmoid,'linew',3);
hold on

syms x
ezplot(1/(1+exp(-0.1*(x+45))), [-100, 0, 0, 1])
title('Non-linear transformation from dB scale to [0-1] scale');
