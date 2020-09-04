figure,
title('Raw position of measurements','FontSize',14);

hold on
gscatter(points(:,1), points(:,2), labels);
hold off;

legend('1', '2', '3', '4', '5', 'Sample Vectors');