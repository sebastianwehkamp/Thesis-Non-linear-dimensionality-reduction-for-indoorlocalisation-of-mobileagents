function [P] = initP(msize, warehouse_width, warehouse_length, wap_locs)
P = {};
temp = [];

dropout = 0.9;

x_step = warehouse_width/(msize(2)-1);
y_step = warehouse_length/(msize(1)-1);

for i=1:msize(2)
    x = (i-1) * x_step;
    for j=1:msize(1)
        y = warehouse_length - (j-1) * y_step;
        if rand(1) > dropout
            P{j,i} = relevance_norm(take_reading([x y 2], wap_locs));
        else
            P{j,i} = []';
    end
end


end

