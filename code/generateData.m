%% Specify parameters

num_aisles = 4;

length_aisle = 60;

width_aisles = 1.5;

width_stand = 2;

width_lane = 2;

height_aisle = 2;

%width_aisles = num_aisles * 3;

grid_step = 1;

num_paths = 100;
max_length = 200;
%% Generate data

% On a grid genrate the positions of the wireless access points
[grid_wap, wap_locs] = simuworld(6, 'y_length', 30);

% Set end locations
starts = zeros(num_aisles, 2);
for i = 1:num_aisles
    starts(i,:) = [(i-1)*width_stand+width_aisles/2+(i-1)*width_aisles length_aisle+height_aisle];
end

% Specify stands (corner coords)
stands = [];
for i = 1:num_aisles-1
   stands = [stands; [i*width_aisles+(i-1)*width_stand width_lane]];
   stands = [stands; [i*width_aisles+(i-1)*width_stand+width_stand width_lane]];
end

% Create grid
Xgrid = [0: grid_step: num_aisles*width_aisles+(num_aisles-1)*width_stand];
Ygrid = [0: grid_step: length_aisle+width_lane];

points = [];
labels = [];
sep_lab = [];
for i = 1:num_aisles
    for j = 1:num_paths
        path = starts(i,:);
        labels = [labels; i];
        curPos = path;
        curGridPos = [ceil(curPos(1)/grid_step) ceil(curPos(2)/grid_step)];
        sep_lab = [sep_lab; curGridPos(1)*curGridPos(2)+1];
        for k = 1:max_length
            % Determine which direction to move
            movementsH = [-1 0 1];
            movementsV = [0 -1];
            moveVert = movementsV(randperm(length(movementsV), 1));
            moveHorz = movementsH(randperm(length(movementsH), 1));

            % Save the old pos
            oldPos = curPos;

            % Move vertically
            if(moveVert == -1)
               curPos(2) = curGridPos(2) - grid_step*rand(1); 
               % check out of bounds
               if(curGridPos(2)+moveVert <= min(Ygrid))
                   moveVert = 0;
                   curPos = oldPos;
               end
            end

            if(moveHorz == -1)
               curPos(1) = curGridPos(1) - grid_step*rand(1); 
               % Out of bounds
               if(curGridPos(1)+moveHorz > max(Xgrid) || curGridPos(1)+moveHorz < min(Xgrid) )
                   moveHorz = 0;
                   curPos = oldPos;
               end
            else
                if(moveHorz == 1)
                curPos(1) = curGridPos(1) + grid_step*rand(1); 
                    % Out of bounds
                    if(curGridPos(1)+moveHorz > max(Xgrid) || curGridPos(1)+moveHorz < min(Xgrid) )
                        moveHorz = 0;
                        curPos = oldPos;
                    end           
                end
            end

            % Goal check
            if(curGridPos(1)<= 0 && curGridPos(2) <= 0)
                break;                       
            end
            
            % Check collision
            l = 1;
            collision = 0;
            for m = 1:num_aisles-1
                if (curPos(1) > stands(l,1) && curPos(1) < stands(l+1,1) && curPos(2) > stands(l,2))
                   curPos = oldPos;
                   collision = 1;
                   break;
                end
                l = l + 2;
            end

            % If no collision update position
            if (~collision)
                curGridPos = [curGridPos(1)+moveHorz curGridPos(2)+moveVert];
            end

            path = [path; curPos];
            
            % Determine label
            if(curPos(2)<width_lane)
                labels = [labels; num_aisles+1];
            else
                labels = [labels; i];
            end
            sep_lab = [sep_lab; curGridPos(1)*curGridPos(2)+1];
        end
        plot(path(:,1), path(:,2));
        hold on
        points = [points; path];
    end
end
grid on;
% Enlarge figure to full screen.
title('Random paths in the warehouse','FontSize',14)
set(gcf, 'Units', 'Normalized', 'Outerposition', [0, 0.05, 1, 0.95]);
axis square;

% Fix heights
points(:,3) = 2;
%% Compute RSSI values
strengths = ones(size(points,1),size(wap_locs,1));

for i = 1:size(points,1)
    strengths(i,:) = take_reading(points(i,:), wap_locs);
end

raw_data = [points, strengths];

strengths_norm = relevance_norm(strengths);

%% Generate label info
