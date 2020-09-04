function [grid, wap_locs] = simuworld(x_length, varargin)

    p = inputParser;
    p.addRequired('x_length', @isfloat); 

    p.addParameter('y_length', 60, @(x)(x ~= x_length & isfloat(x))); %Forcing a rectangular grid
    p.addParameter('height', 5, @(x)(x > 2 & isfloat(x)));
    
    p.FunctionName = 'simuworld';
    % Parse and validate all input arguments.
    p.parse(x_length, varargin{:});
    
    height = p.Results.height;
    y_length = p.Results.y_length;
    %addpath('distmesh');
    
%     fd = inline('drectangle ( p, -1.0, 1.0, -1.0, 1.0 )','p');
%     dbox = [ 0, 0; x_length, y_length ];
%     iteration_max = 50;
%     [ p, t ] = distmesh_2d( fd, @huniform, 0.5, dbox, iteration_max, []);
    
    
    [X, Y, Z] = meshgrid(0:x_length, 0:y_length, 0:height);
    
    %test
    x_length = x_length *2;
    y_length = y_length *2;
    
    x_locs = linspace(0,x_length,x_length+1)';
    y_locs = zeros(x_length+1,1);
    y_locs(1:end) = y_length;
%     wap_locs = ones(x_length+1,2);%[0:x_length, 0:2:y_length,height];
    wap_locs = [x_locs, zeros(x_length+1,1), repmat(height,x_length+1,1)];
    wap_locs(2:2:end,2) = wap_locs(2:2:end,2) + 0.5;
    
    temp_jit_pos = [x_locs, (y_locs/2)-0.5, repmat(height,x_length+1,1)];
    temp_jit_pos(2:2:end,2) = temp_jit_pos(2:2:end,2) + 1;
    wap_locs = [wap_locs; temp_jit_pos];
    
    temp_jit_pos = [x_locs, y_locs, repmat(height,x_length+1,1)];
    temp_jit_pos(2:2:end,2) = temp_jit_pos(2:2:end,2) - 0.5;
    wap_locs = [wap_locs; temp_jit_pos];

    
    grid = X;    

end