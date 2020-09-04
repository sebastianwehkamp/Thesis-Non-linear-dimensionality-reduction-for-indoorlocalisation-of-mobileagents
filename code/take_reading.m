function strengths = take_reading(location, wap_locs)

% assumption of range of 46m

dists = pdist([location; wap_locs]);
dists(size(wap_locs,1)+1:end) = [];

fband = 2.4; %in GHz
N = 20;

ld0 = @() 20 * log10(fband*1000) - 28;
det_strength = @(x) -(ld0() + N * log10(x)) + random('Normal',0,2);

strengths = det_strength(dists);

% x = linspace(


end

