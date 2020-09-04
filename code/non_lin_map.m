function [ L,lo, varargout ] = non_lin_map(data,omegas,r,dim, varargin)
%The algorithm
%   Detailed explanation goes here
%% Parameter handling
    nout = max(nargout,1)-2;
    p = inputParser;   % Create an instance of the class.
    p.addRequired('data', @isfloat);
    p.addRequired('omegas', @iscell);
    p.addRequired('r', @isfloat);
    p.addRequired('dim',@(x)(x>=2 & isfloat(x)));

    p.addParamValue('max_iter', 100, @isfloat);
    p.addParamValue('perplexity',15,@(x)(isfloat(x) && length(x)==1));
    p.addParamValue('epsilon', [0.3,0.03], @(x)(isfloat(x) & (length(x)==2 | length(x)==p.Results.max_iter)) | isa(x,'function_handle'));
    p.addParamValue('alphas', [0.3,0.03], @(x)(isfloat(x) & (length(x)==2 | length(x)==p.Results.max_iter)) | isa(x,'function_handle'));
    p.addParamValue('w', [], @(x)((isfloat(x) & size(x,1)==length(omegas) & size(x,2)==size(omegas{1},2)))|isempty(x));
    p.addParamValue('onlyOffset',1,@(x) isinteger(int8(x)));
    p.addParamValue('Initial_L',[]);
    p.addParamValue('Initial_lo',[],@(x)(size(x,2)==dim));
    p.addParamValue('gamma',1,@(x)(isfloat(x) && length(x)==1));
    p.addParamValue('costs', false, @(x)isa(logical(x),'logical'));
    p.addParamValue('savePath', 'results', @(x)isa(x,'char'));
%     p.addParamValue('saveIter', false, @(x)isa(logical(x),'logical'));
    p.addParamValue('saveToFile',false, @(x)isa(logical(x),'logical'))

    p.CaseSensitive = true;
    p.FunctionName = 'non_lin_map';
    % Parse and validate all input arguments.
    p.parse(data,omegas,r, dim, varargin{:});
    % Display all arguments.
    disp 'List of all arguments:'
    disp(p.Results);
 %% Learning Rates    
    % compute the vector of epochs learning rates alpha for the prototype learning
    if isa(p.Results.epsilon,'function_handle')
        % with a given function specified from the user
        epsilons = arrayfun(p.Results.epsilon, 1:p.Results.max_iter);
    elseif length(p.Results.epsilon)==p.Results.max_iter
        epsilons = p.Results.epsilon;
    else
        % or use an decay with a start and an end value
        eps1 = p.Results.epsilon(1);
        eps2 = p.Results.epsilon(2);	
        epsilons = arrayfun(@(t) eps1 * exp(-(log(eps1/eps2)/p.Results.max_iter)*t), 1:p.Results.max_iter);
    end
    alphas = zeros(1,p.Results.max_iter);
    if isa(p.Results.alphas,'function_handle')
        % with a given function specified from the user
        alphas = arrayfun(p.Results.alphas, 1:p.Results.max_iter);
    elseif length(p.Results.alphas)==p.Results.max_iter
        alphas = p.Results.alphas;
    else
        % or use an decay with a start and an end value
        al1 = p.Results.alphas(1);
        al2 = p.Results.alphas(2);	
        alphas(p.Results.onlyOffset:p.Results.max_iter) = arrayfun(@(x) al1 / (1+(x- p.Results.onlyOffset)*al2), p.Results.onlyOffset:p.Results.max_iter);        
        %alphas(p.Results.onlyOffset:p.Results.max_iter) = arrayfun(@(t) al1 * exp(-(log(al1/al2)/p.Results.max_iter)*t), 1:p.Results.max_iter);
    end
%% Initialize
    n = size(data, 1);
    K = length(omegas);
    if isempty(p.Results.Initial_L)
        L = rand(dim,size(omegas{1},1),K)*2-1;
        disp('random init for L');
    else
        L = p.Results.Initial_L;
    end 
    if isempty(p.Results.Initial_lo)
        lo = rand(K,dim)*2-1;
        disp('random init for lo');
    else
        lo = p.Results.Initial_lo;
    end
    gamma = p.Results.gamma;
    w = p.Results.w;
%     crisp = p.Results.crisp;
if length(unique(r))>2
     crisp = 0;disp('non-crisp responsibilities found');
else
    if unique(r)==[0;1], crisp = 1;disp('crisp responsibilities found');else crisp = 0;end
end
%% additional outputs?
    if ~exist((p.Results.savePath),'dir'), mkdir(p.Results.savePath); end
    all_L = cell(1,p.Results.max_iter+1);
    all_lo= cell(1,p.Results.max_iter+1);
    if nout>=1,% || p.Results.saveIter,
        all_L{1} = L;
        all_lo{1}= lo;
        if p.Results.saveToFile,
            dlmwrite([p.Results.savePath,'/Parameter.txt'], dim, '-append');
            dlmwrite([p.Results.savePath,'/Parameter.txt'], '  ', '-append', 'delimiter', '\n');
            dlmwrite([p.Results.savePath,'/Parameter.txt'], p.Results.perplexity, '-append');
            dlmwrite([p.Results.savePath,'/Parameter.txt'], '  ', '-append', 'delimiter', '\n');            
            dlmwrite([p.Results.savePath,'/Parameter.txt'], p.Results.max_iter);
            dlmwrite([p.Results.savePath,'/Parameter.txt'], '  ', '-append', 'delimiter', '\n');
            dlmwrite([p.Results.savePath,'/Parameter.txt'], p.Results.epsilon, '-append');
            dlmwrite([p.Results.savePath,'/Parameter.txt'], '  ', '-append', 'delimiter', '\n');
%             dlmwrite([p.Results.savePath,'/Parameter.txt'], ['Init_L=[',num2str(p.Results.Initial_L),']'], '-append');
%             dlmwrite([p.Results.savePath,'/Parameter.txt'], '  ', '-append', 'delimiter', '\n');            
%             dlmwrite([p.Results.savePath,'/Parameter.txt'], ['Init_lo=[',num2str(p.Results.Initial_lo),']'], '-append');
%             dlmwrite([p.Results.savePath,'/Parameter.txt'], '  ', '-append', 'delimiter', '\n'); 
        end
    end
%% compute P values
P = x2p(data, p.Results.perplexity, 1e-5);
P = 0.5/n * (P + P'); % symmetrize
P = max(P, eps);
% P = P ./ sum(P(:)); % normalize
% P = P * 4;                    % prevent local minima by lying about P-vals    
allCosts = zeros(1,p.Results.max_iter);
initialDim = size(omegas{1},1);
kX = cell(1,K);
for k=1:K
    if isempty(w)
        kX{k} = (omegas{k}*data')';
    else
        kX{k} = (omegas{k}*bsxfun(@minus,data,w(k,:))')';
    end
end
%% perform epochs
    for iter=1:p.Results.max_iter,
        if mod(iter,50)==0, disp(iter);end
%         if mod(iter,2)==0, disp(iter);end
        % Compute joint probability that point i and j are neighbors
        ykdata = zeros(n,dim,K);
        for k=1:K
            rk = r(:,k);lok = lo(k,:);
            ykdata(:,:,k) = rk(:,ones(1,dim)).*((L(:,:,k)*kX{k}')'+lok(ones(1,n),:));
%             ykdata(:,:,k) = rk(:,ones(1,dim)).*((L(:,:,k)*(omegas{k}*data'))'+lok(ones(1,n),:));
        end        
        ydata = sum(ykdata,3);
        sum_ydata = sum(ydata .^ 2, 2);                                         % precomputation for pairwise distances
        y_dist = bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')));
        brace = 1 + y_dist./gamma;
        num = brace.^(-(gamma+1)/2);  % Student-t distribution
        num(1:n+1:end) = 0;                                                     % set diagonal to zero
        Q = num ./ sum(num(:));                                                 % normalize to get probabilities
%         sumnum=sum(num,2);
%         Q = num./sumnum(:,ones(1,n));
        Q = max(Q, eps);
        
        % Compute the gradients
        stiffnesses = (gamma+1)/gamma * (P - Q) .* 1./brace;%n.*
%         tic;
        
if crisp,
        for k=1:K
                if p.Results.onlyOffset>=iter,test_LK = zeros(size(L,1),size(L,2),n);end
                sumjlo = zeros(n,size(lo,2));
                rk = r(:,k);
                if ~isempty(find(rk==1)),
                    for i=1:n                    
                        diff2 = bsxfun(@minus , ydata(i,:), ydata);                    
                        diff3 = bsxfun(@minus,r(i,k),rk);
                        if p.Results.onlyOffset<iter,
                            if r(i,k)==0,
                                diff1 = zeros(initialDim,n);
                                oneIdx = find(rk==1);
    %                             diff1(:,oneIdx) = bsxfun(@minus,zeros(1,dim),rk(oneIdx,ones(1,dim)).*(omegas{k}*data(oneIdx,:)')')';
                                diff1(:,oneIdx) = bsxfun(@minus,zeros(1,initialDim),rk(oneIdx,ones(1,initialDim)).*kX{k}(oneIdx,:))';
                                testcell = cell([1,n]);
                                testcell(oneIdx) = arrayfun(@(x) diff1(:,x)*diff2(x,:).*stiffnesses(i,x),oneIdx,'UniformOutput',false);
                            else
    %                             diff1 = bsxfun(@minus , r(i,k).*(omegas{k}*data(i,:)')', rk(:,ones(1,dim)).*(omegas{k}*data')')';
                                diff1 = bsxfun(@minus , r(i,k).*kX{k}(i,:), rk(:,ones(1,initialDim)).*kX{k})';
                                testcell = arrayfun(@(x) diff1(:,x)*diff2(x,:).*stiffnesses(i,x),1:n,'UniformOutput',false);
                            end
                            test_LK(:,:,i) = sum(cat(3,testcell{:}),3)';
                        end
                        sts = stiffnesses(i,:)';
                        sumjlo(i,:) = sum(bsxfun(@times,bsxfun(@times,diff2,diff3),sts(:,ones(1,dim))));
                    end
                    if p.Results.onlyOffset<iter,
                        diffLK = sum(test_LK,3);                   
                        L(:,:,k) = L(:,:,k) - epsilons(iter).*diffLK;
                    end
                    difflok = sum(sumjlo);
                    clear testcell test_LK sts diff1 diff2 diff3 sumjlo;                
                    lo(k,:)  = lo(k,:)  - epsilons(iter).*difflok;
                end
        end
else
        for k=1:K
                if p.Results.onlyOffset>=iter,test_LK = zeros(size(L,1),size(L,2),n);end
                sumjlo = zeros(n,size(lo,2));
                rk = r(:,k);
                for i=1:n
                    % Epsilon^i- Epsilon^j
                    diff2 = bsxfun(@minus , ydata(i,:), ydata); 
                    % rik - rjk
                    diff3 = bsxfun(@minus,r(i,k),rk);
                    if p.Results.onlyOffset<iter,
%                             diff1 = bsxfun(@minus , r(i,k).*(omegas{k}*data(i,:)')', rk(:,ones(1,dim)).*(omegas{k}*data')')';
                        diff1 = bsxfun(@minus , r(i,k).*kX{k}(i,:), rk(:,ones(1,initialDim)).*kX{k})';
                        testcell = arrayfun(@(x) diff1(:,x)*diff2(x,:).*stiffnesses(i,x),1:n,'UniformOutput',false);
                        test_LK(:,:,i) = sum(cat(3,testcell{:}),3)';
                    end
                    sts = stiffnesses(i,:)';
                    sumjlo(i,:) = sum(bsxfun(@times,bsxfun(@times,diff2,diff3),sts(:,ones(1,dim))));
                end
                if p.Results.onlyOffset<iter,
                    diffLK = sum(test_LK,3);
                    L(:,:,k) = L(:,:,k) - epsilons(iter).*diffLK;
                end
                difflok = sum(sumjlo);
                clear testcell test_LK sts diff1 diff2 diff3 sumjlo;                
                lo(k,:)  = lo(k,:)  - epsilons(iter).*difflok;
        end
end
%         toc;disp(toc);        
        if p.Results.costs,
%             i_cost = zeros(1,n);
%             for i=1:n
%                 i_cost(i) = sum(P(i,:).*log(P(i,:)./Q(i,:)));
%             end
%             costs = sum(i_cost);clear i_cost;
            costs = sum(sum(P .* log((P + eps) ./ (Q + eps)), 2));
            if mod(iter,50)==0, disp(['iteration ',num2str(iter),' cost: ',num2str(costs)]);end
            allCosts(iter) = costs;
        end        
        if nout>=1,% || p.Results.saveIter,
            all_L{iter+1} = L;
            all_lo{iter+1} = lo;
        end
    end
%% saving?
if p.Results.saveToFile,
    if ~exist((p.Results.savePath),'dir'), mkdir(p.Results.savePath); end
    save([p.Results.savePath,'/non_lin_map_result'],'data','dim','L','lo','all_L','all_lo','epsilons','p','allCosts');
end
%% optional outputs
    varargout = cell(nout);
    for k=1:nout
        switch(k)
            case(1)
                varargout(k) = {all_L};
            case(2)
                varargout(k) = {all_lo};
            case(3)
                varargout(k) = {allCosts};
        end
    end
    
%  langsame Rechnung   
if(0)
        diffA = zeros(size(A));
        for i=1:n
%             y_grads(i,:) = sum(bsxfun(@times, bsxfun(@minus, ydata(i,:), ydata), stiffnesses(:,i)), 1);
            for j=1:n
                diffA = diffA+stiffnesses(i,j).*bsxfun(@times,bsxfun(@minus , data(i,:)*A', data(j,:)*A')',data(i,:)-data(j,:));
            end
        end
end