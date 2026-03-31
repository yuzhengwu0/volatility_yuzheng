function vif = computeVIF(mdl)
    % Get the predictor matrix (excluding intercept)
    X = mdl.Variables{:, mdl.PredictorNames};
    
    p = size(X, 2);
    vif = zeros(p, 1);
    
    for i = 1:p
        % Regress predictor i on all other predictors
        y_i = X(:, i);
        X_others = X(:, setdiff(1:p, i));
        
        % Fit OLS regression
        mdl_i = fitlm(X_others, y_i);
        
        % VIF = 1 / (1 - R^2)
        vif(i) = 1 / (1 - mdl_i.Rsquared.Ordinary);
    end
    
    % Display results
    for i = 1:p
        fprintf('VIF(%s) = %.4f\n', mdl.PredictorNames{i}, vif(i));
    end
end