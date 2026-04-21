%% get individual betas

fitted_models = struct();

for a = 1:length(unique(Correct))
    corr = Correct == a;
    for z = unique(cond)
        volcond = cond == 2;
        for s = unique(subjID)
            sub = subjID == s;
            mask = corr & volcond & sub;
            y = y_var(mask);
            x = x_var(mask);
            fitted_models(a, z, s) = fitlm(x, y);
        end
    end 
end 