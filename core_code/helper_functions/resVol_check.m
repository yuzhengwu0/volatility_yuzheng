%% run this after resVol_mat has been computed
% can just type name of this script after you compute it in the main script
% and it will automatically make the histograms for you

vol_bool = vol~=0; % 0 = low volatility; 1 = high volatility
coh_levels = unique(coh);
vol_levels = unique(vol_bool);

%% two volatility levels on each subplot
vol_string = {'low', 'high'};
colors = {'blue', 'red'};
t = tiledlayout(1, length(coh_levels));
for i = 1:length(coh_levels)
    nexttile;
    hold on
    for j = 1:length(vol_levels)
        c = coh_levels(i);
        v = vol_levels(j);
        v_string = vol_string{v+1};
        mask = vol_bool == v & coh == c;
        histogram(resVol(mask), 'FaceColor', colors{j}, 'BinWidth', 0.25);
        %histogram(resVol(mask), 'FaceColor', colors{j}, 'BinWidth', 1e-6);
        %histogram(resVol_mat(mask), 'FaceColor', colors{j}, 'BinWidth', 0.00001);
        title(sprintf('coh = %.2f', c/100));
    end
end

ylabel(t, 'count')
xlabel(t, 'resVol\_mat (residual motion energy)')
title(t, 'z-scored resVol\_mat')
legend({'low volatility', 'high volatility'}, 'Location', 'northeastoutside')