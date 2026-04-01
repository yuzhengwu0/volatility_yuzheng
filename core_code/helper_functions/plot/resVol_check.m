%% run this after resVol_mat has been computed
% can just type name of this script after you compute it in the main script
% and it will automatically make the histograms for you

vol_bool = vol~=0; % 0 = low volatility; 1 = high volatility
coh_levels = unique(coh);
vol_levels = unique(vol_bool);
% two volatility levels on each subplot
vol_string = {'low', 'high'};
colors = {'blue', 'red'};

%% Figure 1: resVol
% figure;
% t1 = tiledlayout(1, length(coh_levels));
% for i = 1:length(coh_levels)
%     nexttile;
%     hold on
%     for j = 1:length(vol_levels)
%         c = coh_levels(i);
%         v = vol_levels(j);
%         mask = vol_bool == v & coh == c;
%         histogram(resVol(mask), 'FaceColor', colors{j}, 'BinWidth', 0.25);
%         title(sprintf('coh = %.2f', c/100));
%     end
% end
% ylabel(t1, 'count')
% xlabel(t1, 'resVol\_mat (residual motion energy)')
% title(t1, 'resVol\_mat')
% legend({'low volatility', 'high volatility'}, 'Location', 'northeastoutside')

%% Figure 2: log_resVol
figure;
t2 = tiledlayout(1, length(coh_levels));
for i = 1:length(coh_levels)
    nexttile;
    hold on
    for j = 1:length(vol_levels)
        c = coh_levels(i);
        v = vol_levels(j);
        mask = vol_bool == v & coh == c;
        histogram(zlog_vol(mask), 'FaceColor', colors{j}, 'BinWidth', 0.25);
        title(sprintf('coh = %.2f', c/100));
    end
end
ylabel(t2, 'count')
xlabel(t2, 'log\_resVol\_mat (residual motion energy)')
title(t2, 'z-scored log\_resVol\_mat')
legend({'low volatility', 'high volatility'}, 'Location', 'northeastoutside')