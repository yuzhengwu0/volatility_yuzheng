%% compute volatility as difference from expected coherence
all = load(data_path).all;
momentary_coh = all.rdm1_cohframes(idx);
nTrials = length(momentary_coh);
moco_mat = NaN(nTrials, 17);
for t = 1:height(momentary_coh)
    moco_mat(t, :) = momentary_coh{t}';
end

% get direction of stim on each trial
trial_dir = all.rdm1_dir(idx);
trial_dir(trial_dir==180) = -1; 
trial_dir(trial_dir==0) = 1;
% get trial coherence
trial_coh = cfg.coh ./ 1000;
coh_mat = repmat(trial_coh, [1, 17]);
% make trial coherence signed (-1 = left, +1 = right)
coh_mat_dir = coh_mat.* trial_dir;
% use the same signage for momentary coherence
moco_mat_dir = moco_mat .* trial_dir;

% compute frame-by-frame difference in coherence
moco_diff = moco_mat_dir - coh_mat_dir;


% compute vol2: history-dependent operationalization of volatility
vol2_signed = NaN(allTrials, nFrames);
for t = 1:allTrials
    for f = 2:nFrames
        vol2_signed(t, f) = moco(t, f) - moco(t, f-1);
    end
end

vol2_abs = abs(vol2_signed);

%% vanessa to-do: make two plots
% one that could be our third panel of Figure 2 from the original paper: for a low volatility and high volatility trial where coh = 0.128, plot moco_diff over time
% one that is similar to the big motion energy figure you've been making. column 1 is momentary coherence, column 2 is "moco_diff"

%% plot 1: momentary volatility when coh = 0.128
figure
hold on
plot(moco_diff(12,:), 'Color', [1 0 0])
plot(moco_diff(13,:), 'Color', [0 0 1])
title('momentary cohrerence diff single trials')
legend('{high vol}', '{low vol}')

%% plot 2: momentary coherence & moco diff by coh and vol level 
clear yline
figure;
tiledlayout(length(unique(cfg.coh)), 2)

thiscoh = unique(cfg.coh);
for icoh = 1:numel(thiscoh)

    
    % momentary coherence
    nexttile;
    hold on;

    % cond == 2
    idx2 = (cfg.coh == thiscoh(icoh)) & (cfg.vol == 256);
    data2 = moco_mat(idx2, :);
    plot(data2', 'Color', [1 0 0 0.05], 'LineWidth', 0.005);
    % mu2 = mean(data2, 1, 'omitnan');
    % sd2 = std(data2, 0, 1, 'omitnan');

    % cond == 1
    idx1 = (cfg.coh == thiscoh(icoh)) & (cfg.vol == 0);
    data1 = moco_mat(idx1, :);
    plot(data1', 'Color', [0 0 1 0.05], 'LineWidth', 0.005);
    % mu1 = mean(data1, 1, 'omitnan');
    % sd1 = std(data1, 0, 1, 'omitnan');

    % x = 1:size(motion_mat, 2);
    % plot(x, mu1, 'b', 'LineWidth', 1.5)
    % plot(x, mu2, 'r', 'LineWidth', 1.5)
    % errorbar(x, mu1, sd1, 'b', 'LineWidth', 1.5);
    % errorbar(x, mu2, sd2, 'r', 'LineWidth', 1.5);

    % ylim([-0.0005 0.0005])
    title(sprintf('coh = %g', thiscoh(icoh)))
    xlabel('windows')
    ylabel('momentary coh')
    yline(0, 'HandleVisibility', 'off');





    % moco diff
    nexttile;
    hold on;

    % cond == 2
    idx2 = (cfg.coh == thiscoh(icoh)) & (cfg.vol == 256);
    data2 = moco_diff(idx2, :);
    plot(data2', 'Color', [1 0 0 0.05], 'LineWidth', 0.005);
    % mu2 = mean(data2, 1, 'omitnan');
    % sd2 = std(data2, 0, 1, 'omitnan');

    % cond == 1
    idx1 = (cfg.coh == thiscoh(icoh)) & (cfg.vol == 0);
    data1 = moco_diff(idx1, :);
    plot(data1', 'Color', [0 0 1 0.05], 'LineWidth', 0.005);
    % mu1 = mean(data1, 1, 'omitnan');
    % sd1 = std(data1, 0, 1, 'omitnan');

    % x = 1:size(motion_mat, 2);
    % plot(x, mu1, 'b', 'LineWidth', 1.5)
    % plot(x, mu2, 'r', 'LineWidth', 1.5)
    % errorbar(x, mu1, sd1, 'b', 'LineWidth', 1.5);
    % errorbar(x, mu2, sd2, 'r', 'LineWidth', 1.5);

    % ylim([-0.0005 0.0005])
    title(sprintf('coh = %g', thiscoh(icoh)))
    xlabel('windows')
    ylabel('mocoh diff')
    yline(0, 'HandleVisibility', 'off');

    
end
h1 = plot(nan, nan, 'b', 'LineWidth', 2);
h2 = plot(nan, nan, 'r', 'LineWidth', 2);
legend([h1 h2], {'low vol', 'high vol'})