% Compute mean evidence strength per trial
meanme = mean(evidence_strength, 2);

% Define condition masks (volatility)
low_vol  = cond == 1;
high_vol = cond == 2;

% Define coherence masks
coh0   = coh_weuse == 0;
coh32  = coh_weuse == 0.32;
coh64  = coh_weuse == 0.64;
coh128 = coh_weuse == 1.28;
coh256 = coh_weuse == 2.56;

coherence_levels = [0, 0.32, 0.64, 1.28, 2.56];
coh_masks = {coh0, coh32, coh64, coh128, coh256};

n_coh = numel(coherence_levels);

counts_low_neg   = zeros(1, n_coh);  % meanme <= 0
counts_low_pos   = zeros(1, n_coh);  % meanme > 0
counts_high_neg  = zeros(1, n_coh);
counts_high_pos  = zeros(1, n_coh);

for i = 1:n_coh
    trials_low  = meanme(low_vol  & coh_masks{i});
    trials_high = meanme(high_vol & coh_masks{i});

    counts_low_neg(i)  = sum(trials_low  <= 0);
    counts_low_pos(i)  = sum(trials_low  > 0);
    counts_high_neg(i) = sum(trials_high <= 0);
    counts_high_pos(i) = sum(trials_high > 0);
end

% --- Build stacked bar data ---
% Layout: for each coherence level, we have 2 groups (low, high)
% Each group is a stacked bar: [neg_count, pos_count]
% We interleave: columns = [low_coh1, high_coh1, low_coh2, high_coh2, ...]

n_bars = n_coh * 2;
neg_counts = zeros(1, n_bars);
pos_counts = zeros(1, n_bars);

for i = 1:n_coh
    neg_counts(2*i-1) = counts_low_neg(i);
    neg_counts(2*i)   = counts_high_neg(i);
    pos_counts(2*i-1) = counts_low_pos(i);
    pos_counts(2*i)   = counts_high_pos(i);
end

% --- Plot ---
figure;
bar_data = [neg_counts; pos_counts]';  % n_bars x 2, stacked

b = bar(bar_data, 'stacked');

% Bottom (colored): low=blue, high=red — but same color per stack layer
% We'll use one color for the "neg" layer and white for "pos" layer
b(1).FaceColor = 'flat';
b(2).FaceColor = 'flat';

% Assign colors per bar
low_color  = [0.2 0.4 0.8];   % blue for low vol
high_color = [0.8 0.2 0.2];   % red for high vol
white      = [1.0 1.0 1.0];

for i = 1:n_coh
    % Bottom layer (neg): colored by vol type
    b(1).CData(2*i-1, :) = low_color;
    b(1).CData(2*i,   :) = high_color;
    % Top layer (pos): white / empty look
    b(2).CData(2*i-1, :) = white;
    b(2).CData(2*i,   :) = white;
end

% Add black edge to top (pos) layer so bars still visible
b(2).EdgeColor = [0 0 0];
b(1).EdgeColor = [0 0 0];

% X-axis labels: one label per pair, centered
coh_labels = {'0', '0.32', '0.64', '1.28', '2.56'};
tick_positions = 1.5 : 2 : n_bars;   % center between each pair
set(gca, 'XTick', tick_positions, 'XTickLabel', coh_labels);

% Dividers between coherence groups
hold on;
for i = 1:n_coh-1
    xline(2*i + 0.5, '--k', 'Alpha', 0.3);
end

xlabel('Coherence Level');
ylabel('Number of Trials');
title('Trials with meanme \leq 0 (colored) vs > 0 (white) by Coherence & Volatility');

% Custom legend
patch_low  = patch(NaN, NaN, low_color);
patch_high = patch(NaN, NaN, high_color);
patch_pos  = patch(NaN, NaN, white, 'EdgeColor', 'k');
legend([patch_low, patch_high, patch_pos], ...
    {'Low Vol (≤0)', 'High Vol (≤0)', 'meanme > 0'}, ...
    'Location', 'best');

grid on;
box on;