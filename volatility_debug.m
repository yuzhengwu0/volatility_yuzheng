%% compute volatility as difference from expected coherence
all = load(data_path).all;
momentary_coh = all.rdm1_cohframes(idx);
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
trial_coh = repmat(trial_coh, [1, 17]);
% make trial coherence signed (-1 = left, +1 = right)
trial_coh = trial_coh .* trial_dir;
% use the same signage for momentary coherence
moco_mat = moco_mat .* trial_dir;

% compute frame-by-frame difference in coherence
moco_diff = moco_mat - trial_coh;

%% vanessa to-do: make two plots
% one that could be our third panel of Figure 2 from the original paper: for a low volatility and high volatility trial where coh = 0.128, plot moco_diff over time
% one that is similar to the big motion energy figure you've been making. column 1 is momentary coherence, column 2 is "moco_diff"