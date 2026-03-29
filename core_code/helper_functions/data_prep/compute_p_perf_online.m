function [p_perf_online, combination_counter, combination_performance] = compute_p_perf_online(subjID, cond, coh, Correct)
% Compute cumulative performance estimate within each subject, separately for
% each condition × coherence combination.
%
% Inputs:
%   subjID   : subject ID vector
%   cond     : condition vector (volatility - low/high coded as 1/2)
%   coh      : coherence vector
%   Correct  : trial-wise accuracy vector (0/1)
%
% Outputs:
%   p_perf_online           : online performance estimate for each trial
%   combination_counter     : trial counter for each subject × combination
%   combination_performance : accumulated correct count for each subject × combination

    nTrials = numel(subjID);

    % initialize output
    p_perf_online = 0.5 * ones(nTrials, 1);

    % unique subject / condition / coherence values
    subj_list = unique(subjID(~isnan(subjID)));
    cond_list = unique(cond(~isnan(cond)));
    coh_list  = unique(coh(~isnan(coh)));

    nSubj = numel(subj_list);
    total_combinations = numel(cond_list) * numel(coh_list);

    % initialize counters
    combination_counter = zeros(nSubj, total_combinations);
    combination_performance = zeros(nSubj, total_combinations);

    % 2 pseudo-trials with 1 correct -> initial performance = 0.5
    combination_counter(:) = 2;
    combination_performance(:) = 1;

    % loop over subject
    for iSub = 1:nSubj
        thisSub = subj_list(iSub);

        % all trial indices for this subject, in original order
        trial_idx = find(subjID == thisSub);

        for k = 1:numel(trial_idx)
            tr = trial_idx(k);

            this_cond = cond(tr);
            this_coh  = coh(tr);
            this_correct = Correct(tr);

            % skip invalid trial if needed
            if isnan(this_cond) || isnan(this_coh) || isnan(this_correct)
                continue;
            end

            cond_idx = find(cond_list == this_cond, 1);
            coh_idx  = find(coh_list  == this_coh,  1);

            if isempty(cond_idx) || isempty(coh_idx)
                continue;
            end

            % map (cond, coh) to one column index
            combo_idx = (cond_idx - 1) * numel(coh_list) + coh_idx;

            % update counters
            combination_counter(iSub, combo_idx) = combination_counter(iSub, combo_idx) + 1;
            combination_performance(iSub, combo_idx) = combination_performance(iSub, combo_idx) + this_correct;

            % current online estimate
            p_perf_online(tr) = combination_performance(iSub, combo_idx) / combination_counter(iSub, combo_idx);
        end
    end
end