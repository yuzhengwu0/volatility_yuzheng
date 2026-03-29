function p_perf_all = compute_p_perf_all(subjID, cond, coh, Correct)
% Input:
%   subjID   : subject ID vector
%   cond     : condition vector (volatility -> low/high coded as 1/2)
%   coh      : coherence vector 
%   Correct  : trial-wise accuracy vector
%
% Output:
%   p_perf_all : predicted performance for each trial based on
%                subject × condition × coherence mean accuracy
% 
% NOTE: this predicted performance only contains 12 different kinds of values
% assigned to all of the trials based on their stimuli category. 

    nTrials = numel(subjID);
    p_perf_all = nan(nTrials, 1);

    subj_list = unique(subjID(~isnan(subjID)));
    cond_list = unique(cond(~isnan(cond)));
    coh_list  = unique(coh(~isnan(coh)));

    nSubj = numel(subj_list);

    for iSub = 1:nSubj
        thisSub = subj_list(iSub);

        for c = cond_list(:)'
            for h = coh_list(:)'
                mask = (subjID == thisSub) & (cond == c) & (coh == h);
                
                if any(mask)
                    p_perf_all(mask) = mean(Correct(mask), 'omitnan');
                end
            end
        end
    end
end