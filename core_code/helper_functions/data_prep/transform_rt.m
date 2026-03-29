function rtX = transform_rt(rt, subjID)
% Inputs:
% rt       : raw RT vector
% subjID   : subject ID vector
%
% Output:
% ConfY    : z-scored RT within subject

    rtX = nan(size(rt));

    subj_list = unique(subjID);
    nSubj = numel(subj_list);

    for iSub = 1:nSubj
        s = subj_list(iSub);
        idxS = subjID == s;

        rt_sub = rt(idxS);
        rt_log = log(rt_sub);

        mu_rt = mean(rt_log, 'omitnan');
        sd_rt = std(rt_log, 'omitnan');

        if sd_rt == 0 || isnan(sd_rt)
            rtX(idxS) = zeros(size(rt_log));
        else
            rtX(idxS) = (rt_log - mu_rt) ./ sd_rt;
        end
    end
end