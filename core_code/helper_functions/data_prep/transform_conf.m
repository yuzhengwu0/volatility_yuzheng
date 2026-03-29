function ConfY = transform_conf(confCont, subjID)
% Inputs:
% confCont : confidence vector
% subjID   : subject ID vector
%
% Output:
% ConfY    : z-scored confidence within subject

    ConfY = nan(size(confCont));

    subj_list = unique(subjID);
    nSubj = numel(subj_list);

    for iSub = 1:nSubj
        s = subj_list(iSub);
        idxS = subjID == s;

        y = confCont(idxS);

        mu = mean(y, 'omitnan');
        sigma = std(y, 'omitnan');

        if sigma == 0 || isnan(sigma)
            ConfY(idxS) = zeros(size(y));
        else
            ConfY(idxS) = (y - mu) ./ sigma;
        end
    end
end