function [ConfY, confCont] = transform_conf(cfg)
% Inputs:
% confCont : confidence vector
% subjID   : subject ID vector
%
% Output:
% ConfY    : z-scored confidence within subject

confCont = cfg.confCont;
subjID = cfg.subjID;

ConfY = nan(size(confCont));

for i = 1:length(confCont)
    if confCont(i) > 1
        confCont(i) = 1;
    elseif confCont(i) < 0
        confCont(i) = 0;
    else
        confCont(i) = confCont(i);
    end
end

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