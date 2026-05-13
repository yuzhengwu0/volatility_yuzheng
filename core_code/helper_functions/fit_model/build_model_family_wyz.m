function [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
    twoWayNames, twoWayLabels] = build_model_family_wyz()

% model family 2:
% fixed terms: RT (R), accuracy (C), coherence (coh), vol cohition (coh)
% M1: intercept + RT + trial_coh
% M2: intercept + RT + trial_coh + momentary_vol
% M3: intercept + RT + trial_coh + momentary_vol + trial_coh * momentary_vol


baseLabels    = {'b0 (Intercept)','b_{rt}','b_{coh}'};

oneWayNames   = ["V"];
oneWayLabels  = {'b_{vol}'};

twoWayNames   = ["Vxcoh"];
twoWayLabels  = {'b_{vol×coh}'};


% each row:
% [name, use1, use2]
defs = {
    'M1_coh',      0,  0;
    'M2_coh+vol',  1,  0;
    'M3_cohxvol',  1,  1;
    };

nModels    = size(defs, 1);
modelNames = defs(:,1)';

modelSpec = struct( ...
    'use1', cell(1,nModels), ...
    'use2', cell(1,nModels));

for i = 1:nModels
    u1 = false(1, numel(oneWayNames));
    u2 = false(1, numel(twoWayNames)); 
    
    if defs{i,2} == 1
        u1(:) = true;
    end

    if defs{i,3} == 1
        u2(:) = true;
    end


    modelSpec(i).use1 = u1;
    modelSpec(i).use2 = u2;
end

end