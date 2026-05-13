function [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels] = build_model_family_wyzcoh()

% model family 2:
% fixed terms: RT (R), accuracy (C), coherence (coh), vol cohition (coh)
% M1: intercept + RT
% M2: intercept + RT + momentary_vol


baseLabels    = {'b0 (Intercept)','b_{rt}'};

oneWayNames   = ["V"];
oneWayLabels  = {'b_{vol}'};


% each row:
% [name, use1]
defs = {
    'M1_coh',      0;
    'M2_coh+vol',  1;
    };

nModels    = size(defs, 1);
modelNames = defs(:,1)';

modelSpec = struct( ...
    'use1', cell(1,nModels));

for i = 1:nModels
    u1 = false(1, numel(oneWayNames));
    
    if defs{i,2} == 1
        u1(:) = true;
    end


    modelSpec(i).use1 = u1;
end

end