function [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
    twoWayNames, twoWayLabels] = build_model_family_wyzcond()
% model family 2:
% fixed terms: RT (R), trial_cond (cond), momentary vol (Vt)
% M1: intercept + RT
% M2: intercept + RT + cond
% M3: intercept + RT + cond + Vt
% M4: intercept + RT + cond*Vt

baseLabels    = {'b0 (Intercept)','b_{rt}'};

% oneWay terms: cond and V, each toggled independently
oneWayNames   = ["cond", "V"];
oneWayLabels  = {'b_{cond}', 'b_{vol}'};

% twoWay term: interaction
twoWayNames   = ["condxV"];
twoWayLabels  = {'b_{cond×vol}'};

% each row: [name, use1 (1x2: [cond, V]), use2 (1x1: [condxV])]
defs = {
    'M1_rt',          [0 0],  0;
    'M2_rt+cond',     [1 0],  0;
    'M3_cond+vol',    [1 1],  0;
    'M4_condxvol',    [1 1],  1;
    };

nModels    = size(defs, 1);
modelNames = defs(:,1)';

modelSpec = struct( ...
    'use1', cell(1,nModels), ...
    'use2', cell(1,nModels));

for i = 1:nModels
    modelSpec(i).use1 = logical(defs{i,2});
    modelSpec(i).use2 = logical(defs{i,3});
end

end