function [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, twoWayNames, twoWayLabels, ...
    threeWayNames, threeWayLabels] = build_model_family_v2()

baseLabels    = {'b0 (Intercept)','b_{corr}','b_{rt}', 'b_{coh}'};

oneWayNames   = ["P", "V"];
oneWayLabels  = { 'b_{perf}','b_{vol}'};

twoWayNames   = ["PxV"];
twoWayLabels  = {'b_{perf×vol}'};

threeWayNames  = ["PxVxcoh"];
threeWayLabels = {'b_{perf×vol×coh}'};

% each row: [name,  use2 (6 bits),  use3 (4 bits),  use4]
% use2/3/4 written as index lists of which terms to turn ON (empty = none)
defs = {
    'M0_base',          [],          [],    false, ;
    'M1_P',               1,        [],     false;
    'M2_V',                2,        [],   false;
    'M3_all',               1:2,      [],     false;
    'M4_twoWay_PxV',  1:2,    1,   false;
    'M5_PxVxcoh'       1:2,       1,  true;
};
nModels   = size(defs, 1);
modelNames = defs(:, 1)';
modelSpec  = struct('use1', cell(1,nModels), 'use2', cell(1,nModels), 'use3', cell(1,nModels));

for i = 1:nModels
    u1 = false(1,2); u1(defs{i,2}) = true;
    u2 = false(1,2); u2(defs{i,3}) = true;
    modelSpec(i).use1 = u1;
    modelSpec(i).use2 = u2;
    modelSpec(i).use3 = defs{i,4};
end
end