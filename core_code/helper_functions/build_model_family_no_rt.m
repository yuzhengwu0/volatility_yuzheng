function [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_no_rt()
% build model family new version 
% main: baseline, RT, coh
% interaction terms: perf x corr, perf x vol, corr x vol, perf x corr x vol

baseLabels    = {'b0 (Intercept)','b_{rt}','b_{coh}'};
oneWayNames   = ["P", "C", "V"];
oneWayLabels  = { 'b_{perf}', 'b_{corr}','b_{vol}'};
twoWayNames   = ["PxC","PxV","VxC"];
twoWayLabels  = {'b_{perf×corr}','b_{perf×vol}','b_{vol×corr}'};
threeWayNames  = ["PxVxC",];
threeWayLabels = {'b_{perf×vol×corr}'};

% each row: [name,  use2 (6 bits),  use3 (4 bits),  use4]
% use2/3/4 written as index lists of which terms to turn ON (empty = none)
defs = {
    'M0_base',          [],          [],      false;
    'M1_P',                1,         [],      false;
    'M2_C',                2,         [],      false;
    'M3_V',                3,         [],       false;
    'M4_all1',            1:3,       [],      false;
    'M5_2way_PxC',   1:3,         1,    false;
    'M6_2way_PxV',   1:3,         2,    false;
    'M7_2way_VxC',   1:3,         3,    false;
    'M7_all2',             1:3,       1:3,   false;
    'M8_full',              1:3,       1:3,   true;
};

nModels   = size(defs, 1);
modelNames = defs(:, 1)';
modelSpec  = struct('use1', cell(1,nModels), 'use2', cell(1,nModels), 'use3', cell(1,nModels));

for i = 1:nModels
    u1 = false(1,3); u1(defs{i,2}) = true;
    u2 = false(1,3); u2(defs{i,3}) = true;
    modelSpec(i).use1 = u1;
    modelSpec(i).use2 = u2;
    modelSpec(i).use3 = defs{i,4};
end
end