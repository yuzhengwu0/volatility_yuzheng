function [modelNames, modelSpec, baseLabels, withcohNames, withcohLabels, oneWayNames, oneWayLabels, ...
    twoWayNames, twoWayLabels, nocohNames, nocohLabels, threeWayNames, threeWayLabels] = build_model_family_no_rt_withCorrect()
% build model family new version 
% main: baseline, RT, coh
% interaction terms: perf x corr, perf x vol, corr x vol, perf x corr x vol

baseLabels    = {'b0 (Intercept)'};

withcohNames = ["R", "coh", "C"];
withcohLabels ={'b_{rt}','b_{coh}', 'b_{C}'};

oneWayNames   = ["P", "V"];
oneWayLabels  = { 'b_{perf}','b_{vol}'};

twoWayNames   = ["PxV"];
twoWayLabels  = {'b_{perf×vol}'};

nocohLabels = ["R", "C"];
nocohNames = {'b_{rt}', 'b_{C}'};

threeWayNames  = ["PxVxcoh",];
threeWayLabels = {'b_{perf×vol×coh}'};

% each row: [name,  use2 (6 bits),  use3 (4 bits),  use4]
% use2/3/4 written as index lists of which terms to turn ON (empty = none)
defs = {
    'M0_base',          1:3,          [],     [],  [],   false, ;
    'M1_P',               1:3,        1,    [],   [],   false;
    'M2_V',                1:3,        2,    [],    [],   false;
    'M3_all',               1:3,      1:2,    [],     [],   false;
     'M4_all1',            1:3,       1:2,   true,    [],   false;
     'M5_threeway'     []           1:2,    true,    1:2,    true;
};

% M0: baseline: intercept, rt, coh, corr
% M1: baseline + P
% M2: baseline + V
% M3: baseline + P + V
% M4: baseline + P + V + P*V
% M5: intercept, rt, corr + P + V + P*V + P*V*coh

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