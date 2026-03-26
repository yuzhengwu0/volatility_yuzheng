function [modelNames, modelSpec, baseLabels, twoWayNames, twoWayLabels, ...
    threeWayNames, threeWayLabels, fourWayNames, fourWayLabels] = build_model_family()

baseLabels    = {'b0 (Intercept)','b_{perf}','b_{corr}','b_{vol}','b_{rt}'};
twoWayNames   = ["PxC","CxR","RxV","PxV","PxR","VxC"];
twoWayLabels  = {'b_{perf×corr}','b_{perf×vol}','b_{perf×rt}','b_{vol×corr}','b_{corr×rt}','b_{rt×vol}'};
threeWayNames  = ["PxVxC","PxCxR","PxVxR","VxCxR"];
threeWayLabels = {'b_{perf×vol×corr}','b_{perf×corr×rt}','b_{perf×vol×rt}','b_{vol×corr×rt}'};
fourWayNames  = "PxVxCxR";
fourWayLabels = {'b_{perf×vol×corr×rt}'};

% each row: [name,  use2 (6 bits),  use3 (4 bits),  use4]
% use2/3/4 written as index lists of which terms to turn ON (empty = none)
defs = {
    'M0_base',          [],          [],    false;
    'M1_2way_PxC',      1,         [],    false;
    'M2_2way_CxR',      2,         [],    false;
    'M3_2way_RxV',      3,         [],    false;
    'M4_2way_PxV',      4,         [],    false;
    'M5_2way_PxR',      5,         [],    false;
    'M6_2way_VxC',      6,         [],    false;
    'M7_all2',          1:6,       [],    false;
    'M8_all2_all3',     1:6,       1:4, false;
    'M9_full',          1:6,       1:4, true;
};

nModels   = size(defs, 1);
modelNames = defs(:, 1)';
modelSpec  = struct('use2', cell(1,nModels), 'use3', cell(1,nModels), 'use4', cell(1,nModels));

for i = 1:nModels
    u2 = false(1,6); u2(defs{i,2}) = true;
    u3 = false(1,4); u3(defs{i,3}) = true;
    modelSpec(i).use2 = u2;
    modelSpec(i).use3 = u3;
    modelSpec(i).use4 = defs{i,4};
end
end