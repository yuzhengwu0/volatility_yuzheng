function [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_corr()

baseLabels    = {'b0 (Intercept)','b_{rt}','b_{coh}','b_{corr}'};

oneWayNames   = ["P", "V"];
oneWayLabels  = {'b_{perf}','b_{vol}'};

twoWayNames   = ["PxV", "CxV"];
twoWayLabels  = {'b_{perf×vol}','b_{corr×vol}'};

threeWayNames  = ["PxVxcoh"];
threeWayLabels = {'b_{perf×vol×coh}'};

% each row:
% [name, use1, use2, use3]
defs = {
    'M0_base',           [],       [],     false;
    'M1_P',              1,        [],     false;
    'M2_V',              2,        [],     false;
    'M3_all',            1:2,      [],     false;
    'M4_PxV',            1:2,       1,     false;
    'M5_CxV',            1:2,       2,     false;
};

nModels    = size(defs, 1);
modelNames = defs(:,1)';

modelSpec = struct( ...
    'use1', cell(1,nModels), ...
    'use2', cell(1,nModels), ...
    'use3', cell(1,nModels));

for i = 1:nModels
    u1 = false(1, numel(oneWayNames));
    if ~isempty(defs{i,2})
        u1(defs{i,2}) = true;
    end

    u2 = false(1, numel(twoWayNames));
    if ~isempty(defs{i,3})
        u2(defs{i,3}) = true;
    end

    modelSpec(i).use1 = u1;
    modelSpec(i).use2 = u2;
    modelSpec(i).use3 = defs{i,4};
end

end