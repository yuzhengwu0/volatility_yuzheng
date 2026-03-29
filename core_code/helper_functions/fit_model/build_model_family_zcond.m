function [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_zcond()

baseLabels    = {'b0 (Intercept)','b_{corr}','b_{rt}','b_{coh}','b_{cond}'};

oneWayNames   = ["P", "V"];
oneWayLabels  = {'b_{perf}','b_{vol}'};

twoWayNames   = ["PxV"];
twoWayLabels  = {'b_{perf×vol}'};

threeWayNames   = ["PxVxcoh"];
threeWayLabels  = {'b_{perf×volxcoh}'};

% each row:
% [name, use1, use2, use3]
defs = {
    'M0_base',           [],       [],     false;
    'M1_P',              1,        [],     false;
    'M2_V',              2,        [],     false;
    'M3_all',            1:2,      [],     false;
    'M4_twoWay_PxV',     1:2,      1,      false;
};

nModels    = size(defs, 1);
modelNames = defs(:,1)';

modelSpec = struct( ...
    'use1', cell(1,nModels), ...
    'use2', cell(1,nModels));

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
end

end