function [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_zcond()

% model family 2:
% fixed terms: RT (R), accuracy (C), coherence (coh), vol condition (z_cond)
% M0: baseline + R + C + coh + z_cond
% M1: baseline + R + C + coh + z_cond + P
% M2: baseline + R + C + coh + z_cond + V
% M3: baseline + R + C + coh + z_cond + P + V
% M4: baseline + R + C + coh + z_cond + P + V + P*V
% M5: baseline + R + C + coh + z_cond + P + V + C*V
% M6: baseline + R + C + coh + z_cond + P + V + P*V*coh
% M7: baseline + R + C + coh + z_cond + P + V + C*V*coh

baseLabels    = {'b0 (Intercept)','b_{corr}','b_{rt}','b_{coh}','b_{cond}'};

oneWayNames   = ["P", "V"];
oneWayLabels  = {'b_{perf}','b_{vol}'};

twoWayNames   = ["PxV", "CxV"];
twoWayLabels  = {'b_{perf×vol}', 'b_{corrxvol}'};

threeWayNames   = ["PxVxcoh", 'CxVxcoh'];
threeWayLabels  = {'b_{perf×volxcoh}', 'b_{corrxvolxcoh}'};

% each row:
% [name, use1, use2, use3]
defs = {
    'M0_base',           [],       [],     false;
    'M1_P',                 1,        [],     false;
    'M2_V',                 2,        [],     false;
    'M3_P+V',            1:2,      [],     false;
    'M4_PxV',           1:2,      1,      false;
    'M5_CxV',           1:2,     2,         false;
    'M6_PxVxcoh',     1:2,     1:2,       1;
    'M7_CxVxcoh',    1:2,       1:2,      2;
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

    u3 = false(1, numel(threeWayNames));
    if ~isempty(defs{i,4})
        u3(defs{i,4}) = true;
    end

    modelSpec(i).use1 = u1;
    modelSpec(i).use2 = u2;
    modelSpec(i).use3 = u3;
end

end