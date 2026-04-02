function [modelNames, modelSpec, baseLabels, oneWayNames, oneWayLabels, ...
    twoWayNames, twoWayLabels, threeWayNames, threeWayLabels] = build_model_family_blend()

% model family 2:
% fixed terms: RT (R), accuracy (C), coherence (coh), vol cohition (coh)
% M0: intercept + R + C + coh + z_cond
% M1: M0 + V
% M2: M1 + V*Coh
% M3: M1 + V*z_cond
% M4: M1 + V*C + V*coh + V*coh*z_cond

baseLabels    = {'b0 (Intercept)','b_{corr}','b_{rt}','b_{coh}','b_{z_cond}'};

oneWayNames   = ["V"];
oneWayLabels  = {'b_{vol}'};

twoWayNames   = ["Vxcoh", "Vxz_cond"];
twoWayLabels  = {'b_{vol×coh}', 'b_{volxz_cond}'};

threeWayNames   = ['Vxcohxz_cond'];
threeWayLabels  = {'b_{volxcohxz_cond}'};

% each row:
% [name, use1, use2, use3]
defs = {
    'M0_base',              [],        [],     false;
    'M1_V',                   1,        [],     false;
    'M2_Vxcoh',               1,        1,      false;
    'M3_Vxz_cond',      1,         2,       false;
    'M4_Vxcohxz_cond',   1,       1:2,      true;
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