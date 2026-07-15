%% script using cohframes
clear; clc; close all;

addpath(genpath('helper_functions/'));
data_path = '../all_with_me.mat';

%% re-run this chunk if configuration changed
run('cfg_default.m');

% add valid trials filter
[cfg, idx] = prep_main(cfg, data_path);

% redefine accuracy
prep_answer_and_ME;

cond = nan(size(cfg.vol));
cond(cfg.vol == min(cfg.vol)) = -1;
cond(cfg.vol == max(cfg.vol)) = 1;
cfg.cond = cond;

% flipping left trials
if cfg.FLIPPING
    flip_leftward_trials;
end

%% ===== data prep =====

% conf
[ConfY, confCont] = transform_conf(cfg);
cfg.ConfY = ConfY;
cfg.confCont = confCont;

% RT
rtX = transform_rt(cfg);
cfg.rtX = rtX;

% switch between moco_diff and ME
trial_coh = cfg.coh ./ 1000;

if VOL_USE_ME == false

    coh_mat = repmat(trial_coh, [1, 17]);

    moco_mat = NaN(nTrials, 17);

    for t = 1:height(cfg.cohframes)

        frames = cfg.cohframes{t};
        n = numel(frames);

        if n == 17

            moco_mat(t, :) = frames';

        else

            x_orig = linspace(1, 17, n);
            x_new  = 1:17;

            moco_mat(t, :) = interp1( ...
                x_orig, frames, x_new, 'linear');

        end
    end

    trial_dir = cfg.dir;

    trial_dir(trial_dir == 180) = -1;
    trial_dir(trial_dir == 0) = 1;

    coh_mat_dir  = coh_mat .* trial_dir;
    moco_mat_dir = moco_mat .* trial_dir;

    switch cfg.VOLMODE

        case 'vol1'

            moco_diff = moco_mat_dir - coh_mat_dir;
            vol = moco_diff;

        case 'vol2'

            vol2_signed = NaN(nTrials, 17);

            for t = 1:nTrials

                for f = 2:17

                    vol2_signed(t, f) = ...
                        moco_mat_dir(t, f) - ...
                        moco_mat_dir(t, f-1);

                end
            end

            vol = vol2_signed;

        case 'vol3dir'

            vol3 = NaN(nTrials, 17);

            for t = 1:nTrials

                for i = 1:17

                    if i == 1

                        vol3(t, i) = moco_mat_dir(t, i);

                    else

                        vol3(t, i) = ...
                            moco_mat_dir(t, i) + ...
                            vol3(t, i-1);

                    end
                end
            end

            vol = vol3;

        case 'vol3abs'

            vol3 = NaN(nTrials, 17);

            for t = 1:nTrials

                for i = 1:17

                    if i == 1

                        vol3(t, i) = moco_mat_dir(t, i);

                    else

                        vol3(t, i) = ...
                            moco_mat_dir(t, i) + ...
                            vol3(t, i-1);

                    end
                end
            end

            vol = abs(vol3);

        otherwise

            error('Unknown cfg.VOLMODE: %s', cfg.VOLMODE);

    end

else

    thiscoh = unique(trial_coh);
    thiscond = unique(cfg.cond);

    motion_diff = nan(size(motion_mat));

    for icoh = 1:numel(thiscoh)

        for icond = 1:numel(thiscond)

            idx_ME = ...
                (trial_coh == thiscoh(icoh)) & ...
                (cfg.cond == thiscond(icond));

            data = motion_mat(idx_ME, :);

            cond_coh_mean_ME = ...
                mean(data, 1, 'omitnan');

            motion_diff(idx_ME, :) = ...
                data - cond_coh_mean_ME;

        end
    end

    vol = motion_diff;

end

zvol = zscore(vol);

cfg.vol = zvol;
cfg.trial_coh = trial_coh;

%% ============================================================
%  Store fitting / comparison / BIC-selected beta
%
%  Fixed two-column layout:
%
%  coh == 0:
%       left  = all
%       right = blank
%
%  other coherence levels:
%       left  = incorrect
%       right = correct
%% ============================================================

Correct = cfg.Correct;
cfg_orig = cfg;

%% coherence levels

% The order here determines the vertical order in the figure.
% Put 0 last if you want coh == 0 to appear in the bottom row.
coh_levels = [0, 32, 64, 128, 256, 512];

% Other examples:
% coh_levels = [32, 64, 128, 0];
% coh_levels = [64, 128, 256, 0];

% Always use two columns
nAccCols = 2;

RES = struct();
allAbsBeta = [];

% term aliases for extracting beta from BIC-winning model
termAliases = {
    ["R", "rt", "rtX"], ...
    ["cond"], ...
    ["V", "vol", "Vt"], ...
    ["Vxcond", "V:cond", "cond:V", "Vt:cond", "cond:Vt"]
};

for ci = 1:numel(coh_levels)

    thisCoh = coh_levels(ci);

    for ai = 1:nAccCols

        cfg = cfg_orig;

        %% ----- determine accuracy condition -----

        if thisCoh == 0

            if ai == 1

                % coh == 0, left column:
                % pool correct and incorrect trials
                thisAcc = 'all';

            else

                % coh == 0, right column:
                % leave this panel blank
                RES(ci, ai).coh = thisCoh;
                RES(ci, ai).acc = 'blank';
                RES(ci, ai).nTrials = 0;
                RES(ci, ai).isBlank = true;

                continue;

            end

        else

            if ai == 1

                % Other coherence levels, left column
                thisAcc = 'incorr';

            else

                % Other coherence levels, right column
                thisAcc = 'corr';

            end

        end

        cfg.CORR = thisAcc;

        %% ----- coherence filter -----

        cfg.coh_label = num2str(thisCoh);

        % Robust coherence filter:
        % works whether cfg.coh is coded as 32 or 0.032
        if max(abs(cfg.coh), [], 'omitnan') <= 1

            coh_for_filter = round(cfg.coh * 1000);

        else

            coh_for_filter = round(cfg.coh);

        end

        keep_coh = (coh_for_filter == thisCoh);

        %% ----- accuracy filter -----

        switch cfg.CORR

            case 'corr'

                keep_corr = (Correct == 1);

            case 'incorr'

                keep_corr = (Correct == 0);

            case 'all'

                % Include both correct and incorrect trials.
                % Undefined accuracy trials are excluded.
                keep_corr = ...
                    (Correct == 1) | ...
                    (Correct == 0);

            otherwise

                error('Unknown cfg.CORR: %s', cfg.CORR);

        end

        %% ----- combine filters -----

        keep = keep_coh & keep_corr;

        cfg.keep = keep;
        valid = cfg.keep;

        %% ----- update cfg fields -----

        cfg.ConfY     = cfg.ConfY(valid);
        cfg.confCont  = cfg.confCont(valid);
        cfg.Correct   = Correct(valid);
        cfg.rtX       = cfg.rtX(valid);
        cfg.subjID    = cfg.subjID(valid);
        cfg.coh       = cfg.coh(valid);
        cfg.cond      = cfg.cond(valid);
        cfg.vol       = cfg.vol(valid, :);
        cfg.req       = cfg.req(valid);
        cfg.given     = cfg.given(valid);
        cfg.trial_coh = cfg.trial_coh(valid);

        nTrials = sum(cfg.keep);
        cfg.nTrials = nTrials;

        if cfg.nTrials == 0

            error( ...
                'No trials found for condition %s, coh = %d.', ...
                cfg.CORR, thisCoh);

        end

        fprintf('\n====================================\n');
        fprintf('Condition: %s | coh = %d | n = %d\n', ...
            cfg.CORR, thisCoh, cfg.nTrials);
        fprintf('====================================\n');

        %% ===== model family =====

        % Avoid variables carrying over from the previous iteration
        clear twoWayNames twoWayLabels

        switch cfg.OUTCOME

            case 'conf'

                switch cfg.MODEL_FAMILY

                    case 'wyz'

                        [modelNames, modelSpec, ...
                            baseLabels, ...
                            oneWayNames, oneWayLabels, ...
                            twoWayNames, twoWayLabels] = ...
                            build_model_family_wyz();

                    case 'wyzcoh'

                        [modelNames, modelSpec, ...
                            baseLabels, ...
                            oneWayNames, oneWayLabels] = ...
                            build_model_family_wyzcoh();

                    case 'wyzcond'

                        [modelNames, modelSpec, ...
                            baseLabels, ...
                            oneWayNames, oneWayLabels, ...
                            twoWayNames, twoWayLabels] = ...
                            build_model_family_wyzcond();

                    otherwise

                        error( ...
                            'Unknown cfg.MODEL_FAMILY: %s', ...
                            cfg.MODEL_FAMILY);

                end

            otherwise

                error( ...
                    'Unknown cfg.OUTCOME: %s', ...
                    cfg.OUTCOME);

        end

        nModels = numel(modelNames);

        cfg.nModels = nModels;
        cfg.modelNames = modelNames;
        cfg.modelSpec = modelSpec;

        cfg.baseLabels = baseLabels;
        cfg.oneWayLabels = oneWayLabels;
        cfg.oneWayNames = oneWayNames;

        if exist('twoWayNames', 'var')
            cfg.twoWayNames = twoWayNames;
        end

        if exist('twoWayLabels', 'var')
            cfg.twoWayLabels = twoWayLabels;
        end

        cfg.minN = 50;

        % Give each condition a separate dot-plot filename
        cfg.outPDF_ab = fullfile( ...
            '../figure', ...
            sprintf( ...
                'AIC_BIC_bestModel_dots_coh%d_%s.pdf', ...
                thisCoh, cfg.CORR));

        %% ===== model fitting =====

        switch cfg.OUTCOME

            case 'conf'

                switch cfg.MODEL_FAMILY

                    case 'wyz'

                        [Fitted_models, ...
                            AIC_mat, ...
                            BIC_mat, ...
                            Nobs_mat] = ...
                            fit_model_wyz(cfg);

                    case 'wyzcoh'

                        [Fitted_models, ...
                            AIC_mat, ...
                            BIC_mat, ...
                            Nobs_mat] = ...
                            fit_model_wyzcoh(cfg);

                    case 'wyzcond'

                        [Fitted_models, ...
                            AIC_mat, ...
                            BIC_mat, ...
                            Nobs_mat] = ...
                            fit_model_wyzcond(cfg);

                    otherwise

                        error( ...
                            'Unknown cfg.MODEL_FAMILY: %s', ...
                            cfg.MODEL_FAMILY);

                end
        end

        cfg.Fitted_models = Fitted_models;

        %% ===== store after model fitting =====

        RES(ci, ai).coh = thisCoh;
        RES(ci, ai).acc = cfg.CORR;
        RES(ci, ai).nTrials = cfg.nTrials;
        RES(ci, ai).isBlank = false;

        RES(ci, ai).Fitted_models = Fitted_models;
        RES(ci, ai).AIC_mat = AIC_mat;
        RES(ci, ai).BIC_mat = BIC_mat;
        RES(ci, ai).Nobs_mat = Nobs_mat;

        RES(ci, ai).modelNames = modelNames;
        RES(ci, ai).modelSpec = modelSpec;

        %% ===== model comparison =====

        [deltaTbl, score, rankIdx, top4Idx, deltaBIC] = ...
            rank_models( ...
                AIC_mat, ...
                BIC_mat, ...
                cfg.modelNames);

        cfg.rankIdx = rankIdx;
        cfg.top4Idx = top4Idx;

        %% ===== store after model comparison =====

        RES(ci, ai).deltaTbl = deltaTbl;
        RES(ci, ai).score = score;
        RES(ci, ai).rankIdx = rankIdx;
        RES(ci, ai).top4Idx = top4Idx;
        RES(ci, ai).deltaBIC = deltaBIC;

        %% ===== calculate BIC-selected beta =====

        nBins = size(BIC_mat, 1);
        nTerms = numel(termAliases);

        selectedBetas = NaN(nTerms, nBins);
        includedTerm = false(nTerms, nBins);
        bestModelIdx = NaN(nBins, 1);

        minBIC = min(BIC_mat, [], 2, 'omitnan');

        deltaBIC_byBin = ...
            BIC_mat - ...
            repmat(minBIC, 1, size(BIC_mat, 2));

        %% find BIC-winning model for each bin

        for t = 1:nBins

            thisBIC = BIC_mat(t, :);

            if all(isnan(thisBIC))
                continue;
            end

            [~, bestModelIdx(t)] = ...
                min(thisBIC, [], 'omitnan');

        end

        %% extract beta from BIC-winning model only

        for t = 1:nBins

            m = bestModelIdx(t);

            if isnan(m)
                continue;
            end

            if isempty(Fitted_models(t, m).g)
                continue;
            end

            g = Fitted_models(t, m).g;

            coefNames = string(g.CoefficientNames);
            coefEst = g.Coefficients.Estimate;

            for k = 1:nTerms

                aliases = termAliases{k};
                idxCoef = [];

                for a = 1:numel(aliases)

                    idxCoef = find( ...
                        coefNames == aliases(a), ...
                        1);

                    if ~isempty(idxCoef)
                        break;
                    end

                end

                if ~isempty(idxCoef)

                    selectedBetas(k, t) = ...
                        coefEst(idxCoef);

                    includedTerm(k, t) = true;

                end
            end
        end

        RES(ci, ai).selectedBetas = selectedBetas;
        RES(ci, ai).includedTerm = includedTerm;
        RES(ci, ai).bestModelIdx = bestModelIdx;
        RES(ci, ai).deltaBIC_byBin = deltaBIC_byBin;

        %% collect beta values for global scaling

        tmpAbs = abs(selectedBetas);

        tmpVals = tmpAbs( ...
            includedTerm & ...
            ~isnan(tmpAbs));

        allAbsBeta = [allAbsBeta; tmpVals(:)];

        %% optional original dot plot

        if cfg.DO_PLOT_AICBIC_DOTS

            plot_best_model_dots( ...
                AIC_mat, ...
                BIC_mat, ...
                cfg.modelNames, ...
                cfg.outPDF_ab, ...
                cfg);

        end

    end
end

%% ===== save stored results =====

outDir = '../figure';

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

if isempty(allAbsBeta)

    globalBetaMin = 0;
    globalBetaMax = 1;

else

    globalBetaMin = min(allAbsBeta);
    globalBetaMax = max(allAbsBeta);

end

outFile = fullfile( ...
    outDir, ...
    'RES_BIC_beta_mixed_accuracy_layout.mat');

save(outFile, ...
    'RES', ...
    'coh_levels', ...
    'allAbsBeta', ...
    'globalBetaMin', ...
    'globalBetaMax', ...
    '-v7.3');

fprintf('\nSaved RES to:\n%s\n', outFile);

fprintf('Global |beta| min = %.4f, max = %.4f\n', ...
    globalBetaMin, globalBetaMax);