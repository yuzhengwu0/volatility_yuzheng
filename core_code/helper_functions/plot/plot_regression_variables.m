% plot predictors & outcome variable

tiledlayout;
% plot raw confidence
% nexttile;
% histogram(confCont);
% title('raw confidence (not used)')

% plot z-scored confidence (outcome)
nexttile;
histogram(cfg.ConfY);
title('OUTCOME: confidence z-scored within subject');

% plot RT
nexttile;
histogram(cfg.rtX);
title('z-logged RT (R predictor)')

% plot correctness
nexttile;
histogram(cfg.Correct);
title('accuracy (C predictor)');

% plot coherence
nexttile;
histogram(cfg.coh_weuse);
title('coherence (Coh predictor)')

% plot performance term
nexttile;
switch cfg.P_PERF_MODE
    case 'online'
        histogram(cfg.z_perf); 
        title('p perf online (P predictor)'); 
    case 'all'
        histogram(cfg.z_perf);
        title('p perf all (P predictor)');
end

% plot volatility
nexttile;
histogram(cfg.resVol_mat);
title('raw volatility (not used)')

% plot z-logged volatility
nexttile;
histogram(cfg.zlog_vol);
title('zlogged volatility (V predictor)')

