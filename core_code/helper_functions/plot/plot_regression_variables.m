% plot predictors & outcome variable

tiledlayout;
% plot raw confidence
% nexttile;
% histogram(confCont);
% title('raw confidence (not used)')

% plot z-scored confidence (outcome)
nexttile;
histogram(ConfY);
title('OUTCOME: confidence z-scored within subject');

% plot RT
nexttile;
histogram(rtX);
title('z-logged RT (R predictor)')

% plot correctness
nexttile;
histogram(Correct);
title('accuracy (C predictor)');

% plot coherence
nexttile;
histogram(coh_weuse);
title('coherence (Coh predictor)')

% plot performance term
nexttile;
switch P_PERF_MODE
    case 'online'
        histogram(p_perf_online); 
        title('p perf online (P predictor)'); 
    case 'all'
        histogram(p_perf_all);
        title('p perf all (P predictor)');
end

% plot volatility
nexttile;
histogram(resVol_mat);
title('raw volatility (not used)')

% plot z-logged volatility
nexttile;
histogram(zlog_vol);
title('zlogged volatility (V predictor)')

% resVol check: volatility per coherence condition
resVol_check;
