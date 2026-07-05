function [cfg, valid] = prep_main(cfg, data_path)

% ===================== 1. Load & clean data =====================
tmp       = load(data_path, 'all');
allStruct = tmp.all;

coh_all       = allStruct.rdm1_coh(:);
given_all     = allStruct.given_resp(:);        % 1/2
req_all       = allStruct.req_resp(:);
correct_all   = allStruct.correct(:);         % 1/0
confCont_all  = allStruct.confidence(:);      % 0-1
vol_all       = allStruct.rdm1_coh_std(:);
subjID_all    = allStruct.group(:);
session_all   = allStruct.session(:);
motion_energy_all = allStruct.motion_energy(:);
rt_all        = allStruct.rt(:);
cohframes     = allStruct.rdm1_cohframes(:);
dir           = allStruct.rdm1_dir(:);

puntos = allStruct.rdm1_puntos(:);
ncohdots = allStruct.rdm1_ncohdots(:);


valid_basic = ~isnan(coh_all) & ~isnan(correct_all) & ...
            ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ...
            ~isnan(rt_all) & ~isnan(dir) & ~isnan(session_all) & ...
            ~isnan(given_all) & ~isnan(req_all);

valid_fixed = allStruct.times_dots_on == 0.2;

valid_highvol = vol_all == max(vol_all);

idx_notfix = allStruct.times_dots_on ~= 0.2;
% valid_RT =  allStruct.session(idx_notfix);

valid = valid_basic & valid_fixed;

if cfg.ALL
    valid = valid_basic;
end 

if cfg.HIGHVOL
    valid = valid_basic & valid_highvol;
end 

if cfg.RTtask
    valid = valid_basic & idx_notfix;
    if cfg.HIGHVOL
        valid = valid_basic & idx_notfix & valid_highvol;
    end
end



cfg.coh           = coh_all(valid);
cfg.coh_weuse     = cfg.coh/100;
cfg.req           = req_all(valid);
cfg.given         = given_all(valid);
cfg.Correct       = correct_all(valid);
cfg.confCont      = confCont_all(valid);
cfg.vol           = vol_all(valid);
cfg.subjID        = subjID_all(valid);
cfg.session       = session_all(valid);
cfg.motion_energy = motion_energy_all(valid);
cfg.rt            = rt_all(valid);
cfg.puntos        = puntos(valid);
cfg.ncohdots      = ncohdots(valid);
cfg.cohframes     = cohframes(valid);
cfg.dir           = dir(valid);
cfg.valid_fixed   = valid_fixed(valid);

cfg.truesessiontrial = allStruct.trialnum(valid);
cfg.truesession = allStruct.session(valid);




