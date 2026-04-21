function cfg = prep_main(cfg, data_path)

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
motion_energy_all = allStruct.motion_energy;
rt_all        = allStruct.rt(:);

valid_basic = ~isnan(coh_all) & ~isnan(correct_all) & ...
            ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all) & allStruct.times_dots_on == 0.2;



valid = valid_basic;

cfg.coh           = coh_all(valid);
cfg.coh_weuse     = cfg.coh/100;
cfg.req           = req_all(valid);
cfg.given         = given_all(valid);
cfg.Correct       = correct_all(valid);
cfg.confCont      = confCont_all(valid);
cfg.vol           = vol_all(valid);
cfg.subjID        = subjID_all(valid);
cfg.motion_energy = motion_energy_all(valid);
cfg.rt            = rt_all(valid);

cfg.truesessiontrial = allStruct.trialnum(valid);
cfg.truesession = allStruct.session(valid);




