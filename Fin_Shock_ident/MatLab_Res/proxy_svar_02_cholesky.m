% =========================================================================
%  proxy_svar_02_cholesky.m - STAGE B: the monthly instrument used as an
%                             INTERNAL instrument (ordered first, Cholesky)
%                             in a GLP-Minnesota macro BVAR.
%
%  PIPELINE
%    Mori_3var.m   -> mori_3var.mat             daily shocks
%    proxy_svar1.m -> shocks_monthly_<SRC>.mat  Stage A: monthly instrument
%    THIS FILE     -> proxy_svar_02_cholesky_results.mat + IRF figures
%
%  IDENTIFICATION.  The instrument Z is ordered first in the augmented VAR
%  [Z, macro] and the responses are read from the first Cholesky column,
%  normalised per unit of the Z innovation:
%       b_chol = P(2:end,1)/P(1,1),   P = chol(Sigma_aug,'lower').
%  This is the internal-instrument estimator of Plagborg-Moller & Wolf (2021),
%  consistent under non-invertibility and measurement error.  The shocks carry
%  no economic label: EBP, UNC_BEX and RAI_BEX enter as ordinary controls, and
%  the daily FEVD-dominant variable is printed only as an ex-post reading.
%
%  INFERENCE.  Primary bands come from a residual-based moving-block bootstrap
%  of the augmented VAR (valid under conditional heteroskedasticity and fat
%  tails), in which every replication re-estimates the SAME shrunk estimator as
%  the reported point - least squares on [actual; prior dummies], which is the
%  Minnesota posterior mean.  The Gaussian posterior bands are also computed
%  and stored, for comparison.  The unit-effect normalisation is applied inside
%  every draw and every replication, never to the finished bands.
%
%  INPUT   mori_3var.mat, shocks_monthly_<SRC>.mat (Stage A must run first),
%          macro_monthly_ready.csv, bvar_Ferroni toolbox
%  OUTPUT  proxy_svar_02_cholesky_results.mat
%          fig_p02chol_irf_shock<k>_<SRC>.png
%
%  The econometric argument for each choice is in the thesis.
% =========================================================================

clear; clc; close all;
rng(42, 'twister');       % note: bvar_ resets the stream to rng(999) internally

addpath(fullfile(pwd,'bvar_Ferroni','bvartools'));
addpath(fullfile(pwd,'bvar_Ferroni','cmintools'));

%% ---- CONFIGURATION ------------------------------------------------------
SRC           = 'mori_3var.mat';
[~, SRC_BASE] = fileparts(SRC);
MACRO_CSV     = '../../PythonThesis/FinalPy/files/04_output/macro_monthly_ready.csv';

MACRO_NAMES = {'INDPRO','UNRATE','HOUST','CSHPI','PCEPI','NRCREDIT', ...
               'FFR','RAI_BEX','UNC_BEX','EBP'};
I0_VARS     = {'RAI_BEX','UNC_BEX','EBP'};   % white-noise prior mean; the rest random walk

IRF_HOR_MAC = 48;
K_AUG       = 1000;      % posterior draws of each augmented VAR

% Sample end.  The 2025 federal shutdown cancelled the October-2025 CPI and
% household survey, so macro_monthly_ready.csv has a hole at 2025-10 and lags
% are built by row position.  The sample stops before the hole; no imputation.
SAMPLE_END_YM = 202509;

% Lag order, fixed a priori rather than selected: under the Minnesota prior the
% 1/s^2 decay already shrinks the far lags (Banbura-Giannone-Reichlin 2010 use
% 12 for monthly data; here 12 + one quarter of slack).
P_MACRO_FIXED = 15;

% Which (point, band) pair is reported.  They are pairs and are never mixed:
%   'mbb'   Minnesota point + moving-block bootstrap band  [heteroskedasticity-robust]
%   'bayes' Minnesota point + Gaussian posterior band      [prices no heteroskedasticity]
BAND_PRIMARY  = 'mbb';
DO_MBB        = true;            % the bootstrap runs either way (diagnostic)
MBB_ESTIMATOR = 'shrunk';        % every replication re-estimates the Minnesota
                                 % posterior mean, so point and band are the
                                 % same estimator (checked below, see STEP 9)
MBB_B         = 2000;            % replications
MBB_BLOCK     = [];              % [] -> Jentsch-Lunsford rule 5.03*T^(1/4)
MBB_MIN_FRAC  = 0.50;            % minimum share of usable replications

% IRF normalisation.  'refvar' reports the shock that moves a reference
% variable by NORM_REF_SIZE in absolute value on impact (Gertler-Karadi
% convention); the factor is positive, so the direction is left to the upstream
% sign anchor.  'sd' uses one standard deviation of the instrument, 'unitZ' the
% raw per-unit-of-Z scale.  EBP is the reference: all three shocks move it in
% the same direction and its impact varies least across them.
NORM_MODE     = 'refvar';        % 'refvar' | 'sd' | 'unitZ'
NORM_REF_VAR  = 'EBP';
NORM_REF_SIZE = 0.25;

banner(sprintf('proxy_svar_02_cholesky - STAGE B: macro IRFs, internal instrument   SRC = %s', SRC));

%% =========================================================================
%  STEP 1 - DAILY SHOCKS AND THE SIGN CONVENTION INHERITED FROM STAGE A
% =========================================================================
banner('STEP 1 - DAILY SHOCKS (all shocks, no selection)');

S = load(SRC, 'V_ref', 'B_med', 'lam_med', 'VAR_NAMES', 'dates_vec', 'p');
V_ref     = S.V_ref;  [T_eff, M] = size(V_ref);
B_med     = S.B_med;
lam_med   = S.lam_med(:);
VAR_NAMES = cellstr(S.VAR_NAMES(:));
p_daily   = double(S.p);
if ~isfield(S,'dates_vec') || ~isdatetime(S.dates_vec) || numel(S.dates_vec) ~= T_eff
    error('%s carries no usable dates_vec: re-run Mori_3var.m.', SRC);
end
dates_vec = S.dates_vec(:);
clear S

% The sign convention is defined once, in Stage A, and read back here: the
% monthly instrument reused below already carries it.
monthly_file = sprintf('shocks_monthly_%s.mat', SRC_BASE);
if exist(monthly_file,'file') ~= 2
    error('%s not found: run proxy_svar1.m (Stage A) first.', monthly_file);
end
Pm = load(monthly_file, 'MEAN', 'ym_monthly', 'SIGN_CONV');
if ~all(isfield(Pm, {'MEAN','ym_monthly','SIGN_CONV'})) || numel(Pm.SIGN_CONV) ~= M
    error('%s is incomplete: re-run proxy_svar1.m (Stage A).', monthly_file);
end
SIGN_CONV = Pm.SIGN_CONV(:)';
V_ref     = V_ref .* SIGN_CONV;
B_med     = B_med .* SIGN_CONV;

std_m     = sqrt(sum(B_med.^2, 2));
Bstd      = abs(B_med) ./ max(std_m, 1e-12);
shock_tag = arrayfun(@(k) sprintf('v%d', k), 1:M, 'UniformOutput', false);
dom_daily = cell(1,M);
dom_sign  = zeros(1,M);
for k = 1:M
    [~, im]      = max(Bstd(:,k));
    dom_daily{k} = VAR_NAMES{im};
    dom_sign(k)  = sign(B_med(im,k));
end
shock_cols = 1:M;

fprintf('  M=%d  T_eff=%d  p_daily=%d  |  SIGN_CONV = [%s] inherited from Stage A\n', ...
        M, T_eff, p_daily, strjoin(arrayfun(@(s) sprintf('%+d',s), SIGN_CONV, ...
        'UniformOutput', false), ' '));
for k = 1:M
    fprintf('    %-4s  kappa=%6.2f  [FEVD-dominant: %-8s loads %s on impact]\n', ...
            shock_tag{k}, lam_med(k), dom_daily{k}, tern(dom_sign(k)>0,'POSITIVELY','NEGATIVELY'));
end

%% =========================================================================
%  STEP 2 - MACRO MONTHLY DATA  (EBP / UNC_BEX / RAI_BEX are controls)
% =========================================================================
banner('STEP 2 - MACRO MONTHLY DATA');

N_macro       = numel(MACRO_NAMES);
unit_root_vec = double(~ismember(MACRO_NAMES, I0_VARS))';

T_raw_mac = readtable(MACRO_CSV);
Y_macro   = zeros(height(T_raw_mac), N_macro);
for n = 1:N_macro, Y_macro(:,n) = T_raw_mac.(MACRO_NAMES{n}); end
dates_macro = datetime(T_raw_mac.Date,'InputFormat','yyyy-MM-dd');
dates_macro = dates_macro(:);

valid = all(~isnan(Y_macro), 2);
if ~all(valid)
    a = find(valid,1,'first');  b = find(valid,1,'last');
    Y_macro = Y_macro(a:b,:);   dates_macro = dates_macro(a:b);
end
if any(isnan(Y_macro(:)))       % a hole inside the window would misalign the lags
    bad = dates_macro(any(isnan(Y_macro),2));
    error('macro data have missing values inside the sample (%s): fix the CSV.', ...
          strjoin(cellstr(string(bad,'yyyy-MM')), ', '));
end
T_mac = size(Y_macro,1);
fprintf('  Macro: [%d x %d]  %s to %s  (EBP/UNC_BEX/RAI_BEX are CONTROLS)\n', ...
        T_mac, N_macro, string(dates_macro(1),'yyyy-MM'), string(dates_macro(end),'yyyy-MM'));

%% =========================================================================
%  STEP 3 - MONTHLY INSTRUMENT  (read back from Stage A)
% =========================================================================
banner('STEP 3 - INSTRUMENT (signed monthly mean)');

ym_daily    = year(dates_vec)*100 + month(dates_vec);
months_list = unique(ym_daily,'sorted');
T_monthly   = numel(months_list);
dates_monthly = NaT(T_monthly,1);
for mi = 1:T_monthly
    idx_m = find(ym_daily == months_list(mi));
    dates_monthly(mi) = dates_vec(idx_m(end));
end
[tf, locb] = ismember(months_list, Pm.ym_monthly);
if ~all(tf) || size(Pm.MEAN,2) < M
    error('%s does not cover this month/shock set: re-run proxy_svar1.m.', monthly_file);
end
Z_monthly = Pm.MEAN(locb, 1:M);
fprintf('  Instrument read from %s: %d months, %d shocks.\n', monthly_file, T_monthly, M);
clear Pm

%% =========================================================================
%  STEP 4 - SAMPLE ALIGNMENT
% =========================================================================
banner('STEP 4 - SAMPLE ALIGNMENT');

ym_proxy = year(dates_monthly)*100 + month(dates_monthly);
ym_macro = year(dates_macro)*100   + month(dates_macro);
[~, ia, ib]  = intersect(ym_proxy, ym_macro);
Z_aligned    = Z_monthly(ia,:);
Y_aligned    = Y_macro(ib,:);
dates_common = dates_macro(ib);

if ~isempty(SAMPLE_END_YM)
    keep_s = (year(dates_common)*100 + month(dates_common)) <= SAMPLE_END_YM;
    if any(~keep_s)
        fprintf('  Sample end enforced at %d: %d month(s) dropped.\n', SAMPLE_END_YM, sum(~keep_s));
        Z_aligned = Z_aligned(keep_s,:);  Y_aligned = Y_aligned(keep_s,:);
        dates_common = dates_common(keep_s);
    end
end
T_common = size(Y_aligned,1);
fprintf('  Common sample: %d months (%s to %s)\n', T_common, ...
        string(dates_common(1),'yyyy-MM'), string(dates_common(end),'yyyy-MM'));

% Lags are built by row position, so a hole in the monthly calendar would pair
% y_t with the wrong y_{t-1..t-p}: stop rather than report misaligned dynamics.
mnum   = year(dates_common)*12 + month(dates_common);
gap_at = find(diff(mnum) ~= 1);
if ~isempty(gap_at)
    error('calendar gap between %s and %s: fix the macro CSV or truncate the sample.', ...
          string(dates_common(gap_at(1)),'yyyy-MM'), string(dates_common(gap_at(1)+1),'yyyy-MM'));
end
fprintf('  Calendar check: contiguous monthly grid, no missing months.\n');

%% =========================================================================
%  STEP 5 - MACRO-ONLY BVAR (GLP-Minnesota): dynamics and residuals
%    The hyper-parameters are optimised here and reused by the augmented VARs
%    of STEP 8, so the two sets of dynamics are comparable.
% =========================================================================
banner('STEP 5 - MACRO-ONLY BVAR (GLP)');

p_macro = P_MACRO_FIXED;
fprintf('  Lag order fixed a priori: p_macro = %d\n', p_macro);
fprintf('  Parameters per equation: macro-only %d, augmented %d   |   T = %d\n', ...
        N_macro*p_macro+1, (N_macro+1)*p_macro+1, T_common);

TAU_GRID = [0.1,0.5,1,2,5,10,20,50];         % starting values for the optimiser
lnML_tau = zeros(numel(TAU_GRID),1);
opt_tau  = struct('K',10);                   % draws discarded: only logmlike is used
opt_tau.minn_prior_decay = 1;  opt_tau.unit_root_ = unit_root_vec;
for ti = 1:numel(TAU_GRID)
    opt_tau.minn_prior_tau = TAU_GRID(ti);
    lnML_tau(ti) = bvar_(Y_aligned, p_macro, opt_tau).logmlike;
end
[~, bti] = max(lnML_tau);  tau_star = TAU_GRID(bti);

opt_glp = struct('K',2000,'hor',IRF_HOR_MAC);
opt_glp.priors = struct('name','Minnesota');
opt_glp.max_minn_hyper = 1;  opt_glp.max_compute = 2;  opt_glp.index_est = [1,3,4];
opt_glp.minn_prior_tau = tau_star;  opt_glp.minn_prior_decay = 1;
opt_glp.minn_prior_lambda = 5;  opt_glp.minn_prior_mu = 2;  opt_glp.minn_prior_omega = 2;
opt_glp.unit_root_ = unit_root_vec;
opt_glp.lb = [0.05,0.5,0.5];  opt_glp.ub = [100,500,500];
bvar_macro = bvar_(Y_aligned, p_macro, opt_glp);

% Recover the optimised hyper-parameters (lambda = dummy-initial-observation,
% mu = sum-of-coefficients, per rfvar3) so STEP 8 uses the same prior.
lam_star = 5; mu_star = 2; om_star = 2;
if isfield(bvar_macro,'prior') && isfield(bvar_macro.prior,'minn_prior_lambda')
    lam_star = bvar_macro.prior.minn_prior_lambda;
    mu_star  = bvar_macro.prior.minn_prior_mu;
    om_star  = bvar_macro.prior.minn_prior_omega;
    if isfield(bvar_macro.prior,'minn_prior_tau'), tau_star = bvar_macro.prior.minn_prior_tau; end
    fprintf('  Optimised hyper-parameters: tau=%.3g lambda=%.3g mu=%.3g omega=%.3g\n', ...
            tau_star, lam_star, mu_star, om_star);
else
    fprintf(2,'  [!] bvar_ did not expose the optimised hyper-parameters: defaults are used.\n');
end

Phi_pm = mean(bvar_macro.Phi_draws,3);
k_mac  = N_macro*p_macro + 1;
X_mac  = zeros(T_common-p_macro, k_mac);
for l = 1:p_macro
    X_mac(:,(l-1)*N_macro+1 : l*N_macro) = Y_aligned(p_macro-l+1:end-l, :);
end
X_mac(:,end) = 1;
U_mac = Y_aligned(p_macro+1:end,:) - X_mac*Phi_pm;
T_res = size(U_mac,1);
Z_t   = Z_aligned(p_macro+1:end,:);
fprintf('  p_macro=%d  tau*=%.3g  k_mac=%d  T_res=%d  lnML=%.3f\n', ...
        p_macro, tau_star, k_mac, T_res, bvar_macro.logmlike);

%% =========================================================================
%  STEP 6 - RELEVANCE (first-stage F) AND NORMALISATION REFERENCE
%    Not an identification step: it says how strong the instrument is, so the
%    IRFs are read in context.  The reference variable must be fixed BEFORE the
%    draws, because the normalisation is applied inside each one.
% =========================================================================
banner('STEP 6 - INSTRUMENT RELEVANCE (first-stage F)');

i_indpro = find(strcmp(MACRO_NAMES,'INDPRO'),1);
sd_Z = zeros(1,M); Fmax = zeros(1,M); F_indpro = zeros(1,M); Fmax_var = cell(1,M);
for k = 1:M
    Zt = Z_t(:,k) - mean(Z_t(:,k));
    sd_Z(k) = std(Zt);
    Fv = arrayfun(@(n) local_effF(Zt, U_mac(:,n), T_res), 1:N_macro);
    [Fmax(k), im] = max(Fv);
    Fmax_var{k}   = MACRO_NAMES{im};
    F_indpro(k)   = Fv(i_indpro);
end
fprintf('  %-6s %10s %16s %12s\n','shock','sd(Z)','strongest 1st-F','F@INDPRO');
for k = 1:M
    fprintf('  %-6s %10.4f  %7.1f(%-7s) %12.2f%s\n', shock_tag{k}, sd_Z(k), ...
            Fmax(k), Fmax_var{k}, F_indpro(k), tern(Fmax(k)<10,' [WEAK]',''));
end

ir_norm       = zeros(1,M);
norm_ref_name = repmat({'(1 s.d.)'},1,M);
if strcmpi(NORM_MODE,'refvar')
    ir = find(strcmp(MACRO_NAMES, NORM_REF_VAR), 1);
    if isempty(ir), error('NORM_REF_VAR: "%s" is not in MACRO_NAMES.', NORM_REF_VAR); end
    ir_norm(:)       = ir;
    norm_ref_name(:) = {NORM_REF_VAR};
    fprintf('\n  Normalisation reference resolved before the draws: %s (|impact| = %.3g)\n', ...
            NORM_REF_VAR, NORM_REF_SIZE);
end
% Per-draw diagnostics of the normalisation: the spread of the scale factor is
% the uncertainty that rescaling finished bands would throw away.
norm_c_bay  = nan(M,3);  norm_flip_bay = nan(1,M);  raw_imp_bay = nan(1,M);
norm_c_mbb  = nan(M,3);  norm_flip_mbb = nan(1,M);  raw_imp_mbb = nan(1,M);

%% =========================================================================
%  STEP 7 - SHOCK PREDICTABILITY TEST  (Ramey-type, HC1-robust)
%    The instrument is regressed on lags of a small NAMED macro set - testing
%    it against the daily VAR variables would be vacuous, since the shock is
%    orthogonal to them by construction.  I(1) regressors enter in first
%    differences, I(0) ones in levels; this concerns the test only, the VAR is
%    estimated in levels.  A rejection restricts the LABEL of the shock, not
%    the validity of the internal-instrument estimator.
% =========================================================================
banner('STEP 7 - SHOCK PREDICTABILITY TEST (Ramey-type, HC1-robust)');

PRED_VARS = {'INDPRO','UNRATE','PCEPI','EBP'};
PRED_DIFF = [    1   ,    1   ,   1   ,   0  ];   % 1 = first difference
PRED_LAGS = 2;
K_pred    = numel(PRED_VARS);

Wp = nan(T_common, K_pred);
for j = 1:K_pred
    n_j = find(strcmp(MACRO_NAMES, PRED_VARS{j}), 1);
    if isempty(n_j), error('PRED_VARS: "%s" is not in MACRO_NAMES.', PRED_VARS{j}); end
    if PRED_DIFF(j), Wp(2:end,j) = diff(Y_aligned(:,n_j)); else, Wp(:,j) = Y_aligned(:,n_j); end
end
Wp = (Wp - mean(Wp,1,'omitnan')) ./ max(std(Wp,0,1,'omitnan'),1e-12);   % comparable t-stats

Xp    = zeros(T_res, K_pred*PRED_LAGS);
lab_p = cell(1, K_pred*PRED_LAGS);
for l = 1:PRED_LAGS
    Xp(:,(l-1)*K_pred+1 : l*K_pred) = Wp(p_macro+1-l : T_common-l, :);
    for j = 1:K_pred, lab_p{(l-1)*K_pred+j} = sprintf('%s(-%d)', PRED_VARS{j}, l); end
end
if any(~isfinite(Xp(:))), error('Predictability regressors contain non-finite values.'); end
% Row i of z_t is dated dates_common(p_macro+i): check against the calendar that
% the lag-l block really sits l months earlier.
for l = 1:PRED_LAGS
    t_idx = p_macro + (1:T_res);
    if any(mnum(t_idx) - mnum(t_idx-l) ~= l)
        error('Predictability test: the lag-%d block does not span %d calendar months.', l, l);
    end
end

pred_R2 = nan(1,M); pred_F = nan(1,M); pred_p = nan(1,M);
pred_t  = nan(M, K_pred*PRED_LAGS);
q_pred  = size(Xp,2);
fprintf('  regressors: %s  (%d lags, standardised)\n', strjoin(PRED_VARS,', '), PRED_LAGS);
fprintf('  %-6s %9s %9s %10s   %s\n','shock','R2','F(HC1)','p-value','reading');
for k = 1:M
    yp   = Z_t(:,k) - mean(Z_t(:,k));
    Xr   = [ones(T_res,1), Xp];
    bp   = Xr\yp;  ep = yp - Xr*bp;
    dfe  = T_res - size(Xr,2);
    XtXi = inv(Xr'*Xr);
    Vhc  = XtXi * ((Xr.*ep)'*(Xr.*ep)) * XtXi * (T_res/dfe); %#ok<MINV> % HC1
    idx  = 2:size(Xr,2);                                          % drop the constant
    Wald = bp(idx)' * (Vhc(idx,idx) \ bp(idx));                   % ~ chi2(q)
    pred_F(k)  = Wald / q_pred;
    pred_p(k)  = 1 - chi2cdf(Wald, q_pred);
    pred_R2(k) = 1 - (ep'*ep) / max(sum((yp-mean(yp)).^2), 1e-18);
    pred_t(k,:) = (bp(idx) ./ sqrt(max(diag(Vhc(idx,idx)),1e-18)))';
    fprintf('  %-6s %9.4f %9.2f %10.4f   %s\n', shock_tag{k}, pred_R2(k), pred_F(k), pred_p(k), ...
            tern(pred_p(k)<0.05,'PREDICTABLE [!]','not predictable'));
end
fprintf('\n  Robust t-statistics by regressor (* if |t| > 1.96) - which information predicts:\n');
fprintf('  %-6s', 'shock');
for j = 1:q_pred, fprintf(' %13s', lab_p{j}); end
fprintf('\n');
for k = 1:M
    fprintf('  %-6s', shock_tag{k});
    for j = 1:q_pred
        fprintf(' %12.2f%s', pred_t(k,j), tern(abs(pred_t(k,j))>1.96,'*',' '));
    end
    fprintf('\n');
end

%% =========================================================================
%  STEP 8 - INTERNAL INSTRUMENT (Cholesky, Z ordered first): IRF AND FEVD
%
%  IRF and FEVD share the same MA recursion but NOT the same scaling:
%    IRF  uses bchol = P(:,1)/P(1,1)  -> shock normalised to move Z by one unit
%    FEVD uses P(:,1)                 -> unit-variance shock, the only scaling
%                                        under which the squared contributions
%                                        of the Na orthogonal shocks sum to the
%                                        forecast error variance.
%  The FEVD is a ratio of squares of the same object, hence invariant to the
%  presentation normalisation applied to the IRFs; its quantiles are therefore
%  taken on the unnormalised draws.  Row 1 of the stored FEVD is Z itself and
%  must be exactly 1 at h=0: it is checked, not assumed.
% =========================================================================
banner('STEP 8 - INTERNAL INSTRUMENT (Cholesky, ordered first)');

Na          = N_macro + 1;
unit_root_a = [0; unit_root_vec];
opt_aug = struct('K',K_AUG);
opt_aug.priors = struct('name','Minnesota');
opt_aug.max_minn_hyper = 0;
opt_aug.minn_prior_tau = tau_star;   opt_aug.minn_prior_decay = 1;
opt_aug.minn_prior_lambda = lam_star; opt_aug.minn_prior_mu = mu_star;
opt_aug.minn_prior_omega  = om_star;
opt_aug.unit_root_ = unit_root_a;

IRF_int    = zeros(N_macro,IRF_HOR_MAC+1,M);
IRF_int_16 = zeros(N_macro,IRF_HOR_MAC+1,M); IRF_int_84 = zeros(N_macro,IRF_HOR_MAC+1,M);
IRF_int_05 = zeros(N_macro,IRF_HOR_MAC+1,M); IRF_int_95 = zeros(N_macro,IRF_HOR_MAC+1,M);
FEVD_med   = zeros(Na,IRF_HOR_MAC+1,M);      % row 1 = Z (internal check)
FEVD_16    = zeros(Na,IRF_HOR_MAC+1,M);      FEVD_84 = zeros(Na,IRF_HOR_MAC+1,M);
dum_Y = cell(1,M); dum_X = cell(1,M);
Phi_bay = cell(1,M); Phi_sd = cell(1,M); Phi_K = zeros(1,M);

for k = 1:M
    Y_aug = [Z_aligned(:,k), Y_aligned];
    bva   = bvar_(Y_aug, p_macro, opt_aug);
    Phi_a = mean(bva.Phi_draws,3);
    Sig_a = mean(bva.Sigma_draws,3);

    Phi_bay{k} = Phi_a;
    Phi_sd{k}  = std(bva.Phi_draws,0,3);
    Phi_K(k)   = size(bva.Phi_draws,3);
    % Keep the prior dummies: with a conjugate NIW prior written as dummy
    % observations the posterior mean is least squares on [actual; dummies],
    % which is what STEP 9 re-estimates inside every bootstrap replication.
    if isfield(bva,'prior') && isfield(bva.prior,'YYdum') && isfield(bva.prior,'XXdum')
        dum_Y{k} = bva.prior.YYdum;  dum_X{k} = bva.prior.XXdum;
    end

    [IRF_int(:,:,k), FEVD_med(:,:,k)] = chol_irf_fevd(Phi_a, Sig_a, p_macro, IRF_HOR_MAC);

    Kd    = size(bva.Phi_draws,3);
    IRFd  = zeros(N_macro, IRF_HOR_MAC+1, Kd);
    FEVDd = zeros(Na,      IRF_HOR_MAC+1, Kd);
    for d = 1:Kd
        [IRFd(:,:,d), FEVDd(:,:,d)] = chol_irf_fevd(bva.Phi_draws(:,:,d), ...
                                                    bva.Sigma_draws(:,:,d), p_macro, IRF_HOR_MAC);
    end
    FEVD_16(:,:,k) = quantile(FEVDd,0.16,3);
    FEVD_84(:,:,k) = quantile(FEVDd,0.84,3);

    % Unit-effect normalisation INSIDE the draws: each draw is rescaled by its
    % own impact on the reference variable before the quantiles are taken, so
    % the band is a band on the reported ratio (Stock & Watson 2018).
    if ir_norm(k) > 0
        raw_imp_bay(k) = IRF_int(ir_norm(k),1,k);
        bref_d = squeeze(IRFd(ir_norm(k),1,:));
        c_d    = local_norm_fac(bref_d, NORM_MODE, NORM_REF_SIZE, sd_Z(k));
        IRFd   = IRFd .* reshape(c_d,1,1,[]);
        IRF_int(:,:,k) = IRF_int(:,:,k) * ...
                         local_norm_fac(raw_imp_bay(k), NORM_MODE, NORM_REF_SIZE, sd_Z(k));
        fin = isfinite(bref_d) & isfinite(c_d);
        norm_c_bay(k,:)  = quantile(c_d(fin),[0.16 0.50 0.84]);
        norm_flip_bay(k) = mean(sign(bref_d(fin)) ~= sign(raw_imp_bay(k)));
    else
        c_k  = local_norm_fac(1, NORM_MODE, NORM_REF_SIZE, sd_Z(k));   % constant factor
        IRFd = IRFd*c_k;  IRF_int(:,:,k) = IRF_int(:,:,k)*c_k;
    end

    IRF_int_16(:,:,k) = quantile(IRFd,0.16,3);  IRF_int_84(:,:,k) = quantile(IRFd,0.84,3);
    IRF_int_05(:,:,k) = quantile(IRFd,0.05,3);  IRF_int_95(:,:,k) = quantile(IRFd,0.95,3);

    fprintf('  %-6s augmented VAR (%d draws): IRF + posterior bands done  [impact on INDPRO = %+.4f]\n', ...
            shock_tag{k}, Kd, IRF_int(i_indpro,1,k));
    if ir_norm(k) > 0
        fprintf('        scale factor [16,50,84] = [%.2f, %.2f, %.2f], sign flips on %s = %.1f%%\n', ...
                norm_c_bay(k,1), norm_c_bay(k,2), norm_c_bay(k,3), norm_ref_name{k}, ...
                100*norm_flip_bay(k));
    end
end

% --- FEVD table ------------------------------------------------------------
fprintf('\n  FEVD: share of the forecast error variance due to the identified shock.\n');
chk = arrayfun(@(k) FEVD_med(1,1,k), 1:M);
if any(abs(chk-1) > 1e-8)
    fprintf(2,'  [!!] FEVD of Z at h=0 = %s, expected 1: recursion or ordering error.\n', mat2str(chk,6));
else
    fprintf('  Internal check passed: the shock explains 100%% of Z at h=0 for every shock.\n');
end
h_fevd = [0 6 12 24 48];  h_fevd = h_fevd(h_fevd <= IRF_HOR_MAC);
for k = 1:M
    fprintf('\n  %-4s  [sign anchor: %s loads %s on impact]   %% of FEV [16,84]\n', ...
            shock_tag{k}, dom_daily{k}, tern(dom_sign(k)>0,'POSITIVELY','NEGATIVELY'));
    fprintf('  %-10s', 'variable');
    for h = h_fevd, fprintf(' %20s', sprintf('h=%d',h)); end
    fprintf('\n');
    for n = 1:N_macro
        fprintf('  %-10s', MACRO_NAMES{n});
        for h = h_fevd
            fprintf('  %6.1f[%4.1f,%5.1f]', 100*FEVD_med(n+1,h+1,k), ...
                    100*FEVD_16(n+1,h+1,k), 100*FEVD_84(n+1,h+1,k));
        end
        fprintf('\n');
    end
end

%% =========================================================================
%  STEP 9 - MOVING-BLOCK BOOTSTRAP  (primary inference)
%    Residual-based MBB of the augmented VAR: blocks of length
%    5.03*T^(1/4) (Jentsch-Lunsford 2019), recentred, with every replication
%    re-estimating the SAME shrunk estimator as the reported point.  A
%    self-check verifies that this reproduces the Minnesota posterior mean
%    within Monte Carlo error; if it does not, the run stops rather than
%    silently reporting a band that belongs to a different estimator.
% =========================================================================
IRF_mbb_16 = nan(N_macro,IRF_HOR_MAC+1,M); IRF_mbb_84 = nan(N_macro,IRF_HOR_MAC+1,M);
IRF_mbb_05 = nan(N_macro,IRF_HOR_MAC+1,M); IRF_mbb_95 = nan(N_macro,IRF_HOR_MAC+1,M);
IRF_ols_all = nan(N_macro,IRF_HOR_MAC+1,M);   % point estimate of the bootstrap pair
mbb_ok     = zeros(1,M);
mbb_shrunk = false(1,M);

if DO_MBB
banner(sprintf('STEP 9 - MOVING-BLOCK BOOTSTRAP (B=%d)', MBB_B));

Ta    = T_common;
Tu_b  = Ta - p_macro;
L_blk = MBB_BLOCK;  if isempty(L_blk), L_blk = max(2, round(5.03*Tu_b^(1/4))); end
n_blk = ceil(Tu_b/L_blk);
fprintf('  T=%d  block length L=%d  (%d blocks)  B=%d replications\n', Tu_b, L_blk, n_blk, MBB_B);

for k = 1:M
    Y_aug = [Z_aligned(:,k), Y_aligned];
    Xb    = zeros(Tu_b, Na*p_macro+1);
    for l = 1:p_macro, Xb(:,(l-1)*Na+1 : l*Na) = Y_aug(p_macro-l+1:end-l,:); end
    Xb(:,end) = 1;
    Yb0 = Y_aug(p_macro+1:end,:);

    if ~strcmpi(MBB_ESTIMATOR,'shrunk') || isempty(dum_X{k})
        error(['the bootstrap must replicate the shrunk estimator (prior dummies) ' ...
               'to be comparable with the reported point at p = %d.'], p_macro);
    end
    Xd = dum_X{k};  Yd = dum_Y{k};
    Bfun = @(X,Y) [X; Xd] \ [Y; Yd];
    Bols = Bfun(Xb, Yb0);

    % Self-check in units of posterior sd: mean(Phi_draws,3) is itself a Monte
    % Carlo average, so a machine-precision tolerance would reject a correct
    % replication.
    d_sd = max(abs(Bols - Phi_bay{k}) ./ max(Phi_sd{k},1e-12), [], 'all');
    d_mc = d_sd * sqrt(Phi_K(k));
    if d_sd >= 0.25
        error('%s: the replicated estimator does not reproduce the reported point (%.3f posterior sd).', ...
              shock_tag{k}, d_sd);
    end
    fprintf('    %-4s shrunk-estimator check: PASS (max deviation %.3f posterior sd = %.1f MC se)\n', ...
            shock_tag{k}, d_sd, d_mc);
    mbb_shrunk(k) = true;

    Uols = Yb0 - Xb*Bols;
    Sols = (Uols'*Uols) / (Tu_b - size(Xb,2));
    IRF_ols = chol_irf_fevd(Bols, Sols, p_macro, IRF_HOR_MAC);

    IRFb  = nan(N_macro, IRF_HOR_MAC+1, MBB_B);
    nfail = 0;
    for b = 1:MBB_B
        % overlapping blocks of residuals, recentred (Bruggemann-Jentsch-Trenkler)
        starts = randi(Tu_b-L_blk+1, n_blk, 1);
        idxb   = reshape((starts + (0:L_blk-1))', [], 1);
        Ub     = Uols(idxb(1:Tu_b), :);
        Ub     = Ub - mean(Ub,1);
        % recursive DGP under the point estimate
        Ys = zeros(Ta, Na);  Ys(1:p_macro,:) = Y_aug(1:p_macro,:);
        for t = p_macro+1:Ta
            xr = [reshape(flipud(Ys(t-p_macro:t-1,:))', 1, Na*p_macro), 1];
            Ys(t,:) = xr*Bols + Ub(t-p_macro,:);
        end
        % re-estimate with the SAME estimator and recompute the Cholesky-first IRF
        Xs = zeros(Tu_b, Na*p_macro+1);
        for l = 1:p_macro, Xs(:,(l-1)*Na+1 : l*Na) = Ys(p_macro-l+1:end-l,:); end
        Xs(:,end) = 1;
        Ys0 = Ys(p_macro+1:end,:);
        Bs  = Bfun(Xs, Ys0);
        Us  = Ys0 - Xs*Bs;
        Ss  = (Us'*Us) / (Tu_b - size(Xs,2));
        % The VAR is in levels with I(1) variables, so roots at unity are
        % legitimate: only genuinely divergent draws are dropped.
        As = [Bs(1:Na*p_macro,:)'; eye(Na*(p_macro-1)), zeros(Na*(p_macro-1),Na)];
        if max(abs(eig(As))) > 1.02, nfail = nfail + 1; continue; end
        [~, flag] = chol((Ss+Ss')/2,'lower');
        if flag ~= 0, nfail = nfail + 1; continue; end
        IRFb(:,:,b) = chol_irf_fevd(Bs, Ss, p_macro, IRF_HOR_MAC);
        if ~all(isfinite(IRFb(:,:,b)),'all'), IRFb(:,:,b) = NaN; nfail = nfail + 1; end
    end

    % Unit-effect normalisation inside the bootstrap, exactly as in STEP 8.
    if ir_norm(k) > 0
        raw_imp_mbb(k) = IRF_ols(ir_norm(k),1);
        bref_b  = squeeze(IRFb(ir_norm(k),1,:));
        c_b     = local_norm_fac(bref_b, NORM_MODE, NORM_REF_SIZE, sd_Z(k));
        IRFb    = IRFb .* reshape(c_b,1,1,[]);
        IRF_ols = IRF_ols * local_norm_fac(raw_imp_mbb(k), NORM_MODE, NORM_REF_SIZE, sd_Z(k));
        fin = isfinite(bref_b) & isfinite(c_b);
        norm_c_mbb(k,:)  = quantile(c_b(fin),[0.16 0.50 0.84]);
        norm_flip_mbb(k) = mean(sign(bref_b(fin)) ~= sign(raw_imp_mbb(k)));
    else
        c_k  = local_norm_fac(1, NORM_MODE, NORM_REF_SIZE, sd_Z(k));
        IRFb = IRFb*c_k;  IRF_ols = IRF_ols*c_k;
    end

    keep = squeeze(all(all(isfinite(IRFb),1),2));
    mbb_ok(k) = sum(keep);
    IRF_ols_all(:,:,k) = IRF_ols;
    IRF_mbb_16(:,:,k) = quantile(IRFb(:,:,keep),0.16,3);
    IRF_mbb_84(:,:,k) = quantile(IRFb(:,:,keep),0.84,3);
    IRF_mbb_05(:,:,k) = quantile(IRFb(:,:,keep),0.05,3);
    IRF_mbb_95(:,:,k) = quantile(IRFb(:,:,keep),0.95,3);

    w_bay = IRF_int_84(i_indpro,1,k) - IRF_int_16(i_indpro,1,k);
    w_mbb = IRF_mbb_84(i_indpro,1,k) - IRF_mbb_16(i_indpro,1,k);
    fprintf('  %-6s %5d/%d usable replications (%d discarded)  |  68%% width at INDPRO impact: Bayes %.4f vs MBB %.4f (x%.1f)\n', ...
            shock_tag{k}, mbb_ok(k), MBB_B, nfail, w_bay, w_mbb, w_mbb/max(w_bay,1e-12));
end

fprintf('\n  Band-width ratio MBB/Bayesian (68%%), median over variables and horizons:\n');
for k = 1:M
    rw = (IRF_mbb_84(:,:,k)-IRF_mbb_16(:,:,k)) ./ max(IRF_int_84(:,:,k)-IRF_int_16(:,:,k),1e-12);
    fprintf('    %-4s  impact h=0: x%.2f   all horizons: x%.2f\n', ...
            shock_tag{k}, median(rw(:,1)), median(rw(:)));
end
fprintf('  Max |IRF_bootstrap point - IRF_bayes| (same estimator, so ~0 up to Sigma):\n');
for k = 1:M
    fprintf('    %-4s  %.3e\n', shock_tag{k}, max(abs(IRF_ols_all(:,:,k)-IRF_int(:,:,k)),[],'all'));
end
end

%% =========================================================================
%  STEP 10 - REPORTED PAIR, IRF TABLE AND FIGURES
% =========================================================================
banner(sprintf('STEP 10 - INTERNAL-INSTRUMENT IRF  [normalisation: %s]', NORM_MODE));

mbb_usable = DO_MBB && all(mbb_ok >= MBB_MIN_FRAC*MBB_B);
use_mbb    = strcmpi(BAND_PRIMARY,'mbb') && mbb_usable;
if strcmpi(BAND_PRIMARY,'mbb') && ~mbb_usable
    fprintf(2,'  [!] bootstrap unusable (usable replications: %s of %d) -> Bayesian pair.\n', ...
            mat2str(mbb_ok), MBB_B);
end
if use_mbb
    IRF_show = IRF_ols_all; L16 = IRF_mbb_16; U84 = IRF_mbb_84;
    L05 = IRF_mbb_05; U95 = IRF_mbb_95;
    BAND_LABEL = 'Minnesota point + moving-block bootstrap band [heteroskedasticity-robust]';
else
    IRF_show = IRF_int; L16 = IRF_int_16; U84 = IRF_int_84;
    L05 = IRF_int_05; U95 = IRF_int_95;
    BAND_LABEL = 'BVAR point + Gaussian posterior band [NOT heteroskedasticity-robust]';
    fprintf(2,'  NOTE: the Gaussian band prices no conditional heteroskedasticity;\n');
    fprintf(2,'        read it as a lower bound on uncertainty.\n');
end
fprintf('  Reported: %s\n', BAND_LABEL);

% Nothing is rescaled here: every IRF arrived already normalised draw by draw.
% What follows records WHICH factor the reported point carries, so tables and
% figures can state it.  Each pair carries its own factor.
norm_scale   = ones(1,M);
norm_desc    = repmat({''},1,M);
norm_raw_imp = nan(1,M);
norm_c_q     = nan(M,3);
norm_flip    = nan(1,M);
for k = 1:M
    switch lower(NORM_MODE)
      case 'unitz'
        norm_desc{k} = 'per 1 unit of the instrument innovation Z';
      case 'sd'
        norm_scale(k) = sd_Z(k);
        norm_desc{k}  = sprintf('per 1 s.d. of the instrument (sd(Z)=%.3f)', sd_Z(k));
      case 'refvar'
        if use_mbb
            b_ref = raw_imp_mbb(k); norm_c_q(k,:) = norm_c_mbb(k,:); norm_flip(k) = norm_flip_mbb(k);
        else
            b_ref = raw_imp_bay(k); norm_c_q(k,:) = norm_c_bay(k,:); norm_flip(k) = norm_flip_bay(k);
        end
        % first-stage F of the variable actually used, not the selected maximum
        F_ref = local_effF(Z_t(:,k)-mean(Z_t(:,k)), U_mac(:,ir_norm(k)), T_res);
        norm_scale(k)   = NORM_REF_SIZE/abs(b_ref);
        norm_raw_imp(k) = b_ref;
        norm_desc{k}    = sprintf('per a shock moving |%s| by %.3g on impact  [1st-stage F on %s = %.1f]', ...
                                  norm_ref_name{k}, NORM_REF_SIZE, norm_ref_name{k}, F_ref);
      otherwise
        error('NORM_MODE must be ''refvar'', ''sd'' or ''unitZ'' (got ''%s'').', NORM_MODE);
    end
end

fprintf('\n  %-6s %-10s %13s %10s   %s\n','shock','ref var','raw impact','scale','reading');
for k = 1:M
    fprintf('  %-6s %-10s %13.5f %10.2f   %s\n', shock_tag{k}, norm_ref_name{k}, ...
            norm_raw_imp(k), norm_scale(k), norm_desc{k});
end
if strcmpi(NORM_MODE,'refvar')
    fprintf('\n  Scale factor ACROSS DRAWS (the uncertainty an outside rescaling discards):\n');
    fprintf('  %-6s %10s %22s %14s %12s\n','shock','point','[16, 50, 84] of scale','84/16 spread','sign flips');
    for k = 1:M
        fprintf('  %-6s %10.2f   [%5.2f, %5.2f, %5.2f] %13.2fx %11.1f%%\n', shock_tag{k}, ...
                norm_scale(k), norm_c_q(k,1), norm_c_q(k,2), norm_c_q(k,3), ...
                norm_c_q(k,3)/max(norm_c_q(k,1),1e-12), 100*norm_flip(k));
    end
    % After normalisation every draw moves the reference variable by exactly
    % +/- NORM_REF_SIZE at h=0, so the reported point must sit at |NORM_REF_SIZE|
    % and the h=0 band has zero width unless some draws flip the sign.
    dmax = 0; wrep = zeros(1,M);
    for k = 1:M
        dmax    = max(dmax, abs(abs(IRF_show(ir_norm(k),1,k)) - NORM_REF_SIZE));
        wrep(k) = U84(ir_norm(k),1,k) - L16(ir_norm(k),1,k);
    end
    if dmax < 1e-10
        fprintf('  Internal check passed: every reported point moves %s by exactly %.3g at h=0.\n', ...
                NORM_REF_VAR, NORM_REF_SIZE);
    else
        fprintf(2,'  [!!] reported impact on the reference variable is %.4f away from %.3g.\n', ...
                dmax, NORM_REF_SIZE);
    end
    fprintf('  68%% band width of the reference variable at h=0 (0 = no draw flips its sign): %s\n', ...
            mat2str(round(wrep,4)));
end

h_show   = [0 6 12 24 48];  h_show = h_show(h_show <= IRF_HOR_MAC);
key_vars = {'INDPRO','UNRATE','PCEPI'};
for k = 1:M
    fprintf('\n  %s  [sign anchor: %s loads %s on impact - daily FEVD-dominant]\n', ...
            shock_tag{k}, dom_daily{k}, tern(dom_sign(k)>0,'POSITIVELY','NEGATIVELY'));
    fprintf('      %s\n      value [16,84] %s\n', norm_desc{k}, BAND_LABEL);
    fprintf('  %-8s', 'var');
    for h = h_show, fprintf(' %22s', sprintf('h=%d',h)); end
    fprintf('\n');
    for vv = 1:numel(key_vars)
        n = find(strcmp(MACRO_NAMES,key_vars{vv}),1);
        if isempty(n), continue; end
        fprintf('  %-8s', key_vars{vv});
        for h = h_show
            fprintf(' %8.4f[%+.3f,%+.3f]', IRF_show(n,h+1,k), L16(n,h+1,k), U84(n,h+1,k));
        end
        fprintf('\n');
    end
end

horizon = 0:IRF_HOR_MAC;  ncol = 4;  nrow = ceil(N_macro/ncol);
for k = 1:M
    figure('Name',sprintf('proxy_svar_02_cholesky - %s',shock_tag{k}), ...
           'Position',[30 30 1320 max(700,nrow*190)],'Color','w');
    for n = 1:N_macro
        subplot(nrow,ncol,n); hold on;
        fill([horizon fliplr(horizon)], [squeeze(L05(n,:,k)) fliplr(squeeze(U95(n,:,k)))], ...
             [0.98 0.90 0.80],'EdgeColor','none','FaceAlpha',0.8);
        fill([horizon fliplr(horizon)], [squeeze(L16(n,:,k)) fliplr(squeeze(U84(n,:,k)))], ...
             [0.95 0.75 0.50],'EdgeColor','none','FaceAlpha',0.8);
        % dashed = the other point estimate, for contrast only: a band always
        % belongs to the estimator that generated it
        if use_mbb
            plot(horizon, squeeze(IRF_int(n,:,k)),'--','Color',[0.30 0.30 0.70],'LineWidth',1.0);
        elseif DO_MBB && any(mbb_ok > 0)
            plot(horizon, squeeze(IRF_ols_all(n,:,k)),'--','Color',[0.30 0.30 0.70],'LineWidth',1.0);
        end
        plot(horizon, squeeze(IRF_show(n,:,k)),'-','Color',[0.85 0.45 0.10],'LineWidth',1.4);
        yline(0,'k:'); xlim([0 IRF_HOR_MAC]);
        title(MACRO_NAMES{n},'FontSize',8,'Interpreter','none');
        set(gca,'FontSize',7,'Box','on');
    end
    sgtitle({sprintf('%s - internal instrument (Cholesky, Z first)  |  %s', shock_tag{k}, norm_desc{k}), ...
             sprintf('solid + shaded: %s (68%%/90%%) | dashed: the other point estimate | daily FEVD-dominant: %s', ...
                     BAND_LABEL, dom_daily{k})}, 'FontSize',10,'Interpreter','none');
    print(gcf, sprintf('fig_p02chol_irf_shock%d_%s.png', k, SRC_BASE), '-dpng','-r150');
end

%% =========================================================================
%  STEP 11 - MONTIEL OLEA-STOCK-WATSON (2021) ANDERSON-RUBIN SETS
%    The proxy-SVAR estimator is a ratio, so a Wald interval under-covers when
%    the denominator is near zero.  MSW invert a test instead: the confidence
%    set is the set of values not rejected, which reduces to a quadratic
%    inequality.  A > 0 gives a BOUNDED set; otherwise the instrument does not
%    pin down the MAGNITUDE at that (variable, horizon).  These sets belong to
%    the EXTERNAL proxy-SVAR (macro-only dynamics) and are a robustness exhibit,
%    not a confidence set for the internal-instrument point estimate.
% =========================================================================
banner('STEP 11 - MSW(2021) ANDERSON-RUBIN SETS (external proxy-SVAR)');

Psi_mac = var_ma(Phi_pm, p_macro, N_macro, IRF_HOR_MAC);
UU = U_mac'*U_mac;
ZU = zeros(M,N_macro);  SZ_v = zeros(M,1);
for k = 1:M
    zc = Z_t(:,k) - mean(Z_t(:,k));
    ZU(k,:) = zc'*U_mac;  SZ_v(k) = zc'*zc;
end
df_res  = T_res - 2;
F_crits = [finv(0.68,1,df_res), finv(0.90,1,df_res)];
n_ci    = numel(F_crits);
IRF_AR_lo   = nan(N_macro,IRF_HOR_MAC+1,M,n_ci);
IRF_AR_hi   = nan(N_macro,IRF_HOR_MAC+1,M,n_ci);
msw_bounded = false(N_macro,IRF_HOR_MAC+1,M,n_ci);

ar_ref_idx = zeros(1,M);          % same reference variable as the reported IRFs
for k = 1:M
    ir = find(strcmp(MACRO_NAMES, norm_ref_name{k}), 1);
    if isempty(ir), ir = find(strcmp(MACRO_NAMES, Fmax_var{k}), 1); end
    ar_ref_idx(k) = ir;
end

for h = 0:IRF_HOR_MAC
    Psi_h = Psi_mac(:,:,h+1);
    for k = 1:M
        rk    = ar_ref_idx(k);
        b_ref = ZU(k,rk);  S_ref = UU(rk,rk);  Sz_k = SZ_v(k);
        for m = 1:N_macro
            prow = Psi_h(m,:);
            a_hm = prow*ZU(k,:)';
            Q_hm = prow*UU*prow';
            R_hm = prow*UU(:,rk);
            for ci = 1:n_ci
                fc = F_crits(ci);
                Aq = b_ref^2*(df_res+fc) - fc*Sz_k*S_ref;
                Bq = -2*a_hm*b_ref*(df_res+fc) + 2*fc*Sz_k*R_hm;
                Cq = a_hm^2*(df_res+fc) - fc*Sz_k*Q_hm;
                disc = Bq^2 - 4*Aq*Cq;
                if Aq > 0 && disc >= 0
                    r1 = (-Bq-sqrt(disc))/(2*Aq);  r2 = (-Bq+sqrt(disc))/(2*Aq);
                    IRF_AR_lo(m,h+1,k,ci) = min(r1,r2);
                    IRF_AR_hi(m,h+1,k,ci) = max(r1,r2);
                    msw_bounded(m,h+1,k,ci) = true;
                else
                    IRF_AR_lo(m,h+1,k,ci) = -Inf;  IRF_AR_hi(m,h+1,k,ci) = Inf;
                end
            end
        end
    end
end

% Put the sets on the scale of the reported IRFs: MSW parameterise with the
% reference residual moving by +1, so the map is a multiplication by the same
% factor the reported point carries, with the bounds swapped when it is negative.
for k = 1:M
    switch lower(NORM_MODE)
        case 'refvar'
            sgn = sign(norm_raw_imp(k));
            if sgn == 0 || ~isfinite(sgn), sgn = 1; end
            fac = NORM_REF_SIZE*sgn;
        case 'sd',    fac = sd_Z(k);
        otherwise,    fac = 1;
    end
    lo = IRF_AR_lo(:,:,k,:)*fac;  hi = IRF_AR_hi(:,:,k,:)*fac;
    IRF_AR_lo(:,:,k,:) = min(lo,hi);  IRF_AR_hi(:,:,k,:) = max(lo,hi);
end

fprintf('  Share of (variable,horizon) cells with a BOUNDED 90%% AR set:\n');
for k = 1:M
    bnd = msw_bounded(:,:,k,2);
    fprintf('    %-4s  %5.1f%%   [ref var: %s, 1st-stage F=%.1f]\n', shock_tag{k}, ...
            100*mean(bnd(:)), MACRO_NAMES{ar_ref_idx(k)}, Fmax(k));
end
fprintf('\n  90%% AR sets at key (variable,horizon), on the reported scale:\n');
for k = 1:M
    fprintf('  %-4s', shock_tag{k});
    for h = h_show, fprintf(' %20s', sprintf('h=%d',h)); end
    fprintf('\n');
    for vv = 1:numel(key_vars)
        n = find(strcmp(MACRO_NAMES,key_vars{vv}),1);
        if isempty(n), continue; end
        fprintf('  %-8s', key_vars{vv});
        for h = h_show
            if msw_bounded(n,h+1,k,2)
                fprintf(' %9.3f,%+9.3f', IRF_AR_lo(n,h+1,k,2), IRF_AR_hi(n,h+1,k,2));
            else
                fprintf(' %19s', '(-Inf, +Inf)');
            end
        end
        fprintf('\n');
    end
end

%% =========================================================================
%  STEP 12 - SAVE
% =========================================================================
save('proxy_svar_02_cholesky_results.mat', 'SRC','shock_cols','shock_tag','dom_daily', ...
     'MACRO_NAMES','p_macro','tau_star','lam_star','mu_star','om_star', ...
     'T_common','T_res','dates_common','lam_med', ...
     'sd_Z','Fmax','Fmax_var','F_indpro', ...
     'IRF_int','IRF_int_16','IRF_int_84','IRF_int_05','IRF_int_95', ...
     'IRF_mbb_16','IRF_mbb_84','IRF_mbb_05','IRF_mbb_95','mbb_ok','BAND_LABEL', ...
     'IRF_ols_all','BAND_PRIMARY','use_mbb','SAMPLE_END_YM','P_MACRO_FIXED', ...
     'NORM_MODE','NORM_REF_SIZE','norm_scale','norm_ref_name','norm_desc','norm_raw_imp', ...
     'norm_c_q','norm_flip','norm_c_bay','norm_c_mbb','norm_flip_bay','norm_flip_mbb', ...
     'raw_imp_bay','raw_imp_mbb','ir_norm', ...
     'SIGN_CONV','dom_sign','MBB_ESTIMATOR','mbb_shrunk', ...
     'pred_R2','pred_F','pred_p','pred_t','PRED_VARS','PRED_DIFF','PRED_LAGS','lab_p', ...
     'IRF_AR_lo','IRF_AR_hi','msw_bounded','ar_ref_idx','F_crits', ...
     'FEVD_med','FEVD_16','FEVD_84', ...
     'IRF_HOR_MAC','-v7.3');

banner(sprintf(['DONE - Stage B\n  saved: proxy_svar_02_cholesky_results.mat\n' ...
                '  bands reported: %s'], BAND_LABEL));

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function Psi = var_ma(Phi, p, N, H)
% MA coefficient matrices of a VAR(p): Psi_0 = I, Psi_h = sum_j A_j Psi_{h-j}.
% Phi is (N*p+1) x N with the constant last, so Phi(1:N*p,:)' = [A_1 ... A_p].
% Equivalent to J*A_companion^h*J' but without forming the companion powers.
    A   = reshape(Phi(1:N*p,:)', N, N, p);
    Psi = zeros(N, N, H+1);
    Psi(:,:,1) = eye(N);
    for h = 1:H
        Sh = zeros(N);
        for j = 1:min(h,p)
            Sh = Sh + A(:,:,j) * Psi(:,:,h-j+1);
        end
        Psi(:,:,h+1) = Sh;
    end
end

function [ir, fevd] = chol_irf_fevd(Phi, Sigma, p, H)
% Internal-instrument IRF and FEVD of the augmented VAR [Z, macro].
%   ir(n,h+1)   response of macro variable n, per unit of the Z innovation
%               (b = P(:,1)/P(1,1)); the Z row itself is dropped.
%   fevd(:,h+1) share of the forecast error variance of every augmented
%               variable (row 1 = Z) due to the first Cholesky shock, computed
%               with P(:,1), the unit-variance scaling.
    Na  = size(Sigma,1);
    P   = chol((Sigma+Sigma')/2,'lower');
    b   = P(:,1)/P(1,1);
    Psi = var_ma(Phi, p, Na, H);
    ir   = zeros(Na-1, H+1);
    fevd = zeros(Na,   H+1);
    num = zeros(Na,1);  den = zeros(Na,1);
    for h = 0:H
        v          = Psi(:,:,h+1)*b;
        ir(:,h+1)  = v(2:end);
        Th         = Psi(:,:,h+1)*P;
        num        = num + Th(:,1).^2;
        den        = den + sum(Th.^2, 2);
        fevd(:,h+1) = num ./ max(den, 1e-18);
    end
end

function c = local_norm_fac(bref, mode, ref_size, sdz)
% Positive scale factor, one per element of bref: called with a vector of
% per-draw impacts on the reference variable and with a scalar for the point
% estimate, so both are normalised by the same rule applied to their own data.
% The factor is always positive: the direction is left to the upstream anchor.
    switch lower(mode)
        case 'unitz', c = ones(size(bref));
        case 'sd',    c = sdz*ones(size(bref));
        case 'refvar'
            c = ref_size ./ abs(bref);
            c(~isfinite(c)) = NaN;      % a draw with zero reference impact is dropped
        otherwise
            error('NORM_MODE must be ''refvar'', ''sd'' or ''unitZ'' (got ''%s'').', mode);
    end
end

function F = local_effF(z, u, T)
% Heteroskedasticity-robust first-stage F of the instrument in the regression
% of the reduced-form residual u on z.
    z = z - mean(z);
    X = [ones(T,1), z];
    bb = X\u;  e = u - X*bb;
    XtXi = inv(X'*X);
    V = XtXi * ((X.*e)'*(X.*e)) * XtXi * (T/(T-2)); %#ok<MINV>
    F = bb(2)^2 / max(V(2,2), 1e-18);
end

function banner(txt)
    line = repmat('=', 1, 70);
    fprintf('\n%s\n %s\n%s\n', line, txt, line);
end

function out = tern(cond, a, b)
    if cond, out = a; else, out = b; end
end
