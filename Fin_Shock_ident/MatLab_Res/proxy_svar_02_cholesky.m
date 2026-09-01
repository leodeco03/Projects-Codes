% =========================================================================
%  proxy_svar_02_cholesky.m  —  STAGE B (MACRO): the shock used as an INTERNAL
%                               instrument (Cholesky, ordered first) — GLP BVAR.
%
%  DIVISION OF LABOUR (kept separate on purpose)
%    proxy_svar1.m         : (legacy) monthly aggregation + uncertainty validation
%    proxy_svar1_02.m      : STAGE A — builds the AGNOSTIC external instrument
%                            (signed monthly mean) and shows its monthly
%                            correlation with the uncertainty indices (UNC/EPU/…).
%    THIS FILE (Stage B)   : reads that instrument and estimates the MONTHLY
%                            MACRO effects with a GLP-Minnesota BVAR, using the
%                            shock as an INTERNAL instrument (ordered FIRST,
%                            Cholesky = Plagborg-Moller & Wolf 2021).  ONLY the
%                            internal identification is produced here.
%    Script_2_ProxySVAR.m  : the macro stage that anchors each shock to a real
%                            proxy (EBP/UNC_BEX/RAI_BEX) and now carries the
%                            EXTERNAL-vs-INTERNAL comparison (external MR IRF +
%                            internal-instrument overlay).  That is where the two
%                            identifications are compared; this file is the clean
%                            agnostic internal-instrument run.
%
%  AGNOSTIC identification: shocks enter as v1/v2/v3 with NO economic label and
%  NO real proxy; EBP/UNC_BEX/RAI_BEX are ordinary CONTROL variables in the VAR.
%  The daily FEVD-dominant variable is printed only as an a-posteriori reading.
%
%  INTERNAL INSTRUMENT (the focus), normalised PER UNIT of the Z innovation:
%    b_chol = P(2:end,1)/P(1,1),  P = chol(Sigma_aug),  Sigma_aug = cov of the
%    residuals of the augmented VAR [Z, macro] with Z ordered first.
%  Consistent under non-invertibility & measurement error (PMW 2021).  A first-
%  stage F RELEVANCE diagnostic (Step 6) reports how strong/weak the instrument
%  is, so the IRFs are read in context — but no external IRF is drawn here.
%
% 
%  REFERENCES
%    Plagborg-Moller & Wolf (2021, Econometrica) — internal instrument / Cholesky
%    Giannone, Lenza & Primiceri (2015, ReStat)  — GLP Minnesota BVAR
%    Sims, Stock & Watson (1990)                 — levels VAR with mixed I(0)/I(1)
%    Ferroni & Canova (2025)                     — BVAR toolbox
% =========================================================================

clear; clc; close all;
rng(42, 'twister');

% ===================== CONFIG (set by hand) ==============================
SRC          = 'mori_3var.mat';
[~, SRC_BASE]= fileparts(SRC);
% ---- SHOCK SET ----------------------------------------------------------
%  ALL the shocks identified by the daily Mori step travel to the macro stage,
%  in their natural order v1..vM.  No selection step anywhere in the pipeline.
IRF_HOR_MAC  = 48;
K_AUG        = 1000;             % posterior draws for the augmented (internal) VARs
% ---- SAMPLE END ---------------------------------------------------------
%  The 2025 federal shutdown (1 Oct - 12 Nov) stopped the BLS from producing
%  the October-2025 CPI (cancelled outright, prices cannot be collected
%  retroactively) and the October-2025 household survey, so UNRATE for that
%  month does not exist either.  macro_monthly_ready.csv therefore has a HOLE
%  at 2025-10, and lags are built by row position: a hole silently pairs y_t
%  with the wrong y_{t-1..t-p}.  We end the macro sample BEFORE the hole rather
%  than interpolate: the common sample goes from 431 to 429 months (2025-11 and
%  2025-12 dropped; 2025-10 was already absent).  No imputation to defend.
SAMPLE_END_YM = 202509;          % [] -> use everything (and trust the guard)
% ---- LAG ORDER ----------------------------------------------------------
%  FIXED a priori, not selected by marginal likelihood.  Rationale (same as
%  bvar_GLP_strict.m, Block B): under the GLP-Minnesota prior the lag length is
%  not a model-selection problem — the prior variance decays as 1/s^2, so
%  irrelevant lags are shrunk to zero and a generous p costs almost nothing in
%  the posterior while a too-short p imposes a hard, unshrinkable truncation.
%  Banbura-Giannone-Reichlin (2010) use 12 lags for monthly data; GLP (2015)
%  make the same point (results are flat in p once the prior is optimised).
%  We take p=15 = 12 + one quarter of slack.
%  CAVEAT: this is a Bayesian argument.  The moving-block bootstrap of Step 7b
%  re-estimates the augmented VAR by OLS, which does NOT shrink: with Na=11 and
%  p=15 that is 166 coefficients per equation.  Step 7b reports how many
%  replications survive; if that share collapses the frequentist pair is not
%  usable at this p and the Bayesian pair is the one to report.
P_MACRO_FIXED = 15;
% ---- INFERENCE ----------------------------------------------------------
%  The Gaussian-likelihood (inverse-Wishart) posterior bands understate the
%  sampling uncertainty of the impact vector when the instrument is fat-tailed
%  and its informative content is concentrated in a few crisis months: the
%  Gaussian posterior prices no conditional heteroskedasticity.  The PRIMARY
%  inference is therefore a residual-based MOVING-BLOCK BOOTSTRAP of the
%  augmented VAR, which is valid under conditional heteroskedasticity and fat
%  tails (Bruggemann-Jentsch-Trenkler 2016; Jentsch-Lunsford 2019).  The
%  Bayesian bands are still computed and stored, for comparison only.
%  WHICH PAIR IS REPORTED.  The two estimators must be reported as internally
%  consistent PAIRS (point + band), never mixed:
%     'mbb'   : OLS point        + moving-block bootstrap band  (heterosk.-robust
%                                  band, but the point estimate is UNSHRUNK)
%     'bayes' : Minnesota point  + Gaussian posterior band      (smooth, shrunk
%                                  point, but the band prices no heterosk.)
%     'auto'  : 'mbb' whenever enough bootstrap replications survive
%  At p=15 the OLS estimator has 166 coefficients per equation and its IRFs
%  oscillate at short horizons and FLIP SIGN beyond h~20 (see the sign-agreement
%  diagnostic printed in Step 8).  That is estimation noise, not economics.
%
%  RESOLVED.  That dilemma only existed because the bootstrap re-estimated the
%  VAR by OLS.  With MBB_ESTIMATOR='shrunk' every replication re-estimates the
%  MINNESOTA posterior mean instead (least squares on [actual; prior dummies],
%  which IS the posterior mean under a conjugate NIW prior written as dummy
%  observations), so the 'mbb' pair becomes
%       Minnesota point + moving-block bootstrap band
%  i.e. shrunk point AND heteroskedasticity-robust band, with no p=15 objection.
%  A self-check in Step 7b verifies that the replicated estimator reproduces the
%  reported point; if it does not, the code falls back to OLS AND forces
%  BAND_PRIMARY='bayes', because the OLS pair is not defensible at this p.
BAND_PRIMARY = 'mbb';
DO_MBB       = true;             % the bootstrap is computed either way (diagnostic)
MBB_ESTIMATOR= 'shrunk';         % 'shrunk' -> replications re-estimate the Minnesota
                                 %   posterior mean, so point and band are the SAME
                                 %   estimator.  'ols' -> unshrunk (comparison only)
MBB_B        = 2000;             % stage-2 bootstrap replications
MBB_B1       = 500;              % stage-1 replications used to estimate the Kilian bias
MBB_BLOCK    = [];               % [] -> Jentsch-Lunsford rule 5.03*T^(1/4)
% ---- KILIAN (1998) BIAS CORRECTION: OFF, and it is not a tuning choice -------
%  OLS understates persistence in a finite sample, and Kilian (1998) corrects it
%  by simulating from the fitted DGP, measuring the average bias and subtracting
%  it — with a STATIONARITY ADJUSTMENT that shrinks the correction back to zero
%  whenever subtracting it would make the companion matrix explosive.
%  That adjustment is designed for a STATIONARY VAR.  This VAR is in LEVELS with
%  I(1) variables, so max|eig| sits AT ~1 by construction (measured: 1.0018) and
%  the adjustment drove delta to 0 on every shock — the correction was inert,
%  not mis-tuned, and only put a "bias-corrected" label on an uncorrected
%  estimate.  Turned off explicitly.  (The same file already treats roots up to
%  1.02 as legitimate when filtering bootstrap draws: the 1.0 threshold used by
%  the correction was inconsistent with that.)
DO_KILIAN    = false;            % true only for a stationary specification
% ---- IRF NORMALISATION ---------------------------------------------------
%  'unitZ'  : one unit of the instrument innovation Z.  This is what the
%             Cholesky-first algebra returns (b = P(:,1)/P(1,1)), but Z has NO
%             natural unit: sd(Z) ~ 0.21, so "one unit" is a ~4.7 s.d. shock and
%             the responses read as implausibly large (UNRATE -0.60pp on impact).
%  'sd'     : one standard deviation of the instrument.
%  'refvar' : the shock whose IMPACT on a reference macro variable equals
%             NORM_REF_SIZE in absolute value — the Gertler-Karadi convention
%             ("the shock that moves EBP by 25 basis points on impact").
%             The scale factor is |NORM_REF_SIZE / b_ref| and is always POSITIVE:
%             the direction is left to the upstream economic sign anchor and is
%             NOT overwritten here.  Normalising to "+1" instead of "|1|" would
%             silently flip any shock whose reference variable falls.
%             Use the SAME reference variable for every shock, otherwise the
%             panels sit on different yardsticks.  Picking it per shock by max
%             first-stage F is WRONG: F is precision, not magnitude, so a shock
%             that barely moves its reference variable gets a huge scale factor.
%  EBP is the default: among the variables that all three shocks move in the
%  SAME direction at impact, it is the one whose impact varies least across
%  shocks (a factor of 2, against 3.1 for RAI_BEX and 5.2 for UNC_BEX), it is in
%  percentage points, and it is the standard credit-risk price in the literature
%  (Gilchrist-Zakrajsek 2012).  A reference the shocks move in opposite
%  directions (FFR) cannot be used at all, and one that a shock barely moves
%  inflates that shock's responses — the failure mode Gertler-Karadi flag in
%  their own footnote 20.
%
%  WHERE THE NORMALISATION IS APPLIED.  Inside every posterior draw and every
%  bootstrap replication (Steps 7 and 7b), NOT to the finished bands.  The
%  reported response is a RATIO, gamma*Theta_{i,h}/|Theta_{ref,0}|, so its band
%  must price the sampling uncertainty of the denominator as well; Stock &
%  Watson (2018) state that rescaling pre-computed bands by a factor read off
%  the point estimate is incorrect and that the unit-effect normalisation
%  belongs "inside" the bootstrap.  This is also the only scaling the internal
%  instrument identifies: measurement error in the instrument attenuates every
%  response by a common factor which cancels in the ratio but NOT in 'unitZ' or
%  'sd' (Plagborg-Moller & Wolf 2021).  Shapes and signs are unchanged; band
%  WIDTHS are not, and that is the point of the change.
NORM_MODE     = 'refvar';        % 'refvar' | 'sd' | 'unitZ'
NORM_REF_VAR  = 'EBP';           % '' -> per-shock max-F variable (not recommended)
NORM_REF_SIZE = 0.25;            % target |impact| on the reference variable
NORM_LABEL    = 'per unit of the instrument innovation Z';   % overwritten in Step 8
% =========================================================================

addpath(fullfile(pwd,'bvar_Ferroni','bvartools'));
addpath(fullfile(pwd,'bvar_Ferroni','cmintools'));

fprintf('\n%s\n proxy_svar_02_cholesky — STAGE B: macro IRFs, INTERNAL (Cholesky) instrument\n SRC = %s\n%s\n', ...
        repmat('=',1,70), SRC, repmat('=',1,70));

%% =========================================================================
%  STEP 1 — LOAD DAILY SVAR OUTPUT + AGNOSTIC SHOCK SELECTION
% =========================================================================
fprintf('\n%s\n STEP 1 — LOAD DAILY SHOCKS (all shocks, no selection)\n%s\n', repmat('=',1,70), repmat('=',1,70));
S = load(SRC);
V_ref     = S.V_ref;                 [T_eff, M] = size(V_ref);
Vt_draws  = S.Vt_draws;              N_DRAWS    = size(Vt_draws,3);
p         = double(S.p);
B_med     = S.B_med;
lam_med   = S.lam_med(:);
VAR_NAMES = cellstr(S.VAR_NAMES(:));
if isfield(S,'dates_vec') && isdatetime(S.dates_vec) && numel(S.dates_vec)==T_eff
    dates_vec = S.dates_vec(:);
else
    Traw_d    = readtable('financial_data_aligned_corrected.csv');
    d_all     = datetime(Traw_d.Date, 'InputFormat', 'yyyy-MM-dd');
    dates_vec = d_all(end-T_eff+1:end);
end

shock_cols = 1:M;                 % all shocks, natural order — no selection
N_shocks   = M;

% ---- SIGN CONVENTION: INHERITED from Stage A, never redefined here ----------
%  Stage A applies SIGN_CONV to the daily shocks BEFORE aggregating and stores it
%  in shocks_monthly_<SRC>.mat.  Reading it back — rather than hard-coding it in
%  two places — is what keeps the stages from drifting apart: the monthly
%  instrument reused in Step 3 is ALREADY signed, so V_ref here must carry the
%  same convention or the labels would contradict the data.
monthly_file_sig = sprintf('shocks_monthly_%s.mat', SRC_BASE);
SIGN_CONV = ones(1,M);
if exist(monthly_file_sig,'file')==2
    Psig = load(monthly_file_sig,'SIGN_CONV');
    if isfield(Psig,'SIGN_CONV') && numel(Psig.SIGN_CONV)==M
        SIGN_CONV = Psig.SIGN_CONV(:)';
        fprintf('  SIGN_CONV = [%s] inherited from %s\n', ...
                strjoin(arrayfun(@(s)sprintf('%+d',s),SIGN_CONV,'UniformOutput',false),' '), monthly_file_sig);
    else
        fprintf(2,'  [!] %s carries no SIGN_CONV -> upstream anchor (+1 for all).\n', monthly_file_sig);
    end
else
    fprintf(2,'  [!] %s not found -> upstream anchor (+1 for all). Run Stage A first.\n', monthly_file_sig);
end
V_ref = V_ref .* SIGN_CONV;
B_med = B_med .* SIGN_CONV;
for d3 = 1:size(Vt_draws,3), Vt_draws(:,:,d3) = Vt_draws(:,:,d3) .* SIGN_CONV; end

std_m = sqrt(sum(B_med.^2,2));  Bstd = abs(B_med)./max(std_m,1e-12);
dom_daily = cell(1,N_shocks);   shock_tag = cell(1,N_shocks);  dom_sign = zeros(1,N_shocks);
for ki = 1:N_shocks
    [~, im] = max(Bstd(:,shock_cols(ki)));  dom_daily{ki}=VAR_NAMES{im};
    dom_sign(ki)  = sign(B_med(im,shock_cols(ki)));
    shock_tag{ki} = sprintf('v%d', shock_cols(ki));
end
fprintf('  M=%d  T_eff=%d  p=%d  N_DRAWS=%d\n', M, T_eff, p, N_DRAWS);
fprintf('  All %d shocks carried to the macro stage, natural order:\n', N_shocks);
for ki=1:N_shocks
    fprintf('    %-4s  kappa=%6.2f  [FEVD-dominant: %-8s loads %s on impact <- sign anchor]\n', ...
            shock_tag{ki}, lam_med(shock_cols(ki)), dom_daily{ki}, ...
            tern_local(dom_sign(ki)>0,'POSITIVELY','NEGATIVELY'));
end

%% =========================================================================
%  STEP 2 — MACRO MONTHLY DATA  (EBP/UNC_BEX/RAI_BEX = CONTROLS only)
% =========================================================================
fprintf('\n%s\n STEP 2 — MACRO MONTHLY DATA\n%s\n', repmat('=',1,70), repmat('=',1,70));
MACRO_CSV   = '../../PythonThesis/FinalPy/files/04_output/macro_monthly_ready.csv';%  RAI_BEX / UNC_BEX / EBP enter as ORDINARY CONTROLS, never as instruments or
%  anchors: the identification is agnostic, so no shock is matched to a real
%  proxy.  They are mean-reverting risk premia -> white-noise prior mean
%  (delta_i = 0); the remaining seven get the random-walk prior mean.
MACRO_NAMES = {'INDPRO','UNRATE','HOUST','CSHPI','PCEPI','NRCREDIT', ...
               'FFR','RAI_BEX','UNC_BEX','EBP'};
N_macro     = numel(MACRO_NAMES);
I0_VARS     = {'RAI_BEX','UNC_BEX','EBP'};
unit_root_vec = ones(N_macro,1);
for n=1:N_macro, if any(strcmp(MACRO_NAMES{n},I0_VARS)), unit_root_vec(n)=0; end, end
T_raw_mac = readtable(MACRO_CSV);
Y_macro   = zeros(height(T_raw_mac), N_macro);
for n=1:N_macro, Y_macro(:,n) = T_raw_mac.(MACRO_NAMES{n}); end
dates_macro = datetime(T_raw_mac.Date,'InputFormat','yyyy-MM-dd'); dates_macro = dates_macro(:);
valid = all(~isnan(Y_macro),2);
if ~all(valid), a=find(valid,1,'first'); b=find(valid,1,'last'); Y_macro=Y_macro(a:b,:); dates_macro=dates_macro(a:b); end
T_mac = size(Y_macro,1);
fprintf('  Macro: [%d x %d]  %s — %s  (EBP/UNC_BEX/RAI_BEX are CONTROLS)\n', T_mac, N_macro, ...
        datestr(dates_macro(1),'yyyy-mm'), datestr(dates_macro(end),'yyyy-mm'));

%% =========================================================================
%  STEP 3 — READ EXTERNAL INSTRUMENT (from Stage A) + posterior draws
% =========================================================================
fprintf('\n%s\n STEP 3 — INSTRUMENT (signed monthly mean) + draws\n%s\n', repmat('=',1,70), repmat('=',1,70));
ym_daily    = year(dates_vec)*100 + month(dates_vec);
months_list = unique(ym_daily,'sorted');
T_monthly   = numel(months_list);
Z_monthly       = zeros(T_monthly, N_shocks);
dates_monthly   = NaT(T_monthly,1);
for mi=1:T_monthly
    idx_m = find(ym_daily==months_list(mi));
    dates_monthly(mi) = dates_vec(idx_m(end));
    for ki=1:N_shocks
        Z_monthly(mi,ki) = mean(V_ref(idx_m, shock_cols(ki)));
    end
end
% Prefer the instrument written by Stage A, so the two stages cannot diverge.
% Any failure to reuse it is reported explicitly — never silently ignored.
monthly_file = sprintf('shocks_monthly_%s.mat', SRC_BASE);
if exist(monthly_file,'file')==2
    Pm = load(monthly_file,'MEAN','ym_monthly');
    ok_reuse = isfield(Pm,'MEAN') && isfield(Pm,'ym_monthly');
    if ok_reuse
        [tf, locb] = ismember(months_list, Pm.ym_monthly);
        ok_reuse = all(tf) && size(Pm.MEAN,2)>=max(shock_cols);
    end
    if ok_reuse
        for ki=1:N_shocks, Z_monthly(:,ki)=Pm.MEAN(locb,shock_cols(ki)); end
        fprintf('  [REUSE] instrument read from %s (Stage A)\n', monthly_file);
    else
        fprintf(2,'  [!] %s exists but does not cover this month/shock set —\n', monthly_file);
        fprintf(2,'      falling back to the on-the-fly instrument.  Re-run Stage A.\n');
    end
else
    fprintf('  [COMPUTE] %s not found -> instrument computed on the fly.\n', monthly_file);
end

%% =========================================================================
%  STEP 4 — SAMPLE ALIGNMENT
% =========================================================================
ym_proxy = year(dates_monthly)*100 + month(dates_monthly);
ym_macro = year(dates_macro)*100   + month(dates_macro);
[~, ia, ib] = intersect(ym_proxy, ym_macro);
Z_aligned = Z_monthly(ia,:);  Y_aligned = Y_macro(ib,:);  dates_common = dates_macro(ib);
T_common  = size(Y_aligned,1);
fprintf('\n STEP 4 — ALIGNMENT: common months = %d  (%s — %s)\n', T_common, ...
        datestr(dates_common(1),'yyyy-mm'), datestr(dates_common(end),'yyyy-mm'));
% --- SAMPLE END (shutdown 2025) -----------------------------------------
if ~isempty(SAMPLE_END_YM)
    ym_common = year(dates_common)*100 + month(dates_common);
    keep_s    = ym_common <= SAMPLE_END_YM;
    n_cut     = sum(~keep_s);
    if n_cut > 0
        Z_aligned=Z_aligned(keep_s,:); Y_aligned=Y_aligned(keep_s,:);
        dates_common=dates_common(keep_s); T_common=size(Y_aligned,1);
        fprintf('  Sample end enforced at %d: %d month(s) dropped -> %d months (%s — %s)\n', ...
                SAMPLE_END_YM, n_cut, T_common, ...
                datestr(dates_common(1),'yyyy-mm'), datestr(dates_common(end),'yyyy-mm'));
        fprintf('  (2025-10 has no CPI and no UNRATE: the federal shutdown stopped their production.)\n');
    end
end
% --- CALENDAR CONTIGUITY GUARD ------------------------------------------
%  rfvar3/bvar_ (and X_mac below) build the lags BY ROW POSITION, not by date.
%  A hole in the monthly calendar therefore silently pairs y_t with the wrong
%  y_{t-1..t-p} for the p observations that straddle the hole.  Detect it.
mnum   = year(dates_common)*12 + month(dates_common);
gap_at = find(diff(mnum) ~= 1);
if ~isempty(gap_at)
    fprintf(2,'  [!!] CALENDAR GAP in the common sample — lags are built by row position,\n');
    for gg = gap_at(:)'
        fprintf(2,'       missing month(s) between %s and %s (%d-month step)\n', ...
                datestr(dates_common(gg),'yyyy-mm'), datestr(dates_common(gg+1),'yyyy-mm'), mnum(gg+1)-mnum(gg));
    end
    fprintf(2,'       -> the p observations after each hole use MISALIGNED lags.\n');
    fprintf(2,'       Fix the macro CSV or truncate the sample before the gap.\n');
else
    fprintf('  Calendar check: contiguous monthly grid, no missing months.\n');
end

%% =========================================================================
%  STEP 5 — MACRO-ONLY BVAR (GLP-Minnesota) — dynamics + instrument residuals
% =========================================================================
fprintf('\n%s\n STEP 5 — MACRO-ONLY BVAR (GLP)\n%s\n', repmat('=',1,70), repmat('=',1,70));
%  LAG ORDER: FIXED a priori (see CONFIG).  No marginal-likelihood selection:
%  under the Minnesota prior the lag decay (1/s^2) already shrinks the far lags,
%  so p is a prior choice, not a model-selection problem.  Selecting p by ML and
%  THEN optimising the prior hyper-parameters at that p is also inconsistent —
%  the two searches condition on each other.
p_macro = P_MACRO_FIXED;
fprintf('  Lag order FIXED a priori: p_macro = %d (no ML selection; Minnesota decay handles far lags)\n', p_macro);
fprintf('  Parameters per equation: macro-only %d, augmented %d   |   T = %d\n', ...
        N_macro*p_macro+1, (N_macro+1)*p_macro+1, T_common);
if (N_macro+1)*p_macro+1 > 0.5*(T_common-p_macro)
    fprintf(2,'  [!] the augmented VAR has %d coefficients per equation for %d usable observations:\n', ...
            (N_macro+1)*p_macro+1, T_common-p_macro);
    fprintf(2,'      the Minnesota posterior is fine, but the OLS-based bootstrap of Step 7b is not shrunk.\n');
end
TAU_GRID = [0.1,0.5,1,2,5,10,20,50]; lnML_tau = zeros(numel(TAU_GRID),1);
opt_tau = struct('K',200); opt_tau.minn_prior_decay=1; opt_tau.unit_root_=unit_root_vec;
for ti=1:numel(TAU_GRID), opt_tau.minn_prior_tau=TAU_GRID(ti); lnML_tau(ti)=bvar_(Y_aligned,p_macro,opt_tau).logmlike; end
[~,bti]=max(lnML_tau); tau_star=TAU_GRID(bti);
opt_glp = struct('K',2000,'hor',IRF_HOR_MAC); opt_glp.priors=struct('name','Minnesota');
opt_glp.max_minn_hyper=1; opt_glp.max_compute=2; opt_glp.index_est=[1,3,4];
opt_glp.minn_prior_tau=tau_star; opt_glp.minn_prior_decay=1;
opt_glp.minn_prior_lambda=5; opt_glp.minn_prior_mu=2; opt_glp.minn_prior_omega=2;
opt_glp.unit_root_=unit_root_vec; opt_glp.lb=[0.05,0.5,0.5]; opt_glp.ub=[100,500,500];
bvar_macro = bvar_(Y_aligned, p_macro, opt_glp);
% Recover the OPTIMISED Minnesota hyper-parameters so the augmented VAR of
% Step 7 is estimated under the SAME prior as the macro-only VAR validated
% here.  (Without this, lambda and mu silently revert to their hard-coded
% defaults in the augmented run and the two dynamics are not comparable.)
lam_star = 5; mu_star = 2; om_star = 2;
if isfield(bvar_macro,'prior') && isfield(bvar_macro.prior,'minn_prior_lambda')
    lam_star = bvar_macro.prior.minn_prior_lambda;
    mu_star  = bvar_macro.prior.minn_prior_mu;
    om_star  = bvar_macro.prior.minn_prior_omega;
    if isfield(bvar_macro.prior,'minn_prior_tau'), tau_star = bvar_macro.prior.minn_prior_tau; end
    fprintf('  Optimised Minnesota hyper-parameters recovered: tau=%.3g lambda=%.3g mu=%.3g omega=%.3g\n', ...
            tau_star, lam_star, mu_star, om_star);
else
    fprintf(2,'  [!] optimised hyper-parameters not exposed by bvar_ — augmented VAR uses tau* with default lambda/mu.\n');
end
Phi_pm = mean(bvar_macro.Phi_draws,3);
k_mac  = N_macro*p_macro+1;
X_mac  = zeros(T_common-p_macro,k_mac);
for l=1:p_macro, X_mac(:,(l-1)*N_macro+1:l*N_macro)=Y_aligned(p_macro-l+1:end-l,:); end
X_mac(:,end)=1;
U_mac  = Y_aligned(p_macro+1:end,:) - X_mac*Phi_pm;  T_res=size(U_mac,1);
Z_t    = Z_aligned(p_macro+1:end,:);
n_comp = N_macro*p_macro;
fprintf('  p_macro=%d  tau*=%.3g  k_mac=%d  T_res=%d  lnML=%.3f\n', p_macro, tau_star, k_mac, T_res, bvar_macro.logmlike);

%% =========================================================================
%  STEP 6 — INSTRUMENT RELEVANCE DIAGNOSTIC (first-stage F)
%    Not an identification step: reports how strong/weak the instrument is, so
%    the internal-instrument IRFs are read in context.  NO external IRF is
%    computed here — the external-vs-internal COMPARISON lives in
%    Script_2_ProxySVAR.m (which now carries the internal-instrument overlay).
% =========================================================================
fprintf('\n%s\n STEP 6 — INSTRUMENT RELEVANCE DIAGNOSTIC (first-stage F)\n%s\n', repmat('=',1,70), repmat('=',1,70));
sd_Z=zeros(1,N_shocks); Fmax=zeros(1,N_shocks); Fmax_var=cell(1,N_shocks); F_indpro=zeros(1,N_shocks);
i_indpro = find(strcmp(MACRO_NAMES,'INDPRO'),1);
for ki=1:N_shocks
    Zt = Z_t(:,ki)-mean(Z_t(:,ki)); sd_Z(ki)=std(Zt);
    Fv = zeros(N_macro,1);
    for n=1:N_macro, Fv(n)=local_effF(Zt,U_mac(:,n),T_res); end
    [Fmax(ki),im]=max(Fv); Fmax_var{ki}=MACRO_NAMES{im}; F_indpro(ki)=Fv(i_indpro);
end
fprintf('  %-6s %10s %16s %12s\n','shock','sd(Z)','strongest 1st-F','F@INDPRO');
for ki=1:N_shocks
    tagw = ''; if Fmax(ki)<10, tagw=' [WEAK]'; end
    fprintf('  %-6s %10.4f  %7.1f(%-7s) %12.2f%s\n', shock_tag{ki}, sd_Z(ki), Fmax(ki), Fmax_var{ki}, F_indpro(ki), tagw);
end

%% ---- NORMALISATION REFERENCE, RESOLVED ONCE -----------------------------
%  The reference variable must be known BEFORE the draws are generated, because
%  the unit-effect normalisation is applied INSIDE every draw (Steps 7 and 7b)
%  rather than to the finished bands.  Stock & Watson (2018, Section 4.2) state
%  the requirement explicitly: computing a band under one normalisation and then
%  rescaling it by a factor read off the point estimate is incorrect, because the
%  reported response is a RATIO and the band must price the sampling uncertainty
%  of its denominator too.  Their wording is that the normalisation belongs
%  "inside" the bootstrap, not "outside".
ir_norm       = zeros(1,N_shocks);
norm_ref_name = repmat({'(1 s.d.)'},1,N_shocks);
for ki=1:N_shocks
    if strcmpi(NORM_MODE,'refvar')
        if ischar(NORM_REF_VAR) || isstring(NORM_REF_VAR)
            if strlength(string(NORM_REF_VAR))>0, rname=char(NORM_REF_VAR); else, rname=Fmax_var{ki}; end
        elseif iscell(NORM_REF_VAR) && numel(NORM_REF_VAR)>=ki && ~isempty(NORM_REF_VAR{ki})
            rname=NORM_REF_VAR{ki};
        else
            rname=Fmax_var{ki};
        end
        ir = find(strcmp(MACRO_NAMES,rname),1);
        if isempty(ir)
            error('NORM_REF_VAR: "%s" is not in MACRO_NAMES.', rname);
        end
        ir_norm(ki) = ir;  norm_ref_name{ki} = rname;
    end
end
%  Per-draw diagnostics filled in Steps 7 and 7b: the spread of the scale factor
%  across draws IS the uncertainty the old "outside" rescaling threw away, and
%  the share of draws in which the reference variable moves the OTHER way is the
%  honest measure of how well the denominator is pinned down.
norm_c_bay  = nan(N_shocks,3);  norm_flip_bay = nan(1,N_shocks);
norm_c_mbb  = nan(N_shocks,3);  norm_flip_mbb = nan(1,N_shocks);
raw_imp_bay = nan(1,N_shocks);  raw_imp_mbb   = nan(1,N_shocks);
if strcmpi(NORM_MODE,'refvar')
    fprintf('\n  Normalisation reference resolved BEFORE the draws: %s\n', strjoin(norm_ref_name,', '));
    fprintf('  (unit-effect normalisation applied inside every draw — Stock-Watson 2018)\n');
end

%% =========================================================================
%  STEP 6b — SHOCK PREDICTABILITY TEST  (exogeneity / no-anticipation)
%
%  A structural shock must be UNPREDICTABLE from information already available
%  at t-1.  Testing that against the variables of the DAILY VAR would be vacuous
%  (the shock is orthogonal to them by construction), so the informative test
%  uses information the daily VAR does NOT contain: the monthly macro block.
%
%  IMPLEMENTATION.  The instrument is regressed on PRED_LAGS lags of a small set
%  of NAMED macro variables, and joint significance is tested.  This is the
%  standard validity check applied to identified/narrative shock series — see
%  Ramey (2011, QJE; 2016, Handbook of Macroeconomics ch.2, sec. 2.3), and the
%  analogous predictability checks run on high-frequency surprises by Gertler &
%  Karadi (2015).
%
%  WHY NAMED VARIABLES AND NOT PRINCIPAL COMPONENTS.  Factor-based versions of
%  this test (Forni & Gambetti 2014) compress a LARGE panel (~130 FRED-MD
%  series) into a few factors because their lags cannot all fit on the
%  right-hand side.  Here the information set is 10 variables, so no compression
%  is needed, and named regressors buy something the factors destroy: if the
%  test rejects, the individual t-statistics say WHICH information predicts the
%  shock.  That distinction matters for the reading:
%    - predictable from lagged EBP  -> persistence of credit conditions
%                                      themselves (benign; the shock inherits
%                                      the microstructure smoothness of the
%                                      underlying corporate-bond series);
%    - predictable from lagged real activity -> the "financial" shock is partly
%                                      ENDOGENOUS to the business cycle (serious:
%                                      it constrains the structural label).
%
%  STATIONARITY.  Each regressor is stationarised individually: I(1) macro in
%  first differences, I(0) financial variables in levels.  Regressing a
%  stationary instrument on near-integrated regressors would give a
%  non-standard F distribution and an R^2 inflated by shared persistence.
%  NOTE this concerns the TEST ONLY: the VAR itself is estimated in LEVELS with
%  a unit-root Minnesota prior (Sims-Stock-Watson 1990), untouched by this block.
%
%  INFERENCE.  Heteroskedasticity-robust (HC1) Wald test: the instrument
%  inherits conditional heteroskedasticity from the daily shocks, so the
%  homoskedastic F would over-reject.
%
%  CONSEQUENCE OF A REJECTION.  Under non-invertibility the EXTERNAL proxy-SVAR
%  is inconsistent, while the INTERNAL instrument used here (Cholesky, Z ordered
%  first) remains consistent (Plagborg-Moller & Wolf 2021) because it identifies
%  the INNOVATION of Z — the component orthogonal to p lags of the whole system.
%  A rejection therefore argues FOR the design already adopted, and restricts the
%  LABEL that can be attached to the shock, not the validity of the estimator.
% =========================================================================
fprintf('\n%s\n STEP 6b — SHOCK PREDICTABILITY TEST (Ramey-type, HC1-robust)\n%s\n', repmat('=',1,70), repmat('=',1,70));
PRED_VARS = {'INDPRO','UNRATE','PCEPI','EBP'};
PRED_DIFF = [    1   ,    1   ,   1   ,   0  ];   % 1 = first difference (I(1)), 0 = level (I(0))
PRED_LAGS = 2;
K_pred = numel(PRED_VARS);
Wp = nan(T_common, K_pred);
for j=1:K_pred
    n_j = find(strcmp(MACRO_NAMES,PRED_VARS{j}),1);
    if isempty(n_j), error('PRED_VARS: "%s" is not in MACRO_NAMES.', PRED_VARS{j}); end
    if PRED_DIFF(j), Wp(2:end,j) = diff(Y_aligned(:,n_j)); else, Wp(:,j) = Y_aligned(:,n_j); end
end
Wp = (Wp - mean(Wp,1,'omitnan')) ./ max(std(Wp,0,1,'omitnan'),1e-12);   % standardised -> comparable t
Xp = zeros(T_res, K_pred*PRED_LAGS);  lab_p = cell(1, K_pred*PRED_LAGS);
for l=1:PRED_LAGS
    Xp(:,(l-1)*K_pred+1:l*K_pred) = Wp(p_macro+1-l : T_common-l, :);
    for j=1:K_pred
        lab_p{(l-1)*K_pred+j} = sprintf('%s(-%d)', PRED_VARS{j}, l);
    end
end
if any(~isfinite(Xp(:))), error('Predictability regressors contain non-finite values.'); end
%  ALIGNMENT ASSERTION — the whole test is meaningless if a regressor is dated
%  wrongly, and an off-by-one would silently turn a lag into a contemporaneous
%  (or leading) term, manufacturing predictability out of nothing.  Row i of
%  z_t corresponds to calendar date dates_common(p_macro+i); the lag-l block of
%  that row must hold the variable dated EXACTLY l calendar months earlier.
%  Both facts are checked against the date vector, not inferred from the index
%  arithmetic that produced them.
mnum_c = year(dates_common)*12 + month(dates_common);
for l = 1:PRED_LAGS
    for i = unique([1, max(1,round(T_res/2)), T_res])
        t_idx = p_macro + i;
        if max(abs( Wp(t_idx-l,:) - Xp(i,(l-1)*K_pred+1:l*K_pred) )) > 0
            error('Predictability test: lag-%d block is misaligned at row %d.', l, i);
        end
        if mnum_c(t_idx) - mnum_c(t_idx-l) ~= l
            error('Predictability test: row %d lag %d spans %d calendar months, expected %d.', ...
                  i, l, mnum_c(t_idx)-mnum_c(t_idx-l), l);
        end
    end
end
fprintf('  Alignment assertion PASSED: every lag-l regressor is dated exactly l months before z_t.\n');
pred_R2=nan(1,N_shocks); pred_F=nan(1,N_shocks); pred_p=nan(1,N_shocks);
pred_t =nan(N_shocks, K_pred*PRED_LAGS);
q_pred = size(Xp,2);
fprintf('  regressors: %s  (%d lags, standardised; I(1) differenced, I(0) in level)\n', ...
        strjoin(PRED_VARS,', '), PRED_LAGS);
fprintf('  %-6s %9s %9s %10s   %s\n','shock','R2','F(HC1)','p-value','reading');
for ki=1:N_shocks
    yp  = Z_t(:,ki) - mean(Z_t(:,ki));
    Xr  = [ones(T_res,1), Xp];
    bp  = Xr\yp;  ep = yp - Xr*bp;
    kk_ = size(Xr,2);  dfe = T_res - kk_;
    XtXi = inv(Xr'*Xr);
    Vhc  = XtXi * ((Xr.*ep)'*(Xr.*ep)) * XtXi * (T_res/dfe);       % HC1
    idx  = 2:kk_;                                                   % drop the constant
    Wald = bp(idx)' * (Vhc(idx,idx) \ bp(idx));                     % ~ chi2(q)
    pred_F(ki)  = Wald / q_pred;                                    % report as an F
    pred_p(ki)  = 1 - chi2cdf(Wald, q_pred);
    pred_R2(ki) = 1 - (ep'*ep) / max(sum((yp-mean(yp)).^2),1e-18);
    pred_t(ki,:)= (bp(idx) ./ sqrt(max(diag(Vhc(idx,idx)),1e-18)))';
    fprintf('  %-6s %9.4f %9.2f %10.4f   %s\n', shock_tag{ki}, pred_R2(ki), pred_F(ki), pred_p(ki), ...
            tern_local(pred_p(ki)<0.05, 'PREDICTABLE [!]', 'not predictable -> exogeneity not rejected'));
end
fprintf('\n  Robust t-statistics by regressor (|t|>1.96 flagged) — WHICH information predicts:\n');
fprintf('  %-6s', 'shock'); for j=1:q_pred, fprintf(' %13s', lab_p{j}); end, fprintf('\n');
for ki=1:N_shocks
    fprintf('  %-6s', shock_tag{ki});
    for j=1:q_pred
        star = ' '; if abs(pred_t(ki,j))>1.96, star='*'; end
        fprintf(' %12.2f%s', pred_t(ki,j), star);
    end
    fprintf('\n');
end
fprintf('  Reading: rejection driven by EBP lags = persistence of credit conditions themselves;\n');
fprintf('  rejection driven by real-activity lags = the shock is partly endogenous to the cycle.\n');
fprintf('  Either way the INTERNAL instrument identifies the innovation of Z, orthogonal to p lags\n');
fprintf('  of the whole system, and stays consistent under non-invertibility (PMW 2021).\n');

%% =========================================================================
%  STEP 7 — INTERNAL INSTRUMENT (Cholesky, Z ordered first) — THE IDENTIFICATION
% =========================================================================
fprintf('\n%s\n STEP 7 — INTERNAL INSTRUMENT (Cholesky, ordered first)\n%s\n', repmat('=',1,70), repmat('=',1,70));
Na = N_macro+1; na_comp = Na*p_macro;
J_aug = [eye(Na), zeros(Na,Na*(p_macro-1))]; unit_root_a=[0; unit_root_vec];
opt_aug = struct('K',K_AUG); opt_aug.priors=struct('name','Minnesota');
opt_aug.max_minn_hyper=0; opt_aug.minn_prior_tau=tau_star; opt_aug.minn_prior_decay=1;
opt_aug.minn_prior_lambda=lam_star; opt_aug.minn_prior_mu=mu_star; opt_aug.minn_prior_omega=om_star;
opt_aug.unit_root_=unit_root_a;
IRF_int   =zeros(N_macro,IRF_HOR_MAC+1,N_shocks);
IRF_int_16=zeros(N_macro,IRF_HOR_MAC+1,N_shocks); IRF_int_84=zeros(N_macro,IRF_HOR_MAC+1,N_shocks);
IRF_int_05=zeros(N_macro,IRF_HOR_MAC+1,N_shocks); IRF_int_95=zeros(N_macro,IRF_HOR_MAC+1,N_shocks);
dum_Y = cell(1,N_shocks); dum_X = cell(1,N_shocks); Phi_bay = cell(1,N_shocks);
Phi_sd = cell(1,N_shocks); Phi_K = zeros(1,N_shocks);
%  FORECAST ERROR VARIANCE DECOMPOSITION (see STEP 7c for the reading).
%  Stored over the FULL augmented vector [Z; macro] (Na rows): row 1 is Z itself
%  and serves as an exact internal check (must be 1.000 at h=0 by Cholesky).
FEVD_med=zeros(Na,IRF_HOR_MAC+1,N_shocks);
FEVD_16 =zeros(Na,IRF_HOR_MAC+1,N_shocks); FEVD_84=zeros(Na,IRF_HOR_MAC+1,N_shocks);
for ki=1:N_shocks
    Y_aug=[Z_aligned(:,ki), Y_aligned];  bva=bvar_(Y_aug,p_macro,opt_aug);
    Phi_a=mean(bva.Phi_draws,3); Sig_a=mean(bva.Sigma_draws,3);
    % Keep the prior's DUMMY OBSERVATIONS: with a conjugate Normal-inverse-Wishart
    % prior written as dummies, the posterior mean of the coefficients is plain
    % least squares on [actual; dummy] data.  Step 7b re-uses them to re-estimate
    % the SHRUNK VAR inside every bootstrap replication at OLS cost, so that the
    % bootstrap band belongs to the same estimator as the reported point.
    Phi_bay{ki} = Phi_a;
    Phi_sd{ki}  = std(bva.Phi_draws,0,3);
    Phi_K(ki)   = size(bva.Phi_draws,3);
    if isfield(bva,'prior') && isfield(bva.prior,'YYdum') && isfield(bva.prior,'XXdum')
        dum_Y{ki} = bva.prior.YYdum;  dum_X{ki} = bva.prior.XXdum;
    end
    P=chol((Sig_a+Sig_a')/2,'lower'); bchol=P(:,1)/P(1,1);
    Aa=[Phi_a(1:na_comp,:)'; eye(Na*(p_macro-1)), zeros(Na*(p_macro-1),Na)]; Ah=eye(na_comp);
    %  IRF and FEVD share the same recursion.  NOTE THE TWO DIFFERENT SCALINGS:
    %    IRF  uses bchol = P(:,1)/P(1,1)  -> shock normalised to move Z by 1 unit
    %                                        (a reporting convention);
    %    FEVD uses P(:,1) itself          -> the shock with UNIT VARIANCE, which is
    %                                        the only scaling under which the
    %                                        squared contributions of the Na
    %                                        orthogonal shocks sum to the total
    %                                        forecast error variance.
    %  Using bchol in the numerator and the true MSE in the denominator would
    %  inflate every share by 1/P(1,1)^2 without any visible error.
    %  Since Sigma = P*P', Theta_s = Psi_s*P collects the impact of all Na
    %  orthogonal shocks, so  sum_j Theta_s(:,j).^2 = diag(Psi_s*Sigma*Psi_s')
    %  is the per-horizon MSE contribution: one matrix product does both.
    %  The share is a ratio of squares of the SAME object, hence invariant to the
    %  presentation normalisation applied later in Step 8.
    fe_num=zeros(Na,1); fe_den=zeros(Na,1);
    for h=0:IRF_HOR_MAC
        Psi_h_a = J_aug*Ah*J_aug';
        v = Psi_h_a*bchol;  IRF_int(:,h+1,ki)=v(2:end);
        Th     = Psi_h_a*P;
        fe_num = fe_num + Th(:,1).^2;
        fe_den = fe_den + sum(Th.^2,2);
        FEVD_med(:,h+1,ki) = fe_num ./ max(fe_den,1e-18);
        Ah=Ah*Aa;
    end
    Kd=size(bva.Phi_draws,3); IRFd=zeros(N_macro,IRF_HOR_MAC+1,Kd);
    FEVDd=zeros(Na,IRF_HOR_MAC+1,Kd);
    for d=1:Kd
        Phi_d=bva.Phi_draws(:,:,d); Sig_d=bva.Sigma_draws(:,:,d);
        Pd=chol((Sig_d+Sig_d')/2,'lower'); bcd=Pd(:,1)/Pd(1,1);
        Aad=[Phi_d(1:na_comp,:)'; eye(Na*(p_macro-1)), zeros(Na*(p_macro-1),Na)]; Ah=eye(na_comp);
        fn_d=zeros(Na,1); fd_d=zeros(Na,1);
        for h=0:IRF_HOR_MAC
            Psi_d = J_aug*Ah*J_aug';
            v = Psi_d*bcd;  IRFd(:,h+1,d)=v(2:end);
            Thd   = Psi_d*Pd;
            fn_d  = fn_d + Thd(:,1).^2;
            fd_d  = fd_d + sum(Thd.^2,2);
            FEVDd(:,h+1,d) = fn_d ./ max(fd_d,1e-18);
            Ah=Ah*Aad;
        end
    end
    %  The FEVD is a ratio of squares of the SAME object and is invariant to the
    %  normalisation, so its quantiles are taken on the UNNORMALISED draws.
    FEVD_16(:,:,ki)=quantile(FEVDd,0.16,3); FEVD_84(:,:,ki)=quantile(FEVDd,0.84,3);
    % ---- UNIT-EFFECT NORMALISATION, INSIDE THE DRAWS --------------------------
    %  Each draw is rescaled by ITS OWN impact on the reference variable before
    %  the quantiles are taken, so the reported band is a band on the ratio
    %  gamma*Theta_{i,h}/|Theta_{ref,0}| — the only magnitude the internal
    %  instrument identifies (Plagborg-Moller & Wolf 2021: measurement error in
    %  the instrument attenuates every response by the same factor, which cancels
    %  in the ratio).  Rescaling finished bands by a factor read off the point
    %  estimate would report a band on a different object.
    if ir_norm(ki)>0
        raw_imp_bay(ki) = IRF_int(ir_norm(ki),1,ki);
        bref_d = squeeze(IRFd(ir_norm(ki),1,:));
        c_d    = local_norm_fac(bref_d, NORM_MODE, NORM_REF_SIZE, sd_Z(ki));
        IRFd   = IRFd .* reshape(c_d,1,1,[]);
        IRF_int(:,:,ki) = IRF_int(:,:,ki) * ...
                          local_norm_fac(raw_imp_bay(ki), NORM_MODE, NORM_REF_SIZE, sd_Z(ki));
        fin_d = isfinite(bref_d) & isfinite(c_d);
        norm_c_bay(ki,:)   = quantile(c_d(fin_d),[0.16 0.50 0.84]);
        norm_flip_bay(ki)  = mean(sign(bref_d(fin_d))~=sign(raw_imp_bay(ki)));
    elseif strcmpi(NORM_MODE,'sd')
        IRFd = IRFd*sd_Z(ki);  IRF_int(:,:,ki) = IRF_int(:,:,ki)*sd_Z(ki);
    end
    IRF_int_16(:,:,ki)=quantile(IRFd,0.16,3); IRF_int_84(:,:,ki)=quantile(IRFd,0.84,3);
    IRF_int_05(:,:,ki)=quantile(IRFd,0.05,3); IRF_int_95(:,:,ki)=quantile(IRFd,0.95,3);
    fprintf('  %-6s augmented VAR (%d draws): Bayesian IRF + bands done  [impact on INDPRO = %+.4f]\n', ...
            shock_tag{ki}, Kd, IRF_int(i_indpro,1,ki));
    if ir_norm(ki)>0
        fprintf('        normalisation inside the draws: scale [16,50,84] = [%.2f, %.2f, %.2f], sign flips on %s = %.1f%%\n', ...
                norm_c_bay(ki,1), norm_c_bay(ki,2), norm_c_bay(ki,3), norm_ref_name{ki}, 100*norm_flip_bay(ki));
    end
end

%% =========================================================================
%  STEP 7c — FORECAST ERROR VARIANCE DECOMPOSITION
%
%  WHY IT IS REPORTED.  An impulse response says how a variable moves after the
%  shock; it does NOT say whether the shock matters.  A -1.6% response to a
%  disturbance that accounts for 2% of the forecast error variance of industrial
%  production is a true but economically marginal statement; the same response
%  from a shock accounting for 25% is a first-order result.  The FEVD is what
%  separates the two, and it is the standard companion table to IRFs in the SVAR
%  literature.
%
%  WHAT IS DECOMPOSED.  The augmented VAR [Z, macro] with a full Cholesky gives
%  Na ORTHOGONAL shocks, so the h-step forecast error variance splits exactly
%  among them and the shares sum to one.  The first Cholesky shock is the
%  identified one (Z ordered first, PMW 2021).  We report ITS share:
%
%      FEVD_n(h) = sum_{s=0..h} Theta_s(n,1)^2 / sum_{s=0..h} sum_j Theta_s(n,j)^2
%
%  with Theta_s = Psi_s * P.  This is a genuine variance decomposition, not a
%  bound: the remaining Na-1 shocks are orthogonal but carry no economic label,
%  which is why only the first share is reported.
%
%  INTERNAL CHECK.  Row 1 of the stored array is Z itself.  Because P is lower
%  triangular, at h=0 the first Cholesky shock explains Z entirely: the printed
%  value MUST be 1.000.  Any deviation means the recursion or the ordering is
%  wrong, so it is checked rather than assumed.
% =========================================================================
fprintf('\n%s\n STEP 7c — FEVD (share of forecast error variance due to the identified shock)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
h_fevd = [0 6 12 24 48]; h_fevd = h_fevd(h_fevd<=IRF_HOR_MAC);
fevd_check_ok = true;
for ki=1:N_shocks
    chk = FEVD_med(1,1,ki);
    if abs(chk-1) > 1e-8
        fevd_check_ok = false;
        fprintf(2,'  [!!] %s: FEVD of Z at h=0 = %.6f, expected exactly 1 -> recursion/ordering error.\n', ...
                shock_tag{ki}, chk);
    end
end
if fevd_check_ok
    fprintf('  Internal check PASSED: the identified shock explains 100%% of Z at h=0 for every shock.\n');
end
for ki=1:N_shocks
    fprintf('\n  %-4s  [sign anchor: %s loads %s on impact]   share of FEV, %% [16,84 posterior]\n', ...
            shock_tag{ki}, dom_daily{ki}, tern_local(dom_sign(ki)>0,'POSITIVELY','NEGATIVELY'));
    fprintf('  %-10s', 'variable');
    for h=h_fevd, fprintf(' %20s', sprintf('h=%d',h)); end
    fprintf('\n');
    for n=1:N_macro
        fprintf('  %-10s', MACRO_NAMES{n});
        for h=h_fevd
            fprintf('  %6.1f[%4.1f,%5.1f]', 100*FEVD_med(n+1,h+1,ki), ...
                    100*FEVD_16(n+1,h+1,ki), 100*FEVD_84(n+1,h+1,ki));
        end
        fprintf('\n');
    end
end
fprintf('\n  Reading: the share is invariant to the IRF normalisation of Step 8 (ratio of\n');
fprintf('  squares of the same object), so it can be compared across shocks directly.\n');

%% =========================================================================
%  STEP 7b — MOVING-BLOCK BOOTSTRAP  (PRIMARY inference)
%    The Gaussian/inverse-Wishart posterior of Step 7 prices no conditional
%    heteroskedasticity.  The monthly instrument inherits the volatility
%    clustering of the daily shocks (its squares are strongly autocorrelated)
%    and its informative content is concentrated in a few crisis months, so
%    the Gaussian bands can be far too narrow exactly for the fat-tailed
%    shocks.  The residual-based moving-block bootstrap below is valid under
%    conditional heteroskedasticity and fat tails and is the band reported.
%    Bruggemann, Jentsch & Trenkler (2016, J.Econometrics); Jentsch & Lunsford
%    (2019, AER) — block length 5.03*T^(1/4), blocks recentred.
% =========================================================================
IRF_mbb_16=nan(N_macro,IRF_HOR_MAC+1,N_shocks); IRF_mbb_84=nan(N_macro,IRF_HOR_MAC+1,N_shocks);
IRF_mbb_05=nan(N_macro,IRF_HOR_MAC+1,N_shocks); IRF_mbb_95=nan(N_macro,IRF_HOR_MAC+1,N_shocks);
IRF_ols_all=nan(N_macro,IRF_HOR_MAC+1,N_shocks);
mbb_ok = zeros(1,N_shocks);  mbb_delta = nan(1,N_shocks);  mbb_shrunk = false(1,N_shocks);
if DO_MBB
fprintf('\n%s\n STEP 7b — MOVING-BLOCK BOOTSTRAP (primary inference)\n%s\n', repmat('=',1,70), repmat('=',1,70));
Ta = T_common;  Tu_b = Ta - p_macro;
L_blk = MBB_BLOCK; if isempty(L_blk), L_blk = max(2, round(5.03*Tu_b^(1/4))); end
n_blk = ceil(Tu_b/L_blk);
fprintf('  T=%d  block length L=%d  (%d blocks)  B=%d replications\n', Tu_b, L_blk, n_blk, MBB_B);
for ki=1:N_shocks
    Y_aug = [Z_aligned(:,ki), Y_aligned];
    Xb = zeros(Tu_b, Na*p_macro+1);
    for l=1:p_macro, Xb(:,(l-1)*Na+1:l*Na) = Y_aug(p_macro-l+1:end-l,:); end
    Xb(:,end)=1;
    Yb0   = Y_aug(p_macro+1:end,:);
    % ---- WHICH ESTIMATOR THE BOOTSTRAP REPLICATES -------------------------
    %  'shrunk' : least squares on [actual; dummy] data.  With a conjugate NIW
    %             prior written as dummy observations this IS the Minnesota
    %             posterior mean, at the cost of one slightly larger linear
    %             solve — so point estimate and bootstrap band become the SAME
    %             estimator, and the band stays valid under conditional
    %             heteroskedasticity.  This is what removes the p=15 objection:
    %             every replication is shrunk, so nothing is ever estimated with
    %             166 unpenalised coefficients on 414 observations.
    %  'ols'    : the unshrunk estimator (comparison only).
    est_shrunk = strcmpi(MBB_ESTIMATOR,'shrunk') && ~isempty(dum_X{ki});
    if est_shrunk
        Xd = dum_X{ki}; Yd = dum_Y{ki};
        Bfun = @(X,Y) [X; Xd] \ [Y; Yd];
    else
        if strcmpi(MBB_ESTIMATOR,'shrunk')
            fprintf(2,'    [!] %s: bvar_ did not expose the prior dummies -> OLS bootstrap.\n', shock_tag{ki});
        end
        Bfun = @(X,Y) X \ Y;
    end
    Bols  = Bfun(Xb,Yb0);
    % ---- SELF-CHECK: the replicated estimator must reproduce the reported point
    %  If the dummy observations did not match what bvar_ used internally, the
    %  bootstrap would be centred elsewhere and the band would be silently wrong.
    %  NOTE the benchmark mean(Phi_draws,3) is itself a MONTE CARLO average over
    %  K draws, so it does NOT equal the exact posterior mean: it carries a
    %  simulation error of about sd/sqrt(K) per coefficient, and a
    %  machine-precision tolerance would reject a CORRECT replication.  The check
    %  is therefore done in units of POSTERIOR SD; the deviation is also printed
    %  in Monte Carlo standard errors, where a max over ~1800 coefficients is
    %  expected to be around 3.
    if est_shrunk
        sd_k  = max(Phi_sd{ki}, 1e-12);
        d_sd  = max(abs(Bols - Phi_bay{ki}) ./ sd_k, [], 'all');
        d_mc  = max(abs(Bols - Phi_bay{ki}) ./ (sd_k/sqrt(Phi_K(ki))), [], 'all');
        if d_sd < 0.25
            fprintf('    %-4s shrunk-estimator check: PASS  (max dev = %.3f posterior sd = %.1f MC se)\n', ...
                    shock_tag{ki}, d_sd, d_mc);
        else
            fprintf(2,'    [!] %s shrunk-estimator check FAILED (max dev = %.3f posterior sd) -> OLS fallback.\n', ...
                    shock_tag{ki}, d_sd);
            est_shrunk = false; Bfun = @(X,Y) X \ Y; Bols = Bfun(Xb,Yb0);
        end
    end
    mbb_shrunk(ki) = est_shrunk;
    Uols  = Yb0 - Xb*Bols;
    % ---- KILIAN (1998) BIAS CORRECTION (off by default, see CONFIG) -------
    bias_hat = zeros(size(Bols)); delta = 0;
    if DO_KILIAN
        Bsum = zeros(size(Bols)); nb1 = 0;
        for b1 = 1:MBB_B1
            st = randi(Tu_b-L_blk+1, n_blk, 1);
            ix = reshape((st + (0:L_blk-1))', [], 1); ix = ix(1:Tu_b);
            Ub = Uols(ix,:);  Ub = Ub - mean(Ub,1);
            Ys = zeros(Ta, Na); Ys(1:p_macro,:) = Y_aug(1:p_macro,:);
            for tt=p_macro+1:Ta
                xr = [reshape(flipud(Ys(tt-p_macro:tt-1,:))', 1, Na*p_macro), 1];
                Ys(tt,:) = xr*Bols + Ub(tt-p_macro,:);
            end
            Xs = zeros(Tu_b, Na*p_macro+1);
            for l=1:p_macro, Xs(:,(l-1)*Na+1:l*Na) = Ys(p_macro-l+1:end-l,:); end
            Xs(:,end)=1;
            Bsum = Bsum + Bfun(Xs,Ys(p_macro+1:end,:)); nb1 = nb1 + 1;
        end
        bias_hat = Bsum/nb1 - Bols;
        delta = 1; Bbc = Bols - delta*bias_hat;
        while delta > 0
            Ac = [Bbc(1:na_comp,:)'; eye(Na*(p_macro-1)), zeros(Na*(p_macro-1),Na)];
            if max(abs(eig(Ac))) < 1.0, break; end
            delta = delta - 0.02; Bbc = Bols - max(delta,0)*bias_hat;
        end
        fprintf('    %-4s Kilian bias correction: delta=%.2f\n', shock_tag{ki}, max(delta,0));
        Bols = Bbc;  Uols = Yb0 - Xb*Bols;
    end
    mbb_delta(ki) = max(delta,0);
    % Point IRF at the estimator the bootstrap replicates: with 'shrunk' this is
    % (numerically) the Minnesota point of Step 7, so the percentile band below
    % is a heteroskedasticity-robust band around the LINE THAT IS PLOTTED.
    Sols = (Uols'*Uols)/(Tu_b - size(Xb,2));
    Aols = [Bols(1:na_comp,:)'; eye(Na*(p_macro-1)), zeros(Na*(p_macro-1),Na)];
    Pol  = chol((Sols+Sols')/2,'lower'); bco = Pol(:,1)/Pol(1,1);
    IRF_ols = zeros(N_macro, IRF_HOR_MAC+1); Ah = eye(na_comp);
    for h=0:IRF_HOR_MAC, v = J_aug*Ah*J_aug'*bco; IRF_ols(:,h+1)=v(2:end); Ah = Ah*Aols; end
    IRFb  = nan(N_macro, IRF_HOR_MAC+1, MBB_B);
    nfail = 0;
    for b=1:MBB_B
        % --- draw and recentre overlapping blocks of residuals --------------
        starts = randi(Tu_b-L_blk+1, n_blk, 1);
        idxb   = reshape((starts + (0:L_blk-1))', [], 1);
        idxb   = idxb(1:Tu_b);
        Ub     = Uols(idxb,:);
        Ub     = Ub - mean(Ub,1);                 % recentring (BJT 2016)
        % --- recursive DGP under the OLS point estimate ---------------------
        Ys = zeros(Ta, Na);  Ys(1:p_macro,:) = Y_aug(1:p_macro,:);
        for t=p_macro+1:Ta
            xr = [reshape(flipud(Ys(t-p_macro:t-1,:))', 1, Na*p_macro), 1];
            Ys(t,:) = xr*Bols + Ub(t-p_macro,:);
        end
        % --- re-estimate and recompute the Cholesky-first IRF ---------------
        Xs = zeros(Tu_b, Na*p_macro+1);
        for l=1:p_macro, Xs(:,(l-1)*Na+1:l*Na) = Ys(p_macro-l+1:end-l,:); end
        Xs(:,end)=1;
        Ys0 = Ys(p_macro+1:end,:);
        Bs  = Bfun(Xs,Ys0);                       % SAME estimator as the point
        if DO_KILIAN, Bs = Bs - max(delta,0)*bias_hat; end
        Us = Ys0 - Xs*Bs;  Ss = (Us'*Us)/(Tu_b - size(Xs,2));
        As  = [Bs(1:na_comp,:)'; eye(Na*(p_macro-1)), zeros(Na*(p_macro-1),Na)];
        % The VAR is in LEVELS with I(1) variables, so companion roots AT unity
        % are legitimate — requiring max|eig| <= 1 would discard a large share of
        % valid draws and truncate the persistent tail, biasing the band towards
        % transient responses.  Only genuinely divergent draws are dropped.
        if max(abs(eig(As))) > 1.02, nfail = nfail + 1; continue; end
        [Ps, flag] = chol((Ss+Ss')/2,'lower');
        if flag ~= 0, nfail = nfail + 1; continue; end
        bcs = Ps(:,1)/Ps(1,1);  Ah = eye(na_comp);
        for h=0:IRF_HOR_MAC
            v = J_aug*Ah*J_aug'*bcs;  IRFb(:,h+1,b) = v(2:end);  Ah = Ah*As;
        end
        if ~all(isfinite(IRFb(:,:,b)),'all'), IRFb(:,:,b)=NaN; nfail=nfail+1; end
    end
    % ---- UNIT-EFFECT NORMALISATION, INSIDE THE BOOTSTRAP ---------------------
    %  This is the step Stock & Watson (2018) require: the normalisation is
    %  applied to every REPLICATION, so the percentile band below is a band on
    %  the normalised estimator rather than a band on the per-unit-Z estimator
    %  multiplied by a constant.  Draws whose impact on the reference variable is
    %  numerically zero produce a non-finite factor and are dropped by the
    %  finiteness filter on the next line, exactly as explosive draws are.
    if ir_norm(ki)>0
        raw_imp_mbb(ki) = IRF_ols(ir_norm(ki),1);
        bref_b  = squeeze(IRFb(ir_norm(ki),1,:));
        c_b     = local_norm_fac(bref_b, NORM_MODE, NORM_REF_SIZE, sd_Z(ki));
        IRFb    = IRFb .* reshape(c_b,1,1,[]);
        IRF_ols = IRF_ols * local_norm_fac(raw_imp_mbb(ki), NORM_MODE, NORM_REF_SIZE, sd_Z(ki));
        fin = isfinite(bref_b) & isfinite(c_b);
        norm_c_mbb(ki,:)  = quantile(c_b(fin),[0.16 0.50 0.84]);
        norm_flip_mbb(ki) = mean(sign(bref_b(fin))~=sign(raw_imp_mbb(ki)));
    elseif strcmpi(NORM_MODE,'sd')
        IRFb = IRFb*sd_Z(ki);  IRF_ols = IRF_ols*sd_Z(ki);
    end
    keep = squeeze(all(all(isfinite(IRFb),1),2));  mbb_ok(ki) = sum(keep);
    % PLAIN percentile band around the OLS estimator.  Point estimate and band
    % must be the SAME estimator: the Minnesota prior shrinks towards a random
    % walk and yields systematically more persistent responses than the OLS DGP
    % that generates the bootstrap samples, so pairing the BVAR point with an
    % OLS-based band would put the line outside its own interval.  The two
    % estimators are therefore reported as two internally consistent pairs:
    %   frequentist : IRF_ols  + moving-block bootstrap percentiles
    %   Bayesian    : IRF_int  + Gaussian posterior quantiles
    IRF_ols_all(:,:,ki) = IRF_ols;
    % Percentile interval on the BIAS-CORRECTED estimator.  Point estimate and
    % replications now carry the same Kilian (1998) correction, so the interval
    % is centred on the line that is actually plotted.
    IRF_mbb_16(:,:,ki)=quantile(IRFb(:,:,keep),0.16,3); IRF_mbb_84(:,:,ki)=quantile(IRFb(:,:,keep),0.84,3);
    IRF_mbb_05(:,:,ki)=quantile(IRFb(:,:,keep),0.05,3); IRF_mbb_95(:,:,ki)=quantile(IRFb(:,:,keep),0.95,3);
    w_bay = IRF_int_84(i_indpro,1,ki)-IRF_int_16(i_indpro,1,ki);
    w_mbb = IRF_mbb_84(i_indpro,1,ki)-IRF_mbb_16(i_indpro,1,ki);
    fprintf('  %-6s %5d/%d usable draws (%d discarded)  |  68%% width at INDPRO impact: Bayes %.4f vs MBB %.4f  (x%.1f)\n', ...
            shock_tag{ki}, mbb_ok(ki), MBB_B, nfail, w_bay, w_mbb, w_mbb/max(w_bay,1e-12));
end
% --- how much the Gaussian bands understate uncertainty, by shock ---------
fprintf('\n  Band-width ratio MBB/Bayesian (68%%), median over variables and horizons:\n');
for ki=1:N_shocks
    rw = (IRF_mbb_84(:,:,ki)-IRF_mbb_16(:,:,ki)) ./ max(IRF_int_84(:,:,ki)-IRF_int_16(:,:,ki),1e-12);
    fprintf('    %-4s  impact h=0: x%.2f   all horizons: x%.2f\n', ...
            shock_tag{ki}, median(rw(:,1)), median(rw(:)));
end
end

%% =========================================================================
%  STEP 8 — INTERNAL-INSTRUMENT IRF TABLE + FIGURES
% =========================================================================
fprintf('\n%s\n STEP 8 — INTERNAL-INSTRUMENT (Cholesky) IRF  [normalisation: %s]\n%s\n', repmat('=',1,70), NORM_MODE, repmat('=',1,70));
% PRIMARY = frequentist pair (OLS point + moving-block bootstrap band): the
% only pair valid under the conditional heteroskedasticity these instruments
% inherit from the daily shocks.  The Bayesian pair (BVAR point + Gaussian
% posterior band) is overlaid dashed for comparison.
% --- OVERFITTING DIAGNOSTIC ----------------------------------------------
%  Does the UNSHRUNK point estimate agree with the shrunk one?  Both target the
%  same object, so systematic sign disagreement — especially at long horizons,
%  where the companion powers amplify coefficient noise — is evidence that the
%  OLS IRF is tracking estimation error.  Reported so the choice of pair is
%  made on a number, not on eyeballing the figures.
if DO_MBB && ~all(mbb_shrunk)
    fprintf('\n  OLS vs Minnesota point estimate — SIGN DISAGREEMENT (share of cells):\n');
    for ki=1:N_shocks
        D = sign(IRF_ols_all(:,:,ki)) ~= sign(IRF_int(:,:,ki));
        hl = min(25,IRF_HOR_MAC+1):IRF_HOR_MAC+1;
        fprintf('    %-4s  all horizons %3.0f%%   h=0..11 %3.0f%%   h>=24 %3.0f%%\n', shock_tag{ki}, ...
                100*mean(D(:)), 100*mean(reshape(D(:,1:min(12,end)),[],1)), 100*mean(reshape(D(:,hl),[],1)));
    end
    fprintf('    (a large share at long horizons means the unshrunk IRF is estimation noise)\n');
elseif DO_MBB
    fprintf('\n  Bootstrap estimator = SHRUNK (Minnesota posterior mean re-estimated in every\n');
    fprintf('  replication): the bootstrap point and the Bayesian point are the same object,\n');
    fprintf('  so the two bands differ only in HOW they price sampling uncertainty —\n');
    fprintf('  Gaussian posterior vs moving-block bootstrap.  Max |IRF_boot - IRF_bayes|:\n');
    for ki=1:N_shocks
        fprintf('    %-4s  %.3e\n', shock_tag{ki}, max(abs(IRF_ols_all(:,:,ki)-IRF_int(:,:,ki)),[],'all'));
    end
end
%  SAFETY INTERLOCK.  'mbb' is only defensible at this lag length because every
%  replication re-estimates the SHRUNK VAR.  If the shrunk estimator could not be
%  reproduced and the bootstrap fell back to OLS, the 'mbb' pair reverts to
%  "166 unpenalised coefficients on 414 observations" — the configuration the
%  sign-disagreement diagnostic shows to be noise.  Refuse it automatically.
if DO_MBB && strcmpi(MBB_ESTIMATOR,'shrunk') && ~all(mbb_shrunk) && strcmpi(BAND_PRIMARY,'mbb')
    fprintf(2,'  [!] shrunk bootstrap unavailable (OLS fallback) and p=%d -> forcing BAND_PRIMARY=''bayes''.\n', p_macro);
    BAND_PRIMARY = 'bayes';
end
%  FALLBACK RULE.  The bootstrap re-estimates the augmented VAR by OLS, which is
%  unshrunk: at a generous p it can lose replications to explosive draws AND its
%  point estimate can overfit.  A band is only reported together with the point
%  estimate it belongs to, so the choice is made once, here.
MBB_MIN_FRAC = 0.50;
mbb_usable   = DO_MBB && all(mbb_ok >= MBB_MIN_FRAC*MBB_B);
if DO_MBB && ~mbb_usable
    fprintf(2,'  [!] moving-block bootstrap unusable (usable draws: %s of %d; threshold %.0f%%).\n', ...
            mat2str(mbb_ok), MBB_B, 100*MBB_MIN_FRAC);
end
switch lower(BAND_PRIMARY)
    case 'mbb'
        use_mbb = mbb_usable;
        if ~use_mbb, fprintf(2,'  [!] BAND_PRIMARY=''mbb'' requested but not usable -> Bayesian pair.\n'); end
    case 'bayes'
        use_mbb = false;
        fprintf('  BAND_PRIMARY=''bayes'': reporting the SHRUNK pair (see CONFIG for why at p=%d).\n', p_macro);
    otherwise
        use_mbb = mbb_usable;
end
if ~use_mbb
    fprintf(2,'  NOTE: the Gaussian posterior band prices no conditional heteroskedasticity;\n');
    if DO_MBB && any(mbb_ok>0)
        wr = median((IRF_mbb_84(:,1,:)-IRF_mbb_16(:,1,:))./max(IRF_int_84(:,1,:)-IRF_int_16(:,1,:),1e-12),'all','omitnan');
        fprintf(2,'        read it as a LOWER BOUND on uncertainty (the MBB band is x%.2f wider at impact).\n', wr);
    else
        fprintf(2,'        read it as a LOWER BOUND on uncertainty.\n');
    end
end
%% ---- IRF NORMALISATION — BOOKKEEPING ONLY -------------------------------
%  NOTHING IS RESCALED HERE.  Every IRF object arrived from Steps 7 and 7b
%  already normalised draw by draw, which is what makes the bands bands on the
%  reported ratio.  What remains is to record WHICH factor the reported point
%  estimate carries, so the tables, the figure headers and the AR sets of Step 8b
%  can state it.  The two estimator pairs each carry their OWN factor, computed
%  from their OWN impact on the reference variable: pairing the Bayesian point
%  with a factor taken from the bootstrap point would misstate the reading of
%  the dashed line by the ratio of the two impacts.
norm_scale    = ones(1,N_shocks);
norm_desc     = repmat({''},1,N_shocks);
norm_raw_imp  = nan(1,N_shocks);
norm_c_q      = nan(N_shocks,3);   % [16,50,84] of the scale factor across draws
norm_flip     = nan(1,N_shocks);   % share of draws with the reference sign flipped
for ki=1:N_shocks
    switch lower(NORM_MODE)
      case 'unitz'
        norm_scale(ki)=1;
        norm_desc{ki}='per 1 unit of the instrument innovation Z';
      case 'sd'
        norm_scale(ki)=sd_Z(ki);
        norm_desc{ki}=sprintf('per 1 s.d. of the instrument (sd(Z)=%.3f)', sd_Z(ki));
      case 'refvar'
        rname = norm_ref_name{ki};  ir = ir_norm(ki);
        if use_mbb
            b_ref = raw_imp_mbb(ki);  norm_c_q(ki,:) = norm_c_mbb(ki,:);  norm_flip(ki) = norm_flip_mbb(ki);
        else
            b_ref = raw_imp_bay(ki);  norm_c_q(ki,:) = norm_c_bay(ki,:);  norm_flip(ki) = norm_flip_bay(ki);
        end
        % first-stage F of the variable actually used for the normalisation
        % (NOT the max over the 10 macro variables, which is a selected max)
        zc = Z_t(:,ki)-mean(Z_t(:,ki));
        F_ref = local_effF(zc, U_mac(:,ir), T_res);
        norm_scale(ki)  = NORM_REF_SIZE/abs(b_ref);
        norm_raw_imp(ki)= b_ref;
        norm_desc{ki}   = sprintf('per a shock moving |%s| by %.3g on impact  [1st-stage F on %s = %.1f]', ...
                                  rname, NORM_REF_SIZE, rname, F_ref);
      otherwise
        error('NORM_MODE must be ''refvar'', ''sd'' or ''unitZ'' (got ''%s'').', NORM_MODE);
    end
end
NORM_LABEL = sprintf('NORM_MODE=%s (applied inside the draws)', NORM_MODE);
fprintf('\n  IRF NORMALISATION (%s) — applied INSIDE every draw, sign anchor NOT touched:\n', NORM_MODE);
fprintf('  %-6s %-10s %13s %10s   %s\n','shock','ref var','raw impact','scale','reading');
for ki=1:N_shocks
    fprintf('  %-6s %-10s %13.5f %10.2f   %s\n', shock_tag{ki}, norm_ref_name{ki}, ...
            norm_raw_imp(ki), norm_scale(ki), norm_desc{ki});
end
if strcmpi(NORM_MODE,'refvar')
    %  THE NUMBER THE OLD "OUTSIDE" RESCALING THREW AWAY.  A scale factor whose
    %  [16,84] range is wide, or a non-trivial share of draws in which the
    %  reference variable moves the other way, means the denominator of the
    %  reported ratio is itself uncertain — and that uncertainty is now in the
    %  bands instead of being silently replaced by a constant.
    fprintf('\n  Scale factor ACROSS DRAWS (the uncertainty an outside rescaling discards):\n');
    fprintf('  %-6s %10s %22s %14s %12s\n','shock','point','[16, 50, 84] of scale','84/16 spread','sign flips');
    for ki=1:N_shocks
        fprintf('  %-6s %10.2f   [%5.2f, %5.2f, %5.2f] %13.2fx %11.1f%%\n', shock_tag{ki}, ...
                norm_scale(ki), norm_c_q(ki,1), norm_c_q(ki,2), norm_c_q(ki,3), ...
                norm_c_q(ki,3)/max(norm_c_q(ki,1),1e-12), 100*norm_flip(ki));
    end
    %  INTERNAL CHECK.  The scale factor is gamma/|impact on the reference|, so
    %  after normalisation EVERY draw moves the reference variable by exactly
    %  +gamma or -gamma at h=0 — the sign being the draw's own, since the factor
    %  is positive by construction and the sign anchor is not overwritten.  Two
    %  consequences are checked rather than assumed: the point estimate must sit
    %  at |gamma|, and the h=0 band must have zero width when no draw flips the
    %  sign of the reference (and width 2*gamma when some do).
    if use_mbb, Pc=IRF_ols_all; Lc=IRF_mbb_16; Uc=IRF_mbb_84;
    else,       Pc=IRF_int;     Lc=IRF_int_16; Uc=IRF_int_84; end
    dmax = 0; wrep = zeros(1,N_shocks);
    for ki=1:N_shocks
        dmax = max(dmax, abs(abs(Pc(ir_norm(ki),1,ki)) - NORM_REF_SIZE));
        wrep(ki) = Uc(ir_norm(ki),1,ki) - Lc(ir_norm(ki),1,ki);
    end
    if dmax < 1e-10
        fprintf('  Internal check PASSED: every reported point moves %s by exactly %.3g at h=0 (max dev %.1e).\n', ...
                strjoin(unique(norm_ref_name),'/'), NORM_REF_SIZE, dmax);
    else
        fprintf(2,'  [!!] reported impact on the reference variable is %.4f away from %.3g -> normalisation not applied per draw.\n', ...
                dmax, NORM_REF_SIZE);
    end
    fprintf('  68%% band width of the reference variable at h=0 (0 = no draw flips its sign): %s\n', ...
            mat2str(round(wrep,4)));
end

if use_mbb
    IRF_show=IRF_ols_all; L16=IRF_mbb_16; U84=IRF_mbb_84; L05=IRF_mbb_05; U95=IRF_mbb_95;
    if all(mbb_shrunk)
        BAND_LABEL='Minnesota point + moving-block bootstrap band [heteroskedasticity-robust]';
    else
        BAND_LABEL='OLS (UNSHRUNK) point + moving-block bootstrap band';
    end
    if DO_KILIAN && ~all(mbb_delta(isfinite(mbb_delta))==0)
        BAND_LABEL=[BAND_LABEL sprintf(', Kilian delta=%s', mat2str(round(mbb_delta,2)))];
    end
else
    IRF_show=IRF_int;     L16=IRF_int_16; U84=IRF_int_84; L05=IRF_int_05; U95=IRF_int_95;
    BAND_LABEL='BVAR point + Gaussian posterior band [NOT heteroskedasticity-robust]';
end
fprintf('  Reported: %s\n', BAND_LABEL);
% how far the shrinkage estimator sits from the frequentist one (OLS pair only)
if use_mbb && ~all(mbb_shrunk)
    fprintf('  Minnesota shrinkage vs OLS, |IRF_BVAR/IRF_OLS| median over vars (h=12 / h=48):\n');
    for ki=1:N_shocks
        r12=abs(IRF_int(:,13,ki)./max(abs(IRF_ols_all(:,13,ki)),1e-9));
        r48=abs(IRF_int(:,49,ki)./max(abs(IRF_ols_all(:,49,ki)),1e-9));
        fprintf('    %-4s  x%.2f / x%.2f\n', shock_tag{ki}, median(r12), median(r48));
    end
end
% how much wider the robust band is than the Gaussian one it replaces
if use_mbb && all(mbb_shrunk)
    fprintf('  Robust vs Gaussian band width (68%%), median over variables:  h=0 / all horizons\n');
    for ki=1:N_shocks
        rw=(IRF_mbb_84(:,:,ki)-IRF_mbb_16(:,:,ki))./max(IRF_int_84(:,:,ki)-IRF_int_16(:,:,ki),1e-12);
        fprintf('    %-4s  x%.2f / x%.2f\n', shock_tag{ki}, median(rw(:,1)), median(rw(:)));
    end
end
h_show=[0 6 12 24 48]; h_show=h_show(h_show<=IRF_HOR_MAC);
key_vars={'INDPRO','UNRATE','PCEPI'};
for ki=1:N_shocks
    fprintf('\n  %s  [sign anchor: %s loads %s on impact — daily FEVD-dominant]\n', ...
            shock_tag{ki}, dom_daily{ki}, tern_local(dom_sign(ki)>0,'POSITIVELY','NEGATIVELY'));
    fprintf('      %s\n      value [16,84] %s\n', norm_desc{ki}, BAND_LABEL);
    fprintf('  %-8s', 'var'); for h=h_show, fprintf(' %22s', sprintf('h=%d',h)); end, fprintf('\n');
    for vv=1:numel(key_vars)
        n=find(strcmp(MACRO_NAMES,key_vars{vv}),1); if isempty(n), continue; end
        fprintf('  %-8s', key_vars{vv});
        for h=h_show
            fprintf(' %8.4f[%+.3f,%+.3f]', IRF_show(n,h+1,ki), L16(n,h+1,ki), U84(n,h+1,ki));
        end
        fprintf('\n');
    end
end
%% =========================================================================
%  STEP 8b — MONTIEL OLEA-STOCK-WATSON (2021) ANDERSON-RUBIN SETS
%
%  WHY.  The proxy-SVAR estimator is a RATIO: (covariance of the instrument with
%  the residuals) / (relevance of the instrument).  When the denominator is near
%  zero the ratio is not asymptotically normal — it has Cauchy-like tails — so a
%  Wald interval UNDER-covers, and does so invisibly (it just looks narrow).
%  MSW invert a test instead: for each candidate value c0 of the IRF the
%  composite residual  W_t(c0) = (row m of Psi_h)' u_t - c0*u_ref,t  satisfies a
%  moment condition LINEAR in c0, so its AR F-statistic has a known distribution
%  WHATEVER the instrument strength.  The confidence set is the set of c0 not
%  rejected, which reduces to a quadratic inequality A*c0^2 + B*c0 + C <= 0.
%
%  HOW TO READ IT.  A > 0 -> BOUNDED set (the data pin the magnitude down).
%  A <= 0 -> the set is the whole real line: the instrument does not identify the
%  MAGNITUDE at that (variable, horizon), and any narrow interval there would be
%  an artefact.  This is the honest answer to "is your inference driven by the
%  assumption that the instrument is strong?" — here it is tested, not assumed.
%
%  SCOPE.  These sets belong to the EXTERNAL proxy-SVAR (macro-only dynamics,
%  b = Sigma_uZ/sigma^2_Z).  They are NOT a confidence set for the internal
%  instrument point estimate; they answer the separate question of whether the
%  instrument pins the magnitude down at all.  Reported as a robustness exhibit.
%  Reference: Montiel Olea, Stock & Watson (2021, QJE), Theorem 1.
% =========================================================================
fprintf('\n%s\n STEP 8b — MSW(2021) ANDERSON-RUBIN CONFIDENCE SETS (external proxy-SVAR)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
J_mac      = [eye(N_macro), zeros(N_macro,N_macro*(p_macro-1))];
A_comp_mac = [Phi_pm(1:n_comp,:)'; eye(N_macro*(p_macro-1)), zeros(N_macro*(p_macro-1),N_macro)];
UU = U_mac'*U_mac;
ZU = zeros(N_shocks,N_macro);  SZ_v = zeros(N_shocks,1);
for ki=1:N_shocks
    zc = Z_t(:,ki)-mean(Z_t(:,ki));
    ZU(ki,:) = zc'*U_mac;  SZ_v(ki) = zc'*zc;
end
df_res  = T_res - 2;
F_crits = [finv(0.68,1,df_res), finv(0.90,1,df_res)];
n_ci    = numel(F_crits);
IRF_AR_lo = nan(N_macro,IRF_HOR_MAC+1,N_shocks,n_ci);
IRF_AR_hi = nan(N_macro,IRF_HOR_MAC+1,N_shocks,n_ci);
msw_bounded = false(N_macro,IRF_HOR_MAC+1,N_shocks,n_ci);
% normalise the sets on the SAME reference variable used for the reported IRFs
ar_ref_idx = zeros(1,N_shocks);
for ki=1:N_shocks
    ir = find(strcmp(MACRO_NAMES,norm_ref_name{ki}),1);
    if isempty(ir), ir = find(strcmp(MACRO_NAMES,Fmax_var{ki}),1); end
    ar_ref_idx(ki) = ir;
end
A_h = eye(n_comp);
for h=0:IRF_HOR_MAC
    Psi_h = J_mac*A_h*J_mac';
    for ki=1:N_shocks
        rk    = ar_ref_idx(ki);
        b_ref = ZU(ki,rk);  S_ref = UU(rk,rk);  Sz_ki = SZ_v(ki);
        for m=1:N_macro
            prow = Psi_h(m,:);
            a_hm = prow*ZU(ki,:)';
            Q_hm = prow*UU*prow';
            R_hm = prow*UU(:,rk);
            for ci=1:n_ci
                fc = F_crits(ci);
                Aq = b_ref^2*(df_res+fc) - fc*Sz_ki*S_ref;
                Bq = -2*a_hm*b_ref*(df_res+fc) + 2*fc*Sz_ki*R_hm;
                Cq = a_hm^2*(df_res+fc) - fc*Sz_ki*Q_hm;
                disc = Bq^2 - 4*Aq*Cq;
                if Aq > 0 && disc >= 0
                    r1=(-Bq-sqrt(disc))/(2*Aq); r2=(-Bq+sqrt(disc))/(2*Aq);
                    IRF_AR_lo(m,h+1,ki,ci)=min(r1,r2); IRF_AR_hi(m,h+1,ki,ci)=max(r1,r2);
                    msw_bounded(m,h+1,ki,ci)=true;
                else
                    IRF_AR_lo(m,h+1,ki,ci)=-Inf; IRF_AR_hi(m,h+1,ki,ci)=Inf;
                end
            end
        end
    end
    A_h = A_h*A_comp_mac;
end
% Put the sets on the SAME scale as the reported IRFs.  MSW parameterises with
% the reference residual moving by +1; the reported IRFs use |impact on the
% reference variable| = NORM_REF_SIZE with the sign left to the upstream anchor.
% The map is a multiplication by NORM_REF_SIZE*sign(b_ref), bounds swapped when
% that sign is negative.
for ki=1:N_shocks
    sgn = sign(norm_raw_imp(ki)); if sgn==0 || ~isfinite(sgn), sgn = 1; end
    fac = NORM_REF_SIZE*sgn;
    lo = IRF_AR_lo(:,:,ki,:)*fac;  hi = IRF_AR_hi(:,:,ki,:)*fac;
    IRF_AR_lo(:,:,ki,:) = min(lo,hi);  IRF_AR_hi(:,:,ki,:) = max(lo,hi);
end
fprintf('  Share of (variable,horizon) cells with a BOUNDED 90%% AR set:\n');
for ki=1:N_shocks
    bnd = msw_bounded(:,:,ki,2);
    fprintf('    %-4s  %5.1f%%   [ref var: %s, 1st-stage F=%.1f]\n', shock_tag{ki}, ...
            100*mean(bnd(:)), MACRO_NAMES{ar_ref_idx(ki)}, Fmax(ki));
end
fprintf('\n  90%% AR sets at key (variable,horizon), on the reported scale:\n');
for ki=1:N_shocks
    fprintf('  %-4s', shock_tag{ki});
    for h=h_show, fprintf(' %20s', sprintf('h=%d',h)); end, fprintf('\n');
    for vv=1:numel(key_vars)
        n=find(strcmp(MACRO_NAMES,key_vars{vv}),1); if isempty(n), continue; end
        fprintf('  %-8s', key_vars{vv});
        for h=h_show
            if msw_bounded(n,h+1,ki,2)
                fprintf(' %9.3f,%+9.3f', IRF_AR_lo(n,h+1,ki,2), IRF_AR_hi(n,h+1,ki,2));
            else
                fprintf(' %19s', '(-Inf, +Inf)');
            end
        end
        fprintf('\n');
    end
end

horizon=0:IRF_HOR_MAC; ncol=4; nrow=ceil(N_macro/ncol);
for ki=1:N_shocks
    figure('Name',sprintf('proxy_svar_02_cholesky — %s',shock_tag{ki}), ...
           'Position',[30 30 1320 max(700,nrow*190)],'Color','w');
    for n=1:N_macro
        subplot(nrow,ncol,n); hold on;
        fill([horizon fliplr(horizon)],[squeeze(L05(n,:,ki)) fliplr(squeeze(U95(n,:,ki)))],[0.98 0.90 0.80],'EdgeColor','none','FaceAlpha',0.8);
        fill([horizon fliplr(horizon)],[squeeze(L16(n,:,ki)) fliplr(squeeze(U84(n,:,ki)))],[0.95 0.75 0.50],'EdgeColor','none','FaceAlpha',0.8);
        % dashed = THE OTHER point estimate, for contrast only (never with a band:
        % a band belongs to the estimator that generated it)
        if use_mbb
            plot(horizon,squeeze(IRF_int(n,:,ki)),'--','Color',[0.30 0.30 0.70],'LineWidth',1.0);
        elseif DO_MBB && any(mbb_ok>0)
            plot(horizon,squeeze(IRF_ols_all(n,:,ki)),'--','Color',[0.30 0.30 0.70],'LineWidth',1.0);
        end
        plot(horizon,squeeze(IRF_show(n,:,ki)),'-','Color',[0.85 0.45 0.10],'LineWidth',1.4);
        yline(0,'k:'); xlim([0 IRF_HOR_MAC]); title(MACRO_NAMES{n},'FontSize',8,'Interpreter','none');
        set(gca,'FontSize',7,'Box','on');
    end
    dash_lbl = tern_local(use_mbb, 'Minnesota-BVAR point', 'unshrunk OLS point');
    sgtitle({sprintf('%s — internal instrument (Cholesky, Z first, PMW 2021)  |  %s  |  sign anchor: %s %s 0 on impact', ...
                     shock_tag{ki}, norm_desc{ki}, dom_daily{ki}, tern_local(dom_sign(ki)>0,'>','<')), ...
             sprintf('daily FEVD-dominant: %s (ex-post reading) | solid+shaded: %s (68%%/90%%) | dashed: %s', ...
                     dom_daily{ki}, BAND_LABEL, dash_lbl)}, ...
            'FontSize',10,'Interpreter','none');
    print(gcf, sprintf('fig_p02chol_irf_shock%d_%s.png', shock_cols(ki), SRC_BASE), '-dpng','-r150');
end

%% =========================================================================
%  STEP 9 — SAVE
% =========================================================================
save('proxy_svar_02_cholesky_results.mat', 'SRC','shock_cols','shock_tag','dom_daily', ...
     'MACRO_NAMES','p_macro','tau_star','lam_star','mu_star','om_star', ...
     'T_common','T_res','dates_common','lam_med', ...
     'sd_Z','Fmax','Fmax_var','F_indpro', ...
     'IRF_int','IRF_int_16','IRF_int_84','IRF_int_05','IRF_int_95', ...
     'IRF_mbb_16','IRF_mbb_84','IRF_mbb_05','IRF_mbb_95','mbb_ok','BAND_LABEL', ...
     'IRF_ols_all','BAND_PRIMARY','use_mbb','SAMPLE_END_YM','P_MACRO_FIXED','mbb_delta', ...
     'NORM_MODE','NORM_REF_SIZE','norm_scale','norm_ref_name','norm_desc','norm_raw_imp', ...
     'norm_c_q','norm_flip','norm_c_bay','norm_c_mbb','norm_flip_bay','norm_flip_mbb', ...
     'raw_imp_bay','raw_imp_mbb','ir_norm', ...
     'SIGN_CONV','dom_sign','MBB_ESTIMATOR','mbb_shrunk','DO_KILIAN', ...
     'pred_R2','pred_F','pred_p','pred_t','PRED_VARS','PRED_DIFF','PRED_LAGS','lab_p', ...
     'IRF_AR_lo','IRF_AR_hi','msw_bounded','ar_ref_idx','F_crits', ...
     'FEVD_med','FEVD_16','FEVD_84', ...
     'IRF_HOR_MAC','-v7.3');
fprintf('\n%s\n DONE — saved proxy_svar_02_cholesky_results.mat\n  bands reported: %s\n  internal IV (Cholesky, Z first); external-vs-internal comparison lives in Script_2_ProxySVAR.m.\n%s\n', ...
        repmat('=',1,70), BAND_LABEL, repmat('=',1,70));

% =========================================================================
function c = tern_local(cond,a,b), if cond, c=a; else, c=b; end, end

function c = local_norm_fac(bref, mode, ref_size, sdz)
%  POSITIVE scale factor, one per element of bref.  Called with a VECTOR of
%  per-draw impacts on the reference variable (Steps 7 and 7b) and with a SCALAR
%  for the point estimate, so the point and its band are normalised by the same
%  rule applied to their own data — the "inside the bootstrap" requirement of
%  Stock & Watson (2018).  The factor is always positive: the direction of the
%  response is left to the upstream sign anchor and is never overwritten here.
    switch lower(mode)
        case 'unitz'
            c = ones(size(bref));
        case 'sd'
            c = sdz*ones(size(bref));
        case 'refvar'
            c = ref_size ./ abs(bref);
            %  A draw whose reference impact is numerically zero carries no
            %  information about the ratio.  It is turned into NaN and dropped by
            %  the finiteness filter rather than floored, which would silently
            %  invent a finite response where the data pin down none.
            c(~isfinite(c)) = NaN;
        otherwise
            error('NORM_MODE must be ''refvar'', ''sd'' or ''unitZ'' (got ''%s'').', mode);
    end
end

function F = local_effF(z,u,T)
    z=z-mean(z); X=[ones(T,1),z]; bb=X\u; e=u-X*bb;
    XtXi=inv(X'*X); meat=(X.*e)'*(X.*e); V=XtXi*meat*XtXi*(T/(T-2));
    F=bb(2)^2/max(V(2,2),1e-18);
end
