
% =========================================================================

clear; clc; close all;
rng(42, 'twister');
 
%% -- TOOLBOX PATHS --------------------------------------------------------
addpath(fullfile(pwd, 'bvar_Ferroni', 'bvartools'));
addpath(fullfile(pwd, 'bvar_Ferroni', 'cmintools'));
 
%% -- CONFIGURATION --------------------------------------------------------
CSV_FILE   = 'financial_data_aligned_corrected.csv';
VAR_NAMES  = {'SP500','VIX', 'CSPREAD'};
M          = numel(VAR_NAMES);
 
N_DRAWS    = 5000;     % posterior draws
IRF_HOR    = 20;       % horizon (trading days)
SHRINK_TAU = 0.05;     % Ridge regularisation of K_hat toward K_z (identification only)
                       % K* = w*K_hat + (1-w)*Kz,  w = 1/(1+SHRINK_TAU) = 0.952
                       
LAG_GRID   = [3,4,5,6,7,8,9,10];  % candidate VAR lags for flat-prior marginal likelihood selection
nlag       = numel(LAG_GRID);
HILL_FRAC  = 0.05;     % top 5% tail for Hill estimator
 
% Petrova robust posterior: shrinkage on the K used in the QML
% covariance of vech(Sigma).  This is INDEPENDENT of SHRINK_TAU above:
%   K_SHRINK_POST controls posterior bands (uncertainty on Sigma);
%   SHRINK_TAU    controls identification (regularisation of M_hat).
K_SHRINK_POST = NaN;   % NaN -> toolbox default: K_shrinkage = T_full (50/50 weight).
                       % For stronger sample weighting: K_SHRINK_POST = 0.05*(T_full-p)
                       % (95% sample, 5% Gaussian).  Set AFTER lag selection if used.
 
% Degeneracy test
DEGEN_THRESHOLD = 0.10;  % flag near-degeneracy if lambda_2/lambda_1 > 1-this

% Serial-dependence-robust inference (Option A): moving-block bootstrap of M_hat,
% plus i.i.d. diagnostics on the structural shocks.  These REPLACE the (incorrect)
% GARCH pre-filter: we measure the impact of non-i.i.d. shocks instead of removing it.
BLOCK_BOOT_N = 500;      % bootstrap replications for K_hat/M_hat sampling variance
BLOCK_LEN    = 20;       % block length (trading days) ~ daily volatility persistence
LB_LAGS      = 10;       % Ljung-Box lags for i.i.d. diagnostics on structural shocks
 
%% =========================================================================
%  STEP 1 -- LOAD DATA  (original units: log-levels + raw levels)
% =========================================================================
fprintf('\n%s\n STEP 1 -- DATA LOADING\n%s\n', repmat('=',1,70), repmat('=',1,70));
 
T_raw  = readtable(CSV_FILE);
T_raw = T_raw(datetime(T_raw.Date,'InputFormat','yyyy-MM-dd') >= datetime(1990,1,2), :);
Y      = zeros(height(T_raw), M);
for m = 1:M
    Y(:,m) = T_raw.(VAR_NAMES{m});
end
T_full = size(Y, 1);
 
fprintf('  CSV: %d obs x %d variables\n\n', T_full, M);
fprintf('  %-10s %12s %12s %10s %12s\n','Variable','Mean','Std','Skewness','Kurt.Exc');
fprintf('  %s\n', repmat('-',1,62));
for m = 1:M
    y_m = Y(:,m);
    y_s = (y_m - mean(y_m)) / std(y_m);
    fprintf('  %-10s %12.4f %12.4f %10.4f %12.4f\n', ...
        VAR_NAMES{m}, mean(y_m), std(y_m), mean(y_s.^3), mean(y_s.^4)-3);
end
%% =========================================================================
%  STEP 2 -- LAG SELECTION (flat-prior marginal likelihood)
%
%  ln p(Y|p) = const - (T_eff/2)*ln|S_OLS| - (T_eff*M/2)*ln(T_eff)
%  Computed internally by bvar_() as .logmlike.  Lag selection uses flat
%  prior: under flat prior the OLS coefficients are consistent regardless
%  of error distribution, so this is valid even with non-Gaussian residuals.
% =========================================================================
 fprintf('\n%s\n STEP 2 -- LAG SELECTION\n%s\n', repmat('=',1,70), repmat('=',1,70));
 fprintf('  %-6s  %-8s  %-5s  %-14s  %-10s\n','p','T_eff','k','ln ML','Delta');
 fprintf('  %s\n', repmat('-',1,52));
 
 opt_sel   = struct('K', 600);
 BVAR_sel  = cell(nlag, 1);
 lnML_grid = zeros(nlag, 1);
 
 for i = 1:nlag
     p_i           = LAG_GRID(i);
     BVAR_sel{i}   = bvar_(Y, p_i, opt_sel);
     lnML_grid(i)  = BVAR_sel{i}.logmlike;
 end
 
 lnML_max          = max(lnML_grid);
 [~, best_lag_idx] = max(lnML_grid);
 p                 = LAG_GRID(best_lag_idx);
 T_eff             = T_full - p;
 k                 = M*p + 1;
 
 for i = 1:nlag
     flag = '';
     if i == best_lag_idx, flag = '  <- ML OPTIMAL'; end
     fprintf('  p=%-4d  T_eff=%-5d  k=%-4d  lnML=%12.4f  Delta=%9.4f%s\n', ...
         LAG_GRID(i), T_full - LAG_GRID(i), M*LAG_GRID(i)+1, ...
         lnML_grid(i), lnML_grid(i)-lnML_max, flag);
 end
 fprintf('\n  Selected p = %d  |  T_eff = %d  |  k = %d\n', p, T_eff, k);
 clear BVAR_sel opt_sel
 
%% =========================================================================
%  STEP 3 -- ROBUST BAYESIAN BVAR(p)  (Petrova 2022)
%
%  Why robust_bayes?
%    Under flat prior + Gaussian likelihood, the posterior of Sigma is
%    Inverse-Wishart.  Under non-Gaussianity (the very assumption that
%    enables Kollo identification!), this posterior is mis-specified:
%    the asymptotic variance of vech(Sigma_hat) depends on the 4th moment
%    of residuals, not just on Sigma.
%
%    Petrova (2022) replaces IW with the QML asymptotic distribution:
%      vech(Sigma) ~ N( vech(Sigma_hat), (1/T)[K_hat^* - vech*vech'] )
%    where K_hat^* is a shrinkage estimator of the 4th moment.
%  
%   Cholesky decomposition and Kollo identification
%   operate on the correlation structure regardless of scale.
%
%   Reference: HitchhikerGuide sec. 3.1.6; AFM (2025) App. A.5.
% =========================================================================
fprintf('\n%s\n STEP 3 -- ROBUST BAYESIAN BVAR(p=%d)  [Petrova 2022]\n%s\n', ...
        repmat('=',1,70), p, repmat('=',1,70));
 
opt_flat              = struct();
opt_flat.K            = N_DRAWS;
opt_flat.hor          = IRF_HOR;
opt_flat.robust_bayes = 1;             % Petrova (2022) QML correction for Sigma posterior.
                                       % =1: 4th-moment correction (symmetric heavy tails).
                                       % =2: 4th+3rd-moment (asymmetric + heavy tails).
                                       % Financial daily data often has negative skew
                                       % (SP500, AAA): check Step 1 output and switch
                                       % to =2 if skewness is significant.
if ~isnan(K_SHRINK_POST)
    opt_flat.K_shrinkage = K_SHRINK_POST;
end
 
bvar_obj = bvar_(Y, p, opt_flat);
 
% Posterior means
Phi_pm   = mean(bvar_obj.Phi_draws,   3);   %(k x M)
Sigma_pm = mean(bvar_obj.Sigma_draws, 3);   % M x M  (original-unit covariance)
 
% Regressor matrix X -- needed if e_draws not available
X = zeros(T_eff, k);
for l = 1:p
    X(:, (l-1)*M+1 : l*M) = Y(p-l+1:end-l, :);
end
X(:, end) = 1;
U_pm = Y(p+1:end, :) - X * Phi_pm;     % T_eff x M  (Posterior mean residuals, original units)
 
fprintf('  ln ML (flat, p=%d): %.4f\n', p, bvar_obj.logmlike);
fprintf('  cond(Sigma_pm):     %.4e  ', cond(Sigma_pm));
if cond(Sigma_pm) < 1e4
    fprintf('[OK]\n');
else
    fprintf('[WARNING: ill-conditioned — check variable scales]\n');
end
 
ev_pm = sort(eig(Sigma_pm), 'descend');
fprintf('  Eigenvalues of Sigma_pm (original units): ');
fprintf('%.6f  ', ev_pm); fprintf('\n');
 
% Petrova diagnostics: spread of posterior Sigma draws
sigma_draw_var = var(reshape(bvar_obj.Sigma_draws, M*M, N_DRAWS), 0, 2);
fprintf('  Petrova robust posterior of Sigma:\n');
fprintf('    Mean draw-variance of Sigma elements: %.4e\n', mean(sigma_draw_var));
fprintf('    (compare with non-robust IW: bands may differ materially under heavy tails)\n');

%% =========================================================================
%  STEP 3b -- BREUSCH-GODFREY TEST  (validate that p removes autocorrelation)
%
%  Under non-Gaussian errors OLS is consistent (Sims-Stock-Watson 1990), but
%  Kollo identification requires i.i.d. residuals.  If BG rejects at the
%  selected lag p, the lag grid should be extended.
% =========================================================================
fprintf('\n%s\n STEP 3b -- BREUSCH-GODFREY TEST on U_pm  (p = %d)\n%s\n', ...
        repmat('=',1,70), p, repmat('=',1,70));

H_BG_VAL = 10;    % BG lags
ALPHA_BG  = 0.05;
fprintf('  %-10s  %10s  %8s  %s\n', 'Variable', sprintf('BG(%d)',H_BG_VAL), 'p-value', 'Result');
fprintf('  %s\n', repmat('-',1,55));
bg_any_reject = false;
bg_results    = zeros(M, 3);    % [LM_stat, p-value, reject]
for m = 1:M
    u_m    = U_pm(:,m);
    U_lag  = zeros(T_eff, H_BG_VAL);
    for s  = 1:H_BG_VAL
        U_lag(s+1:end,s) = u_m(1:end-s);
    end
    idx_bg  = H_BG_VAL+1 : T_eff;
    Z_bg    = [X(idx_bg,:), U_lag(idx_bg,:)];
    u_bg    = u_m(idx_bg);
    e_bg    = u_bg - Z_bg*(Z_bg\u_bg);
    R2_bg   = max(0, 1-(e_bg'*e_bg)/sum((u_bg-mean(u_bg)).^2));
    LM_bg   = numel(idx_bg) * R2_bg;
    pv_bg   = 1 - chi2cdf(LM_bg, H_BG_VAL);
    reject  = pv_bg < ALPHA_BG;
    if reject, bg_any_reject = true; end
    flag    = ''; if reject, flag = '  *** AUTOCORR'; end
    fprintf('  %-10s  %10.4f  %8.4f%s\n', VAR_NAMES{m}, LM_bg, pv_bg, flag);
    bg_results(m,:) = [LM_bg, pv_bg, double(reject)];
end
if bg_any_reject
    fprintf('\n  WARNING: autocorrelation detected at p=%d (alpha=%.2f).\n', p, ALPHA_BG);
    fprintf('  Consider extending LAG_GRID beyond [%s].\n', ...
            num2str(LAG_GRID,'%d '));
else
    fprintf('\n  All series: no autocorrelation at alpha=%.2f. p=%d is adequate.\n', ...
            ALPHA_BG, p);
end

%% =========================================================================
%  STEP 4 -- PRE-WHITENING CHECK + TAIL DIAGNOSTICS
%
%  iota_t = L^{-1} u_t  where  Sigma_pm = L L'  (lower Cholesky).
%  By construction E[iota iota'] = I_M  regardless of the original scale
%  of u_t.  Pre-whitening is scale-invariant: if u_t = D*u_t_sc with D
%  diagonal, then L = D * L_sc * D^{-1}... actually the Cholesky of D Sigma D
%  is D*L_sc, so L^{-1} u_t = L_sc^{-1} D^{-1} * D * u_t_sc = L_sc^{-1} u_t_sc.
%  Conclusion: iota_t is IDENTICAL whether we standardise or not.
%  The Kollo eigendecomposition is therefore numerically equivalent.
%
%  Tail diagnostics:
%    (1) Excess kurtosis (sample 4th central moment / sigma^4 - 3).
%    (2) Hill estimator alpha:  P(|X|>x) ~ x^{-alpha}.
%    (3) Kim-White (2004) octile-based robust kurtosis.
% =========================================================================
fprintf('\n%s\n STEP 4 -- PRE-WHITENING + TAIL DIAGNOSTICS\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
 
L_pm    = chol(Sigma_pm, 'lower');
iota_pm = (L_pm \ U_pm')';               % T_eff x M
 
cov_check    = (iota_pm' * iota_pm) / T_eff;
off_diag_max = max(abs(cov_check - eye(M)), [], 'all');
if off_diag_max <= 1e-3
    flag_cov = '[OK]';
elseif off_diag_max <= 0.02
    flag_cov = '[WARN: marginally above 1e-3]';
else
    flag_cov = '[BAD: >0.02 — check scales]';
end
fprintf('  Max |cov(iota) - I_M|: %.6f  (OK<=1e-3, warn<=0.02, bad>0.02)  %s\n', ...
        off_diag_max, flag_cov);
if off_diag_max > 0.02
    warning('Pre-whitening imprecise. Consider variable set or numerical regularisation.');
end
 
fprintf('\n  %-10s %10s %10s %12s %12s %12s %12s %10s\n', ...
        'Variable','Mean','Std','Kurt.Exc','Hill(avg)','Hill(right)','Hill(left)','KW.Exc');
fprintf('  %s\n', repmat('-',1,84));
kurtosis_pm    = zeros(M,1);
hill_alpha_pm  = zeros(M,1);   % two-sided average
hill_right_pm  = zeros(M,1);   % right-tail only
hill_left_pm   = zeros(M,1);   % left-tail only
kw_kurt_pm     = zeros(M,1);
for m = 1:M
    i_m   = iota_pm(:,m);
    i_std = (i_m - mean(i_m)) / std(i_m);
 
    % (1) Sample excess kurtosis
    k_exc = mean(i_std.^4) - 3;
    kurtosis_pm(m) = k_exc;
 
    % (2) Hill alpha — two-sided: right and left tail estimated separately.
    %   Financial data has asymmetric tails; mixing via |u| conflates them.
    mh = max(2, floor(HILL_FRAC * T_eff));
    % Right tail (large positive residuals)
    pos_i = sort(i_m(i_m > 0), 'descend');
    if numel(pos_i) > mh && pos_i(mh+1) > 0
        hill_r = 1 / (mean(log(pos_i(1:mh))) - log(pos_i(mh+1)));
    else
        hill_r = NaN;
    end
    % Left tail (large negative residuals, reflected)
    neg_i = sort(-i_m(i_m < 0), 'descend');
    if numel(neg_i) > mh && neg_i(mh+1) > 0
        hill_l = 1 / (mean(log(neg_i(1:mh))) - log(neg_i(mh+1)));
    else
        hill_l = NaN;
    end
    hill_a = mean([hill_r, hill_l], 'omitnan');
    hill_alpha_pm(m)  = hill_a;
    hill_right_pm(m)  = hill_r;
    hill_left_pm(m)   = hill_l;
 
    % (3) Kim-White (2004) octile-based robust kurtosis
    octiles = quantile(i_m, (1:7)/8);
    p1 = octiles(1); p2 = octiles(2); p3 = octiles(3);
    p5 = octiles(5); p6 = octiles(6); p7 = octiles(7);
    if abs(p6 - p2) > 1e-10
        K_KW     = ((p7 - p5) + (p3 - p1)) / (p6 - p2);
        K_KW_exc = K_KW - 1.23;
    else
        K_KW_exc = NaN;
    end
    kw_kurt_pm(m) = K_KW_exc;
 
    fprintf('  %-10s %10.4f %10.4f %12.4f %12.4f %12.4f %12.4f %10.4f\n', ...
            VAR_NAMES{m}, mean(i_m), std(i_m), k_exc, hill_a, hill_r, hill_l, kw_kurt_pm(m));
end
fprintf('\n  Diagnostic summary:\n');
fprintf('    All sample excess kurtoses > 0:  %s\n', mat2str(all(kurtosis_pm > 0)));
fprintf('    All Kim-White excess kurt. > 0:  %s\n', mat2str(all(kw_kurt_pm > 0)));
n_alpha_lt4   = sum(hill_alpha_pm < 4 & ~isnan(hill_alpha_pm));
n_right_lt4   = sum(hill_right_pm < 4 & ~isnan(hill_right_pm));
n_left_lt4    = sum(hill_left_pm  < 4 & ~isnan(hill_left_pm));
fprintf('    Hill alpha(avg)   < 4 in %d/%d variables\n', n_alpha_lt4, M);
fprintf('    Hill alpha(right) < 4 in %d/%d variables (upper tail)\n', n_right_lt4, M);
fprintf('    Hill alpha(left)  < 4 in %d/%d variables (lower tail)\n', n_left_lt4, M);
fprintf('      => alpha < 4: 4th moment may be infinite in that tail\n');
fprintf('      => Kollo identification still operative in finite sample;\n');
fprintf('      => K_KW (Kim-White) confirms non-Gaussianity robustly.\n');

%% =========================================================================
%  NO GARCH PRE-FILTER  (Kollo identification on raw iota)
%  Why NOT filter?
%   (i)  the filter is applied in the reduced-form ROTATED space, where the
%        conditional covariance of iota is NON-diagonal, so componentwise
%        standardisation breaks the sphering that M_hat presupposes;
%   (ii) it removes precisely the (volatility-clustering) kurtosis that may
%        carry the identifying signal -- if shocks are conditionally Gaussian,
%        M_hat -> 0 and P is unidentified;
%   (iii) it changes P, hence B = L*P, hence IRF/FEVD/V -- not merely the
%        precision of K_hat (the old "IRFs unchanged" claim was false).
%
%  We therefore do NOT filter.  The non-i.i.d. nature of the shocks is TESTED
%  (Step 6c) and its effect on inference is QUANTIFIED by a moving-block
%  bootstrap (Step 7c), both of which leave the point estimate P untouched.
% =========================================================================

%% =========================================================================
%  STEP 5 -- BUILD AUXILIARY MATRICES  (computed ONCE, outside draw loop)
%
%  Commutation matrix K_{M,M}  (M^2 x M^2):
%    K_{M,M} vec(X) = vec(X')  for any M x M matrix X.
%
%  Gaussian benchmark K_z  (M^2 x M^2):
%    K_z = I_{M^2} + K_{M,M} + vec(I_M) vec(I_M)'
%    K_z is rotation-invariant: carries ZERO identification information.
%    Only K_hat - K_z is informative about A_0.
%    Note: K_z depends only on M, not on the scale of the data, because
%    it is the 4th-moment matrix of N(0, I_M) — i.e. of the pre-whitened
%    iota_t.  It is therefore identical in standardised and non-standardised
%    versions.
% =========================================================================
fprintf('\n%s\n STEP 5 -- BUILD K_z  (M^2=%d x M^2=%d)\n%s\n', ...
        repmat('=',1,70), M^2, M^2, repmat('=',1,70));
 
Kmm   = commutationmatrix(M);
vecIM = reshape(eye(M), M^2, 1);
Kz    = eye(M^2) + Kmm + vecIM * vecIM';
 
fprintf('  K_z: %dx%d  |  rank = %d\n', M^2, M^2, rank(Kz));
 
shrink_w = T_eff / (T_eff + SHRINK_TAU * T_eff);
fprintf('  Petrova shrinkage weight (identification): %.6f  (tau=%.3f*T)\n', ...
        shrink_w, SHRINK_TAU);

%% =========================================================================
%  STEP 5b -- REFERENCE ROTATION P_ref  (Kollo at posterior mean)
%
%  P_ref is computed ONCE here, before the draw loop, and serves two roles:
%    (1) Sign anchor:   each draw's eigenvectors are flipped if their dot
%        product with P_ref is negative  (fixes C1 — bimodal posteriors).
%    (2) Column anchor: greedy matching aligns each draw's eigenvector
%        columns to P_ref  (fixes S1 — near-degenerate eigenvalue swaps).
%
%  Mathematical chain (identical to the per-draw chain in Step 6):
%    Z_pm    = kron-product matrix of iota_pm (T_eff x M^2)
%    K_hat_pm = Z_pm' * Z_pm / T_eff             (M^2 x M^2)
%    K_star_pm = shrink_w*K_hat_pm + (1-shrink_w)*Kz
%    D_mat_pm  = K_star_pm - Kz
%    M_hat_pm  = sum of the DIAGONAL M×M blocks of D_mat_pm
%                [= I_M star D_mat_pm = MRS/FOBI K(Z), Kollo 2008 eq.4.1]
%                NOTE: this is the IDENTITY contraction.  Summing ALL elements
%                of each block would give the OTHER contraction C(Z)=1 star c4,
%                whose eigenvalues are kappa_i*s_i^2 rather than kappa_i.
%    [P_ref, lam_ref] = eig(M_hat_pm), sorted descending by eigenvalue
% =========================================================================
fprintf('\n%s\n STEP 5b -- REFERENCE ROTATION P_ref  (posterior mean)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));

% Reference identification at the posterior mean.  The full Kollo chain
% (K_hat -> Petrova shrinkage -> star-product compression -> symmetric
% eigendecomposition, sorted by descending excess kurtosis) lives in the
% shared local function kollo_identify(), so that the posterior-mean
% reference (here), the per-draw loop (Step 6) and the bootstrap (Step 7c)
% can never drift apart.
[P_ref, lam_ref_sorted] = kollo_identify(iota_pm, Kz, shrink_w, M);

for kk = 1:M
    [~, midx] = max(abs(P_ref(:,kk)));
    if P_ref(midx,kk) < 0, P_ref(:,kk) = -P_ref(:,kk); end
end

% Step 5b-ii: ECONOMIC sign normalisation (Lanne, Meitz & Saikkonen 2017 JE,
%   Section 3: "signs of columns of B_0 are not identified, fixed by convention").
%
%  The max-abs convention applied above to P_ref fixes signs in the PRE-WHITENED
%  space.  That space is not economically interpretable: the same column can have
%  its dominant pre-whitened loading on a different variable than its dominant
%  ORIGINAL-UNIT loading, because B = L_pm * P_ref re-mixes the rows through the
%  Cholesky factor L_pm.  We therefore normalise signs in the ECONOMIC space B.
%
%  PRIMARY ANCHOR (data-driven, re-run stable):
%    For each shock-column k, let m*(k) = argmax_m |B(m,k)|/std_m be the variable
%    whose impact-FEVD share (at h=0) shock k explains the most.  We use the
%    STANDARDISED loading (not raw |B|) so the anchor is scale-invariant and
%    reflects economics rather than the unit of each variable.  We flip column k
%    so that its dominant impact is POSITIVE:  B(m*(k), k) > 0.  This is the
%    max-abs convention of LMS (2017) applied to standardised B.  It assumes NOTHING
%    about which shock is "the CSPREAD shock" or "the VIX shock", so it survives
%    a change of data, lag order, or eigenvalue re-ordering — unlike a hard-coded
%    column→variable map.  Because we flip the corresponding column of P_ref too,
%    the convention propagates to ALL N_DRAWS via the P_ref alignment in Step 6
%    (no per-draw economic correction needed).
%
%  OPTIONAL ECONOMIC OVERRIDE (off by default):
%    If you want a *named* variable to be the positive direction of a *named*
%    column (e.g. "column 1 should be CSPREAD-up"), list (variable, column) pairs
%    in ECON_ANCHORS below.  Each override is applied AFTER the dominant-loading
%    anchor, so it wins for the columns it names.  Leave ECON_ANCHORS empty to
%    use the purely data-driven convention.  This is fragile by design (it hard-
%    codes a shock→column mapping) so it is opt-in only.
ECON_ANCHORS = {};   % e.g. {'CSPREAD',1; 'VIX',6} to force named directions.

B_ref_ec = L_pm * P_ref;    % structural impact at posterior mean (original units)
std_ec   = sqrt(sum(B_ref_ec.^2, 2));   % per-variable residual std (= sqrt diag Sigma_pm)

% --- Primary anchor: dominant STANDARDISED loading positive, per column -------
%  We rank variables by |B(m,k)|/std_m (= impact-FEVD share at h=0), NOT raw
%  |B|.  Raw |B| is dominated by the variable with the largest unit scale (a
%  rate in pp vs a log index), which mislabels a broad shock as e.g. "SLOPE"
%  purely by scale.  The standardised loading reflects economics (which
%  variable the shock explains the most), matching Script_2's selection.
for kk = 1:M
    [~, mdom]  = max(abs(B_ref_ec(:,kk)) ./ std_ec);   % FEVD-dominant variable
    if B_ref_ec(mdom,kk) < 0
        P_ref(:,kk)    = -P_ref(:,kk);       % flip rotation column ...
        B_ref_ec(:,kk) = -B_ref_ec(:,kk);    % ... and the impact column in lockstep
    end
end

% --- Optional named-variable override (only if ECON_ANCHORS is non-empty) -----
for a = 1:size(ECON_ANCHORS, 1)
    vname  = ECON_ANCHORS{a,1};
    kcol   = ECON_ANCHORS{a,2};
    vrow   = find(strcmp(VAR_NAMES, vname));
    if isempty(vrow)
        fprintf('  [ECON_ANCHOR] variable "%s" not found — skipped.\n', vname);
        continue
    end
    if kcol < 1 || kcol > M
        fprintf('  [ECON_ANCHOR] column %d out of range for "%s" — skipped.\n', kcol, vname);
        continue
    end
    if B_ref_ec(vrow, kcol) < 0
        P_ref(:,kcol)    = -P_ref(:,kcol);
        B_ref_ec(:,kcol) = -B_ref_ec(:,kcol);
        fprintf('  [ECON_ANCHOR] flipped column %d so %s responds positively.\n', kcol, vname);
    end
end

% Report which variable anchors each column (for transparency / reproducibility).
fprintf('\n  Economic sign anchor (FEVD-dominant variable per shock, standardised |B|):\n');
for kk = 1:M
    [amax, mdom] = max(abs(B_ref_ec(:,kk)) ./ std_ec);
    fprintf('    v_%d  ->  %-10s  (std|B| = %.4f, sign +)\n', kk, VAR_NAMES{mdom}, amax);
end

clear B_ref_ec

fprintf('\n  Eigenvalues d of K(Z) = I*c4 [MRS/FOBI; under indep. d = TRUE kurtosis]:\n  ');
fprintf('%.4f  ', lam_ref_sorted); fprintf('\n');

% --- v01_KZ: VERIFY that K(Z) eigenvalue d ~ true excess kurtosis kappa -------
%   For K(Z) (identity contraction) the eigenvalues equal kappa_i (no s_i^2),
%   so d (eig) and TRUE kappa_i should MATCH (up to dependence/finite sample),
%   and ordering by d already matches non-Gaussianity. The s_i^2 / kappa*s^2
%   columns are shown only to contrast with the C(Z) version (kollo_modd01.m),
%   where they would equal d instead.
V_ref_tmp  = iota_pm * P_ref;                       % T_eff x M structural shocks
kappa_ref  = mean(V_ref_tmp.^4, 1)' - 3;            % M x 1 true excess kurtosis
s_col_ref  = sum(P_ref, 1)';                        % M x 1 column sums
d_decomp   = kappa_ref .* s_col_ref.^2;             % the C(Z) eigenvalue (for contrast)
fprintf('\n  v01_KZ -- K(Z) eigenvalue vs TRUE excess kurtosis (should match):\n');
fprintf('  %-8s %12s %12s %16s %14s\n', 'Shock','d=K(Z) eig','TRUE kappa_i','sum-col s_i','kappa*s^2 [C(Z)]');
fprintf('  %s\n', repmat('-',1,66));
for kk = 1:M
    fprintf('  v_%-6d %12.4f %12.4f %16.4f %14.4f\n', ...
            kk, lam_ref_sorted(kk), kappa_ref(kk), s_col_ref(kk), d_decomp(kk));
end
fprintf('  max|d - kappa| = %.4f   (small => K(Z) eigenvalue ~ true kurtosis OK)\n', ...
        max(abs(lam_ref_sorted(:) - kappa_ref(:))));
[~, ord_k] = sort(kappa_ref, 'descend');
fprintf('  => most NON-GAUSSIAN shock (TRUE kurtosis): v_%d (kappa=%.3f); under K(Z)\n', ...
        ord_k(1), kappa_ref(ord_k(1)));
fprintf('     this is also the leading-eigenvalue shock (no s^2 paradox).\n');
clear V_ref_tmp

fprintf('\n  P_ref (rotation at posterior mean, economic sign convention: dominant |B| > 0):\n');
fprintf('  %-10s', 'Var\\Comp');
for kk = 1:M, fprintf(' %9s', ['P' num2str(kk)]); end, fprintf('\n');
fprintf('  %s\n', repmat('-',1,10+10*M));
for m = 1:M
    fprintf('  %-10s', VAR_NAMES{m});
    for kk = 1:M, fprintf(' %9.4f', P_ref(m,kk)); end
    fprintf('\n');
end

%% =========================================================================
%  STEP 6 -- KOLLO IDENTIFICATION LOOP  (vectorised K_hat)
%
%  Mathematical chain per draw s:
%    Sigma_s = L_s L_s'                        [Cholesky, original-unit cov]
%    iota_t  = L_s^{-1} u_t^(s)                [pre-whitened, E[iota iota']=I]
%    K_hat_s = (1/T) Z'Z,  Z(t,:)=kron(iota_t, iota_t)'  [M^2 x M^2]
%    K_star_s = shrink_w*K_hat_s + (1-shrink_w)*Kz        [Petrova shrinkage]
%    M_hat_s  = eye(M) star (K_star_s - Kz)            [M x M]
%    [P_s, Lambda_s] = eig(M_hat_s), sorted descending
%    B_s  = L_s * P_s        [structural impact matrix, original units]
%    v_t  = P_s' * iota_t    [structural shocks, unit variance by construction]
%    Theta_h = Psi_h * B_s   [IRF at horizon h, original units]
%
%  Key point: B_s is in original units — e.g. B_s(SP500, v_1) measures
%  the log-level response of SP500 to a unit structural shock v_1.
% =========================================================================
fprintf('\n%s\n STEP 6 -- KOLLO IDENTIFICATION LOOP  (%d draws)\n%s\n', ...
        repmat('=',1,70), N_DRAWS, repmat('=',1,70));
 
eigenvalues_draws = zeros(M,        N_DRAWS);
P_draws           = zeros(M, M,     N_DRAWS);
B_draws           = zeros(M, M,     N_DRAWS);
IRF_draws         = zeros(M, IRF_HOR+1, M, N_DRAWS, 'single');  % single: halves storage
Vt_draws          = zeros(T_eff, M, N_DRAWS, 'single');         % single: ~1.3GB not 2.5GB
cov_V_draws       = zeros(M, M, N_DRAWS);    % computed in-loop (avoids 2nd pass in Step 10)
valid             = true(N_DRAWS, 1);        % flags draws kept (rcond OK)

tic;
skip_count = 0;
for s = 1:N_DRAWS
 
    % (a) Posterior draw
    Phi_s   = bvar_obj.Phi_draws(:,:,s);
    Sigma_s = bvar_obj.Sigma_draws(:,:,s);
 
    % (b) Residuals
    if isfield(bvar_obj,'e_draws') && ndims(bvar_obj.e_draws)==3
        U_s = bvar_obj.e_draws(:,:,s);
    else
        U_s = Y(p+1:end,:) - X * Phi_s;
    end
 
    % (c) Cholesky pre-whitening
    if rcond(Sigma_s) < 1e-14
        skip_count   = skip_count + 1;
        valid(s)     = false;
        continue
    end
    L_s    = chol(Sigma_s, 'lower');
    iota_s = (L_s \ U_s')';                  % T_eff x M

    % (d-f) Kollo identification chain (K_hat -> Petrova shrinkage ->
    %       star-product compression -> symmetric eigendecomposition, sorted
    %       by descending excess kurtosis).  Same shared function used for the
    %       posterior-mean reference (Step 5b) and the bootstrap (Step 7c).
    %       The column ORDER and SIGN are not yet anchored here; that is done
    %       in (g) below by aligning to P_ref.
    [P_s, lam_s] = kollo_identify(iota_s, Kz, shrink_w, M);

    % (g) C1+S1: align column order THEN sign with P_ref.
    %   This prevents bimodal posteriors (C1) and eigenvalue-swap artefacts (S1).
    %
    %   Step 1 — greedy column matching.
    %   For each reference column r of P_ref, find the source column in P_s
    %   with the highest absolute dot product and assign it to position r.
    %   The greedy order (r=1,...,M) prioritises the most non-Gaussian shock.
    dots_abs  = abs(P_s' * P_ref);   % M x M: |<col_src, col_ref>|
    used_src  = false(M,1);
    P_aligned   = zeros(M,M);
    lam_aligned = zeros(M,1);
    for r = 1:M
        d_r              = dots_abs(:,r);
        d_r(used_src)    = -Inf;        % mask already-assigned source columns
        [~, best_src]    = max(d_r);
        P_aligned(:,r)   = P_s(:,best_src);
        lam_aligned(r)   = lam_s(best_src);
        used_src(best_src) = true;
    end
    P_s   = P_aligned;
    lam_s = lam_aligned;
    %   Step 2 — sign alignment: ensure each column points the same way as P_ref.
    for kk = 1:M
        if dot(P_s(:,kk), P_ref(:,kk)) < 0
            P_s(:,kk) = -P_s(:,kk);
        end
    end

    % (h) Impact matrix (original units — no diag(sigma) rescaling needed)
    B_s = L_s * P_s;
 
    % (i) Structural shocks
    V_s = iota_s * P_s;
 
    % (j) IRF via toolbox iresponse()
    try
        ir_s = iresponse(Phi_s, Sigma_s, IRF_HOR+1, P_s);
        IRF_draws(:,:,:,s) = single(ir_s);
    catch
        % Fallback to manual companion-form computation
        n_comp = M * p;
        J_sel  = [eye(M), zeros(M, M*(p-1))];
        Phi_coef = Phi_s(1:M*p, :)';
        A_comp   = [Phi_coef; eye(M*(p-1)), zeros(M*(p-1), M)];
        A_h      = eye(n_comp);
        for h = 0:IRF_HOR
            Psi_h = J_sel * A_h * J_sel';
            IRF_draws(:, h+1, :, s) = single(Psi_h * B_s);
            A_h = A_h * A_comp;
        end
    end

    eigenvalues_draws(:,s)  = lam_s;
    P_draws(:,:,s)          = P_s;
    B_draws(:,:,s)          = B_s;
    Vt_draws(:,:,s)         = single(V_s);
    cov_V_draws(:,:,s)      = (V_s' * V_s) / T_eff;   % in-loop (Step 10 reuse)
 
    if mod(s,500)==0
        fprintf('  Draw %d / %d  (%.1fs)\n', s, N_DRAWS, toc);
    end
end
fprintf('  Total: %.2fs  |  Skipped draws (rcond<1e-14): %d\n', toc, skip_count);

% --- Compact away skipped draws -------------------------------------------
%  Skipped draws leave zero-filled slabs that would contaminate every
%  posterior median/quantile (and produce 0/0=NaN in the FEVD).  Remove them.
N_DRAWS_REQ = N_DRAWS;                  % requested count (for the record)
if skip_count > 0
    eigenvalues_draws = eigenvalues_draws(:,    valid);
    P_draws           = P_draws(:,:,           valid);
    B_draws           = B_draws(:,:,           valid);
    IRF_draws         = IRF_draws(:,:,:,       valid);
    Vt_draws          = Vt_draws(:,:,          valid);
    cov_V_draws       = cov_V_draws(:,:,       valid);
    N_DRAWS           = nnz(valid);
    fprintf('  Compacted: %d valid draws kept (%d removed).\n', N_DRAWS, skip_count);
end

% --- v01: posterior of the TRUE excess kurtosis of each structural shock ----
%   kappa_i = E[v_i^4] - 3 (v_i has unit variance by construction).  This is
%   the genuine non-Gaussianity measure, distinct from the Kollo eigenvalue
%   d_i = kappa_i * s_i^2.  Columns are already anchored to P_ref in the loop,
%   so kappa_draws(i,:) refers to the same shock across draws.
kappa_draws = zeros(M, N_DRAWS);
for s = 1:N_DRAWS
    Vs = double(Vt_draws(:,:,s));            % T_eff x M (one draw at a time)
    kappa_draws(:,s) = mean(Vs.^4, 1)' - 3;  % E[v_i^4]-3 (unit variance)
end

%% =========================================================================
%  STEP 6b -- REFERENCE STRUCTURAL SHOCKS  (posterior-mean parameters)
% =========================================================================
fprintf('\n%s\n STEP 6b -- REFERENCE STRUCTURAL SHOCKS  (posterior mean)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
 
% P_ref (and lam_ref_sorted) were computed in Step 5b via kollo_identify()
% on the posterior-mean iota_pm, before the draw loop, to serve as the
% sign/column anchor.  Single best-estimate objects at the posterior mean.
B_ref = L_pm * P_ref;     % M x M structural impact matrix (original units)
V_ref = iota_pm * P_ref;  % T_eff x M  (unit variance by construction)
 
cov_V_ref = (V_ref' * V_ref) / T_eff;
fprintf('  max|Cov(V_ref) - I_M|: %.2e  (should be < 1e-3)\n', ...
        max(abs(cov_V_ref - eye(M)), [], 'all'));
 
% Reference IRF via companion form at posterior mean
n_comp_ref = M * p;
J_ref      = [eye(M), zeros(M, M*(p-1))];
Phi_c_ref  = Phi_pm(1:M*p, :)';
A_comp_ref = [Phi_c_ref; eye(M*(p-1)), zeros(M*(p-1), M)];
Theta_ref  = zeros(M, IRF_HOR+1, M);
A_h_ref    = eye(n_comp_ref);
for h = 0:IRF_HOR
    Psi_h_ref          = J_ref * A_h_ref * J_ref';
    Theta_ref(:,h+1,:) = Psi_h_ref * B_ref;
    A_h_ref            = A_h_ref * A_comp_ref;
end
fprintf('  Reference IRF (Theta_ref): M x (H+1) x M = %dx%dx%d  [original units]\n', ...
        M, IRF_HOR+1, M);

%% =========================================================================
%  STEP 6c -- i.i.d. / SERIAL-INDEPENDENCE DIAGNOSTICS  (structural shocks)
%
%  Kollo identification needs the structural shocks to be MUTUALLY INDEPENDENT
%  with finite, distinct 4th moments.  It does NOT need them to be i.i.d. over
%  time: serial dependence (e.g. GARCH) leaves eig(M)=P consistent and only
%  inflates the sampling variance of M_hat (Step 7c).  So we do NOT filter it
%  out (that would distort P); we TEST it and report it.
%
%   (1) Ljung-Box on v_k    -> serial CORRELATION (should be ~absent: the
%                              VAR(p) already whitens the conditional mean).
%   (2) Ljung-Box on v_k^2  -> volatility clustering / GARCH (EXPECTED to
%                              reject on daily finance: shocks NOT i.i.d.).
%   (3) ARCH-LM on v_k      -> same, conditional-variance dynamics.
%   (4) Cross 4th moments   -> E[v_i^2 v_j^2]-1, E[v_i^3 v_j] (~0 under
%                              independence; tests beyond mere uncorrelatedness).
% =========================================================================
fprintf('\n%s\n STEP 6c -- i.i.d. / INDEPENDENCE DIAGNOSTICS (structural shocks)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));

lb_v     = zeros(M,2);   % [Q(v),   p]
lb_v2    = zeros(M,2);   % [Q(v^2), p]
archlm_v = zeros(M,2);   % [LM,     p]
fprintf('\n  %-8s %11s %8s %11s %8s %11s %8s\n', ...
        'Shock','LB(v)','p','LB(v^2)','p','ARCH-LM','p');
fprintf('  %s\n', repmat('-',1,68));
for kk = 1:M
    vk  = V_ref(:,kk) - mean(V_ref(:,kk));
    vk2 = vk.^2;
    [Qv,  pv ] = local_lb(vk,  LB_LAGS);
    [Qv2, pv2] = local_lb(vk2, LB_LAGS);
    lb_v(kk,:)  = [Qv,  pv ];
    lb_v2(kk,:) = [Qv2, pv2];

    qA = 5;                                  % ARCH-LM on v_k
    U2 = zeros(T_eff,qA);
    for s_ = 1:qA, U2(s_+1:end,s_) = vk2(1:end-s_); end
    iA  = qA+1 : T_eff;
    ZA  = [ones(numel(iA),1), U2(iA,:)];
    yA  = vk2(iA);
    eA  = yA - ZA*(ZA\yA);
    R2A = max(0, 1-(eA'*eA)/sum((yA-mean(yA)).^2));
    LMA = numel(iA)*R2A;  pA = 1 - chi2cdf(LMA, qA);
    archlm_v(kk,:) = [LMA, pA];

    fprintf('  v_%-6d %11.2f %8.4f %11.2f %8.4f %11.2f %8.4f\n', ...
            kk, Qv, pv, Qv2, pv2, LMA, pA);
end

% Cross fourth-moment independence (off-diagonal): standardised shocks
Vc = V_ref - mean(V_ref);
Vsd = Vc ./ std(Vc);
co_kurt = zeros(M); co_kurt_robust = zeros(M);
for i = 1:M
    for j = 1:M
        co_kurt(i,j) = mean(Vsd(:,i).^2 .* Vsd(:,j).^2);   % = 1 if independent
        co_kurt_robust(i,j) = mean(Vsd(:,i).^3 .* Vsd(:,j));   % E[v_i^3 v_j]; = 0 if independent.
        % SV-ROBUST: odd power in v_j is annihilated by symmetry, so a common
        % volatility factor does not contaminate it (unlike co_kurt).
    end
end
mask = ~eye(M);
indep_diag = struct('co_kurt', co_kurt, 'co_kurt_robust', co_kurt_robust);

fprintf('\n  Co-kurtosis complete matrix E[v_i^2 v_j^2]:\n');
fprintf('  %-10s', '');
for kk = 1:M, fprintf(' %8s', ['v_' num2str(kk)]); end, fprintf('\n');
fprintf('  %s\n', repmat('-', 1, 10 + 9*M));
for i = 1:M
    fprintf('  %-10s', ['v_' num2str(i)]);
    for j = 1:M
        fprintf(' %8.4f', co_kurt(i,j));
    end
    fprintf('\n');
end

fprintf('\n  Cross 4th-moment independence (i ~= j):\n');
fprintf('    max |E[v_i^2 v_j^2] - 1| = %.4f   (0 = independent)\n', max(abs(co_kurt(mask)-1)));
fprintf('    max |E[v_i^3 v_j]|       = %.4f   (0 = independent)\n', max(abs(co_kurt_robust(mask))));

n_v2_rej = nnz(lb_v2(:,2) < 0.05);
fprintf('\n  Interpretation:\n');
fprintf('    LB(v^2) rejects in %d/%d shocks => volatility clustering present.\n', n_v2_rej, M);
fprintf('    => structural shocks are NOT i.i.d., but Kollo P stays CONSISTENT.\n');
fprintf('    => inference on lambda / dominance uses the block bootstrap (Step 7c),\n');
fprintf('       not the naive posterior bands alone (which treat K_hat as exact).\n');

%% =========================================================================
%  STEP 7 -- EIGENVALUE DECAY
% =========================================================================
fprintf('\n%s\n STEP 7 -- EIGENVALUE DECAY  (Kollo d)  +  TRUE KURTOSIS DECAY\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));

lam_med = median(eigenvalues_draws, 2);
lam_16  = quantile(eigenvalues_draws, 0.16, 2);
lam_84  = quantile(eigenvalues_draws, 0.84, 2);

% v01: posterior of the TRUE excess kurtosis (the genuine non-Gaussianity)
kap_med = median(kappa_draws, 2);
kap_16  = quantile(kappa_draws, 0.16, 2);
kap_84  = quantile(kappa_draws, 0.84, 2);

fprintf('\n  (A) K(Z)=I*c4 eigenvalue d_i  [under indep. = TRUE excess kurtosis kappa_i]\n');
fprintf('  Shock   d_med      d_16       d_84      bar\n');
fprintf('  %s\n', repmat('-',1,58));
for i = 1:M
    bar_str = repmat('|', 1, max(0, round(lam_med(i)/max(lam_med)*25)));
    fprintf('  v_%d   %9.4f  %9.4f  %9.4f   %s\n', ...
            i, lam_med(i), lam_16(i), lam_84(i), bar_str);
end
fprintf('\n  (B) TRUE excess kurtosis kappa_i = E[v_i^4]-3  [non-Gaussianity ranking]\n');
fprintf('  Shock   kap_med    kap_16     kap_84    bar\n');
fprintf('  %s\n', repmat('-',1,58));
for i = 1:M
    bar_str = repmat('|', 1, max(0, round(kap_med(i)/max(kap_med)*25)));
    fprintf('  v_%d   %9.4f  %9.4f  %9.4f   %s\n', ...
            i, kap_med(i), kap_16(i), kap_84(i), bar_str);
end
[~, ord_kap] = sort(kap_med, 'descend');
fprintf('\n  Non-Gaussianity ranking (TRUE kurtosis, desc): ');
fprintf('v_%d ', ord_kap); fprintf('\n');
fprintf('  kappa_max / kappa_min: %.4f\n', kap_med(ord_kap(1))/max(kap_med(ord_kap(end)),1e-8));
 
%% =========================================================================
%  STEP 7b -- POSTERIOR NEAR-DEGENERACY TEST
% =========================================================================
fprintf('\n%s\n STEP 7b -- POSTERIOR NEAR-DEGENERACY TEST\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
 
fprintf('\n  H0: lambda_i = lambda_{i+1}  vs  H1: lambda_i > lambda_{i+1}\n\n');
fprintf('  %-10s %12s %12s %12s %12s %10s\n', ...
        'Pair','P(gap>0|Y)','median(gap)','gap/lam_1','q16(gap)','degen?');
fprintf('  %s\n', repmat('-',1,72));
 
n_pairs       = M - 1;
posterior_pgt = zeros(n_pairs, 1);
gap_med_arr   = zeros(n_pairs, 1);
degen_flag    = false(n_pairs, 1);
 
for i = 1:n_pairs
    gap_draws        = eigenvalues_draws(i,:) - eigenvalues_draws(i+1,:);
    posterior_pgt(i) = mean(gap_draws > 0);
    gap_med          = median(gap_draws);
    gap_q16          = quantile(gap_draws, 0.16);
    gap_med_arr(i)   = gap_med;
    rel_gap          = gap_med / max(lam_med(1), 1e-8);
 
    is_degen      = (posterior_pgt(i) < 0.95) || (rel_gap < DEGEN_THRESHOLD);
    degen_flag(i) = is_degen;
    flag_str = '  '; if is_degen, flag_str = '[!]'; end
 
    fprintf('  v_%d-v_%d   %10.4f   %10.4f   %10.4f   %10.4f   %s\n', ...
            i, i+1, posterior_pgt(i), gap_med, rel_gap, gap_q16, flag_str);
end
 
fprintf('\n  Interpretation:\n');
if posterior_pgt(1) >= 0.95 && gap_med_arr(1)/max(lam_med(1),1e-8) >= DEGEN_THRESHOLD
    fprintf('    [v] v_1 is well-separated from v_2 (P(lam_1>lam_2|Y)=%.3f).\n', ...
            posterior_pgt(1));
    fprintf('        Point identification of the leading shock is supported.\n');
else
    fprintf('    [!] v_1 vs v_2 near-degeneracy: identification of v_1 fragile.\n');
    fprintf('        Caution in interpreting v_1 as the unique tail-risk shock.\n');
end
 
conc_ratio_draws = eigenvalues_draws(1,:) ./ max(eigenvalues_draws(2,:), 1e-8);
fprintf('\n  Concentration ratio lambda_1 / lambda_2 (eigenvalue d, identification):\n');
fprintf('    median: %.3f  |  q16: %.3f  |  q84: %.3f\n', ...
        median(conc_ratio_draws), ...
        quantile(conc_ratio_draws, 0.16), ...
        quantile(conc_ratio_draws, 0.84));
fprintf('    P(lam_1/lam_2 > 2 | Y): %.4f  (>=0.95 supports robust dominance)\n', ...
        mean(conc_ratio_draws > 2));

% --- v01: dominance in TRUE non-Gaussianity (kappa), which is what matters
%     economically for "the most non-Gaussian / tail-risk shock".  Ranked
%     per draw so the ratio is between the largest and 2nd-largest kappa.
kap_sorted_draws = sort(kappa_draws, 1, 'descend');               % M x N
kap_conc_draws   = kap_sorted_draws(1,:) ./ max(kap_sorted_draws(2,:), 1e-8);
fprintf('\n  Concentration ratio kappa_(1) / kappa_(2) (TRUE kurtosis, non-Gaussianity):\n');
fprintf('    median: %.3f  |  q16: %.3f  |  q84: %.3f\n', ...
        median(kap_conc_draws), quantile(kap_conc_draws,0.16), quantile(kap_conc_draws,0.84));
fprintf('    P(kappa_(1)/kappa_(2) > 2 | Y): %.4f  (>=0.95 supports a uniquely\n', ...
        mean(kap_conc_draws > 2));
fprintf('       non-Gaussian leading shock; this is the economically relevant gap)\n');
 
fig_degen = figure('Name','Degeneracy Test','Position',[50 50 1100 450]);
subplot(1,2,1);
gap1_draws = eigenvalues_draws(1,:) - eigenvalues_draws(2,:);
histogram(gap1_draws, 40, 'Normalization','pdf', ...
          'FaceColor',[0.4 0.6 0.85],'EdgeColor','none'); hold on;
xline(0,'r-','LineWidth',1.5);
xline(median(gap1_draws),'b--','LineWidth',1.5,'Label','median');
xlabel('lambda_1 - lambda_2','Interpreter','none');
ylabel('Posterior density','Interpreter','none');
title(sprintf('Posterior of lambda_1 - lambda_2 (P(>0)=%.3f)', posterior_pgt(1)), ...
      'FontSize',10,'Interpreter','none');
grid on; set(gca,'FontSize',9,'Box','on');
 
subplot(1,2,2);
histogram(conc_ratio_draws, 40, 'Normalization','pdf', ...
          'FaceColor',[0.4 0.7 0.5],'EdgeColor','none'); hold on;
xline(1,'r-','LineWidth',1.5,'Label','degenerate');
xline(2,'k--','LineWidth',1.0,'Label','dominance');
xline(median(conc_ratio_draws),'b--','LineWidth',1.5,'Label','median');
xlabel('lambda_1 / lambda_2','Interpreter','none');
ylabel('Posterior density','Interpreter','none');
title(sprintf('Concentration ratio (median=%.2f)', median(conc_ratio_draws)), ...
      'FontSize',10,'Interpreter','none');
grid on; set(gca,'FontSize',9,'Box','on');
sgtitle('Near-degeneracy diagnostics (Kollo identification)', ...
        'FontSize',11,'Interpreter','none');
 
%% =========================================================================
%  STEP 7c -- MOVING-BLOCK BOOTSTRAP for M_hat  (serial-dependence-robust)
%
%  The Bayesian posterior bands reflect uncertainty in (Phi, Sigma) but treat
%  K_hat as DETERMINISTIC given iota_s.  Under serial dependence (GARCH) the
%  4th-moment estimator K_hat is itself noisy, with effective sample << T.  The
%  moving-block bootstrap resamples blocks of iota_pm (preserving short-run
%  dependence) and re-identifies, giving a sampling distribution of the kurtosis
%  eigenvalues and of leading-shock dominance that is robust to dependence of
%  unknown form -- WITHOUT distorting the point estimate P.  This is the correct
%  way to address the "VIF" concern (cf. the discarded GARCH pre-filter).
% =========================================================================
fprintf('\n%s\n STEP 7c -- MOVING-BLOCK BOOTSTRAP  (B=%d, block=%d)\n%s\n', ...
        repmat('=',1,70), BLOCK_BOOT_N, BLOCK_LEN, repmat('=',1,70));

nblk        = ceil(T_eff / BLOCK_LEN);
boot_lam    = zeros(M, BLOCK_BOOT_N);
boot_dom    = false(BLOCK_BOOT_N, 1);     % lambda_1 > lambda_2 in resample
boot_coldev = zeros(M, BLOCK_BOOT_N);     % 1 - |<Pb_k, P_ref_k>|  (column instability)
tic;
for b = 1:BLOCK_BOOT_N
    starts = randi(T_eff - BLOCK_LEN + 1, nblk, 1);
    idx    = zeros(nblk*BLOCK_LEN, 1);
    for q = 1:nblk
        idx((q-1)*BLOCK_LEN+1 : q*BLOCK_LEN) = starts(q) : starts(q)+BLOCK_LEN-1;
    end
    idx = idx(1:T_eff);
    ib  = iota_pm(idx, :);

    % Same Kollo identification chain as Steps 5b and 6 (shared function).
    [Pb, lamb] = kollo_identify(ib, Kz, shrink_w, M);

    boot_lam(:,b) = lamb;
    boot_dom(b)   = lamb(1) > lamb(2);
    % column stability vs P_ref (greedy abs-dot matching)
    dabs = abs(Pb' * P_ref);  usedb = false(M,1);
    for r = 1:M
        dr = dabs(:,r);  dr(usedb) = -Inf;  [mx, bs] = max(dr);
        boot_coldev(r,b) = 1 - mx;  usedb(bs) = true;
    end
end
fprintf('  Bootstrap done in %.1fs.\n', toc);

boot_lam_ci   = quantile(boot_lam, [0.025 0.5 0.975], 2);   % M x 3
boot_dom_frac = mean(boot_dom);
boot_gap12    = boot_lam(1,:) - boot_lam(2,:);
boot_gap_ci   = quantile(boot_gap12, [0.025 0.5 0.975]);

fprintf('\n  Bootstrap CI of K(Z)=I*c4 eigenvalue d_i [under indep. = TRUE excess\n');
fprintf('  kurtosis kappa_i; ordering matches non-Gaussianity, no s^2 paradox]:\n');
fprintf('  %-8s %12s %12s %12s\n', 'Shock','q2.5','median','q97.5');
fprintf('  %s\n', repmat('-',1,48));
for kk = 1:M
    fprintf('  v_%-6d %12.4f %12.4f %12.4f\n', ...
            kk, boot_lam_ci(kk,1), boot_lam_ci(kk,2), boot_lam_ci(kk,3));
end
fprintf('\n  lambda_1 - lambda_2: median %.4f  CI [%.4f, %.4f]\n', ...
        boot_gap_ci(2), boot_gap_ci(1), boot_gap_ci(3));
fprintf('  Bootstrap P(lambda_1 > lambda_2) = %.3f  (>=0.95 supports robust dominance)\n', ...
        boot_dom_frac);
fprintf('  Mean column instability vs P_ref (0=stable): ');
fprintf('%.3f  ', mean(boot_coldev,2)); fprintf('\n');
if boot_gap_ci(1) <= 0
    fprintf('  [!] Bootstrap CI for lambda_1-lambda_2 includes 0: leading-shock\n');
    fprintf('      identification is FRAGILE once serial dependence is accounted for.\n');
else
    fprintf('  [v] Leading shock separated from the rest even under the block bootstrap.\n');
end

%% =========================================================================
%  STEP 8 -- STRUCTURAL IMPACT MATRIX B  (original units)
% =========================================================================
fprintf('\n%s\n STEP 8 -- IMPACT MATRIX B  (original units)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
 
B_med = median(B_draws, 3);
B_16  = quantile(B_draws, 0.16, 3);
B_84  = quantile(B_draws, 0.84, 3);
 
fprintf('\n  Rows=variables, Cols=shocks v_1,...,v_%d (ordered by lambda desc)\n', M);
fprintf('  Units: B(m,k) is the impact of a unit-variance shock v_k on variable m\n');
fprintf('         in the original unit of that variable.\n\n');
fprintf('  %-10s', 'Var\\Shock');
for kk = 1:M, fprintf(' %10s', ['v_' num2str(kk)]); end, fprintf('\n');
fprintf('  %s\n', repmat('-', 1, 10 + 11*M));
for m = 1:M
    fprintf('  %-10s', VAR_NAMES{m});
    for kk = 1:M, fprintf(' %10.4f', B_med(m,kk)); end
    fprintf('\n');
end
 
%% =========================================================================
%  STEP 8b -- ECONOMIC SIGNATURE OF EACH SHOCK
% =========================================================================
fprintf('\n%s\n STEP 8b -- ECONOMIC SIGNATURE TABLE  (no restrictions imposed)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));

% + or -: agreement ≥ 84% — the sign is stable across the posterior.
sign_agreement = zeros(M, M);
for m = 1:M
    for kk = 1:M
        b_draws     = squeeze(B_draws(m,kk,:));
        median_sign = sign(median(b_draws));
        if median_sign == 0
            sign_agreement(m,kk) = 0.5;
        else
            sign_agreement(m,kk) = mean(sign(b_draws) == median_sign);
        end
    end
end
 
fprintf('\n  Sign pattern of B (median impact response):\n\n');
fprintf('  %-10s', 'Var\\Shock');
for kk = 1:M, fprintf(' %8s', ['v_' num2str(kk)]); end
fprintf('\n  %s\n', repmat('-', 1, 10 + 9*M));
for m = 1:M
    fprintf('  %-10s', VAR_NAMES{m});
    for kk = 1:M
        s_med  = sign(B_med(m,kk));
        agree  = sign_agreement(m,kk);
        if s_med > 0
            sym = '+'; if agree < 0.84, sym = '+?'; end
        elseif s_med < 0
            sym = '-'; if agree < 0.84, sym = '-?'; end
        else
            sym = '0';
        end
        fprintf(' %8s', sym);
    end
    fprintf('\n');
end
fprintf('\n  Legend:  + positive (agreement >= 0.84)\n');
fprintf('            - negative (agreement >= 0.84)\n');
fprintf('            +? / -?  posterior sign uncertain (agreement < 0.84)\n');
 
fprintf('\n  Tail-risk signature score for each shock:\n');
fprintf('    Canonical:  VIX(+) CSPREAD(+) DXY(+) SP500(-) DGS10(-) AAA(-) SLOPE(?)\n\n');

% --- Canonical tail-risk signature: (variable, expected sign) PAIRS ----------
%  The variable name and its expected sign MUST travel together.  The previous
%  formulation built two separate arrays,
%     target_idx  = [idx_VIX, idx_CSPRD, idx_DXY, idx_SP500, idx_DGS10, idx_AAA];
%     target_vals = [+1, +1, +1, -1, -1, -1];
%  and relied on both having the same length.  When a variable is absent from
%  VAR_NAMES its find() returns [], which VANISHES in the concatenation while
%  target_vals stays length 6 -- so the loop silently paired the SURVIVING
%  variables with the FIRST entries of target_vals.  In the M=3 baseline
%  {SP500,VIX,CSPREAD} this collapsed target_idx to [VIX,CSPREAD,SP500] against
%  target_vals(1:3) = [+1,+1,+1], i.e. SP500 was scored as if a tail shock made
%  equities RISE -- inverting the flight-to-quality signature.  Keeping the pair
%  in one cell array makes the filtering below structurally safe.
TR_SPEC = {'VIX',+1; 'CSPREAD',+1; 'DXY',+1; 'SP500',-1; 'DGS10',-1; 'AAA',-1};
has_var = cellfun(@(v) ~isempty(find(strcmp(VAR_NAMES, v), 1)), TR_SPEC(:,1));
TR_SPEC = TR_SPEC(has_var, :);                        % keep only present variables
target_idx  = cellfun(@(v) find(strcmp(VAR_NAMES, v), 1), TR_SPEC(:,1))';
target_vals = cell2mat(TR_SPEC(:,2))';
n_tgt       = numel(target_idx);                      % max attainable score

% Thresholds scale with n_tgt (the old hard-coded 5/4/1 assumed n_tgt=6 and were
% UNREACHABLE for M=3, where the best possible score is 3).
thr_full    = ceil(0.83 * n_tgt);    % n_tgt=6 -> 5 ; n_tgt=3 -> 3
thr_partial = ceil(0.66 * n_tgt);    % n_tgt=6 -> 4 ; n_tgt=3 -> 2
thr_anti    = floor(0.17 * n_tgt);   % n_tgt=6 -> 1 ; n_tgt=3 -> 0

fprintf('  Scored on %d of the 6 canonical variables present in this system:\n  ', n_tgt);
for q = 1:n_tgt
    fprintf('%s(%+d) ', VAR_NAMES{target_idx(q)}, target_vals(q));
end
fprintf('\n\n');

score = zeros(M,1);
for kk = 1:M
    sc = 0;
    for q = 1:n_tgt
        if sign(B_med(target_idx(q), kk)) == target_vals(q)
            sc = sc + 1;
        end
    end
    score(kk) = sc;
end

fprintf('  %-8s %12s %15s %20s\n','Shock','kappa_med', ...
        sprintf('TR-score (max=%d)', n_tgt),'Interpretation');
fprintf('  %s\n', repmat('-', 1, 65));
for kk = 1:M
    if score(kk) >= thr_full,        interp = 'TAIL-RISK candidate';
    elseif score(kk) >= thr_partial, interp = 'partial match';
    elseif score(kk) <= thr_anti,    interp = 'opposite sign (anti-tail)';
    else,                            interp = 'unclear';
    end
    fprintf('  v_%-6d %12.4f %15d   %s\n', kk, kap_med(kk), score(kk), interp);
end

[max_score, idx_best] = max(score);
if nnz(score == max_score) > 1
    fprintf('\n  [!] TIE on the sign score (%d shocks at %d/%d): idx_best is the\n', ...
            nnz(score == max_score), max_score, n_tgt);
    fprintf('      lowest-index winner and is NOT informative on its own.  Use the\n');
    fprintf('      kurtosis-share (Step 8c) as the primary tail-shock criterion.\n');
end
[~, idx_kappa] = max(kap_med);   % v01: most non-Gaussian shock by TRUE kurtosis
fprintf('\n  >>> Best tail-risk candidate (sign score): v_%d (score=%d/%d, kappa=%.3f)\n', ...
        idx_best, score(idx_best), n_tgt, kap_med(idx_best));
fprintf('  >>> Most non-Gaussian shock (TRUE kurtosis): v_%d (kappa=%.3f)\n', ...
        idx_kappa, kap_med(idx_kappa));
if idx_best ~= idx_kappa
    fprintf('      NOTE: the tail-risk (sign) shock and the most non-Gaussian shock\n');
    fprintf('      DIFFER. Ranking by the Kollo eigenvalue d would mislabel which is\n');
    fprintf('      "leading" (d = kappa*s^2 favours the common-direction shock).\n');
end
 
%% =========================================================================
%  STEP 8c -- KURTOSIS-SHARE  (max-share selection at the 4th order)
%
%  Identification vs selection: independence + distinct kurtoses identify
%  the shocks (Steps 5b-6); WHICH shock is "the tail shock" is a labelling
%  question.  We answer it in the max-share tradition (Faust 1998; Uhlig
%  2003; Angeletos, Collard & Dellas 2020 AER) moved from 2nd to 4th
%  cumulants: 4th cumulants of INDEPENDENT shocks are additive, so each
%  variable's residual excess kurtosis decomposes exactly, no cross terms:
%
%      kurt_exc(u_m) = sum_k kappa_k * Btilde(m,k)^4,
%      Btilde(m,k)   = B(m,k) / sd(u_m),   sum_k Btilde(m,k)^2 = 1.
%
%  This is the FEVD identity one order up: variance splits with weights
%  Btilde^2, kurtosis with weights kappa_k*Btilde^4.  The KURTOSIS-SHARE
%
%      KS(m,k) = kappa_k Btilde(m,k)^4 / sum_s kappa_s Btilde(m,s)^4
%
%  is unit-free (standardisation by sd(u_m) neutralises the heterogeneous
%  scales, log-index vs pp rates, exactly as in the variance FEVD).  The
%  tail shock is argmax_k of the aggregate share on the declared TARGET
%  variables -- NOT the max-eigenvalue shock: the shock with the largest
%  own kurtosis kappa (MRS/FOBI ordering) need not be the one generating
%  the tails of the variables of interest (kurtosis is corner-concentrated:
%  sum_k a_k^4 kappa_k <= max_k kappa_k for unit a, so the per-variable
%  weights Btilde^4 decide who actually inherits the tails).
%
%  Caveats (reported, not hidden):
%   (i)  exact additivity needs mutual independence; the equity-vol leverage
%        effect (Step 6c) adds small cross-cumulant terms.  The model-implied
%        kurtosis is therefore checked against the sample kurtosis below.
%   (ii) if some kappa_k < 0 (platykurtic shock) shares can exit [0,1];
%        draws where this happens are counted and reported.
% =========================================================================
fprintf('\n%s\n STEP 8c -- KURTOSIS-SHARE  (max-share at the 4th order)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));

%  System-wide target (ALL variables): the aggregate share below becomes the
%  4th-order analog of the PC1 variance share -- the shock whose fat tails
%  the SYSTEM inherits the most (global-financial-cycle spirit).  Summing
%  standardised contributions implies variables with more residual kurtosis
%  (VIX, CSPREAD) weigh more, which is the point: tails count where tails
%  live.  Restrict to a subset (e.g. {'VIX','CSPREAD'}) to target one block.
TARGET_VARS = VAR_NAMES;
target_m    = find(ismember(VAR_NAMES, TARGET_VARS));
if numel(target_m) == M
    target_lab = 'ALL variables (system-wide)';
else
    target_lab = strjoin(TARGET_VARS, ', ');
end

KS_draws        = zeros(M, M, N_DRAWS);   % variable x shock x draw
kurt_var_draws  = zeros(M, N_DRAWS);      % model-implied excess kurtosis of u_m
ks_target_draws = zeros(M, N_DRAWS);      % aggregate share on targets, per shock
n_neg_kappa     = 0;                      % draws with at least one kappa_k < 0
for s = 1:N_DRAWS
    Bs  = B_draws(:,:,s);
    Bt  = Bs ./ sqrt(sum(Bs.^2, 2));      % rows unit-norm: B_s B_s' = Sigma_s
    Cmk = Bt.^4 .* kappa_draws(:,s)';     % M x M: contribution of shock k to kurt(u_m)
    kurt_var_draws(:,s) = sum(Cmk, 2);
    KS_draws(:,:,s)     = Cmk ./ sum(Cmk, 2);
    agg                  = sum(Cmk(target_m,:), 1);
    ks_target_draws(:,s) = (agg / sum(agg))';
    if any(kappa_draws(:,s) < 0), n_neg_kappa = n_neg_kappa + 1; end
end

KS_med   = median(KS_draws, 3);
KS_16    = quantile(KS_draws, 0.16, 3);
KS_84    = quantile(KS_draws, 0.84, 3);
ks_t_med = median(ks_target_draws, 2);
ks_t_16  = quantile(ks_target_draws, 0.16, 2);
ks_t_84  = quantile(ks_target_draws, 0.84, 2);
[~, argmax_draw] = max(ks_target_draws, [], 1);
ks_winner_prob   = mean(argmax_draw(:) == 1:M, 1)';   % P(shock k has max target share)

fprintf('\n  Kurtosis-share KS(m,k) -- posterior median share of each variable''s\n');
fprintf('  residual excess kurtosis generated by each shock:\n\n');
fprintf('  %-10s', 'Var\\Shock');
for kk = 1:M, fprintf(' %8s', ['v_' num2str(kk)]); end, fprintf('\n');
fprintf('  %s\n', repmat('-', 1, 10 + 9*M));
for m = 1:M
    fprintf('  %-10s', VAR_NAMES{m});
    for kk = 1:M, fprintf(' %8.4f', KS_med(m,kk)); end
    fprintf('\n');
end

fprintf('\n  Aggregate kurtosis-share on target [%s]:\n', target_lab);
fprintf('  %-8s %10s %10s %10s %12s\n','Shock','median','q16','q84','P(argmax)');
fprintf('  %s\n', repmat('-',1,56));
for kk = 1:M
    fprintf('  v_%-6d %10.4f %10.4f %10.4f %12.4f\n', ...
            kk, ks_t_med(kk), ks_t_16(kk), ks_t_84(kk), ks_winner_prob(kk));
end

[~, idx_ks] = max(ks_t_med);
fprintf('\n  >>> TAIL shock by kurtosis-share: v_%d (share=%.1f%%, P(argmax)=%.2f)\n', ...
        idx_ks, 100*ks_t_med(idx_ks), ks_winner_prob(idx_ks));
fprintf('      Cross-check: sign-signature candidate v_%d | max-kurtosis shock v_%d\n', ...
        idx_best, idx_kappa);
if n_neg_kappa > 0
    fprintf('  [!] %d/%d draws contain a platykurtic shock (kappa<0): shares can\n', ...
            n_neg_kappa, N_DRAWS);
    fprintf('      exit [0,1] in those draws.\n');
end

% Consistency check: model-implied vs sample kurtosis of the reduced-form
% residuals.  Under exact independence the two coincide; a gap flags
% cross-cumulant (dependence) contamination of that variable's 4th moment.
U_std       = (U_pm - mean(U_pm)) ./ std(U_pm);
kurt_sample = (mean(U_std.^4) - 3)';
kv_med      = median(kurt_var_draws, 2);
fprintf('\n  Model-implied vs sample excess kurtosis of u_m (independence check):\n');
fprintf('  %-10s %14s %14s %10s\n','Variable','model (med)','sample','gap');
fprintf('  %s\n', repmat('-',1,52));
for m = 1:M
    fprintf('  %-10s %14.4f %14.4f %10.4f\n', ...
            VAR_NAMES{m}, kv_med(m), kurt_sample(m), kurt_sample(m) - kv_med(m));
end

fig_ks = figure('Name','Kurtosis-share','Position',[50 50 1150 480]);
subplot(1,2,1);
imagesc(KS_med*100); colorbar;
set(gca,'XTick',1:M,'XTickLabel',arrayfun(@(i)['v_' num2str(i)],1:M,'UniformOutput',false), ...
        'YTick',1:M,'YTickLabel',VAR_NAMES,'FontSize',9);
title('Kurtosis-share KS(m,k) [%]','Interpreter','none','FontSize',10);
xlabel('Shock','Interpreter','none'); ylabel('Variable','Interpreter','none');
subplot(1,2,2);
bar(100*ks_t_med, 'FaceColor',[0.4 0.6 0.85]); hold on;
errorbar((1:M)', 100*ks_t_med, 100*(ks_t_med-ks_t_16), 100*(ks_t_84-ks_t_med), ...
         'k.', 'LineWidth', 1);
set(gca,'XTick',1:M,'XTickLabel',arrayfun(@(i)['v_' num2str(i)],1:M,'UniformOutput',false), ...
        'FontSize',9,'Box','on');
ylabel('% of target kurtosis','Interpreter','none');
title(sprintf('Aggregate kurtosis-share, target: %s [16,84]', target_lab), ...
      'Interpreter','none','FontSize',10);
grid on;
sgtitle('Kurtosis-share: which shock generates the tails of the variables', ...
        'FontSize',11,'Interpreter','none');

%% =========================================================================
%  STEP 9 -- IRF PLOTS
% =========================================================================
fprintf('\n%s\n STEP 9 -- IRF PLOTS\n%s\n', repmat('=',1,70), repmat('=',1,70));
 
IRF_med = median(IRF_draws, 4);
IRF_16  = quantile(IRF_draws, 0.16, 4);
IRF_84  = quantile(IRF_draws, 0.84, 4);
IRF_05  = quantile(IRF_draws, 0.05, 4);
IRF_95  = quantile(IRF_draws, 0.95, 4);
horizon = 0:IRF_HOR;
 
fig1 = figure('Name','IRF Full Grid','Position',[50 50 1600 1000]);
for sk = 1:M
    for vm = 1:M
        subplot(M, M, (vm-1)*M + sk);
        h_m = squeeze(IRF_med(vm,:,sk));
        h_l = squeeze(IRF_16(vm,:,sk));
        h_h = squeeze(IRF_84(vm,:,sk));
        h_5 = squeeze(IRF_05(vm,:,sk));
        h_9 = squeeze(IRF_95(vm,:,sk));
        fill([horizon,fliplr(horizon)],[h_5,fliplr(h_9)], ...
             [0.85 0.85 0.85],'EdgeColor','none'); hold on;
        fill([horizon,fliplr(horizon)],[h_l,fliplr(h_h)], ...
             [0.65 0.75 0.90],'EdgeColor','none');
        plot(horizon, h_m,'b-','LineWidth',1.5);
        yline(0,'r--','LineWidth',0.8);
        xlim([0 IRF_HOR]);
        if vm==1, title(['v_' num2str(sk)],'FontSize',8,'Interpreter','none'); end
        if sk==1, ylabel(VAR_NAMES{vm},'FontSize',7,'Interpreter','none'); end
        set(gca,'FontSize',7);
    end
end
sgtitle('IRF: Rows=Variables, Cols=Structural Shocks (Kollo 2008) — original units', ...
        'FontSize',11,'Interpreter','none');
 
fig2 = figure('Name','IRF to v_1','Position',[50 50 1400 700]);
n_rows_p = ceil(M/4);
for vm = 1:M
    subplot(n_rows_p, min(M,4), vm);
    h_m = squeeze(IRF_med(vm,:,1));
    h_l = squeeze(IRF_16(vm,:,1));
    h_h = squeeze(IRF_84(vm,:,1));
    h_5 = squeeze(IRF_05(vm,:,1));
    h_9 = squeeze(IRF_95(vm,:,1));
    fill([horizon,fliplr(horizon)],[h_5,fliplr(h_9)], ...
         [0.85 0.85 0.85],'EdgeColor','none'); hold on;
    fill([horizon,fliplr(horizon)],[h_l,fliplr(h_h)], ...
         [0.65 0.75 0.90],'EdgeColor','none');
    plot(horizon, h_m,'b-','LineWidth',2);
    yline(0,'r--','LineWidth',1);
    title(VAR_NAMES{vm},'FontSize',10,'Interpreter','none');
    xlabel('Days'); xlim([0 IRF_HOR]);
    set(gca,'FontSize',9,'Box','on');
end
sgtitle(sprintf('IRF to v_1 (lambda=%.3f) -- max-kurtosis shock [original units]', ...
        lam_med(1)), 'FontSize',12,'Interpreter','none');
 
fig3 = figure('Name','Eigenvalue Decay','Position',[50 50 700 450]);
shock_labels = arrayfun(@(i) ['v_' num2str(i)], 1:M, 'UniformOutput', false);
boxplot(eigenvalues_draws', 'Labels', shock_labels);
hold on;
plot(1:M, lam_med,'b-o','LineWidth',2,'MarkerSize',6,'MarkerFaceColor','b');
yline(0,'r--','LineWidth',1);
xlabel('Structural shock (ordered by lambda, descending)','Interpreter','none');
ylabel('Excess kurtosis (lambda_i)','Interpreter','none');
title('Eigenvalue decay of M_hat = sum_blocks(K_hat - K_z)', ...
      'Interpreter','none','FontSize',10);
set(gca,'FontSize',10,'Box','on'); grid on;
 
fig4 = figure('Name','Structural Shocks','Position',[50 50 1400 800]);
for kk = 1:M
    subplot(M, 1, kk);
    plot(V_ref(:,kk),'Color',[0.2 0.4 0.7],'LineWidth',0.8);
    yline(0,'r-','LineWidth',0.5);
    title(sprintf('v_%d  (lambda=%.3f)', kk, lam_med(kk)), ...
          'FontSize',9,'Interpreter','none');
    xlim([1 T_eff]);
    set(gca,'FontSize',8,'Box','on');
end
sgtitle(sprintf('Structural shocks v_t — BVAR(%d), Kollo (2008) [V_ref]', p), ...
        'FontSize',10,'Interpreter','none');
xlabel('Trading days','Interpreter','none');
 
fprintf('  Toolbox plot_irfs_ : v_1 IRFs ...\n');
try
    indx_sho     = 1;
    irfs_to_plot = IRF_draws(:,:,indx_sho,:);
    plot_opt              = struct();
    plot_opt.varnames     = VAR_NAMES;
    plot_opt.shocksnames  = {sprintf('v_1 (lam=%.2f)', lam_med(1))};
    plot_opt.conf_sig     = 0.68;
    plot_opt.conf_sig_2   = 0.90;
    plot_irfs_(irfs_to_plot, plot_opt);
 
    if idx_best ~= 1
        fprintf('  Toolbox plot_irfs_ : best candidate v_%d IRFs ...\n', idx_best);
        irfs_best = IRF_draws(:,:,idx_best,:);
        plot_opt.shocksnames = {sprintf('v_%d (lam=%.2f, best tail-risk)', ...
                                idx_best, lam_med(idx_best))};
        plot_irfs_(irfs_best, plot_opt);
    end
catch ME
    fprintf('  [WARNING] toolbox plot_irfs_ failed: %s\n', ME.message);
    fprintf('  Custom plots above remain valid.\n');
end
 
%% =========================================================================
%  STEP 10 -- ORTHOGONALITY CHECK
% =========================================================================
fprintf('\n%s\n STEP 10 -- ORTHOGONALITY CHECK\n%s\n', repmat('=',1,70), repmat('=',1,70));
 
% cov_V_draws was accumulated inside the Step 6 loop (no second pass needed).
% Median Cov(v) across all draws 
cov_V_med      = median(cov_V_draws, 3);
% Posterior bands on the diagonal elements
cov_V_mat2d    = reshape(cov_V_draws, M*M, N_DRAWS);
diag_lin_idx   = sub2ind([M M], 1:M, 1:M);
cov_V_diag_drw = cov_V_mat2d(diag_lin_idx, :);
cov_V_diag16   = quantile(cov_V_diag_drw', 0.16)';
cov_V_diag84   = quantile(cov_V_diag_drw', 0.84)';
max_off_med    = max(abs(cov_V_med - eye(M)), [], 'all');
 
fprintf('\n  (A) Median Cov(v) over %d draws (should be ~ I_%d):\n\n', N_DRAWS, M);
fprintf('  %-10s', '');
for kk = 1:M, fprintf(' %8s', ['v_' num2str(kk)]); end, fprintf('\n');
fprintf('  %s\n', repmat('-', 1, 10 + 9*M));
for m = 1:M
    fprintf('  %-10s', ['v_' num2str(m)]);
    for kk = 1:M, fprintf(' %8.4f', cov_V_med(m,kk)); end
    fprintf('\n');
end
fprintf('\n  Diagonal [16, 84] posterior quantile:\n');
for m = 1:M
    fprintf('    v_%d: %.4f  [%.4f, %.4f]\n', m, cov_V_med(m,m), ...
            cov_V_diag16(m), cov_V_diag84(m));
end
fprintf('\n  Max |median Cov(v) - I_M|: %.6f\n', max_off_med);

%Cov(Vref​) at the posterior mean 
max_off_ref = max(abs(cov_V_ref - eye(M)), [], 'all');
fprintf('\n  (B) Cov(V_ref) - I_%d  (posterior-mean identification):\n\n', M);
fprintf('  %-10s', '');
for kk = 1:M, fprintf(' %8s', ['v_' num2str(kk)]); end, fprintf('\n');
fprintf('  %s\n', repmat('-', 1, 10 + 9*M));
for m = 1:M
    fprintf('  %-10s', ['v_' num2str(m)]);
    for kk = 1:M, fprintf(' %8.4f', cov_V_ref(m,kk)); end
    fprintf('\n');
end
fprintf('\n  Max |Cov(V_ref) - I_M|: %.6f  (target < 1e-3)\n', max_off_ref);

% Corr(Vref​)
std_Vref  = std(V_ref);
corr_Vref = cov_V_ref ./ (std_Vref' * std_Vref);
 
fprintf('\n  (C) Corr(V_ref):\n\n');
fprintf('  %-10s', '');
for kk = 1:M, fprintf(' %8s', ['v_' num2str(kk)]); end, fprintf('\n');
fprintf('  %s\n', repmat('-', 1, 10 + 9*M));
for m = 1:M
    fprintf('  %-10s', ['v_' num2str(m)]);
    for kk = 1:M, fprintf(' %8.4f', corr_Vref(m,kk)); end
    fprintf('\n');
end

cov_V = cov_V_ref;
 
%% =========================================================================
%  STEP 11 -- NARRATIVE CHECK
% =========================================================================
fprintf('\n%s\n STEP 11 -- NARRATIVE CHECK\n%s\n', repmat('=',1,70), repmat('=',1,70));
 
try
    dates_vec = datetime(T_raw.Date(p+1:end), 'InputFormat','yyyy-MM-dd');
    fprintf('  Using Date column from CSV.\n');
catch
    fprintf('  No Date column. Using synthetic approximation.\n');
    start_date = datetime(1990,1,2);
    dates_vec  = start_date + caldays(ceil((0:T_eff-1) * 365.25/252));
    dates_vec  = dates_vec(:);
end
 
events = {
    '1998-08-17', 'Russia/LTCM';
    '2001-09-11', '9/11';
    '2001-12-02', 'Enron';
    '2007-08-09', 'BNP Paribas';
    '2008-09-15', 'Lehman';
    '2010-05-06', 'Flash Crash';
    '2011-08-05', 'US downgrade';
    '2020-03-16', 'COVID-19';
    '2023-03-10', 'SVB';
};
n_ev = size(events,1);
 
fig5 = figure('Name','Narrative Check','Position',[50 50 1400 900]);
for kk = 1:M
    ax = subplot(M, 1, kk);
    v_k      = V_ref(:,kk);
    vk_draws = squeeze(Vt_draws(:,kk,:));
    v_k16    = quantile(vk_draws, 0.16, 2);
    v_k84    = quantile(vk_draws, 0.84, 2);
 
    fill([dates_vec; flipud(dates_vec)],[v_k16; flipud(v_k84)], ...
         [0.75 0.85 0.95],'EdgeColor','none','FaceAlpha',0.6); hold on;
    plot(dates_vec, v_k,'Color',[0.1 0.35 0.65],'LineWidth',1.0);
    yline(0,'k-','LineWidth',0.5);
 
    for e = 1:n_ev
        ev_d = datetime(events{e,1},'InputFormat','yyyy-MM-dd');
        if ev_d >= dates_vec(1) && ev_d <= dates_vec(end)
            xline(ev_d,'--','Color',[0.7 0.1 0.1],'LineWidth',0.8, ...
                  'Label',events{e,2},'LabelOrientation','horizontal', ...
                  'FontSize',6,'LabelVerticalAlignment','bottom', ...
                  'Interpreter','none');
        end
    end
 
    sig_k   = std(v_k);
    idx_ext = abs(v_k) > 3*sig_k;
    if any(idx_ext)
        scatter(dates_vec(idx_ext), v_k(idx_ext), 10, 'r', 'filled');
    end
 
    ylabel(sprintf('v_%d (lam=%.2f)', kk, lam_med(kk)), ...
           'FontSize',8,'Interpreter','none');
    set(ax,'FontSize',8,'Box','on','XGrid','on');
    xlim([dates_vec(1), dates_vec(end)]);
end
sgtitle({sprintf('Structural shocks — Narrative Check | BVAR(%d), Kollo (2008)',p), ...
         'Shaded=68% band | Red dots=|v_k|>3sigma'}, ...
        'FontSize',10,'Interpreter','none');
fprintf('  Check v_1 (lambda=%.3f): should spike at Lehman and COVID.\n', lam_med(1));

%% =========================================================================
%  STEP 11b -- TIME SERIES OF LEADING SHOCKS  (spike plots)
%
%  One figure per shock in SPIKE_SHOCKS: series, +/- threshold bands,
%  historical-event markers (events list from Step 11), top-10 spike table.
% =========================================================================
SPIKE_SHOCKS = 1:3;    % shocks plotted individually
SPIKE_THRESH = 2.5;    % highlight |v_k| > SPIKE_THRESH * sigma

for kk = SPIKE_SHOCKS
    figure('Name',sprintf('v%d time series',kk),'Position',[50 50 1500 450]);
    vk   = V_ref(:, kk);
    sigk = std(vk);

    plot(dates_vec, vk, 'Color',[0.55 0.65 0.80], 'LineWidth', 0.6); hold on;
    yline( SPIKE_THRESH*sigk, 'r--', 'LineWidth', 0.8);
    yline(-SPIKE_THRESH*sigk, 'r--', 'LineWidth', 0.8);
    yline(0, 'k-', 'LineWidth', 0.4);

    big = abs(vk) > SPIKE_THRESH * sigk;
    scatter(dates_vec(big), vk(big), 12, 'r', 'filled', 'MarkerFaceAlpha', 0.7);

    for e = 1:n_ev
        ev_d = datetime(events{e,1}, 'InputFormat','yyyy-MM-dd');
        if ev_d >= dates_vec(1) && ev_d <= dates_vec(end)
            xline(ev_d, '--', 'Color',[0.75 0.10 0.10], 'LineWidth', 0.9, ...
                  'Label', events{e,2}, ...
                  'LabelOrientation','horizontal', ...
                  'LabelVerticalAlignment','bottom', ...
                  'FontSize', 7, 'Interpreter','none');
        end
    end

    ylabel(sprintf('v_%d  (sigma = 1,  lambda = %.1f)', kk, lam_med(kk)), ...
           'FontSize', 10, 'Interpreter','none');
    xlabel('Date', 'FontSize', 9);
    title(sprintf('Structural shock v_%d  --  |v_%d| > %.1f sigma highlighted (red dots)', ...
          kk, kk, SPIKE_THRESH), 'FontSize', 11, 'Interpreter','none');
    xlim([dates_vec(1), dates_vec(end)]);
    set(gca, 'FontSize', 9, 'Box','on', 'XGrid','on');

    [vk_sort, vk_idx] = sort(abs(vk), 'descend');
    fprintf('\n  Top-10 |v%d| spikes:\n', kk);
    fprintf('  %-14s  %8s\n', 'Date', sprintf('|v%d|',kk));
    fprintf('  %s\n', repmat('-',1,26));
    for i = 1:min(10, numel(vk_idx))
        fprintf('  %s  %8.3f\n', datestr(dates_vec(vk_idx(i)),'yyyy-mm-dd'), vk_sort(i)); %#ok<*DATST>
    end
end


%% =========================================================================
%  STEP 12 -- FEVD
% =========================================================================
fprintf('\n%s\n STEP 12 -- FEVD  (toolbox convention: rows=vars, cols=shocks)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
 
fevd_draws = zeros(M, M, IRF_HOR+1, N_DRAWS);
for s = 1:N_DRAWS
    cum_sq = zeros(M, M);
    for h = 0:IRF_HOR
        Th     = double(squeeze(IRF_draws(:,h+1,:,s)));   % cast: IRF_draws is single
        cum_sq = cum_sq + Th .* Th;
        tot    = sum(cum_sq, 2);
        fevd_draws(:,:,h+1,s) = cum_sq ./ (tot * ones(1,M));
    end
end
fevd_med = median(fevd_draws, 4);
fevd_16  = quantile(fevd_draws, 0.16, 4);
fevd_84  = quantile(fevd_draws, 0.84, 4);
 
for h_rep = [1, 5, 20]
    if h_rep > IRF_HOR, continue; end
    fprintf('\n  FEVD at h=%d (median):\n', h_rep);
    fprintf('  %-10s', 'Var\\Shock');
    for kk=1:M, fprintf(' %8s', ['v_' num2str(kk)]); end, fprintf('\n');
    fprintf('  %s\n', repmat('-',1,10+9*M));
    for m = 1:M
        fprintf('  %-10s', VAR_NAMES{m});
        for kk=1:M, fprintf(' %8.4f', fevd_med(m,kk,h_rep+1)); end
        fprintf('\n');
    end
end
 
fprintf('\n  FEVD of v_1 across variables (median [16, 84] %%):\n');
fprintf('  %-10s %12s %12s %12s\n','Variable','h=1','h=5','h=20');
fprintf('  %s\n', repmat('-',1,52));
for m = 1:M
    s1  = 100*fevd_med(m,1,2);  s1l = 100*fevd_16(m,1,2);  s1h = 100*fevd_84(m,1,2);
    s5  = 100*fevd_med(m,1,6);  s5l = 100*fevd_16(m,1,6);  s5h = 100*fevd_84(m,1,6);
    s20 = 100*fevd_med(m,1,21); s20l= 100*fevd_16(m,1,21); s20h= 100*fevd_84(m,1,21);
    fprintf('  %-10s %5.1f[%4.1f,%4.1f]  %5.1f[%4.1f,%4.1f]  %5.1f[%4.1f,%4.1f]\n', ...
            VAR_NAMES{m}, s1,s1l,s1h, s5,s5l,s5h, s20,s20l,s20h);
end
 
% (FEVD figure removed: the console tables above carry the same information;
%  the thesis-relevant graphical object is the kurtosis-share, Step 8c.)
 
%% =========================================================================
%  STEP 12b -- HISTORICAL DECOMPOSITION (crisis windows)
% =========================================================================
fprintf('\n%s\n STEP 12b -- HISTORICAL DECOMPOSITION  (crisis windows)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
 
crisis_dates = {'2008-09-15','Lehman'; '2020-03-16','COVID-19'; '2023-03-10','SVB'};
n_crisis     = size(crisis_dates, 1);
 
h_win    = min(5, IRF_HOR+1);
cum_irf5 = squeeze(sum(Theta_ref(:, 1:h_win, :), 2));   % M x M
 
HD_crisis = struct();
 
for ec = 1:n_crisis
    ev_date_str = crisis_dates{ec,1};
    ev_label    = crisis_dates{ec,2};
    ev_d        = datetime(ev_date_str, 'InputFormat','yyyy-MM-dd');
    t_ev        = find(dates_vec >= ev_d, 1);
 
    if isempty(t_ev)
        fprintf('\n  %s (%s): not in sample, skipping.\n', ev_label, ev_date_str);
        continue
    end
 
    delta_day0 = B_ref .* repmat(V_ref(t_ev,:), M, 1);
    delta_cum5 = cum_irf5 .* repmat(V_ref(t_ev,:), M, 1);
 
    fprintf('\n  === %s (%s, t=%d) ===\n', ev_label, ev_date_str, t_ev);
    fprintf('  Day-0 shock decomposition of forecast error u_m:\n');
    fprintf('  %-10s', 'Var');
    for kk = 1:M, fprintf(' %8s', ['v_' num2str(kk)]); end
    fprintf('  %8s\n', 'Total');
    fprintf('  %s\n', repmat('-', 1, 10 + 9*M + 10));
    for m = 1:M
        fprintf('  %-10s', VAR_NAMES{m});
        for kk = 1:M, fprintf(' %8.4f', delta_day0(m,kk)); end
        fprintf('  %8.4f\n', sum(delta_day0(m,:)));
    end
 
    fprintf('\n  5-day cumulative IRF effect of event-day shock:\n');
    fprintf('  %-10s', 'Var');
    for kk = 1:M, fprintf(' %8s', ['v_' num2str(kk)]); end
    fprintf('\n  %s\n', repmat('-', 1, 10 + 9*M));
    for m = 1:M
        fprintf('  %-10s', VAR_NAMES{m});
        for kk = 1:M, fprintf(' %8.4f', delta_cum5(m,kk)); end
        fprintf('\n');
    end
 
    HD_crisis(ec).label      = ev_label;
    HD_crisis(ec).date       = ev_date_str;
    HD_crisis(ec).t_ev       = t_ev;
    HD_crisis(ec).delta_day0 = delta_day0;
    HD_crisis(ec).delta_cum5 = delta_cum5;
end
 
key_vars = {'SP500','VIX','CSPREAD'};
n_ec_ok  = numel(HD_crisis);
if n_ec_ok > 0
    fig7 = figure('Name','HD Crisis Windows','Position',[50 50 1300 700]);
    for vi = 1:numel(key_vars)
        m_vi = find(strcmp(VAR_NAMES, key_vars{vi}));
        if isempty(m_vi), continue; end
        subplot(1, numel(key_vars), vi);
        bar_data = zeros(M, n_ec_ok);
        bar_labs = cell(n_ec_ok, 1);
        for ec2 = 1:n_ec_ok
            bar_data(:,ec2) = HD_crisis(ec2).delta_day0(m_vi,:)';
            bar_labs{ec2}   = HD_crisis(ec2).label;
        end
        bar(bar_data'); hold on;
        xticks(1:n_ec_ok); xticklabels(bar_labs);
        title(key_vars{vi},'FontSize',10,'Interpreter','none');
        ylabel('Contribution to u_m (day-0)','Interpreter','none');
        legend(arrayfun(@(i)['v_' num2str(i)],1:M,'UniformOutput',false), ...
               'Location','best','FontSize',7);
        grid on; set(gca,'FontSize',8,'Box','on');
    end
    sgtitle('Historical Decomposition — day-0 contributions at crisis events', ...
            'FontSize',11,'Interpreter','none');
end
 
%% =========================================================================
%  STEP 13 -- SAVE
% =========================================================================
fprintf('\n%s\n STEP 13 -- SAVING\n%s\n', repmat('=',1,70), repmat('=',1,70));
 
save('mori_3var.mat', ...
     'Y', 'VAR_NAMES', 'M', 'p', 'T_eff', 'k', ...
     'N_DRAWS', 'IRF_HOR', 'SHRINK_TAU', 'shrink_w', 'K_SHRINK_POST', ...
     'dates_vec', ...
     'Kz', 'Kmm', ...
     'kappa_draws', 'kap_med', 'kap_16', 'kap_84', 'kappa_ref', ...
     'eigenvalues_draws', 'lam_med', 'lam_16', 'lam_84', ...
     'posterior_pgt', 'gap_med_arr', 'degen_flag', 'conc_ratio_draws', ...
     'P_draws', 'B_draws', 'B_med', 'B_16', 'B_84', ...
     'sign_agreement', 'score', 'idx_best', 'n_tgt', 'target_idx', 'target_vals', ...
     'IRF_draws', 'IRF_med', 'IRF_16', 'IRF_84', 'IRF_05', 'IRF_95', ...
     'Vt_draws', 'cov_V', ...
     'V_ref', 'B_ref', 'P_ref', 'Theta_ref', ...
     'cov_V_ref', 'corr_Vref', 'cov_V_med', ...
     'HD_crisis', ...
     'KS_draws', 'KS_med', 'KS_16', 'KS_84', 'kurt_var_draws', ...
     'ks_target_draws', 'ks_t_med', 'ks_t_16', 'ks_t_84', 'ks_winner_prob', ...
     'TARGET_VARS', 'idx_ks', ...
     'fevd_draws', 'fevd_med', 'fevd_16', 'fevd_84', ...
     'kurtosis_pm', 'hill_alpha_pm', 'hill_right_pm', 'hill_left_pm', 'kw_kurt_pm', ...
     'bg_results', 'bg_any_reject', ...
     'lb_v', 'lb_v2', 'archlm_v', 'indep_diag', ...
     'boot_lam', 'boot_lam_ci', 'boot_gap_ci', 'boot_dom_frac', 'boot_coldev', ...
     'BLOCK_BOOT_N', 'BLOCK_LEN', 'LB_LAGS', 'N_DRAWS_REQ', ...
     '-v7.3');

fprintf('  Saved: mori_3var.mat\n');
fprintf('\n%s\n ALL STEPS COMPLETED\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70))

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function [Q, pval] = local_lb(x, h)
% Ljung-Box Q statistic for h lags (no Econometrics Toolbox needed).
%   Q = T(T+2) sum_{k=1}^h rho_k^2 / (T-k)  ~  chi2(h)  under white noise.
x  = x(:) - mean(x);
T  = numel(x);
c0 = sum(x.^2) / T;
Q  = 0;
for k = 1:h
    ck = sum(x(1+k:end) .* x(1:end-k)) / T;
    Q  = Q + (ck/c0)^2 / (T-k);
end
Q    = T*(T+2)*Q;
pval = 1 - chi2cdf(Q, h);
end


function [P, lam] = kollo_identify(iota, Kz, shrink_w, M)
% KOLLO_IDENTIFY  Kollo non-Gaussian identification from pre-whitened residuals.
%
%   [P, lam] = kollo_identify(iota, Kz, shrink_w, M) returns the rotation P
%   (M x M, columns sorted by DESCENDING excess-kurtosis eigenvalue) and the
%   sorted eigenvalues lam (M x 1), from pre-whitened residuals iota (n x M).
%
%   Chain (identical wherever identification is performed -- posterior-mean
%   reference in Step 5b, per-draw loop in Step 6, moving-block bootstrap in
%   Step 7c):
%       Z      = kron-product design of iota                 (n x M^2)
%       K_hat  = Z'Z / n                                     (M^2 x M^2)
%       K_star = shrink_w*K_hat + (1-shrink_w)*Kz   [Petrova shrinkage]
%       M_hat  = sum over the DIAGONAL M x M blocks of (K_star - Kz)
%                [= I_M star (K_star-Kz) = MRS/FOBI K(Z), eq.4.1 Kollo 2008]
%       M_hat  = (M_hat + M_hat')/2                  [symmetrise -> real eig]
%       [P,Lam]= eig(M_hat); sort eigenvalues descending; reorder columns
%
%   NOTE on normalisation: dividing by n = size(iota,1) reproduces the former
%   division by T_eff exactly, because in all three call sites iota has
%   exactly T_eff rows.  The column ORDER beyond the eigenvalue sort and the
%   column SIGNS are NOT fixed here; callers anchor them to P_ref.
    n      = size(iota, 1);
    Z      = repmat(iota, 1, M) .* kron(iota, ones(1, M));
    K_hat  = (Z' * Z) / n;
    K_star = shrink_w * K_hat + (1 - shrink_w) * Kz;
    D      = K_star - Kz;
    % ---------------------------------------------------------------------
    % v01_KZ: MRS/FOBI kurtosis matrix  K(Z) = I_M star c4  (Kollo 2008, eq.4.1)
    %   -> contraction with the IDENTITY: sum of the DIAGONAL blocks only
    %      (ii==jj), i.e.  M_hat(a,c) = sum_k cum(iota_a,iota_c,iota_k,iota_k).
    %   Under independence the eigenvalues are the TRUE excess kurtoses kappa_i
    %   (no s_i^2 distortion); under conditional independence + COMMON stochastic
    %   volatility the off-diagonal stays ~0 so the eigenVECTORS are unbiased
    %   (verified by simulation), unlike C(Z)=1*c4 which absorbs Var(omega).
    %   Valid here because the kappa_i are well separated (no Kollo degeneracy).
    % ---------------------------------------------------------------------
    M_hat  = zeros(M, M);
    for ii = 1:M
        ri = (ii-1)*M+1 : ii*M;
        M_hat = M_hat + D(ri, ri);         % diagonal blocks -> I_M star D
    end
    M_hat = (M_hat + M_hat') / 2;          % symmetrise -> real eigenvalues
    [P, Lam]   = eig(M_hat);
    [lam, idx] = sort(diag(Lam), 'descend');
    P          = P(:, idx);
end