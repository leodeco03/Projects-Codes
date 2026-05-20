% =========================================================================
%  KOLLO (2008) SVAR IDENTIFICATION — ROBUST BAYESIAN BVAR
%
%  Variables enter the VAR in their original units: log-levels for
%  price/volatility variables (SP500, VIX, DXY), raw % points for
%  rates/spreads (AAA, DGS10, SLOPE, CSPREAD).
%  Pre-whitening (Cholesky sphering) is applied inside each draw to
%  decorrelate and unit-normalise the residuals before computing K_hat,
%  so the kurtosis matrix M_hat is always well-defined.
%
%
%  REFERENCES:
%    Kollo (2008)                  JMVA 99(10): multivariate kurtosis + ICA
%    Lanne, Meitz, Saikkonen (2017) JEconometrics: non-Gaussian SVAR id
%    Andrade, Ferroni, Melosi (2025) FRB Boston WP: App. A.2.3-A.2.4, A.5
%    Petrova (2022)                JEconometrics: robust QML BVAR
%    Kim & White (2004)            Finance Research Letters: robust kurtosis
%    Ferroni & Canova (2025)       BVAR Hitchhiker's Guide (toolbox)
% =========================================================================
 
clear; clc; close all;
rng(42, 'twister');
 
%% -- TOOLBOX PATHS --------------------------------------------------------
addpath(fullfile(pwd, 'bvar_Ferroni', 'bvartools'));
addpath(fullfile(pwd, 'bvar_Ferroni', 'cmintools'));
 
%% -- CONFIGURATION --------------------------------------------------------
CSV_FILE   = 'financial_data_aligned_log.csv';
VAR_NAMES  = {'SP500','VIX','CSPREAD','AAA','DGS10','SLOPE','DXY'};
M          = numel(VAR_NAMES);
 
N_DRAWS    = 5000;     % posterior draws
IRF_HOR    = 20;       % horizon (trading days)
SHRINK_TAU = 0.05;     % Ridge regularisation of K_hat toward K_z (identification only)
                       %   K* = w*K_hat + (1-w)*Kz,  w = 1/(1+SHRINK_TAU) = 0.952
                       %   NOT the Petrova (2022) posterior correction (that is robust_bayes)
LAG_GRID   = [2,3,4,5,6];
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
 
%% =========================================================================
%  STEP 1 -- LOAD DATA  (original units: log-levels + raw levels)
% =========================================================================
fprintf('\n%s\n STEP 1 -- DATA LOADING\n%s\n', repmat('=',1,70), repmat('=',1,70));
 
T_raw  = readtable(CSV_FILE);
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
Phi_pm   = mean(bvar_obj.Phi_draws,   3);   % (k x M)
Sigma_pm = mean(bvar_obj.Sigma_draws, 3);   % M x M  (original-unit covariance)
 
% Regressor matrix X -- needed if e_draws not available
X = zeros(T_eff, k);
for l = 1:p
    X(:, (l-1)*M+1 : l*M) = Y(p-l+1:end-l, :);
end
X(:, end) = 1;
U_pm = Y(p+1:end, :) - X * Phi_pm;     % T_eff x M  (residuals, original units)
 
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
%    M_hat_pm(a,b) = sum of all elements of (a,b)-th M×M block of D_mat_pm
%    [P_ref, lam_ref] = eig(M_hat_pm), sorted descending by eigenvalue
% =========================================================================
fprintf('\n%s\n STEP 5b -- REFERENCE ROTATION P_ref  (posterior mean)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));

Z_pm      = repmat(iota_pm, 1, M) .* kron(iota_pm, ones(1, M));
K_hat_pm  = (Z_pm' * Z_pm) / T_eff;
K_star_pm = shrink_w * K_hat_pm + (1 - shrink_w) * Kz;
D_mat_pm  = K_star_pm - Kz;
M_hat_pm  = zeros(M, M);
for ii = 1:M
    ri_pm = (ii-1)*M+1 : ii*M;
    for jj = 1:M
        ci_pm = (jj-1)*M+1 : jj*M;
        M_hat_pm = M_hat_pm + D_mat_pm(ri_pm, ci_pm);
    end
end
M_hat_pm = (M_hat_pm + M_hat_pm') / 2;

[P_ref, Lam_ref]  = eig(M_hat_pm);
[lam_ref_sorted, idx_ref] = sort(diag(Lam_ref), 'descend');
P_ref             = P_ref(:, idx_ref);
for kk = 1:M
    [~, midx] = max(abs(P_ref(:,kk)));
    if P_ref(midx,kk) < 0, P_ref(:,kk) = -P_ref(:,kk); end
end

fprintf('  Eigenvalues of M_hat_pm (excess kurtosis of pre-whitened components):\n  ');
fprintf('%.4f  ', lam_ref_sorted); fprintf('\n');
fprintf('  P_ref (rotation at posterior mean, max-abs sign convention):\n');
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
%    M_hat_s  = ones(M,M) star (K_star_s - Kz)            [M x M]
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
IRF_draws      = zeros(M, IRF_HOR+1, M, N_DRAWS);
Vt_draws          = zeros(T_eff, M, N_DRAWS);
 
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
        skip_count = skip_count + 1;
        continue
    end
    L_s    = chol(Sigma_s, 'lower');
    iota_s = (L_s \ U_s')';                  % T_eff x M
 
    % (d) K_hat  (M^2 x M^2) -- VECTORISED
    Z = repmat(iota_s, 1, M) .* kron(iota_s, ones(1, M));
    K_hat_s = (Z' * Z) / T_eff;
 
    % (d') Petrova shrinkage on identification matrix
    K_star_s = shrink_w * K_hat_s + (1 - shrink_w) * Kz;
 
    % (e) Star-product compression  M x M
    D_mat   = K_star_s - Kz;
    M_hat_s = zeros(M, M);
    for ii = 1:M
        ri = (ii-1)*M+1 : ii*M;
        for jj = 1:M
            ci = (jj-1)*M+1 : jj*M;
            M_hat_s = M_hat_s + D_mat(ri, ci);
        end
    end
    M_hat_s = (M_hat_s + M_hat_s') / 2;
 
    % (f) Eigendecomposition
    [P_s, Lam_s]   = eig(M_hat_s);
    [lam_s, idx_s] = sort(diag(Lam_s), 'descend');
    P_s            = P_s(:, idx_s);
 
    % (g) Sign normalisation: max-abs element positive (local anchor)
    for kk = 1:M
        [~, max_idx] = max(abs(P_s(:,kk)));
        if P_s(max_idx, kk) < 0
            P_s(:,kk) = -P_s(:,kk);
        end
    end

    % (g2) C1+S1: align column order THEN sign with P_ref.
    %   This prevents bimodal posteriors (C1) and eigenvalue-swap artefacts (S1).
    %
    %   Step 1 — greedy column matching.
    %   For each reference column r of P_ref, find the source column in P_s
    %   with the highest absolute dot product and assign it to position r.
    %   The greedy order (r=1,...,M) prioritises the most non-Gaussian shock.
    dots_abs  = abs(P_s' * P_ref);    % M×M: row=src col in P_s, col=ref col in P_ref
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
        IRF_draws(:,:,:,s) = ir_s;
    catch
        % Fallback to manual companion-form computation
        n_comp = M * p;
        J_sel  = [eye(M), zeros(M, M*(p-1))];
        Phi_coef = Phi_s(1:M*p, :)';
        A_comp   = [Phi_coef; eye(M*(p-1)), zeros(M*(p-1), M)];
        A_h      = eye(n_comp);
        for h = 0:IRF_HOR
            Psi_h = J_sel * A_h * J_sel';
            IRF_draws(:, h+1, :, s) = Psi_h * B_s;
            A_h = A_h * A_comp;
        end
    end
 
    eigenvalues_draws(:,s)  = lam_s;
    P_draws(:,:,s)          = P_s;
    B_draws(:,:,s)       = B_s;
    Vt_draws(:,:,s)         = V_s;
 
    if mod(s,500)==0
        fprintf('  Draw %d / %d  (%.1fs)\n', s, N_DRAWS, toc);
    end
end
fprintf('  Total: %.2fs  |  Skipped draws (rcond<1e-14): %d\n', toc, skip_count);
 
%% =========================================================================
%  STEP 6b -- REFERENCE STRUCTURAL SHOCKS  (posterior-mean parameters)
% =========================================================================
fprintf('\n%s\n STEP 6b -- REFERENCE STRUCTURAL SHOCKS  (posterior mean)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
 
% Z_pm, K_hat_pm, K_star_pm, D_mat_pm, M_hat_pm and P_ref were computed in
% Step 5b (before the draw loop) to serve as the sign/column anchor.
% Re-use them here directly — no recomputation needed.
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
%  STEP 7 -- EIGENVALUE DECAY
% =========================================================================
fprintf('\n%s\n STEP 7 -- EIGENVALUE DECAY\n%s\n', repmat('=',1,70), repmat('=',1,70));
 
lam_med = median(eigenvalues_draws, 2);
lam_16  = quantile(eigenvalues_draws, 0.16, 2);
lam_84  = quantile(eigenvalues_draws, 0.84, 2);
 
fprintf('\n  Shock   lam_med    lam_16     lam_84    bar\n');
fprintf('  %s\n', repmat('-',1,58));
for i = 1:M
    bar_str = repmat('|', 1, max(0, round(lam_med(i)/max(lam_med)*25)));
    fprintf('  v_%d   %9.4f  %9.4f  %9.4f   %s\n', ...
            i, lam_med(i), lam_16(i), lam_84(i), bar_str);
end
fprintf('\n  lambda_1 / lambda_M: %.4f\n', lam_med(1)/max(lam_med(end),1e-8));
 
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
fprintf('\n  Concentration ratio lambda_1 / lambda_2:\n');
fprintf('    median: %.3f  |  q16: %.3f  |  q84: %.3f\n', ...
        median(conc_ratio_draws), ...
        quantile(conc_ratio_draws, 0.16), ...
        quantile(conc_ratio_draws, 0.84));
fprintf('    P(lam_1/lam_2 > 2 | Y): %.4f  (>=0.95 supports robust dominance)\n', ...
        mean(conc_ratio_draws > 2));
 
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
 
idx_VIX   = find(strcmp(VAR_NAMES, 'VIX'));
idx_CSPRD = find(strcmp(VAR_NAMES, 'CSPREAD'));
idx_DXY   = find(strcmp(VAR_NAMES, 'DXY'));
idx_SP500 = find(strcmp(VAR_NAMES, 'SP500'));
idx_DGS10 = find(strcmp(VAR_NAMES, 'DGS10'));
idx_AAA   = find(strcmp(VAR_NAMES, 'AAA'));
 
target_idx  = [idx_VIX, idx_CSPRD, idx_DXY, idx_SP500, idx_DGS10, idx_AAA];
target_vals = [+1, +1, +1, -1, -1, -1];
 
score = zeros(M,1);
for kk = 1:M
    sc = 0;
    for q = 1:length(target_idx)
        if sign(B_med(target_idx(q), kk)) == target_vals(q)
            sc = sc + 1;
        end
    end
    score(kk) = sc;
end
 
fprintf('  %-8s %12s %15s %20s\n','Shock','lam_med','TR-score (max=6)','Interpretation');
fprintf('  %s\n', repmat('-', 1, 65));
for kk = 1:M
    if score(kk) >= 5,    interp = 'TAIL-RISK candidate';
    elseif score(kk) >= 4, interp = 'partial match';
    elseif score(kk) <= 1, interp = 'opposite sign (anti-tail)';
    else,                  interp = 'unclear';
    end
    fprintf('  v_%-6d %12.4f %15d   %s\n', kk, lam_med(kk), score(kk), interp);
end
 
[~, idx_best] = max(score);
fprintf('\n  >>> Best tail-risk candidate: v_%d (score=%d/6, lambda=%.3f)\n', ...
        idx_best, score(idx_best), lam_med(idx_best));
if idx_best ~= 1
    fprintf('      WARNING: best candidate is NOT v_1 (the most non-Gaussian shock).\n');
    fprintf('      v_1 may capture a different non-Gaussian feature (e.g. uncertainty).\n');
end
 
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
sgtitle(sprintf('IRF to v_1 (lambda=%.3f) -- tail-risk candidate [original units]', ...
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
 
cov_V_draws = zeros(M, M, N_DRAWS);
for s = 1:N_DRAWS
    V_s = squeeze(Vt_draws(:,:,s));
    cov_V_draws(:,:,s) = (V_s' * V_s) / T_eff;
end
cov_V_med      = median(cov_V_draws, 3);
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
%  STEP 12 -- FEVD
% =========================================================================
fprintf('\n%s\n STEP 12 -- FEVD  (toolbox convention: rows=vars, cols=shocks)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
 
fevd_draws = zeros(M, M, IRF_HOR+1, N_DRAWS);
for s = 1:N_DRAWS
    cum_sq = zeros(M, M);
    for h = 0:IRF_HOR
        Th     = squeeze(IRF_draws(:,h+1,:,s));
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
 
fig6 = figure('Name','FEVD of v_1','Position',[50 50 1200 500]);
for m = 1:M
    subplot(2, ceil(M/2), m);
    fm = squeeze(fevd_med(m,1,:))*100;
    fl = squeeze(fevd_16(m,1,:))*100;
    fh = squeeze(fevd_84(m,1,:))*100;
    fill([horizon,fliplr(horizon)],[fl',fliplr(fh')], ...
         [0.7 0.85 0.95],'EdgeColor','none'); hold on;
    plot(horizon, fm,'b-','LineWidth',2);
    title(VAR_NAMES{m},'FontSize',9,'Interpreter','none');
    xlabel('h'); ylabel('% var explained');
    ylim([0 100]); xlim([0 IRF_HOR]); grid on;
    set(gca,'FontSize',8,'Box','on');
end
sgtitle(sprintf('FEVD: contribution of v_1 (lambda=%.3f) to each variable', ...
        lam_med(1)),'FontSize',11,'Interpreter','none');
 
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
%  STEP 12c -- TAIL-FEVD  (contemporaneous, h=0)
%
%  The bottom-5% tail of Y_eff(:,m) is in original units, which is
%  directly interpretable in terms of the variable's economic scale.
% =========================================================================
fprintf('\n%s\n STEP 12c -- TAIL-FEVD  (h=0, bottom 5%%)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
 
Y_eff    = Y(p+1:end, :);   % T_eff x M  (original units, aligned with V_ref)
TailFEVD = zeros(M, M);
UncFEVD0 = zeros(M, M);
 
for m = 1:M
    q05_m    = quantile(Y_eff(:,m), 0.05);
    tail_idx = Y_eff(:,m) < q05_m;
    n_tail   = sum(tail_idx);
 
    sq_tail = zeros(M,1);
    sq_unc  = zeros(M,1);
    for kk = 1:M
        contrib     = B_ref(m,kk) * V_ref(:,kk);
        sq_tail(kk) = mean(contrib(tail_idx).^2);
        sq_unc(kk)  = mean(contrib.^2);
    end
    TailFEVD(m,:) = (sq_tail / sum(sq_tail))';
    UncFEVD0(m,:) = (sq_unc  / sum(sq_unc))';
 
    fprintf('  %s: %d tail obs (bottom 5%%)  |  q05 = %.4f (original units)\n', ...
            VAR_NAMES{m}, n_tail, q05_m);
end
 
fprintf('\n  Tail-FEVD (h=0, conditional on y_m < q_05):\n');
fprintf('  %-10s', 'Var');
for kk = 1:M, fprintf(' %8s', ['v_' num2str(kk)]); end, fprintf('\n');
fprintf('  %s\n', repmat('-', 1, 10 + 9*M));
for m = 1:M
    fprintf('  %-10s', VAR_NAMES{m});
    for kk = 1:M, fprintf(' %8.4f', TailFEVD(m,kk)); end
    fprintf('\n');
end
 
fprintf('\n  Unconditional FEVD (h=0):\n');
fprintf('  %-10s', 'Var');
for kk = 1:M, fprintf(' %8s', ['v_' num2str(kk)]); end, fprintf('\n');
fprintf('  %s\n', repmat('-', 1, 10 + 9*M));
for m = 1:M
    fprintf('  %-10s', VAR_NAMES{m});
    for kk = 1:M, fprintf(' %8.4f', UncFEVD0(m,kk)); end
    fprintf('\n');
end
 
fprintf('\n  Tail uplift (TailFEVD / UncFEVD - 1):\n');
fprintf('  %-10s', 'Var');
for kk = 1:M, fprintf(' %8s', ['v_' num2str(kk)]); end, fprintf('\n');
fprintf('  %s\n', repmat('-', 1, 10 + 9*M));
for m = 1:M
    fprintf('  %-10s', VAR_NAMES{m});
    for kk = 1:M
        uplift = TailFEVD(m,kk) / max(UncFEVD0(m,kk), 1e-6) - 1;
        fprintf(' %8.4f', uplift);
    end
    fprintf('\n');
end
 
fig8 = figure('Name','Tail-FEVD','Position',[50 50 900 600]);
subplot(1,2,1);
imagesc(TailFEVD*100); colorbar;
set(gca,'XTick',1:M,'XTickLabel',arrayfun(@(i)['v_' num2str(i)],1:M,'UniformOutput',false), ...
        'YTick',1:M,'YTickLabel',VAR_NAMES,'FontSize',9);
title('Tail-FEVD h=0 (bottom 5%) [%]','Interpreter','none','FontSize',10);
xlabel('Shock','Interpreter','none'); ylabel('Variable','Interpreter','none');
 
subplot(1,2,2);
imagesc((TailFEVD - UncFEVD0)*100); colorbar;
clim_val = max(abs((TailFEVD - UncFEVD0)*100), [], 'all');
if clim_val > 0, clim([-clim_val clim_val]); end
n_rb = 32;
cmap_rb = [linspace(0.1,1,n_rb)', linspace(0.1,1,n_rb)', ones(n_rb,1); ...
           ones(n_rb,1), linspace(1,0.1,n_rb)', linspace(1,0.1,n_rb)'];
colormap(gca, cmap_rb);
set(gca,'XTick',1:M,'XTickLabel',arrayfun(@(i)['v_' num2str(i)],1:M,'UniformOutput',false), ...
        'YTick',1:M,'YTickLabel',VAR_NAMES,'FontSize',9);
title('Tail uplift: TailFEVD - UncFEVD [pp]','Interpreter','none','FontSize',10);
xlabel('Shock','Interpreter','none'); ylabel('Variable','Interpreter','none');
sgtitle('Tail-FEVD vs Unconditional FEVD (h=0)','FontSize',11,'Interpreter','none');
 
%% =========================================================================
%  STEP 13 -- SAVE
% =========================================================================
fprintf('\n%s\n STEP 13 -- SAVING\n%s\n', repmat('=',1,70), repmat('=',1,70));
 
save('kollo_svar_results.mat', ...
     'Y', 'VAR_NAMES', 'M', 'p', 'T_eff', 'k', ...
     'N_DRAWS', 'IRF_HOR', 'SHRINK_TAU', 'shrink_w', 'K_SHRINK_POST', ...
     'Kz', 'Kmm', ...
     'eigenvalues_draws', 'lam_med', 'lam_16', 'lam_84', ...
     'posterior_pgt', 'gap_med_arr', 'degen_flag', 'conc_ratio_draws', ...
     'P_draws', 'B_draws', 'B_med', 'B_16', 'B_84', ...
     'sign_agreement', 'score', 'idx_best', ...
     'IRF_draws', 'IRF_med', 'IRF_16', 'IRF_84', 'IRF_05', 'IRF_95', ...
     'Vt_draws', 'cov_V', ...
     'V_ref', 'B_ref', 'P_ref', 'Theta_ref', ...
     'cov_V_ref', 'corr_Vref', 'cov_V_med', ...
     'HD_crisis', 'TailFEVD', 'UncFEVD0', ...
     'fevd_draws', 'fevd_med', 'fevd_16', 'fevd_84', ...
     'kurtosis_pm', 'hill_alpha_pm', 'hill_right_pm', 'hill_left_pm', 'kw_kurt_pm', ...
     'bg_results', 'bg_any_reject', ...
     '-v7.3');

fprintf('  Saved: kollo_svar_results.mat\n');
fprintf('\n%s\n ALL STEPS COMPLETED\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));