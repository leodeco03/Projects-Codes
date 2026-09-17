% =========================================================================
%  Mori_3var.m - daily SVAR identified by non-Gaussianity (Mori/Kollo FOBI).
%
%  PIPELINE
%     Mori_3var.m              -> mori_3var.mat            (this file)
%     Bootstrap_kollo.m        -> independence diagnostics of the shocks
%     proxy_svar1.m            -> shocks_monthly_*.mat     (Stage A, instrument)
%     proxy_svar_02_cholesky.m -> macro IRFs               (Stage B)
%
%  WHAT THIS FILE DOES
%    1. BVAR(p) on [SP500, VIX, CSPREAD] in log-levels: flat prior, Petrova
%       (2022) robust posterior for Sigma, p chosen by marginal likelihood.
%    2. Pre-whitens the residuals, iota = L^{-1} u with Sigma = L L', and
%       identifies the rotation from the fourth moment:
%          K_hat = E[vec(iota iota') vec(iota iota')'],
%          M_hat = I_M star (K_hat - K_z)   [FOBI/Kollo K(Z), Kollo 2008 eq.4.1]
%       Eigenvectors of M_hat give P, eigenvalues give the excess kurtoses.
%    3. Repeats the chain on every posterior draw, with columns matched and
%       signed against the posterior-mean rotation P_ref, and runs a
%       moving-block bootstrap that prices the serial dependence of K_hat.
%    4. Diagnostics (tails, serial dependence, independence, degeneracy),
%       IRF, FEVD, kurtosis-share, narrative check, crisis decomposition.
%
%  INPUT   financial_data_aligned_corrected.csv  (Date + the three series)
%          bvar_Ferroni toolbox (Ferroni & Canova)
%  OUTPUT  mori_3var.mat
%
%  The econometric argument behind each choice is in the thesis; the comments
%  here say what the code does, not why it is the right thing to do.
% =========================================================================

clear; clc; close all;
rng(42, 'twister');       % note: bvar_ resets the stream to rng(999) internally

addpath(fullfile(pwd, 'bvar_Ferroni', 'bvartools'));
addpath(fullfile(pwd, 'bvar_Ferroni', 'cmintools'));

%% ---- CONFIGURATION ------------------------------------------------------
CSV_FILE   = 'financial_data_aligned_corrected.csv';
VAR_NAMES  = {'SP500','VIX','CSPREAD'};
M          = numel(VAR_NAMES);

N_DRAWS    = 5000;                % posterior draws
IRF_HOR    = 20;                  % IRF horizon (trading days)
LAG_GRID   = [3,4,5,6,7,8,9,10];  % candidate lags for the marginal likelihood
HILL_FRAC  = 0.05;                % top 5% of each tail for the Hill estimator

SHRINK_TAU    = 0;    % identification: K* = w*K_hat + (1-w)*Kz, w = 1/(1+tau)
K_SHRINK_POST = NaN;  % Petrova posterior: NaN -> toolbox default (K = T_full)

DEGEN_THRESHOLD = 0.10;  % near-degeneracy flag if (lam_i - lam_i+1)/lam_i < this

BLOCK_BOOT_N = 500;   % moving-block bootstrap replications for the spectrum
BLOCK_LEN    = 20;    % block length (trading days)
LB_LAGS      = 10;    % Ljung-Box lags on the structural shocks
H_BG         = 10;    % Breusch-Godfrey lags
ARCH_Q       = 5;     % ARCH-LM lags
ALPHA_TEST   = 0.05;

shock_lab = arrayfun(@(i) sprintf('v_%d', i), 1:M, 'UniformOutput', false);

%% =========================================================================
%  STEP 1 - DATA  (log-levels and raw levels, as in the CSV)
% =========================================================================
banner('STEP 1 - DATA LOADING');

T_raw  = readtable(CSV_FILE);
T_raw  = T_raw(datetime(T_raw.Date,'InputFormat','yyyy-MM-dd') >= datetime(1990,1,2), :);
Y      = zeros(height(T_raw), M);
for m = 1:M
    Y(:,m) = T_raw.(VAR_NAMES{m});
end
T_full = size(Y, 1);

fprintf('  CSV: %d obs x %d variables\n\n', T_full, M);
fprintf('  %-10s %12s %12s %10s %12s\n','Variable','Mean','Std','Skewness','Kurt.Exc');
fprintf('  %s\n', repmat('-',1,62));
for m = 1:M
    y_s = (Y(:,m) - mean(Y(:,m))) / std(Y(:,m));
    fprintf('  %-10s %12.4f %12.4f %10.4f %12.4f\n', ...
            VAR_NAMES{m}, mean(Y(:,m)), std(Y(:,m)), mean(y_s.^3), mean(y_s.^4)-3);
end

%% =========================================================================
%  STEP 2 - LAG SELECTION  (flat-prior marginal likelihood, bvar_.logmlike)
%
%  CAVEAT: under the flat (improper) prior the marginal likelihood is compared
%  across models of different dimension and on samples of different length
%  (T_full - p), so it ranks lags but is not a proper Bayes factor.  The
%  surface is flat here; the thesis discusses the choice.
% =========================================================================
banner('STEP 2 - LAG SELECTION');

opt_sel   = struct('K', 20);     % draws are discarded: only logmlike is used
lnML_grid = zeros(numel(LAG_GRID), 1);
for i = 1:numel(LAG_GRID)
    lnML_grid(i) = bvar_(Y, LAG_GRID(i), opt_sel).logmlike;
end
[lnML_max, best_lag_idx] = max(lnML_grid);
p     = LAG_GRID(best_lag_idx);
T_eff = T_full - p;
k     = M*p + 1;

fprintf('  %-6s  %-8s  %-5s  %-14s  %-10s\n','p','T_eff','k','ln ML','Delta');
fprintf('  %s\n', repmat('-',1,52));
for i = 1:numel(LAG_GRID)
    flag = ''; if i == best_lag_idx, flag = '  <- ML OPTIMAL'; end
    fprintf('  p=%-4d  T_eff=%-5d  k=%-4d  lnML=%12.4f  Delta=%9.4f%s\n', ...
            LAG_GRID(i), T_full - LAG_GRID(i), M*LAG_GRID(i)+1, ...
            lnML_grid(i), lnML_grid(i)-lnML_max, flag);
end
fprintf('\n  Selected p = %d  |  T_eff = %d  |  k = %d\n', p, T_eff, k);
clear opt_sel

%% =========================================================================
%  STEP 3 - BVAR(p) WITH PETROVA (2022) ROBUST POSTERIOR
%    robust_bayes = 1 replaces the inverse-Wishart posterior of Sigma with the
%    QML asymptotic distribution, whose variance depends on the fourth moment
%    of the residuals - the moment identification relies on.
% =========================================================================
banner(sprintf('STEP 3 - ROBUST BAYESIAN BVAR(p=%d)  [Petrova 2022]', p));

opt_flat              = struct();
opt_flat.K            = N_DRAWS;
opt_flat.hor          = IRF_HOR;
opt_flat.robust_bayes = 1;          % 1 = 4th-moment correction, 2 = also 3rd
if ~isnan(K_SHRINK_POST), opt_flat.K_shrinkage = K_SHRINK_POST; end

bvar_obj = bvar_(Y, p, opt_flat);

Phi_pm   = mean(bvar_obj.Phi_draws,   3);   % k x M
Sigma_pm = mean(bvar_obj.Sigma_draws, 3);   % M x M, original units

X = zeros(T_eff, k);                        % [y_{t-1} ... y_{t-p}, 1]
for l = 1:p
    X(:, (l-1)*M+1 : l*M) = Y(p-l+1:end-l, :);
end
X(:, end) = 1;
U_pm = Y(p+1:end, :) - X * Phi_pm;          % T_eff x M

fprintf('  ln ML (flat, p=%d): %.4f\n', p, bvar_obj.logmlike);
fprintf('  cond(Sigma_pm):     %.4e  %s\n', cond(Sigma_pm), ...
        tern(cond(Sigma_pm) < 1e4, '[OK]', '[WARNING: check variable scales]'));
fprintf('  Eigenvalues of Sigma_pm: ');
fprintf('%.6f  ', sort(eig(Sigma_pm),'descend')); fprintf('\n');
fprintf('  Mean draw-variance of Sigma elements: %.4e\n', ...
        mean(var(reshape(bvar_obj.Sigma_draws, M*M, N_DRAWS), 0, 2)));

%% =========================================================================
%  STEP 3b - BREUSCH-GODFREY TEST  (is p enough to whiten the residuals?)
% =========================================================================
banner(sprintf('STEP 3b - BREUSCH-GODFREY TEST on U_pm  (p = %d)', p));

fprintf('  %-10s  %10s  %8s  %s\n', 'Variable', sprintf('BG(%d)',H_BG), 'p-value', 'Result');
fprintf('  %s\n', repmat('-',1,55));
bg_results = zeros(M, 3);            % [LM, p-value, reject]
for m = 1:M
    u_m     = U_pm(:,m);
    U_lag   = lagmat(u_m, H_BG);
    idx_bg  = H_BG+1 : T_eff;
    [LM_bg, pv_bg] = lm_test(u_m(idx_bg), [X(idx_bg,:), U_lag(idx_bg,:)], H_BG);
    reject  = pv_bg < ALPHA_TEST;
    bg_results(m,:) = [LM_bg, pv_bg, double(reject)];
    fprintf('  %-10s  %10.4f  %8.4f%s\n', VAR_NAMES{m}, LM_bg, pv_bg, ...
            tern(reject, '  *** AUTOCORR', ''));
end
bg_any_reject = any(bg_results(:,3) > 0);
if bg_any_reject
    fprintf('\n  WARNING: autocorrelation at p=%d (alpha=%.2f): consider a longer LAG_GRID.\n', ...
            p, ALPHA_TEST);
else
    fprintf('\n  No autocorrelation at alpha=%.2f: p=%d is adequate.\n', ALPHA_TEST, p);
end

%% =========================================================================
%  STEP 4 - PRE-WHITENING + TAIL DIAGNOSTICS
%    iota = L^{-1} u with Sigma_pm = L L', so E[iota iota'] = I_M and the
%    whole chain is invariant to the scale of the variables.
%    Tails: sample excess kurtosis, two-sided Hill index, Kim-White (2004)
%    octile-based robust kurtosis.
% =========================================================================
banner('STEP 4 - PRE-WHITENING + TAIL DIAGNOSTICS');

L_pm    = chol(Sigma_pm, 'lower');
iota_pm = (L_pm \ U_pm')';                    % T_eff x M

off_diag_max = max(abs((iota_pm'*iota_pm)/T_eff - eye(M)), [], 'all');
fprintf('  Max |cov(iota) - I_M|: %.6f  (OK <= 1e-3)\n', off_diag_max);
if off_diag_max > 0.02
    warning('Pre-whitening imprecise: check the variable set or regularise.');
end

fprintf('\n  %-10s %10s %10s %12s %12s %12s %12s %10s\n', ...
        'Variable','Mean','Std','Kurt.Exc','Hill(avg)','Hill(right)','Hill(left)','KW.Exc');
fprintf('  %s\n', repmat('-',1,84));
kurtosis_pm   = zeros(M,1);
hill_alpha_pm = zeros(M,1);   hill_right_pm = zeros(M,1);   hill_left_pm = zeros(M,1);
kw_kurt_pm    = zeros(M,1);
mh = max(2, floor(HILL_FRAC * T_eff));        % order statistics in each tail
for m = 1:M
    i_m   = iota_pm(:,m);
    i_std = (i_m - mean(i_m)) / std(i_m);
    kurtosis_pm(m) = mean(i_std.^4) - 3;

    hill_r = hill(i_m(i_m > 0),  mh);         % right tail
    hill_l = hill(-i_m(i_m < 0), mh);         % left tail, reflected
    hill_right_pm(m) = hill_r;
    hill_left_pm(m)  = hill_l;
    hill_alpha_pm(m) = mean([hill_r, hill_l], 'omitnan');

    oct = quantile(i_m, (1:7)/8);             % Kim-White octile kurtosis
    if abs(oct(6) - oct(2)) > 1e-10
        kw_kurt_pm(m) = ((oct(7)-oct(5)) + (oct(3)-oct(1))) / (oct(6)-oct(2)) - 1.23;
    else
        kw_kurt_pm(m) = NaN;
    end

    fprintf('  %-10s %10.4f %10.4f %12.4f %12.4f %12.4f %12.4f %10.4f\n', ...
            VAR_NAMES{m}, mean(i_m), std(i_m), kurtosis_pm(m), ...
            hill_alpha_pm(m), hill_r, hill_l, kw_kurt_pm(m));
end
fprintf('\n  All sample excess kurtoses > 0: %s  |  all Kim-White > 0: %s\n', ...
        mat2str(all(kurtosis_pm > 0)), mat2str(all(kw_kurt_pm > 0)));
fprintf('  Hill alpha < 4 (4th moment possibly infinite): avg %d/%d, right %d/%d, left %d/%d\n', ...
        sum(hill_alpha_pm < 4), M, sum(hill_right_pm < 4), M, sum(hill_left_pm < 4), M);

%% =========================================================================
%  STEP 5 - GAUSSIAN BENCHMARK K_z  (rotation-invariant: carries no
%  identification; only K_hat - K_z is informative about the rotation)
% =========================================================================
banner(sprintf('STEP 5 - BUILD K_z  (%d x %d)', M^2, M^2));

Kmm   = commutationmatrix(M);
vecIM = reshape(eye(M), M^2, 1);
Kz    = eye(M^2) + Kmm + vecIM * vecIM';
shrink_w = 1 / (1 + SHRINK_TAU);

fprintf('  K_z: %dx%d  |  rank = %d  |  shrinkage weight w = %.4f (tau = %.3f)\n', ...
        M^2, M^2, rank(Kz), shrink_w, SHRINK_TAU);

%% =========================================================================
%  STEP 5b - REFERENCE ROTATION P_ref  (identification at the posterior mean)
%
%  P_ref anchors the whole posterior: every draw's columns are matched to it
%  (order) and signed against it.  Sign convention: each column is flipped so
%  that its impact on the variable it explains the most - argmax_m
%  |B(m,k)|/std_m, i.e. the impact-FEVD dominant variable - is positive.  It
%  is data-driven, so it survives a change of sample or lag order.
% =========================================================================
banner('STEP 5b - REFERENCE ROTATION P_ref  (posterior mean)');

[P_ref, lam_ref_sorted] = kollo_identify(iota_pm, Kz, shrink_w);

B_ref_ec = L_pm * P_ref;                       % impact matrix, original units
std_ec   = sqrt(sum(B_ref_ec.^2, 2));          % = sqrt(diag(Sigma_pm))
for kk = 1:M
    [~, mdom] = max(abs(B_ref_ec(:,kk)) ./ std_ec);
    if B_ref_ec(mdom,kk) < 0
        P_ref(:,kk)    = -P_ref(:,kk);
        B_ref_ec(:,kk) = -B_ref_ec(:,kk);
    end
end

fprintf('  Sign anchor (FEVD-dominant variable per shock, standardised |B|):\n');
for kk = 1:M
    [amax, mdom] = max(abs(B_ref_ec(:,kk)) ./ std_ec);
    fprintf('    %-5s ->  %-10s  (std|B| = %.4f, sign +)\n', shock_lab{kk}, VAR_NAMES{mdom}, amax);
end
clear B_ref_ec

% Under independence the eigenvalues of K(Z) ARE the excess kurtoses of the
% shocks, so d and kappa must agree up to dependence and sampling error.
V_tmp     = iota_pm * P_ref;
kappa_ref = mean(V_tmp.^4, 1)' - 3;
fprintf('\n  K(Z) eigenvalue d_i vs realised excess kurtosis kappa_i:\n');
fprintf('  %-8s %14s %14s\n', 'Shock', 'd = K(Z) eig', 'kappa_i');
fprintf('  %s\n', repmat('-',1,38));
for kk = 1:M
    fprintf('  %-8s %14.4f %14.4f\n', shock_lab{kk}, lam_ref_sorted(kk), kappa_ref(kk));
end
fprintf('  max|d - kappa| = %.4f\n', max(abs(lam_ref_sorted(:) - kappa_ref(:))));
clear V_tmp

fprintf('\n  P_ref (rotation at the posterior mean):\n');
print_table(VAR_NAMES, compose_labels('P', M), P_ref);

%% =========================================================================
%  STEP 6 - IDENTIFICATION ON EVERY POSTERIOR DRAW
%    Per draw: Sigma_s = L_s L_s' -> iota_s -> kollo_identify -> columns
%    matched and signed against P_ref -> B_s = L_s P_s, v_t = P_s' iota_t,
%    IRF = Psi_h B_s.
% =========================================================================
banner(sprintf('STEP 6 - IDENTIFICATION LOOP  (%d draws)', N_DRAWS));

eigenvalues_draws = zeros(M, N_DRAWS);
P_draws           = zeros(M, M, N_DRAWS);
B_draws           = zeros(M, M, N_DRAWS);
IRF_draws         = zeros(M, IRF_HOR+1, M, N_DRAWS, 'single');
Vt_draws          = zeros(T_eff, M, N_DRAWS, 'single');
cov_V_draws       = zeros(M, M, N_DRAWS);
kappa_draws       = zeros(M, N_DRAWS);
valid             = true(N_DRAWS, 1);

tic; skip_count = 0;
for s = 1:N_DRAWS
    Phi_s   = bvar_obj.Phi_draws(:,:,s);
    Sigma_s = bvar_obj.Sigma_draws(:,:,s);

    if isfield(bvar_obj,'e_draws') && ndims(bvar_obj.e_draws) == 3
        U_s = bvar_obj.e_draws(:,:,s);
    else
        U_s = Y(p+1:end,:) - X * Phi_s;
    end

    if rcond(Sigma_s) < 1e-14
        skip_count = skip_count + 1;  valid(s) = false;  continue
    end
    L_s    = chol(Sigma_s, 'lower');
    iota_s = (L_s \ U_s')';

    [P_s, lam_s] = kollo_identify(iota_s, Kz, shrink_w);

    ord   = match_columns(P_s, P_ref);            % order, then sign
    P_s   = P_s(:, ord);   lam_s = lam_s(ord);
    flip  = sign(sum(P_s .* P_ref, 1));  flip(flip == 0) = 1;
    P_s   = P_s .* flip;

    B_s = L_s * P_s;
    V_s = iota_s * P_s;

    eigenvalues_draws(:,s) = lam_s;
    P_draws(:,:,s)         = P_s;
    B_draws(:,:,s)         = B_s;
    IRF_draws(:,:,:,s)     = single(companion_irf(Phi_s, B_s, p, IRF_HOR));
    Vt_draws(:,:,s)        = single(V_s);
    cov_V_draws(:,:,s)     = (V_s' * V_s) / T_eff;
    kappa_draws(:,s)       = mean(V_s.^4, 1)' - 3;   % unit variance by construction

    if mod(s,500) == 0, fprintf('  Draw %d / %d  (%.1fs)\n', s, N_DRAWS, toc); end
end
fprintf('  Total: %.2fs  |  skipped draws (rcond < 1e-14): %d\n', toc, skip_count);

% Drop skipped draws: their zero-filled slabs would contaminate every quantile.
N_DRAWS_REQ = N_DRAWS;
if skip_count > 0
    eigenvalues_draws = eigenvalues_draws(:,   valid);
    P_draws           = P_draws(:,:,          valid);
    B_draws           = B_draws(:,:,          valid);
    IRF_draws         = IRF_draws(:,:,:,      valid);
    Vt_draws          = Vt_draws(:,:,         valid);
    cov_V_draws       = cov_V_draws(:,:,      valid);
    kappa_draws       = kappa_draws(:,        valid);
    N_DRAWS           = nnz(valid);
    fprintf('  Compacted: %d valid draws kept.\n', N_DRAWS);
end

%% =========================================================================
%  STEP 6b - REFERENCE SHOCKS AND IRF  (posterior-mean parameters)
% =========================================================================
banner('STEP 6b - REFERENCE SHOCKS  (posterior mean)');

B_ref     = L_pm * P_ref;          % M x M impact matrix, original units
V_ref     = iota_pm * P_ref;       % T_eff x M structural shocks
cov_V_ref = (V_ref' * V_ref) / T_eff;
Theta_ref = companion_irf(Phi_pm, B_ref, p, IRF_HOR);

fprintf('  max|Cov(V_ref) - I_M|: %.2e\n', max(abs(cov_V_ref - eye(M)), [], 'all'));
fprintf('  Theta_ref: %d x %d x %d  [variable x horizon x shock]\n', M, IRF_HOR+1, M);

%% =========================================================================
%  STEP 6c - SERIAL DEPENDENCE AND INDEPENDENCE DIAGNOSTICS
%    (1) Ljung-Box on v_k and on v_k^2, (2) ARCH-LM on v_k, (3) cross fourth
%    moments.  Serial dependence leaves P consistent and is priced by the
%    block bootstrap of Step 7c; it is tested, not filtered out.
% =========================================================================
banner('STEP 6c - i.i.d. / INDEPENDENCE DIAGNOSTICS');

lb_v = zeros(M,2); lb_v2 = zeros(M,2); archlm_v = zeros(M,2);
fprintf('\n  %-8s %11s %8s %11s %8s %11s %8s\n', ...
        'Shock','LB(v)','p','LB(v^2)','p','ARCH-LM','p');
fprintf('  %s\n', repmat('-',1,68));
for kk = 1:M
    vk  = V_ref(:,kk) - mean(V_ref(:,kk));
    vk2 = vk.^2;
    [Qv,  pv ] = local_lb(vk,  LB_LAGS);
    [Qv2, pv2] = local_lb(vk2, LB_LAGS);
    lb_v(kk,:) = [Qv, pv];  lb_v2(kk,:) = [Qv2, pv2];

    U2   = lagmat(vk2, ARCH_Q);
    iA   = ARCH_Q+1 : T_eff;
    [LMA, pA] = lm_test(vk2(iA), [ones(numel(iA),1), U2(iA,:)], ARCH_Q);
    archlm_v(kk,:) = [LMA, pA];

    fprintf('  %-8s %11.2f %8.4f %11.2f %8.4f %11.2f %8.4f\n', ...
            shock_lab{kk}, Qv, pv, Qv2, pv2, LMA, pA);
end

% Cross fourth moments of the standardised shocks: E[v_i^2 v_j^2] = 1 and
% E[v_i^3 v_j] = 0 under independence.  The second is robust to a common
% stochastic volatility factor; Bootstrap_kollo.m attaches intervals to both.
Vsd = (V_ref - mean(V_ref)) ./ std(V_ref);
co_kurt        = (Vsd.^2)' * (Vsd.^2) / T_eff;
co_kurt_robust = (Vsd.^3)' *  Vsd     / T_eff;
mask       = ~eye(M);
indep_diag = struct('co_kurt', co_kurt, 'co_kurt_robust', co_kurt_robust);

fprintf('\n  Co-kurtosis E[v_i^2 v_j^2]:\n');
print_table(shock_lab, shock_lab, co_kurt);
fprintf('\n  Off-diagonal: max |E[v_i^2 v_j^2] - 1| = %.4f,  max |E[v_i^3 v_j]| = %.4f\n', ...
        max(abs(co_kurt(mask)-1)), max(abs(co_kurt_robust(mask))));
fprintf('  LB(v^2) rejects in %d/%d shocks: volatility clustering present, so\n', ...
        nnz(lb_v2(:,2) < ALPHA_TEST), M);
fprintf('  inference on the eigenvalues uses the block bootstrap of Step 7c.\n');

%% =========================================================================
%  STEP 7 - SPECTRUM: EIGENVALUES d AND REALISED KURTOSES kappa
% =========================================================================
banner('STEP 7 - EIGENVALUE DECAY  +  KURTOSIS DECAY');

lam_med = median(eigenvalues_draws, 2);
lam_16  = quantile(eigenvalues_draws, 0.16, 2);
lam_84  = quantile(eigenvalues_draws, 0.84, 2);
kap_med = median(kappa_draws, 2);
kap_16  = quantile(kappa_draws, 0.16, 2);
kap_84  = quantile(kappa_draws, 0.84, 2);

print_decay('(A) K(Z) eigenvalue d_i', shock_lab, lam_med, lam_16, lam_84);
print_decay('(B) realised excess kurtosis kappa_i', shock_lab, kap_med, kap_16, kap_84);

[~, ord_kap] = sort(kap_med, 'descend');
fprintf('\n  Non-Gaussianity ranking (kappa, descending): ');
fprintf('%s ', shock_lab{ord_kap}); fprintf('\n');

%% =========================================================================
%  STEP 7b - NEAR-DEGENERACY TEST  (H0: lam_i = lam_i+1)
% =========================================================================
banner('STEP 7b - POSTERIOR NEAR-DEGENERACY TEST');

fprintf('  %-10s %12s %12s %12s %12s %10s\n', ...
        'Pair','P(gap>0|Y)','median(gap)','gap/lam_i','q16(gap)','degen?');
fprintf('  %s\n', repmat('-',1,72));

n_pairs       = M - 1;
posterior_pgt = zeros(n_pairs, 1);
gap_med_arr   = zeros(n_pairs, 1);
degen_flag    = false(n_pairs, 1);
for i = 1:n_pairs
    gap_draws        = eigenvalues_draws(i,:) - eigenvalues_draws(i+1,:);
    posterior_pgt(i) = mean(gap_draws > 0);
    gap_med_arr(i)   = median(gap_draws);
    rel_gap          = gap_med_arr(i) / max(lam_med(i), 1e-8);
    degen_flag(i)    = (posterior_pgt(i) < 0.95) || (rel_gap < DEGEN_THRESHOLD);
    fprintf('  %-5s-%-4s %10.4f   %10.4f   %10.4f   %10.4f   %s\n', ...
            shock_lab{i}, shock_lab{i+1}, posterior_pgt(i), gap_med_arr(i), ...
            rel_gap, quantile(gap_draws, 0.16), tern(degen_flag(i), '[!]', '  '));
end

conc_ratio_draws = eigenvalues_draws(1,:) ./ max(eigenvalues_draws(2,:), 1e-8);
kap_sorted_draws = sort(kappa_draws, 1, 'descend');
kap_conc_draws   = kap_sorted_draws(1,:) ./ max(kap_sorted_draws(2,:), 1e-8);
fprintf('\n  Concentration ratio lam_1/lam_2: median %.3f  [%.3f, %.3f]   P(>2|Y) = %.4f\n', ...
        median(conc_ratio_draws), quantile(conc_ratio_draws,0.16), ...
        quantile(conc_ratio_draws,0.84), mean(conc_ratio_draws > 2));
fprintf('  Concentration ratio kappa_(1)/kappa_(2): median %.3f  [%.3f, %.3f]   P(>2|Y) = %.4f\n', ...
        median(kap_conc_draws), quantile(kap_conc_draws,0.16), ...
        quantile(kap_conc_draws,0.84), mean(kap_conc_draws > 2));

figure('Name','Degeneracy Test','Position',[50 50 1100 450]);
subplot(1,2,1);
gap1_draws = eigenvalues_draws(1,:) - eigenvalues_draws(2,:);
histogram(gap1_draws, 40, 'Normalization','pdf', 'FaceColor',[0.4 0.6 0.85],'EdgeColor','none');
hold on; xline(0,'r-','LineWidth',1.5);
xline(median(gap1_draws),'b--','LineWidth',1.5,'Label','median');
xlabel('lambda_1 - lambda_2','Interpreter','none'); ylabel('Posterior density');
title(sprintf('Posterior of lambda_1 - lambda_2 (P(>0)=%.3f)', posterior_pgt(1)), ...
      'FontSize',10,'Interpreter','none');
grid on; set(gca,'FontSize',9,'Box','on');
subplot(1,2,2);
histogram(conc_ratio_draws, 40, 'Normalization','pdf','FaceColor',[0.4 0.7 0.5],'EdgeColor','none');
hold on; xline(1,'r-','LineWidth',1.5,'Label','degenerate');
xline(2,'k--','LineWidth',1.0,'Label','dominance');
xline(median(conc_ratio_draws),'b--','LineWidth',1.5,'Label','median');
xlabel('lambda_1 / lambda_2','Interpreter','none'); ylabel('Posterior density');
title(sprintf('Concentration ratio (median=%.2f)', median(conc_ratio_draws)), ...
      'FontSize',10,'Interpreter','none');
grid on; set(gca,'FontSize',9,'Box','on');
sgtitle('Near-degeneracy diagnostics','FontSize',11,'Interpreter','none');

%% =========================================================================
%  STEP 7c - MOVING-BLOCK BOOTSTRAP OF THE SPECTRUM
%    The posterior treats K_hat as deterministic given iota; under volatility
%    clustering K_hat is itself noisy.  Blocks of iota_pm are resampled and
%    the identification is re-run on each resample.
%
%    The resampled eigenvalues are matched to P_ref with the same greedy rule
%    used in Step 6 BEFORE anything is computed from them: kollo_identify
%    returns them sorted, so comparing lam(1) with lam(2) directly would test
%    a statement that is true by construction rather than the separation of
%    the two identified directions.
% =========================================================================
banner(sprintf('STEP 7c - MOVING-BLOCK BOOTSTRAP  (B=%d, block=%d)', BLOCK_BOOT_N, BLOCK_LEN));

nblk        = ceil(T_eff / BLOCK_LEN);
boot_lam    = zeros(M, BLOCK_BOOT_N);
boot_dom    = false(BLOCK_BOOT_N, 1);
boot_coldev = zeros(M, BLOCK_BOOT_N);        % 1 - |<Pb_k, P_ref_k>|
tic;
for b = 1:BLOCK_BOOT_N
    idx = block_indices(T_eff, BLOCK_LEN, nblk);
    [Pb, lamb]  = kollo_identify(iota_pm(idx,:), Kz, shrink_w);
    [ord, dots] = match_columns(Pb, P_ref);
    boot_lam(:,b)    = lamb(ord);
    boot_coldev(:,b) = 1 - dots;
    boot_dom(b)      = boot_lam(1,b) > boot_lam(2,b);
end
fprintf('  Bootstrap done in %.1fs.\n', toc);

boot_lam_ci   = quantile(boot_lam, [0.025 0.5 0.975], 2);   % M x 3
boot_dom_frac = mean(boot_dom);
boot_gap_ci   = quantile(boot_lam(1,:) - boot_lam(2,:), [0.025 0.5 0.975]);

fprintf('\n  Bootstrap interval of the eigenvalue of each identified direction:\n');
fprintf('  %-8s %12s %12s %12s\n', 'Shock','q2.5','median','q97.5');
fprintf('  %s\n', repmat('-',1,48));
for kk = 1:M
    fprintf('  %-8s %12.4f %12.4f %12.4f\n', shock_lab{kk}, ...
            boot_lam_ci(kk,1), boot_lam_ci(kk,2), boot_lam_ci(kk,3));
end
fprintf('\n  lambda_1 - lambda_2: median %.4f  CI [%.4f, %.4f]\n', ...
        boot_gap_ci(2), boot_gap_ci(1), boot_gap_ci(3));
fprintf('  Bootstrap P(lambda_1 > lambda_2) = %.3f\n', boot_dom_frac);
fprintf('  Mean column instability vs P_ref (0 = stable): ');
fprintf('%.3f  ', mean(boot_coldev,2)); fprintf('\n');
if boot_gap_ci(1) <= 0
    fprintf('  [!] The bootstrap interval of the gap includes 0.\n');
end

%% =========================================================================
%  STEP 8 - IMPACT MATRIX B  (original units)
% =========================================================================
banner('STEP 8 - IMPACT MATRIX B  (original units)');

B_med = median(B_draws, 3);
B_16  = quantile(B_draws, 0.16, 3);
B_84  = quantile(B_draws, 0.84, 3);

fprintf('  B(m,k) = impact of a unit-variance shock v_k on variable m,\n');
fprintf('  in the original unit of that variable.\n\n');
print_table(VAR_NAMES, shock_lab, B_med);

%% =========================================================================
%  STEP 8b - SIGN SIGNATURE OF EACH SHOCK  (no restriction is imposed)
%    A sign is reported as stable when the posterior agrees in >= 84% of the
%    draws; the tail-risk score counts how many canonical signs a shock
%    matches among the variables present in the system.
% =========================================================================
banner('STEP 8b - ECONOMIC SIGNATURE TABLE');

sign_agreement = zeros(M, M);
for m = 1:M
    for kk = 1:M
        b_draws = squeeze(B_draws(m,kk,:));
        s_med   = sign(median(b_draws));
        if s_med == 0
            sign_agreement(m,kk) = 0.5;
        else
            sign_agreement(m,kk) = mean(sign(b_draws) == s_med);
        end
    end
end

fprintf('  Sign pattern of B  (+/- : agreement >= 0.84,  +?/-? : uncertain)\n\n');
fprintf('  %-10s', 'Var\Shock');
for kk = 1:M, fprintf(' %8s', shock_lab{kk}); end
fprintf('\n  %s\n', repmat('-', 1, 10 + 9*M));
for m = 1:M
    fprintf('  %-10s', VAR_NAMES{m});
    for kk = 1:M
        s_med = sign(B_med(m,kk));
        if s_med > 0,     sym = tern(sign_agreement(m,kk) >= 0.84, '+', '+?');
        elseif s_med < 0, sym = tern(sign_agreement(m,kk) >= 0.84, '-', '-?');
        else,             sym = '0';
        end
        fprintf(' %8s', sym);
    end
    fprintf('\n');
end

% Canonical tail-risk signature: (variable, expected sign) pairs kept together,
% then filtered to the variables of this system.
TR_SPEC     = {'VIX',+1; 'CSPREAD',+1; 'DXY',+1; 'SP500',-1; 'DGS10',-1; 'AAA',-1};
has_var     = cellfun(@(v) any(strcmp(VAR_NAMES, v)), TR_SPEC(:,1));
TR_SPEC     = TR_SPEC(has_var, :);
target_idx  = cellfun(@(v) find(strcmp(VAR_NAMES, v), 1), TR_SPEC(:,1))';
target_vals = cell2mat(TR_SPEC(:,2))';
n_tgt       = numel(target_idx);
thr_full    = ceil(0.83 * n_tgt);     % thresholds scale with the score range
thr_partial = ceil(0.66 * n_tgt);
thr_anti    = floor(0.17 * n_tgt);

fprintf('\n  Scored on %d canonical variables: ', n_tgt);
for q = 1:n_tgt, fprintf('%s(%+d) ', VAR_NAMES{target_idx(q)}, target_vals(q)); end
fprintf('\n\n');

score = zeros(M,1);
for kk = 1:M
    score(kk) = sum(sign(B_med(target_idx, kk))' == target_vals);
end
fprintf('  %-8s %12s %15s %20s\n','Shock','kappa_med', ...
        sprintf('TR-score (max=%d)', n_tgt),'Reading');
fprintf('  %s\n', repmat('-', 1, 65));
for kk = 1:M
    if     score(kk) >= thr_full,    interp = 'TAIL-RISK candidate';
    elseif score(kk) >= thr_partial, interp = 'partial match';
    elseif score(kk) <= thr_anti,    interp = 'opposite sign (anti-tail)';
    else,                            interp = 'unclear';
    end
    fprintf('  %-8s %12.4f %15d   %s\n', shock_lab{kk}, kap_med(kk), score(kk), interp);
end

[max_score, idx_best] = max(score);
[~, idx_kappa]        = max(kap_med);
if nnz(score == max_score) > 1
    fprintf('\n  [!] Tie on the sign score (%d shocks at %d/%d): use the kurtosis-share\n', ...
            nnz(score == max_score), max_score, n_tgt);
    fprintf('      of Step 8c as the primary criterion.\n');
end
fprintf('\n  Best tail-risk candidate (sign score): %s  |  most non-Gaussian (kappa): %s\n', ...
        shock_lab{idx_best}, shock_lab{idx_kappa});

%% =========================================================================
%  STEP 8c - KURTOSIS-SHARE  (max-share selection at the fourth order)
%
%  Fourth cumulants of independent shocks are additive, so each variable's
%  residual excess kurtosis splits with no cross terms:
%      kurt_exc(u_m) = sum_k kappa_k * Btilde(m,k)^4,  Btilde = B / sd(u_m),
%      KS(m,k)       = kappa_k Btilde(m,k)^4 / sum_s kappa_s Btilde(m,s)^4.
%  This is the FEVD identity one order up; the tail shock is the argmax of the
%  aggregate share on the target variables.  Exact additivity needs mutual
%  independence, so the model-implied kurtosis is checked against the sample
%  one below; shares can leave [0,1] in draws with a platykurtic shock.
% =========================================================================
banner('STEP 8c - KURTOSIS-SHARE');

TARGET_VARS = VAR_NAMES;                       % subset to target one block
target_m    = find(ismember(VAR_NAMES, TARGET_VARS));
target_lab  = tern(numel(target_m) == M, 'ALL variables (system-wide)', strjoin(TARGET_VARS, ', '));

KS_draws        = zeros(M, M, N_DRAWS);        % variable x shock x draw
kurt_var_draws  = zeros(M, N_DRAWS);           % model-implied kurtosis of u_m
ks_target_draws = zeros(M, N_DRAWS);           % aggregate share on the targets
n_neg_kappa     = 0;
for s = 1:N_DRAWS
    Bs  = B_draws(:,:,s);
    Bt  = Bs ./ sqrt(sum(Bs.^2, 2));           % rows unit-norm
    Cmk = Bt.^4 .* kappa_draws(:,s)';
    kurt_var_draws(:,s)  = sum(Cmk, 2);
    KS_draws(:,:,s)      = Cmk ./ sum(Cmk, 2);
    agg                  = sum(Cmk(target_m,:), 1);
    ks_target_draws(:,s) = (agg / sum(agg))';
    n_neg_kappa          = n_neg_kappa + any(kappa_draws(:,s) < 0);
end

KS_med   = median(KS_draws, 3);
KS_16    = quantile(KS_draws, 0.16, 3);
KS_84    = quantile(KS_draws, 0.84, 3);
ks_t_med = median(ks_target_draws, 2);
ks_t_16  = quantile(ks_target_draws, 0.16, 2);
ks_t_84  = quantile(ks_target_draws, 0.84, 2);
[~, argmax_draw] = max(ks_target_draws, [], 1);
ks_winner_prob   = mean(argmax_draw(:) == 1:M, 1)';

fprintf('  KS(m,k): posterior median share of each variable''s residual excess\n');
fprintf('  kurtosis generated by each shock.\n\n');
print_table(VAR_NAMES, shock_lab, KS_med);

fprintf('\n  Aggregate kurtosis-share on target [%s]:\n', target_lab);
fprintf('  %-8s %10s %10s %10s %12s\n','Shock','median','q16','q84','P(argmax)');
fprintf('  %s\n', repmat('-',1,56));
for kk = 1:M
    fprintf('  %-8s %10.4f %10.4f %10.4f %12.4f\n', shock_lab{kk}, ...
            ks_t_med(kk), ks_t_16(kk), ks_t_84(kk), ks_winner_prob(kk));
end
[~, idx_ks] = max(ks_t_med);
fprintf('\n  Tail shock by kurtosis-share: %s (share %.1f%%, P(argmax)=%.2f)\n', ...
        shock_lab{idx_ks}, 100*ks_t_med(idx_ks), ks_winner_prob(idx_ks));
if n_neg_kappa > 0
    fprintf('  [!] %d/%d draws contain a platykurtic shock (kappa<0).\n', n_neg_kappa, N_DRAWS);
end

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

figure('Name','Kurtosis-share','Position',[50 50 1150 480]);
subplot(1,2,1);
imagesc(KS_med*100); colorbar;
set(gca,'XTick',1:M,'XTickLabel',shock_lab,'YTick',1:M,'YTickLabel',VAR_NAMES,'FontSize',9);
title('Kurtosis-share KS(m,k) [%]','Interpreter','none','FontSize',10);
xlabel('Shock'); ylabel('Variable');
subplot(1,2,2);
bar(100*ks_t_med, 'FaceColor',[0.4 0.6 0.85]); hold on;
errorbar((1:M)', 100*ks_t_med, 100*(ks_t_med-ks_t_16), 100*(ks_t_84-ks_t_med), 'k.', 'LineWidth', 1);
set(gca,'XTick',1:M,'XTickLabel',shock_lab,'FontSize',9,'Box','on');
ylabel('% of target kurtosis');
title(sprintf('Aggregate kurtosis-share, target: %s [16,84]', target_lab), ...
      'Interpreter','none','FontSize',10);
grid on;
sgtitle('Kurtosis-share: which shock generates the tails','FontSize',11,'Interpreter','none');

%% =========================================================================
%  STEP 9 - IRF PLOTS
% =========================================================================
banner('STEP 9 - IRF PLOTS');

IRF_med = median(IRF_draws, 4);
IRF_16  = quantile(IRF_draws, 0.16, 4);
IRF_84  = quantile(IRF_draws, 0.84, 4);
IRF_05  = quantile(IRF_draws, 0.05, 4);
IRF_95  = quantile(IRF_draws, 0.95, 4);
horizon = 0:IRF_HOR;

figure('Name','IRF Full Grid','Position',[50 50 1600 1000]);
for sk = 1:M
    for vm = 1:M
        subplot(M, M, (vm-1)*M + sk);
        plot_band(horizon, squeeze(IRF_med(vm,:,sk)), squeeze(IRF_16(vm,:,sk)), ...
                  squeeze(IRF_84(vm,:,sk)), squeeze(IRF_05(vm,:,sk)), squeeze(IRF_95(vm,:,sk)), 1.5);
        xlim([0 IRF_HOR]);
        if vm == 1, title(shock_lab{sk},'FontSize',8,'Interpreter','none'); end
        if sk == 1, ylabel(VAR_NAMES{vm},'FontSize',7,'Interpreter','none'); end
        set(gca,'FontSize',7);
    end
end
sgtitle('IRF: rows = variables, columns = structural shocks (original units)', ...
        'FontSize',11,'Interpreter','none');

figure('Name','IRF to v_1','Position',[50 50 1400 700]);
for vm = 1:M
    subplot(ceil(M/4), min(M,4), vm);
    plot_band(horizon, squeeze(IRF_med(vm,:,1)), squeeze(IRF_16(vm,:,1)), ...
              squeeze(IRF_84(vm,:,1)), squeeze(IRF_05(vm,:,1)), squeeze(IRF_95(vm,:,1)), 2);
    title(VAR_NAMES{vm},'FontSize',10,'Interpreter','none');
    xlabel('Days'); xlim([0 IRF_HOR]); set(gca,'FontSize',9,'Box','on');
end
sgtitle(sprintf('IRF to %s (d = %.3f) - most non-Gaussian shock', shock_lab{1}, lam_med(1)), ...
        'FontSize',12,'Interpreter','none');

figure('Name','Eigenvalue Decay','Position',[50 50 700 450]);
boxplot(eigenvalues_draws', 'Labels', shock_lab); hold on;
plot(1:M, lam_med,'b-o','LineWidth',2,'MarkerSize',6,'MarkerFaceColor','b');
yline(0,'r--','LineWidth',1);
xlabel('Structural shock'); ylabel('Excess kurtosis (d_i)','Interpreter','none');
title('Eigenvalue decay of M_hat = I star (K_hat - K_z)','Interpreter','none','FontSize',10);
set(gca,'FontSize',10,'Box','on'); grid on;

%% =========================================================================
%  STEP 10 - ORTHOGONALITY CHECK
% =========================================================================
banner('STEP 10 - ORTHOGONALITY CHECK');

cov_V_med      = median(cov_V_draws, 3);
cov_V_mat2d    = reshape(cov_V_draws, M*M, N_DRAWS);
cov_V_diag_drw = cov_V_mat2d(sub2ind([M M], 1:M, 1:M), :);
cov_V_diag16   = quantile(cov_V_diag_drw', 0.16)';
cov_V_diag84   = quantile(cov_V_diag_drw', 0.84)';
corr_Vref      = cov_V_ref ./ (std(V_ref)' * std(V_ref));
cov_V          = cov_V_ref;                 % alias kept for downstream scripts

fprintf('  (A) Median Cov(v) over %d draws (should be ~ I_%d):\n\n', N_DRAWS, M);
print_table(shock_lab, shock_lab, cov_V_med);
fprintf('\n  Diagonal [16, 84] posterior quantiles:\n');
for m = 1:M
    fprintf('    %-5s %.4f  [%.4f, %.4f]\n', shock_lab{m}, cov_V_med(m,m), ...
            cov_V_diag16(m), cov_V_diag84(m));
end
fprintf('  Max |median Cov(v) - I_M|: %.6f\n', max(abs(cov_V_med - eye(M)), [], 'all'));

fprintf('\n  (B) Cov(V_ref) at the posterior mean:\n\n');
print_table(shock_lab, shock_lab, cov_V_ref);
fprintf('\n  Max |Cov(V_ref) - I_M|: %.6f\n', max(abs(cov_V_ref - eye(M)), [], 'all'));

fprintf('\n  (C) Corr(V_ref):\n\n');
print_table(shock_lab, shock_lab, corr_Vref);

%% =========================================================================
%  STEP 11 - NARRATIVE CHECK  (shocks against historical events)
% =========================================================================
banner('STEP 11 - NARRATIVE CHECK');

dates_vec = datetime(T_raw.Date(p+1:end), 'InputFormat','yyyy-MM-dd');
dates_vec = dates_vec(:);

events = {'1998-08-17','Russia/LTCM'; '2001-09-11','9/11'; '2001-12-02','Enron';
          '2007-08-09','BNP Paribas'; '2008-09-15','Lehman'; '2010-05-06','Flash Crash';
          '2011-08-05','US downgrade'; '2020-03-16','COVID-19'; '2023-03-10','SVB'};

figure('Name','Narrative Check','Position',[50 50 1400 900]);
for kk = 1:M
    ax = subplot(M, 1, kk);
    v_k      = V_ref(:,kk);
    vk_draws = squeeze(Vt_draws(:,kk,:));
    v_k16    = quantile(vk_draws, 0.16, 2);
    v_k84    = quantile(vk_draws, 0.84, 2);

    fill([dates_vec; flipud(dates_vec)], [v_k16; flipud(v_k84)], ...
         [0.75 0.85 0.95],'EdgeColor','none','FaceAlpha',0.6); hold on;
    plot(dates_vec, v_k,'Color',[0.1 0.35 0.65],'LineWidth',1.0);
    yline(0,'k-','LineWidth',0.5);
    add_event_lines(events, dates_vec, 6);

    idx_ext = abs(v_k) > 3*std(v_k);
    if any(idx_ext), scatter(dates_vec(idx_ext), v_k(idx_ext), 10, 'r', 'filled'); end

    ylabel(sprintf('%s (d=%.2f)', shock_lab{kk}, lam_med(kk)),'FontSize',8,'Interpreter','none');
    set(ax,'FontSize',8,'Box','on','XGrid','on');
    xlim([dates_vec(1), dates_vec(end)]);
end
sgtitle({sprintf('Structural shocks - narrative check | BVAR(%d)', p), ...
         'Shaded = 68% band | red dots = |v_k| > 3 sigma'},'FontSize',10,'Interpreter','none');

%% =========================================================================
%  STEP 11b - ONE FIGURE PER SHOCK  (spikes and the months behind them)
% =========================================================================
SPIKE_THRESH = 2.5;                 % highlight |v_k| > SPIKE_THRESH * sigma

for kk = 1:M
    figure('Name',sprintf('%s time series',shock_lab{kk}),'Position',[50 50 1500 450]);
    vk   = V_ref(:, kk);
    sigk = std(vk);

    plot(dates_vec, vk, 'Color',[0.55 0.65 0.80], 'LineWidth', 0.6); hold on;
    yline( SPIKE_THRESH*sigk, 'r--', 'LineWidth', 0.8);
    yline(-SPIKE_THRESH*sigk, 'r--', 'LineWidth', 0.8);
    yline(0, 'k-', 'LineWidth', 0.4);
    big = abs(vk) > SPIKE_THRESH * sigk;
    scatter(dates_vec(big), vk(big), 12, 'r', 'filled', 'MarkerFaceAlpha', 0.7);
    add_event_lines(events, dates_vec, 7);

    ylabel(sprintf('%s  (sigma = 1, d = %.1f)', shock_lab{kk}, lam_med(kk)), ...
           'FontSize',10,'Interpreter','none');
    xlabel('Date','FontSize',9);
    title(sprintf('Structural shock %s - |v| > %.1f sigma highlighted', shock_lab{kk}, SPIKE_THRESH), ...
          'FontSize',11,'Interpreter','none');
    xlim([dates_vec(1), dates_vec(end)]); set(gca,'FontSize',9,'Box','on','XGrid','on');

    [vk_sort, vk_idx] = sort(abs(vk), 'descend');
    fprintf('\n  Top-10 |%s| spikes:\n', shock_lab{kk});
    fprintf('  %-14s  %8s\n', 'Date', 'abs value');
    fprintf('  %s\n', repmat('-',1,26));
    for i = 1:min(10, numel(vk_idx))
        fprintf('  %-14s  %8.3f\n', string(dates_vec(vk_idx(i)),'yyyy-MM-dd'), vk_sort(i));
    end
end

%% =========================================================================
%  STEP 12 - FEVD  (share of the h-step forecast error variance, horizons 0..h)
% =========================================================================
banner('STEP 12 - FEVD  (rows = variables, columns = shocks)');

fevd_draws = zeros(M, M, IRF_HOR+1, N_DRAWS);
for s = 1:N_DRAWS
    cum_sq = zeros(M, M);
    for h = 0:IRF_HOR
        Th     = double(squeeze(IRF_draws(:,h+1,:,s)));
        cum_sq = cum_sq + Th.^2;
        fevd_draws(:,:,h+1,s) = cum_sq ./ sum(cum_sq, 2);
    end
end
fevd_med = median(fevd_draws, 4);
fevd_16  = quantile(fevd_draws, 0.16, 4);
fevd_84  = quantile(fevd_draws, 0.84, 4);

for h_rep = [1, 5, 20]
    if h_rep > IRF_HOR, continue; end
    fprintf('\n  FEVD at h=%d (median):\n', h_rep);
    print_table(VAR_NAMES, shock_lab, fevd_med(:,:,h_rep+1));
end

fprintf('\n  FEVD of %s across variables (median [16, 84], %%):\n', shock_lab{1});
fprintf('  %-10s %12s %12s %12s\n','Variable','h=1','h=5','h=20');
fprintf('  %s\n', repmat('-',1,52));
for m = 1:M
    fprintf('  %-10s', VAR_NAMES{m});
    for h = [1 5 20]
        fprintf(' %5.1f[%4.1f,%4.1f]', 100*fevd_med(m,1,h+1), ...
                100*fevd_16(m,1,h+1), 100*fevd_84(m,1,h+1));
    end
    fprintf('\n');
end

%% =========================================================================
%  STEP 12b - CRISIS DECOMPOSITION  (day-0 contributions B * v_t)
% =========================================================================
banner('STEP 12b - CRISIS DECOMPOSITION');

crisis_dates = {'2008-09-15','Lehman'; '2020-03-16','COVID-19'; '2023-03-10','SVB'};
h_win        = min(5, IRF_HOR+1);
cum_irf5     = squeeze(sum(Theta_ref(:, 1:h_win, :), 2));      % M x M
HD_crisis    = struct('label',{},'date',{},'t_ev',{},'delta_day0',{},'delta_cum5',{});

for ec = 1:size(crisis_dates, 1)
    ev_d = datetime(crisis_dates{ec,1}, 'InputFormat','yyyy-MM-dd');
    t_ev = find(dates_vec >= ev_d, 1);
    if isempty(t_ev)
        fprintf('\n  %s (%s): not in sample, skipped.\n', crisis_dates{ec,2}, crisis_dates{ec,1});
        continue
    end
    delta_day0 = B_ref     .* repmat(V_ref(t_ev,:), M, 1);
    delta_cum5 = cum_irf5  .* repmat(V_ref(t_ev,:), M, 1);

    fprintf('\n  === %s (%s, t=%d) ===\n', crisis_dates{ec,2}, crisis_dates{ec,1}, t_ev);
    fprintf('  Day-0 decomposition of the forecast error u_m:\n');
    print_table(VAR_NAMES, shock_lab, delta_day0);
    fprintf('  %d-day cumulative effect of the event-day shock:\n', h_win);
    print_table(VAR_NAMES, shock_lab, delta_cum5);

    HD_crisis(end+1) = struct('label', crisis_dates{ec,2}, 'date', crisis_dates{ec,1}, ...
                              't_ev', t_ev, 'delta_day0', delta_day0, ...
                              'delta_cum5', delta_cum5); %#ok<SAGROW>
end

if ~isempty(HD_crisis)
    figure('Name','Crisis decomposition','Position',[50 50 1300 700]);
    for vi = 1:M
        subplot(1, M, vi);
        bar_data = cell2mat(arrayfun(@(e) e.delta_day0(vi,:)', HD_crisis, 'UniformOutput', false));
        bar(bar_data'); hold on;
        xticks(1:numel(HD_crisis)); xticklabels({HD_crisis.label});
        title(VAR_NAMES{vi},'FontSize',10,'Interpreter','none');
        ylabel('Contribution to u_m (day 0)','Interpreter','none');
        legend(shock_lab,'Location','best','FontSize',7);
        grid on; set(gca,'FontSize',8,'Box','on');
    end
    sgtitle('Day-0 contributions at crisis events','FontSize',11,'Interpreter','none');
end

%% =========================================================================
%  STEP 13 - SAVE
% =========================================================================
banner('STEP 13 - SAVING');

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
banner('ALL STEPS COMPLETED');

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function [P, lam] = kollo_identify(iota, Kz, shrink_w)
% KOLLO_IDENTIFY  Rotation from the fourth moment of pre-whitened residuals.
%   Z      = kron-product design of iota                  (n x M^2)
%   K_hat  = Z'Z / n
%   K_star = shrink_w*K_hat + (1-shrink_w)*Kz             [Petrova shrinkage]
%   M_hat  = sum of the DIAGONAL M x M blocks of (K_star - Kz)
%            = I_M star (K_star - Kz) = FOBI/Kollo K(Z), Kollo (2008) eq. 4.1
%   Columns of P are sorted by descending eigenvalue; order and sign are
%   anchored by the caller against P_ref.
    [n, M] = size(iota);
    Z      = repmat(iota, 1, M) .* kron(iota, ones(1, M));
    K_hat  = (Z' * Z) / n;
    D      = shrink_w * K_hat + (1 - shrink_w) * Kz - Kz;

    M_hat = zeros(M, M);
    for ii = 1:M
        ri    = (ii-1)*M+1 : ii*M;
        M_hat = M_hat + D(ri, ri);
    end
    M_hat      = (M_hat + M_hat') / 2;          % symmetrise -> real eigenvalues
    [P, Lam]   = eig(M_hat);
    [lam, idx] = sort(diag(Lam), 'descend');
    P          = P(:, idx);
end

function [ord, dots] = match_columns(P, P_ref)
% Greedy assignment of the columns of P to those of P_ref by largest absolute
% inner product, taken in reference order so the most non-Gaussian direction is
% matched first.  ord(r) is the column of P assigned to reference column r;
% dots(r) is the absolute inner product of that match (1 = same direction).
    M    = size(P_ref, 2);
    d    = abs(P' * P_ref);
    used = false(M, 1);
    ord  = zeros(M, 1);
    dots = zeros(M, 1);
    for r = 1:M
        col = d(:,r);  col(used) = -Inf;
        [dots(r), ord(r)] = max(col);
        used(ord(r)) = true;
    end
end

function ir = companion_irf(Phi, B, p, H)
% IRF of a VAR(p) with impact matrix B: ir(:,h+1,:) = J A^h J' * B.
% Phi is (M*p+1) x M with the constant in the last row.
    M    = size(B, 1);
    J    = [eye(M), zeros(M, M*(p-1))];
    A    = [Phi(1:M*p, :)'; eye(M*(p-1)), zeros(M*(p-1), M)];
    ir   = zeros(M, H+1, M);
    A_h  = eye(M*p);
    for h = 0:H
        ir(:, h+1, :) = (J * A_h * J') * B;
        A_h = A_h * A;
    end
end

function idx = block_indices(T, len, nblk)
% Overlapping moving blocks: nblk starting points, trimmed to T observations.
    starts = randi(T - len + 1, nblk, 1);
    idx    = reshape((starts + (0:len-1))', [], 1);
    idx    = idx(1:T);
end

function XL = lagmat(x, q)
% x lagged 1..q, zero-padded at the top: the caller drops the first q rows.
    n  = numel(x);
    XL = zeros(n, q);
    for s = 1:q
        XL(s+1:end, s) = x(1:end-s);
    end
end

function [LM, pval] = lm_test(y, Z, df)
% LM = n*R^2 of the auxiliary regression of y on Z (constant included in Z),
% chi2(df) under the null of no explanatory power.  Used by the
% Breusch-Godfrey test (Step 3b) and the ARCH-LM test (Step 6c).
    e    = y - Z*(Z\y);
    R2   = max(0, 1 - (e'*e) / sum((y - mean(y)).^2));
    LM   = numel(y) * R2;
    pval = 1 - chi2cdf(LM, df);
end

function [Q, pval] = local_lb(x, h)
% Ljung-Box statistic: Q = T(T+2) sum_k rho_k^2/(T-k) ~ chi2(h).
    x  = x(:) - mean(x);
    T  = numel(x);
    c0 = sum(x.^2) / T;
    Q  = 0;
    for kk = 1:h
        ck = sum(x(1+kk:end) .* x(1:end-kk)) / T;
        Q  = Q + (ck/c0)^2 / (T-kk);
    end
    Q    = T*(T+2)*Q;
    pval = 1 - chi2cdf(Q, h);
end

function a = hill(x, mh)
% Hill tail index from the mh largest observations of the positive vector x.
    x = sort(x, 'descend');
    if numel(x) > mh && x(mh+1) > 0
        a = 1 / (mean(log(x(1:mh))) - log(x(mh+1)));
    else
        a = NaN;
    end
end

function banner(txt)
    line = repmat('=', 1, 70);
    fprintf('\n%s\n %s\n%s\n', line, txt, line);
end

function print_table(row_lab, col_lab, A)
% Labelled matrix, one row per row_lab and one column per col_lab.
    fprintf('  %-10s', '');
    for j = 1:numel(col_lab), fprintf(' %9s', col_lab{j}); end
    fprintf('\n  %s\n', repmat('-', 1, 10 + 10*numel(col_lab)));
    for i = 1:numel(row_lab)
        fprintf('  %-10s', row_lab{i});
        fprintf(' %9.4f', A(i,:));
        fprintf('\n');
    end
end

function print_decay(title_str, lab, med, q16, q84)
% Posterior median and [16,84] band of a spectrum, with a bar chart.
    fprintf('\n  %s\n', title_str);
    fprintf('  %-8s %10s %10s %10s   %s\n', 'Shock','median','q16','q84','');
    fprintf('  %s\n', repmat('-',1,58));
    for i = 1:numel(med)
        bar_str = repmat('|', 1, max(0, round(med(i)/max(med)*25)));
        fprintf('  %-8s %10.4f %10.4f %10.4f   %s\n', lab{i}, med(i), q16(i), q84(i), bar_str);
    end
end

function plot_band(h, med, lo68, hi68, lo90, hi90, lw)
% Median response with 68% and 90% bands.
    fill([h, fliplr(h)], [lo90, fliplr(hi90)], [0.85 0.85 0.85], 'EdgeColor','none'); hold on;
    fill([h, fliplr(h)], [lo68, fliplr(hi68)], [0.65 0.75 0.90], 'EdgeColor','none');
    plot(h, med, 'b-', 'LineWidth', lw);
    yline(0, 'r--', 'LineWidth', 0.8);
end

function add_event_lines(events, dates, fs)
% Vertical markers for the historical events inside the sample.
    for e = 1:size(events,1)
        ev_d = datetime(events{e,1}, 'InputFormat','yyyy-MM-dd');
        if ev_d >= dates(1) && ev_d <= dates(end)
            xline(ev_d, '--', 'Color',[0.75 0.10 0.10], 'LineWidth', 0.9, ...
                  'Label', events{e,2}, 'LabelOrientation','horizontal', ...
                  'LabelVerticalAlignment','bottom', 'FontSize', fs, 'Interpreter','none');
        end
    end
end

function out = tern(cond, a, b)
    if cond, out = a; else, out = b; end
end

function lab = compose_labels(prefix, n)
    lab = arrayfun(@(i) sprintf('%s%d', prefix, i), 1:n, 'UniformOutput', false);
end
