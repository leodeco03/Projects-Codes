% =========================================================================
%  BVAR IN LEVELS — DIAGNOSTICS + MINNESOTA PRIOR (GLP FULL SEQUENTIAL)
%  Using Ferroni & Canova (2025) bvar_ toolbox
% =========================================================================
%
%  STRUCTURE:
%    BLOCK A (Steps 1-6):  Flat-prior BVAR at p in LAG_GRID.
%                          OLS behaviour, residual diagnostics,
%                          lag selection via marginal likelihood.
%
%    BLOCK B (Steps 7a-f): Minnesota BVAR at fixed p_MN.
%                          STRICT-GLP (2015) hyperparameter selection.
%                          Following GLP p.439-440, the hyperparameters
%                          treated as random are  (lambda, mu, delta).
%                          The lag decay theta is FIXED at 1, as in
%                          GLP (p.439, covariance formula proportional
%                          to 1/s^2 where s is the lag).
%                          Via bvar_max_hyper:
%                            Stage 1: optimise tau only (1-D, fast)
%                            Stage 2: optimise (tau, lambda, mu)
%                                     jointly, warm-start from Stage 1
%                                     [decay fixed at theta=1, GLP-strict]
%                            Stage 3: multi-start robustness check
%                          The BVAR posterior in Steps 7b-f is estimated
%                          at the STAGE-2 optimal hyperparameters.
%                          A grid search on tau (legacy Block A approach)
%                          is ALSO reported for comparability.
%
%  DEVIATION FROM GLP (justified):
%    - omega (toolbox) is a scalar weight on IW prior dummies, NOT the
%      vector psi(n x 1) that GLP treat as an additional hyperparameter.
%      True psi-vector implementation would require modifying bvar_.m
%      internals.  We keep omega at its default (2).  With T_eff >> omega,
%      the posterior of Sigma is dominated by the data (Sigma_bar ~=
%      Sigma_OLS), so this simplification has negligible impact on IRF
%      identification via ICA.  Declared limitation.
%
%  CONVENTION — minn_prior_tau in Ferroni-Canova toolbox:
%    Var(phi_{jj,1}) = Sigma_jj / (tau * s_j)^2
%    LARGER tau = TIGHTER prior (stronger RW shrinkage)
%    SMALLER tau = LOOSER prior (approaches OLS)
%    Equivalence: lambda_Litterman = 1/tau
%
%  GLP (2015) HYPERPARAMETER MAP (toolbox → paper):
%    minn_prior_tau    = 1/lambda   (overall tightness)
%    minn_prior_decay  = theta      (lag decay)
%    minn_prior_lambda = delta      (dummy-initial-obs prior / co-persistence)
%    minn_prior_mu     = mu         (sum-of-coefficients prior)
%    minn_prior_omega  = psi        (prior on Sigma)
%
%  References:
%    Giannone, Lenza & Primiceri (2015, RESTUD) — sequential empirical Bayes
%    Banbura, Giannone & Reichlin (2010)        — large BVAR with Minnesota
%    Sims, Stock & Watson (1990)                — OLS in levels, I(1) consistency
%    Kilian & Lutkepohl (2017) ch.2,5           — lag selection + Bayesian VARs
%    Litterman (1986)                           — Minnesota prior
%    Lanne, Meitz & Saikkonen (2017)            — ICA identification
% =========================================================================

clear; clc; close all;

% =========================================================================
%  TOOLBOX PATH SETUP
% =========================================================================
addpath(fullfile(pwd, 'bvar_Ferroni', 'bvartools'));
addpath(fullfile(pwd, 'bvar_Ferroni', 'cmintools'));

% ── helpers ──────────────────────────────────────────────────────────────
flag_str    = @(p,a,lbl) repmat(lbl, 1, double(p < a));
logdet_chol = @(A) 2*sum(log(diag(chol(A))));

% =========================================================================
%  CONFIGURATION  (all user-tunable parameters in one place)
% =========================================================================
CSV_FILE  = 'financial_data_aligned_log.csv';

% ── Block A ──────────────────────────────────────────────────────────────
LAG_GRID  = [3, 4, 5];          % candidate lags for flat-prior selection

% ── Block B ──────────────────────────────────────────────────────────────
p_MN      = 22;                 % fixed a priori: 22 trading days ≈ 1 month
%   Rationale: Banbura-Giannone-Reichlin (2010) use 12 lags for monthly
%   data; the analogous choice for daily financial data is ~22 trading days.

% ── GLP hyperparameter starting values (toolbox default convention) ───────
%   Mapping (Ferroni-Canova toolbox  ->  GLP 2015 paper):
%     tau    = 1/lambda     OPTIMISED   (overall tightness)
%     decay  = theta        FIXED at 1  (GLP: Var prop to 1/s^2)
%     lambda = delta        OPTIMISED   (dummy-initial-obs prior / co-persistence)
%     mu     = mu           OPTIMISED   (sum-of-coefficients prior)
%     omega  ~ psi (approx) FIXED at 2  (toolbox limitation, see header)
HP0(1) = 3;    % tau       overall tightness       (1/lambda in GLP)
HP0(2) = 1;    % decay     GLP-FIXED at theta = 1  (Litterman, GLP p.439)
HP0(3) = 5;    % lambda    dummy-initial-obs       (delta in GLP) -- optimised
HP0(4) = 2;    % mu        sum-of-coefficients     (mu in GLP)    -- optimised
HP0(5) = 2;    % omega     prior on Sigma (scalar approx of psi)  -- fixed

% ── Grid search (legacy, kept for robustness comparison) ─────────────────
TAU_GRID  = [0.1, 1, 5, 8, 10, 14, 16, 18, 19, 20, 22, 25];
nTau      = numel(TAU_GRID);
DECAY_FIX = 1;                 % GLP-FIXED at theta = 1 (paper p.439)

% ── Common settings ───────────────────────────────────────────────────────
H_BG      = 10;                % Breusch-Godfrey lags
Q_ARCH    = 5;                 % ARCH-LM lags
ALPHA     = 0.05;              % significance level
HILL_FRAC = 0.05;              % Hill estimator tail fraction (top 10%)
IRF_HOR   = 20;                % IRF horizon (trading days)
N_DRAWS   = 1000;              % posterior draws (increased for MC accuracy)
%   Rationale for N_DRAWS=1000:
%   With k=M*p_MN+1=199 regressors per equation, 500 draws gives
%   non-negligible MC variance in posterior means. 1000 draws reduces
%   this by factor sqrt(2) at modest computational cost.

% ── Multi-start grid for Stage 2 robustness ──────────────────────────────
%   Rows: [tau_init, delta_init, mu_init]  (decay FIXED at 1 GLP-strict)
%         col1=tau, col2=lambda(=delta_GLP dummy-init), col3=mu(=mu_GLP sum-of-coef)
MULTISTART = [
  % punti vicini ai default letteratura
  3.0,   5.0,   2.0;
  % punti vicini all'ottimo Stage-1 (tau noto, dummy ai default)
  7.0,   5.0,   2.0;
  % punti nella zona media dello spazio
  5.0,   50.0,  100.0;
  % punti vicini all'ottimo atteso (informati dall'analisi)
  7.0,   90.0,  350.0;
  % punti agli estremi dei bounds
  10.0,  90.0,  200.0;
  3.0,   20.0,  50.0 ];
VAR_NAMES = {'SP500','VIX','CSPREAD','AAA', ...
             'DGS10','SLOPE','DXY'};
M    = numel(VAR_NAMES);
nlag = numel(LAG_GRID);

% =========================================================================
%  STEP 1 — DATA LOADING (LEVELS, NO TRANSFORMATION)
% =========================================================================
fprintf('\n%s\n STEP 1 — DATA LOADING (LEVELS)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));

T_raw    = readtable(CSV_FILE);
Y_levels = zeros(height(T_raw), M);
for m = 1:M
    Y_levels(:,m) = T_raw.(VAR_NAMES{m});
end
T_full = size(Y_levels, 1);

fprintf(' CSV loaded: %d rows x %d columns\n', height(T_raw), width(T_raw));
fprintf(' Variables: %d  |  Observations: %d\n', M, T_full);
fprintf('\n %-10s %10s %10s %10s %10s\n','Variable','Mean','Std','Min','Max');
fprintf(' %s\n', repmat('-',1,56));
for m = 1:M
    fprintf(' %-10s %10.4f %10.4f %10.4f %10.4f\n', ...
        VAR_NAMES{m}, mean(Y_levels(:,m)), std(Y_levels(:,m)), ...
        min(Y_levels(:,m)), max(Y_levels(:,m)));
end

% =========================================================================
%  STEP 2 — FLAT-PRIOR BVAR(p)  [Block A, OLS posterior]
% =========================================================================
fprintf('\n%s\n STEP 2 — FLAT-PRIOR BVAR(p)  [Block A, OLS posterior]\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
fprintf(' p      T_eff   k      ln|Sigma|         cond(X''X)\n');
fprintf(' %s\n', repmat('-',1,60));

BVAR_flat = cell(nlag, 1);
opt_flat  = struct('K', N_DRAWS);

for i = 1:nlag
    p     = LAG_GRID(i);
    T_eff = T_full - p;
    k     = M*p + 1;

    BVAR_flat{i} = bvar_(Y_levels, p, opt_flat);

    B_pm   = mean(BVAR_flat{i}.Phi_draws, 3);
    Sig_pm = mean(BVAR_flat{i}.Sigma_draws, 3);

    X = zeros(T_eff, k);
    for l = 1:p
        X(:, (l-1)*M+1 : l*M) = Y_levels(p-l+1:end-l, :);
    end
    X(:, end) = 1;

    fprintf(' p=%-3d  T_eff=%-5d  k=%-4d  ln|Sig|=%9.4f  cond=%9.2e\n', ...
            p, T_eff, k, logdet_chol(Sig_pm + 1e-14*eye(M)), cond(X'*X));

    BVAR_flat{i}.XX     = X;
    BVAR_flat{i}.T_eff  = T_eff;
    BVAR_flat{i}.k      = k;
    BVAR_flat{i}.p      = p;
    BVAR_flat{i}.B_pm   = B_pm;
    BVAR_flat{i}.Sig_pm = Sig_pm;
    BVAR_flat{i}.e_pm   = squeeze(mean(BVAR_flat{i}.e_draws, 3));
end

% =========================================================================
%  STEP 3 — LAG SELECTION  (flat-prior marginal likelihood, Block A)
% =========================================================================
fprintf('\n%s\n STEP 3 — LAG SELECTION  (flat-prior ML, Block A)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
fprintf(' NOTE: p_flat is for Block A diagnostics only.\n');
fprintf('       Block B uses fixed p_MN = %d (GLP-style).\n\n', p_MN);

lnML_flat = zeros(nlag, 1);
for i = 1:nlag
    lnML_flat(i) = BVAR_flat{i}.logmlike;
end
[~, best_idx] = max(lnML_flat);
p_flat        = LAG_GRID(best_idx);

fprintf(' %-6s %-8s %-22s %-10s\n','p','nu','ln ML','Delta');
fprintf(' %s\n', repmat('-',1,50));
for i = 1:nlag
    T_eff = BVAR_flat{i}.T_eff;
    k     = BVAR_flat{i}.k;
    nu    = T_eff - k;
    flag  = '';
    if i == best_idx, flag = ' <- BEST (Block A)'; end
    fprintf(' p=%-3d  nu=%-6d  ln ML=%12.4f  Delta=%10.4f%s\n', ...
            LAG_GRID(i), nu, lnML_flat(i), lnML_flat(i)-max(lnML_flat), flag);
end
fprintf('\n Block A optimal lag: p_flat = %d\n', p_flat);
fprintf(' Block B fixed lag:   p_MN   = %d  (22 trading days, GLP-style)\n', p_MN);
fprintf(' Information Criteria (p_flat=%d):\n', p_flat);
disp(BVAR_flat{best_idx}.InfoCrit);

U_flat_opt = BVAR_flat{best_idx}.e_pm;

% =========================================================================
%  STEP 4 — RESIDUAL DIAGNOSTICS  (Block A, all lags)
% =========================================================================
fprintf('\n%s\n STEP 4 — RESIDUAL DIAGNOSTICS\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
fprintf(' Tests: BG(%d) | JB | ARCH(%d)   (alpha = %.2f)\n\n', H_BG, Q_ARCH, ALPHA);

results_diag = zeros(nlag*M, 12);
row_idx      = 0;

for i = 1:nlag
    p     = BVAR_flat{i}.p;
    U     = BVAR_flat{i}.e_pm;
    X     = BVAR_flat{i}.XX;
    T_eff = BVAR_flat{i}.T_eff;
    k     = BVAR_flat{i}.k;

    fprintf(' +-- p = %d  (T_eff=%d, k=%d)', p, T_eff, k);
    if i == best_idx, fprintf('  <- Block A optimal'); end
    fprintf('\n');

    for j = 1:M
        u_j = U(:,j);

        % (A) BREUSCH-GODFREY: LM test for serial correlation
        U_lag_bg = zeros(T_eff, H_BG);
        for s = 1:H_BG
            U_lag_bg(s+1:end, s) = u_j(1:end-s);
        end
        idx_bg  = H_BG+1 : T_eff;
        Z_bg    = [X(idx_bg,:), U_lag_bg(idx_bg,:)];
        u_bg    = u_j(idx_bg);
        e_bg    = u_bg - Z_bg*(Z_bg\u_bg);
        R2_bg   = max(0, 1-(e_bg'*e_bg)/sum((u_bg-mean(u_bg)).^2));
        LM_bg   = numel(idx_bg)*R2_bg;
        pval_bg = 1 - chi2cdf(LM_bg, H_BG);

        % (B) JARQUE-BERA: test for normality
        u_std   = (u_j - mean(u_j)) / std(u_j);
        S_j     = mean(u_std.^3);
        K_j     = mean(u_std.^4);
        JB_j    = (T_eff/6)*(S_j^2 + (K_j-3)^2/4);
        pval_jb = 1 - chi2cdf(JB_j, 2);

        % (C) ARCH-LM: test for conditional heteroskedasticity
        u2     = u_j.^2;
        U2_lag = zeros(T_eff, Q_ARCH);
        for s = 1:Q_ARCH
            U2_lag(s+1:end, s) = u2(1:end-s);
        end
        idx_arch  = Q_ARCH+1 : T_eff;
        Z_arch    = [ones(numel(idx_arch),1), U2_lag(idx_arch,:)];
        y_arch    = u2(idx_arch);
        e_arch    = y_arch - Z_arch*(Z_arch\y_arch);
        R2_arch   = max(0, 1-(e_arch'*e_arch)/sum((y_arch-mean(y_arch)).^2));
        LM_arch   = numel(idx_arch)*R2_arch;
        pval_arch = 1 - chi2cdf(LM_arch, Q_ARCH);

        fprintf(' |  %-10s  BG p=%6.4f%s  JB p=%6.4f (S=%+5.2f K=%5.2f)%s  ARCH p=%6.4f%s\n', ...
            VAR_NAMES{j}, ...
            pval_bg,   flag_str(pval_bg,   ALPHA,' ***AUTOCORR '), ...
            pval_jb,   S_j, K_j, flag_str(pval_jb, ALPHA,' ***NONGAUSS'), ...
            pval_arch, flag_str(pval_arch, ALPHA,' ***ARCH    '));

        row_idx = row_idx + 1;
        results_diag(row_idx,:) = ...
            [p,j,LM_bg,pval_bg,JB_j,pval_jb,S_j,K_j,LM_arch,pval_arch,T_eff,k];
    end
    fprintf(' +%s\n', repmat('-',1,68));
end

% =========================================================================
%  STEP 5 — RESIDUAL DISTRIBUTION + HILL TAIL INDEX  (at p_flat)
% =========================================================================
fprintf('\n%s\n STEP 5 — RESIDUAL DISTRIBUTION  (p_flat = %d)\n%s\n', ...
        repmat('=',1,70), p_flat, repmat('=',1,70));
fprintf(' %-10s %10s %10s %10s %12s %10s\n', ...
        'Variable','Mean','Std','Skewness','Kurt.Exc','Hill alpha');
fprintf(' %s\n', repmat('-',1,62));

tail_flat = zeros(M, 5);
for j = 1:M
    tail_flat(j,:) = residual_tail_stats(U_flat_opt(:,j), HILL_FRAC, VAR_NAMES{j});
end

fprintf('\n Tail index interpretation:\n');
fprintf('   alpha < 2 : infinite variance (very heavy tail)\n');
fprintf('   2 < alpha < 4 : finite variance, infinite kurtosis\n');
fprintf('   alpha > 4 : finite kurtosis (approaches Gaussian tails)\n');
n_heavy = sum(tail_flat(:,5) < 4 & ~isnan(tail_flat(:,5)));
fprintf(' Variables with alpha < 4: %d / %d', n_heavy, M);
if n_heavy > M/2
    fprintf('  -> FAVOURABLE for ICA (Lanne-Meitz-Saikkonen 2017)\n');
else
    fprintf('  -> CHECK: fewer than half have heavy tails\n');
end

% =========================================================================
%  STEP 6 — FIGURES  (histograms + QQ-plots, flat prior at p_flat)
% =========================================================================
fprintf('\n%s\n STEP 6 — FIGURES\n%s\n', repmat('=',1,70), repmat('=',1,70));

fig_hist = figure('Name','Residual Distributions (Flat)','Position',[50 50 1400 900]);
fig_qq   = figure('Name','QQ-Plots (Flat)',              'Position',[50 50 1400 900]);

for j = 1:M
    u_j   = U_flat_opt(:,j);
    u_std = (u_j - mean(u_j)) / std(u_j);

    figure(fig_hist);
    subplot(2,5,j);
    histogram(u_std, 60, 'Normalization','pdf', ...
              'FaceColor',[0.2 0.45 0.7],'EdgeColor','none','FaceAlpha',0.75);
    hold on;
    xr = linspace(-6,6,500);
    plot(xr, normpdf(xr,0,1), 'r-', 'LineWidth',2);
    xlim([-6 6]); xlabel('Std. residual');
    title(sprintf('%s  (K_{exc}=%.2f)', VAR_NAMES{j}, tail_flat(j,4)-3), 'FontSize',9);
    if j==1, legend('Empirical','N(0,1)','Location','northwest'); end
    grid on; box off;

    figure(fig_qq);
    subplot(2,5,j);
    qqplot(u_std);
    title(sprintf('%s  (alpha=%.2f)', VAR_NAMES{j}, tail_flat(j,5)), 'FontSize',9);
    grid on; box off;
end

sgtitle(fig_hist, sprintf('Residual Distributions  |  Flat BVAR(%d)', p_flat), ...
        'FontSize',12,'FontWeight','bold');
sgtitle(fig_qq, sprintf('QQ-Plots vs Normal  |  Flat BVAR(%d)', p_flat), ...
        'FontSize',12,'FontWeight','bold');
saveas(fig_hist, 'residual_histograms_flat.png');
saveas(fig_qq,   'residual_qqplots_flat.png');
fprintf(' Figures saved (flat prior, p_flat=%d).\n', p_flat);

% =========================================================================
%  ████████████████████████████  BLOCK B  ████████████████████████████████
%
%  STRICT-GLP SEQUENTIAL HYPERPARAMETER OPTIMISATION
%  (Giannone, Lenza & Primiceri 2015, Section III)
%
%  Criterion: maximise ln p(Y | gamma, p_MN)  w.r.t. gamma = (tau, mu_sc, delta)
%  where gamma corresponds to GLP (lambda, mu, delta).  decay theta and psi
%  are FIXED per GLP.  The log-ML is the Normal-IW marginal likelihood in
%  closed form (GLP 2015 eq. 5; Kilian & Lutkepohl 2017, eq. 5.2.15-17).
%  Under flat hyperpriors this is empirical-Bayes (GLP p.441), which also
%  maximises one-step OOS forecasting performance.
%
%  STAGE 1:  max_{tau}        ln p(Y | tau, theta=1, mu_sc=5, delta=2)   [1-D]
%  STAGE 2:  max_{tau,mu_sc,delta} ln p(Y | tau, 1, mu_sc, delta)        [3-D]
%  STAGE 3:  multi-start robustness (4 starting points)
%            -> selects global best across all runs
%  =========================================================================

fprintf('\n%s\n BLOCK B — GLP SEQUENTIAL HYPERPARAMETER OPTIMISATION\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
fprintf(' p_MN = %d  (fixed, 22 trading days = 1 month)\n', p_MN);
fprintf(' Toolbox: bvar_max_hyper  (Ferroni & Canova 2025)\n');
fprintf(' Algorithm: fmincon (constrained gradient, MATLAB)\n\n');

% ── Stage 1 — tau only ────────────────────────────────────────────────────
fprintf(' ── STAGE 1: optimise tau only (decay=1 GLP-fixed, lam, mu, omega at defaults) ──\n');
fprintf(' Starting: tau0=%.2f  theta=%.2f (GLP-fixed)  lam=%.2f  delta=%.2f\n\n', ...
        HP0(1), HP0(2), HP0(3), HP0(4));

opts1.index_est   = 1;       % parameter 1 = tau
opts1.max_compute = 2;       % fmincon
opts1.lb          = 0.5;
opts1.ub          = 50;

[mode_s1, lnML_s1, ~] = bvar_max_hyper(HP0, Y_levels, p_MN, opts1);
tau_s1 = mode_s1(1);

fprintf('\n Stage 1 result:\n');
fprintf('   tau*  = %.4f\n', tau_s1);
fprintf('   lnML  = %.4f\n', lnML_s1);
fprintf('   lambda_L = 1/tau* = %.4f\n\n', 1/tau_s1);

% ── Stage 2 — (tau, lambda, mu) jointly ───────────────────────────────────
fprintf(' ── STAGE 2: joint optimisation (tau, lambda_DIO, mu_SC)  [theta=1 fixed] ──\n');
fprintf(' Warm start from Stage 1 mode.\n');
fprintf(' Bounds: tau[0.05,50], lambda_DIO[0.05,100], mu_SC[0.05,200]\n\n');

opts2.index_est   = [1 3 4];          % tau, lambda (=delta_GLP), mu (=mu_GLP)
opts2.max_compute = 2;                % fmincon
opts2.lb          = [0.05, 0.05, 0.05];
opts2.ub          = [50,   150,   500];

[mode_s2_main, lnML_s2_main, ~] = bvar_max_hyper(mode_s1, Y_levels, p_MN, opts2);

fprintf('\n Stage 2 result (fmincon, warm start):\n');
fprintf('   tau*      = %.4f  (1/lambda_GLP)\n',          mode_s2_main(1));
fprintf('   theta     = %.4f  (GLP-FIXED)\n',              mode_s2_main(2));
fprintf('   lambda_DIO= %.4f  (= delta_GLP, dummy-init-obs)\n', mode_s2_main(3));
fprintf('   mu_SC     = %.4f  (= mu_GLP, sum-of-coef)\n',     mode_s2_main(4));
fprintf('   lnML      = %.4f\n', lnML_s2_main);
fprintf('   DeltalnML (Stage2 - Stage1) = %.4f\n', lnML_s2_main - lnML_s1);

% ── Stage 3 — Multi-start robustness for Stage 2 ─────────────────────────
%   fmincon can get stuck in flat regions with complex likelihoods.
%   We run Stage 2 from MULTISTART rows and keep the global best.
fprintf('\n ── STAGE 3: multi-start robustness for Stage 2 ──\n');
fprintf(' %d starting points. Best lnML across all runs = global candidate.\n\n', ...
        size(MULTISTART,1));
fprintf(' %-4s  %-8s  %-10s  %-10s  %-10s  %-10s  %-12s  %-12s  %-12s\n', ...
        'Run','tau0','lam_dio0','mu_sc0','tau*','theta','lam_dio*','mu_sc*','lnML');
fprintf(' %s\n', repmat('-',1,100));

lnML_ms  = zeros(size(MULTISTART,1)+1, 1);
mode_ms  = zeros(size(MULTISTART,1)+1, 5);

% Row 1 = Stage 2 warm-start result (already computed)
lnML_ms(1)   = lnML_s2_main;
mode_ms(1,:) = mode_s2_main;
fprintf(' %-4d  %-8.2f  %-10.2f  %-10.2f  %-10.4f  %-10.4f  %-12.4f  %-12.4f  %-12.4f  (warm-S1)\n', ...
        0, mode_s1(1), mode_s1(3), mode_s1(4), ...
        mode_s2_main(1), mode_s2_main(2), mode_s2_main(3), mode_s2_main(4), lnML_s2_main);

for r = 1:size(MULTISTART,1)
    HP_r        = HP0;
    HP_r(1)     = MULTISTART(r,1);   % tau
    HP_r(3)     = MULTISTART(r,2);   % lambda_DIO (= delta_GLP, dummy-init-obs)
    HP_r(4)     = MULTISTART(r,3);   % mu_SC (= mu_GLP, sum-of-coeff)
    % HP_r(2)   remains HP0(2) = 1   (decay FIXED per GLP-strict)
    % HP_r(5)   remains HP0(5) = 2   (omega FIXED, toolbox default)

    [mode_r, lnML_r, ~] = bvar_max_hyper(HP_r, Y_levels, p_MN, opts2);
    lnML_ms(r+1)   = lnML_r;
    mode_ms(r+1,:) = mode_r;

    fprintf(' %-4d  %-8.2f  %-10.2f  %-10.2f  %-10.4f  %-10.4f  %-12.4f  %-12.4f  %-12.4f\n', ...
            r, HP_r(1), HP_r(3), HP_r(4), ...
            mode_r(1), mode_r(2), mode_r(3), mode_r(4), lnML_r);
end

[lnML_s2, best_ms] = max(lnML_ms);
mode_s2            = mode_ms(best_ms,:);

tau_opt    = mode_s2(1);
decay_opt  = mode_s2(2);        % =1 by construction (GLP-fixed)
delta_opt = mode_s2(3);        % = delta_GLP (dummy-initial-obs / co-persistence)
mu_opt     = mode_s2(4);        % = mu_GLP (sum-of-coefficients) -- NOW optimised
omega_opt  = HP0(5);            % kept at default (toolbox limitation)

fprintf('\n Global best across all runs: Run %d\n', best_ms-1);
fprintf('   tau*       = %.4f  (1/lambda_GLP overall tightness)\n', tau_opt);
fprintf('   theta      = %.4f  (GLP-FIXED: lag decay)\n',           decay_opt);
fprintf('   lambda_DIO*= %.4f  (= delta_GLP, dummy-init-obs)\n',     delta_opt);
fprintf('   mu_SC*     = %.4f  (= mu_GLP, sum-of-coef)\n',           mu_opt);
fprintf('   lnML       = %.4f\n', lnML_s2);
fprintf('   DeltalnML (GLP_strict - Stage1) = %.4f\n', lnML_s2 - lnML_s1);
fprintf('   DeltalnML (GLP_strict - Stage2_main) = %.4f\n', lnML_s2 - lnML_s2_main);

% Jeffreys-Kass-Raftery evidence scale
delta_vs_s1 = lnML_s2 - lnML_s1;
if     delta_vs_s1 >  5,   evid_str = 'DECISIVE (>> 150 BF)';
elseif delta_vs_s1 >  3,   evid_str = 'STRONG   (20-150 BF)';
elseif delta_vs_s1 >  1,   evid_str = 'POSITIVE (3-20 BF)';
else,                      evid_str = 'NEGLIGIBLE'; 
end
fprintf('\n Evidence for GLP-strict (3-D) vs Stage-1 (tau-only): %s\n', evid_str);
fprintf('   -> If DeltalnML > 5: joint optimisation is necessary.\n');
fprintf('   -> If DeltalnML < 3: tau-only is sufficient approximation.\n');

% ── Lag decay declaration (GLP-strict: fixed at theta=1) ─────────────────
fprintf('\n ── LAG DECAY (GLP-strict) ──\n');
fprintf(' theta = 1  FIXED per Giannone-Lenza-Primiceri (2015), p.439.\n');
fprintf(' Prior variance Var(phi_{ij,l}) proportional to 1/l^2.\n');
fprintf(' %-6s  %-14s\n', 'Lag l', 'Relative var');
fprintf(' %s\n', repmat('-',1,24));
for l = [1 2 3 5 10 15 22]
    fprintf(' l=%-4d  %-14.6f\n', l, l^(-2));
end

% =========================================================================
%  STEP 7a — GRID SEARCH ON TAU  (legacy, theta=1 GLP-fixed, for comparison)
%  Purpose: shows the 1-D marginal likelihood profile and confirms that
%           the Stage-1 continuous optimum is consistent with the grid.
% =========================================================================
fprintf('\n%s\n STEP 7a — TAU GRID SEARCH  (p_MN=%d, theta=1 GLP-fixed, legacy)\n%s\n', ...
        repmat('=',1,70), p_MN, repmat('=',1,70));
fprintf(' Kept for comparability with prior literature.\n');
fprintf(' The POSTERIOR in Steps 7b-f uses GLP-optimal params, not this grid.\n\n');
fprintf(' %-10s  %-14s  %-14s  %-10s\n', 'tau','lambda_L=1/tau','ln ML (MN)','Delta');
fprintf(' %s\n', repmat('-',1,54));

opt_gs          = struct('K', 1);
lnML_tau_grid   = zeros(nTau, 1);
for g = 1:nTau
    opt_gs.priors.name      = 'Minnesota';
    opt_gs.minn_prior_tau   = TAU_GRID(g);
    opt_gs.minn_prior_decay = DECAY_FIX;
    tmp              = bvar_(Y_levels, p_MN, opt_gs);
    lnML_tau_grid(g) = tmp.logmlike;
end

[lnML_grid_max, best_g] = max(lnML_tau_grid);
tau_grid_star           = TAU_GRID(best_g);

if best_g == nTau
    warning('tau* at upper boundary of TAU_GRID (%.1f). Consider extending.', TAU_GRID(end));
end

for g = 1:nTau
    flag = '';
    if g == best_g, flag = '  <- GRID OPTIMAL'; end
    fprintf(' tau=%-8.3f  lam_L=%-8.4f  lnML=%12.4f  Delta=%10.4f%s\n', ...
            TAU_GRID(g), 1/TAU_GRID(g), lnML_tau_grid(g), ...
            lnML_tau_grid(g)-lnML_grid_max, flag);
end

fprintf('\n Grid optimal:  tau_grid* = %.3f  (lambda_L = %.4f)\n', ...
        tau_grid_star, 1/tau_grid_star);
fprintf(' Stage-1 cont.: tau_s1*   = %.4f  (lambda_L = %.4f)\n', ...
        tau_s1, 1/tau_s1);
fprintf(' GLP full:      tau_opt*  = %.4f  (lambda_L = %.4f)\n', ...
        tau_opt, 1/tau_opt);
fprintf(' -> Consistency check: grid and Stage-1 should agree closely.\n');

% Robustness: tau_grid* at p_MN +/- 2 lags
fprintf('\n ROBUSTNESS: grid tau* at p_MN +/- 2 lags (theta=1)\n');
fprintf(' %-6s  %-10s  %-14s  %-10s\n', 'p', 'tau_grid*', 'ln ML (MN)', 'lambda_L');
fprintf(' %s\n', repmat('-',1,46));
p_rob = [p_MN-2, p_MN-1, p_MN, p_MN+1, p_MN+2];
p_rob = p_rob(p_rob > 1);
for r = 1:numel(p_rob)
    p_r  = p_rob(r);
    ml_r = zeros(nTau, 1);
    for g = 1:nTau
        opt_gs.minn_prior_tau = TAU_GRID(g);
        tmp     = bvar_(Y_levels, p_r, opt_gs);
        ml_r(g) = tmp.logmlike;
    end
    [ml_best, gi] = max(ml_r);
    flag = '';
    if p_r == p_MN, flag = '  <- reference'; end
    fprintf(' p=%-4d  tau*=%-8.3f  lnML=%10.4f  lam_L=%.4f%s\n', ...
            p_r, TAU_GRID(gi), ml_best, 1/TAU_GRID(gi), flag);
end

% =========================================================================
%  STEP 7b — FULL GLP-OPTIMAL MINNESOTA POSTERIOR + BASELINES
%
%  Three models estimated at the SAME lag p_MN for valid comparison:
%    [FLAT]  : flat prior BVAR(p_MN)  — OLS baseline
%    [MN-S1] : Minnesota, tau from Stage 1 (tau_s1, theta=1, delta=5, mu=2)
%    [MN-GLP]: Minnesota, GLP full optimal (tau_opt, decay_opt, delta_opt)
%              -> THIS IS THE MAIN MODEL FOR ALL SUBSEQUENT ANALYSIS
% =========================================================================
fprintf('\n%s\n STEP 7b — GLP-OPTIMAL POSTERIOR  (p_MN=%d)\n%s\n', ...
        repmat('=',1,70), p_MN, repmat('=',1,70));
fprintf(' Estimating three models at p_MN=%d for model comparison:\n', p_MN);
fprintf('   [FLAT]   flat prior (OLS)\n');
fprintf('   [MN-S1]  Minnesota tau=%.4f, theta=1, lam=5, dlt=2\n', tau_s1);
fprintf('   [MN-GLP] Minnesota tau=%.4f, d=%.4f, lam=%.4f  <- MAIN MODEL\n\n', ...
        tau_opt, decay_opt, delta_opt);

% ── [FLAT] full posterior ─────────────────────────────────────────────────
opt_flat_full    = struct('K', N_DRAWS);
BVAR_flat_pMN    = bvar_(Y_levels, p_MN, opt_flat_full);
B_flat_pMN       = mean(BVAR_flat_pMN.Phi_draws,   3);
Sig_flat_pMN     = mean(BVAR_flat_pMN.Sigma_draws, 3);
E_flat_pMN       = squeeze(mean(BVAR_flat_pMN.e_draws, 3));
lnML_flat_pMN_full = BVAR_flat_pMN.logmlike;

% ── [MN-S1] Minnesota at Stage-1 params (for comparison) ─────────────────
opt_MN_s1                   = struct();
opt_MN_s1.priors.name       = 'Minnesota';
opt_MN_s1.minn_prior_tau    = tau_s1;
opt_MN_s1.minn_prior_decay  = 1;       % GLP-fixed
opt_MN_s1.minn_prior_lambda = 5;   % delta_GLP (dummy-init-obs) = 5 (default HP0)
opt_MN_s1.K                 = N_DRAWS;

BVAR_MN_s1  = bvar_(Y_levels, p_MN, opt_MN_s1);
Sig_MN_s1   = mean(BVAR_MN_s1.Sigma_draws, 3);
lnML_MN_s1  = BVAR_MN_s1.logmlike;

% ── [MN-GLP] Minnesota at GLP-optimal params — MAIN MODEL ────────────────
opt_MN_glp                   = struct();
opt_MN_glp.priors.name       = 'Minnesota';
opt_MN_glp.minn_prior_tau    = tau_opt;
opt_MN_glp.minn_prior_decay  = decay_opt;
opt_MN_glp.minn_prior_lambda = delta_opt;
opt_MN_glp.minn_prior_mu     = mu_opt;
opt_MN_glp.minn_prior_omega  = omega_opt;
opt_MN_glp.K                 = N_DRAWS;
opt_MN_glp.hor               = IRF_HOR;

BVAR_MN_glp  = bvar_(Y_levels, p_MN, opt_MN_glp);
B_MN_pm      = mean(BVAR_MN_glp.Phi_draws,   3);
Sig_MN_pm    = mean(BVAR_MN_glp.Sigma_draws, 3);
E_MN_pm      = squeeze(mean(BVAR_MN_glp.e_draws, 3));
lnML_MN_glp  = BVAR_MN_glp.logmlike;

% ── Build regressor matrix X at p_MN (for condition number) ──────────────
T_eff_MN = T_full - p_MN;
k_MN     = M*p_MN + 1;
X_MN     = zeros(T_eff_MN, k_MN);
for l = 1:p_MN
    X_MN(:, (l-1)*M+1 : l*M) = Y_levels(p_MN-l+1:end-l, :);
end
X_MN(:, end) = 1;
XtX          = X_MN' * X_MN;
cond_flat    = cond(XtX);

% Pre-sample std for prior variance computation
s_var = std(Y_levels(1:p_MN, :));
s_var(s_var < 1e-10) = 1e-10;

% ── Condition numbers for GLP-optimal and Stage-1 priors ─────────────────
%   V_diag(row) = Sigma_jj(flat) / (tau * s_j * l^decay)^2
lags_col    = (1:p_MN)';
h_glp       = lags_col .^ decay_opt;   % GLP-optimal decay per lag

V_diag_glp = zeros(k_MN, 1);
V_diag_s1  = zeros(k_MN, 1);
for j = 1:M
    rows = (lags_col - 1)*M + j;
    V_diag_glp(rows) = Sig_flat_pMN(j,j) ./ (tau_opt * s_var(j) * h_glp   ).^2;
    V_diag_s1(rows)  = Sig_flat_pMN(j,j) ./ (tau_s1  * s_var(j) * lags_col).^2;
end
V_diag_glp(end) = 1e6;   % intercept: near-diffuse
V_diag_s1(end)  = 1e6;

cond_MN_glp = cond(XtX + diag(1./V_diag_glp));
cond_MN_s1  = cond(XtX + diag(1./V_diag_s1));

fprintf('\n ── CONDITION NUMBERS  (p_MN=%d) ──\n', p_MN);
fprintf(' %-40s  %14s\n', 'Model', 'cond number');
fprintf(' %s\n', repmat('-',1,58));
fprintf(' %-40s  %14.4e\n', 'Flat OLS  [X''X]',              cond_flat);
fprintf(' %-40s  %14.4e\n', 'MN-S1     [X''X + V^{-1}(S1)]', cond_MN_s1);
fprintf(' %-40s  %14.4e\n', 'MN-GLP    [X''X + V^{-1}(GLP)]',cond_MN_glp);
fprintf(' Ratio flat / MN-GLP: %.2f\n', cond_flat/cond_MN_glp);
fprintf(' NOTE: cond < 10^12 desirable; > 10^14 = near-singular.\n');

% ── Prior variance table (own lag-1) ─────────────────────────────────────
fprintf('\n ── MINNESOTA PRIOR — OWN LAG-1 DIAGONAL [GLP-optimal] ──\n');
fprintf(' tau*=%.4f  decay*=%.4f  delta*(=lambda_DIO)*=%.4f\n', tau_opt, decay_opt, delta_opt);
fprintf(' %-10s  %14s  %14s  %14s\n', ...
        'Variable','Var_prior','Std_prior','Sigma_jj(OLS)');
fprintf(' %s\n', repmat('-',1,58));
for j = 1:M
    var_prior_glp = Sig_flat_pMN(j,j) / (tau_opt * s_var(j))^2;
    fprintf(' %-10s  %14.6f  %14.6f  %14.6f\n', ...
            VAR_NAMES{j}, var_prior_glp, sqrt(var_prior_glp), Sig_flat_pMN(j,j));
end

% ── Residual correlation matrices ────────────────────────────────────────
corr_from_sigma = @(S) diag(1./sqrt(diag(S))) * S * diag(1./sqrt(diag(S)));
Corr_flat = corr_from_sigma(Sig_flat_pMN);
Corr_s1   = corr_from_sigma(Sig_MN_s1);
Corr_glp  = corr_from_sigma(Sig_MN_pm);

print_corr = @(Corr, lbl) fprintf_corr(Corr, VAR_NAMES, M, lbl);

fprintf('\n ── RESIDUAL CORRELATION MATRICES ──\n');
print_corr(Corr_flat, sprintf('FLAT OLS (p_MN=%d)', p_MN));
print_corr(Corr_s1,   sprintf('MN-S1 (tau=%.2f, theta=1, lam=5, dlt=2)', tau_s1));
print_corr(Corr_glp,  sprintf('MN-GLP (tau=%.2f, d=%.2f, lam=%.2f)', ...
                               tau_opt, decay_opt, delta_opt));

% ── Model comparison table (Bayes Factors) ───────────────────────────────
fprintf('\n ── MODEL COMPARISON (Bayes Factors, all at p_MN=%d) ──\n', p_MN);
fprintf(' %-44s  %10s  %10s\n', 'Model', 'ln|Sigma|', 'ln ML');
fprintf(' %s\n', repmat('-',1,68));
fprintf(' %-44s  %10.4f  %10.4f\n', ...
        sprintf('FLAT  OLS           (p_MN=%d)', p_MN), ...
        logdet_chol(Sig_flat_pMN+1e-14*eye(M)), lnML_flat_pMN_full);
fprintf(' %-44s  %10.4f  %10.4f\n', ...
        sprintf('MN-S1 tau=%.2f, theta=1, lam=5, dlt=2', tau_s1), ...
        logdet_chol(Sig_MN_s1+1e-14*eye(M)), lnML_MN_s1);
fprintf(' %-44s  %10.4f  %10.4f  <- MAIN\n', ...
        sprintf('MN-GLP tau=%.2f, d=%.2f, delta=%.2f', tau_opt,decay_opt,delta_opt), ...
        logdet_chol(Sig_MN_pm+1e-14*eye(M)), lnML_MN_glp);

fprintf('\n BF [MN-S1 / Flat]  = exp(%.4f) = ', lnML_MN_s1 - lnML_flat_pMN_full);
print_bf(lnML_MN_s1 - lnML_flat_pMN_full);
fprintf(' BF [MN-GLP / Flat] = exp(%.4f) = ', lnML_MN_glp - lnML_flat_pMN_full);
print_bf(lnML_MN_glp - lnML_flat_pMN_full);
fprintf(' BF [MN-GLP / MN-S1]= exp(%.4f) = ', lnML_MN_glp - lnML_MN_s1);
print_bf(lnML_MN_glp - lnML_MN_s1);
fprintf('\n (Proper BFs: both marginal likelihoods under their respective\n');
fprintf('  priors, same sample T_eff=%d.)\n', T_eff_MN);

% =========================================================================
%  STEP 7c — OLS vs MN-GLP  (own lag-1 coefficients)
% =========================================================================
fprintf('\n%s\n STEP 7c — OLS vs MN-GLP  (own lag-1, p_MN=%d)\n%s\n', ...
        repmat('=',1,70), p_MN, repmat('=',1,70));
fprintf(' Prior mean: phi_{jj,1} = 1  (Random Walk)\n');
fprintf(' shrink_frac = (MN_jj - OLS_jj) / (1 - OLS_jj)\n');
fprintf('   ~ 0: data dominate  |  ~ 1: prior dominates  |  > 1: OLS explosive\n\n');
fprintf(' %-10s  %10s  %10s  %10s  %12s\n', ...
        'Variable','OLS','MN-GLP','Prior=1','Shrink_frac');
fprintf(' %s\n', repmat('-',1,58));

for j = 1:M
    ols_jj = B_flat_pMN(j, j);
    mn_jj  = B_MN_pm(j, j);
    denom  = 1 - ols_jj;
    sfrac  = NaN;
    if abs(denom) > 1e-6
        sfrac = (mn_jj - ols_jj) / denom;
    end
    fprintf(' %-10s  %10.5f  %10.5f  %10.5f  %12.4f\n', ...
            VAR_NAMES{j}, ols_jj, mn_jj, 1.0, sfrac);
end

n_explosive = sum(diag(B_flat_pMN(1:M,1:M)) > 1);
if n_explosive > 0
    fprintf('\n WARNING: %d variable(s) have OLS own-lag-1 > 1 at p_MN=%d.\n', ...
            n_explosive, p_MN);
    fprintf(' Minnesota shrinkage toward 1 is stabilising for these.\n');
else
    fprintf('\n All OLS own-lag-1 <= 1 at p_MN=%d.\n', p_MN);
end

% =========================================================================
%  STEP 7d — MINNESOTA (GLP) RESIDUAL DISTRIBUTION + TAIL INDEX + QQ-PLOTS
% =========================================================================
fprintf('\n%s\n STEP 7d — MN-GLP RESIDUAL DISTRIBUTION  (p_MN=%d)\n%s\n', ...
        repmat('=',1,70), p_MN, repmat('=',1,70));
fprintf(' %-10s %10s %10s %10s %12s %10s\n', ...
        'Variable','Mean','Std','Skewness','Kurt.Exc','Hill alpha');
fprintf(' %s\n', repmat('-',1,62));

tail_MN_glp = zeros(M, 5);
for j = 1:M
    tail_MN_glp(j,:) = residual_tail_stats(E_MN_pm(:,j), HILL_FRAC, VAR_NAMES{j});
end

n_heavy_glp = sum(tail_MN_glp(:,5) < 4 & ~isnan(tail_MN_glp(:,5)));
fprintf('\n ICA check [MN-GLP residuals]:\n');
fprintf(' Variables with Hill alpha < 4: %d / %d\n', n_heavy_glp, M);
if n_heavy_glp >= M-1
    fprintf(' -> At most 1 Gaussian component: ICA feasible (LMS 2017)\n');
else
    fprintf(' -> WARNING: multiple near-Gaussian; ICA identification may be weak.\n');
end

% ── Residual diagnostics at p_MN (GLP residuals) ─────────────────────────
%   This step is NEW vs the original code — it verifies that the
%   MN-GLP residuals at p_MN=22 are free of linear autocorrelation.
%   Note: ARCH rejection is expected (not a VAR misspecification issue).
fprintf('\n ── BG + JB + ARCH on MN-GLP RESIDUALS at p_MN=%d ──\n', p_MN);
fprintf(' (Confirms that p_MN=22 removes linear autocorrelation)\n\n');
fprintf(' %-10s  BG(%d)     JB         ARCH(%d)\n', 'Variable', H_BG, Q_ARCH);
fprintf(' %s\n', repmat('-',1,64));

for j = 1:M
    u_j = E_MN_pm(:,j);
    T_e = numel(u_j);

    % BG test — condition on X_MN (same regressors as VAR), consistent with Block A
    U_lag_bg2 = zeros(T_e, H_BG);
    for s = 1:H_BG
        U_lag_bg2(s+1:end,s) = u_j(1:end-s);
    end
    idx2   = H_BG+1:T_e;
    Z2     = [X_MN(idx2,:), U_lag_bg2(idx2,:)];   % same structure as Block A BG
    u2_bg  = u_j(idx2);
    e2_bg  = u2_bg - Z2*(Z2\u2_bg);
    R2_2   = max(0, 1-(e2_bg'*e2_bg)/sum((u2_bg-mean(u2_bg)).^2));
    LM2_bg = numel(idx2)*R2_2;
    pv_bg2 = 1 - chi2cdf(LM2_bg, H_BG);

    % JB
    u_s2   = (u_j-mean(u_j))/std(u_j);
    Sj2    = mean(u_s2.^3); Kj2 = mean(u_s2.^4);
    JB2    = (T_e/6)*(Sj2^2+(Kj2-3)^2/4);
    pv_jb2 = 1-chi2cdf(JB2,2);

    % ARCH-LM
    u2sq  = u_j.^2;
    UL2   = zeros(T_e, Q_ARCH);
    for s=1:Q_ARCH, UL2(s+1:end,s)=u2sq(1:end-s); end
    ia2   = Q_ARCH+1:T_e;
    Za2   = [ones(numel(ia2),1), UL2(ia2,:)];
    ya2   = u2sq(ia2);
    ea2   = ya2-Za2*(Za2\ya2);
    R2_a2 = max(0,1-(ea2'*ea2)/sum((ya2-mean(ya2)).^2));
    LMa2  = numel(ia2)*R2_a2;
    pv_a2 = 1-chi2cdf(LMa2,Q_ARCH);

    fprintf(' %-10s  p=%6.4f%s  p=%6.4f%s  p=%6.4f%s\n', ...
            VAR_NAMES{j}, ...
            pv_bg2, flag_str(pv_bg2,  ALPHA,'***'), ...
            pv_jb2, flag_str(pv_jb2,  ALPHA,'***'), ...
            pv_a2,  flag_str(pv_a2,   ALPHA,'***'));
end
fprintf('\n Note: JB and ARCH rejections are structural (heavy tails, clustering).\n');
fprintf(' BG *** at p_MN=22 would indicate lag misspecification.\n');

% ── Figures: QQ-plots and comparison histograms ───────────────────────────
fig_qq_glp = figure('Name','QQ-Plots MN-GLP','Position',[50 50 1400 900]);
for j = 1:M
    u_std = (E_MN_pm(:,j)-mean(E_MN_pm(:,j)))/std(E_MN_pm(:,j));
    subplot(2,5,j);
    qqplot(u_std);
    title(sprintf('%s (alpha=%.2f)', VAR_NAMES{j}, tail_MN_glp(j,5)), 'FontSize',9);
    grid on; box off;
end
sgtitle(fig_qq_glp, sprintf('QQ-Plots vs N(0,1)  |  MN-GLP BVAR(%d)', p_MN), ...
        'FontSize',12,'FontWeight','bold');
saveas(fig_qq_glp, 'residual_qqplots_MN_GLP.png');

% Three-way histogram: Flat vs MN-S1 vs MN-GLP
E_MN_s1   = squeeze(mean(BVAR_MN_s1.e_draws, 3));   % precomputed once
fig_comp3 = figure('Name','Residual Comparison 3-way','Position',[50 50 1400 900]);
for j = 1:M
    u_fl = (E_flat_pMN(:,j)-mean(E_flat_pMN(:,j)))/std(E_flat_pMN(:,j));
    u_s1 = (E_MN_s1(:,j)-mean(E_MN_s1(:,j)))/std(E_MN_s1(:,j));
    u_gp = (E_MN_pm(:,j)-mean(E_MN_pm(:,j)))/std(E_MN_pm(:,j));

    subplot(2,5,j);
    histogram(u_fl, 50, 'Normalization','pdf', ...
              'FaceColor',[0.2 0.45 0.7],'EdgeColor','none','FaceAlpha',0.45);
    hold on;
    histogram(u_s1, 50, 'Normalization','pdf', ...
              'FaceColor',[0.85 0.3 0.1],'EdgeColor','none','FaceAlpha',0.45);
    histogram(u_gp, 50, 'Normalization','pdf', ...
              'FaceColor',[0.1 0.65 0.3],'EdgeColor','none','FaceAlpha',0.45);
    xr = linspace(-6,6,500);
    plot(xr, normpdf(xr,0,1), 'k--', 'LineWidth',1.5);
    xlim([-6 6]);
    title(VAR_NAMES{j}, 'FontSize',9);
    if j==1
        legend('Flat','MN-S1','MN-GLP','N(0,1)', ...
               'Location','northwest','FontSize',7);
    end
    grid on; box off;
end
sgtitle(fig_comp3, sprintf('Flat vs MN-S1 vs MN-GLP Residuals  |  BVAR(%d)', p_MN), ...
        'FontSize',12,'FontWeight','bold');
saveas(fig_comp3, 'residual_comparison_3way.png');

% =========================================================================
%  STEP 8 — EXPORT
% =========================================================================
fprintf('\n%s\n STEP 8 — EXPORT\n%s\n', repmat('=',1,70), repmat('=',1,70));

% Diagnostics table (Block A)
col_names = {'p','eq','LM_bg','pval_bg','JB','pval_jb', ...
             'Skewness','Kurtosis','LM_arch','pval_arch','T_eff','k'};
T_diag = array2table(results_diag(1:row_idx,:), 'VariableNames', col_names);
T_diag.Variable = repmat(VAR_NAMES', nlag, 1);
writetable(T_diag, 'bvar_diagnostics_results.csv');

% Minnesota results table (three models)
T_mn = table(VAR_NAMES', ...
    round(diag(Sig_flat_pMN), 6), ...
    round(diag(Sig_MN_s1),    6), ...
    round(diag(Sig_MN_pm),    6), ...
    'VariableNames', {'Variable','Sigma_Flat','Sigma_MN_S1','Sigma_MN_GLP'});
writetable(T_mn, 'minnesota_results.csv');

% Hyperparameter optimisation results (GLP-strict)
T_hyp = table({'Stage1_tau_only';'Stage2_main';'GLP_global_best'}, ...
    [tau_s1;     mode_s2_main(1); tau_opt], ...
    [1;          mode_s2_main(2); decay_opt], ...       % theta FIXED at 1
    [HP0(3);     mode_s2_main(3); delta_opt], ...
    [HP0(4);     mode_s2_main(4); mu_opt], ...
    [lnML_s1;    lnML_s2_main;    lnML_s2], ...
    'VariableNames', {'Stage','tau','theta_fixed','lambda_DIO_delta_GLP','mu_SC_mu_GLP','lnML'});
writetable(T_hyp, 'glp_hyperparameter_results.csv');

fprintf(' Files: bvar_diagnostics_results.csv\n');
fprintf('        minnesota_results.csv\n');
fprintf('        glp_hyperparameter_results.csv\n');
fprintf(' Figs (Block A): residual_histograms_flat.png\n');
fprintf('                 residual_qqplots_flat.png\n');
fprintf(' Figs (Block B): residual_qqplots_MN_GLP.png\n');
fprintf('                 residual_comparison_3way.png\n');

% =========================================================================
%  FINAL SUMMARY
% =========================================================================
fprintf('\n ══════════════════════════════════════════════════════════════════\n');
fprintf(' FINAL SUMMARY\n');
fprintf(' ══════════════════════════════════════════════════════════════════\n');
fprintf(' BLOCK A\n');
fprintf('   Flat-prior optimal lag:    p_flat = %d\n',   p_flat);
fprintf('   (for diagnostics only; Block B uses p_MN)\n\n');
fprintf(' BLOCK B — GLP-STRICT SEQUENTIAL OPTIMISATION\n');
fprintf('   Fixed lag:                 p_MN   = %d  (22 trading days)\n', p_MN);
fprintf('   Stage 1 (tau only):\n');
fprintf('     tau_s1*  = %.4f  (lambda_GLP = %.4f)\n',  tau_s1, 1/tau_s1);
fprintf('     lnML     = %.4f\n',                        lnML_s1);
fprintf('   GLP-strict (tau, lambda_DIO, mu_SC)   [theta fixed at 1]:\n');
fprintf('     tau*        = %.4f  (lambda_GLP = %.4f)\n', tau_opt, 1/tau_opt);
fprintf('     theta       = %.4f  (GLP-FIXED)\n',         decay_opt);
fprintf('     lambda_DIO* = %.4f  (= delta_GLP, dummy-init-obs)\n', delta_opt);
fprintf('     mu_SC*      = %.4f  (= mu_GLP, sum-of-coef)\n',      mu_opt);
fprintf('     omega       = %.4f  (FIXED, toolbox scalar)\n',   omega_opt);
fprintf('     lnML        = %.4f\n',                     lnML_s2);
fprintf('   DeltalnML (GLP_strict - Stage1) = %.4f\n',   lnML_s2 - lnML_s1);
fprintf('   Evidence class: %s\n',                        evid_str);
fprintf('\n MODEL COMPARISON (all at p_MN=%d)\n', p_MN);
fprintf('   lnML  Flat    = %.4f\n', lnML_flat_pMN_full);
fprintf('   lnML  MN-S1   = %.4f\n', lnML_MN_s1);
fprintf('   lnML  MN-GLP  = %.4f  <- MAIN\n', lnML_MN_glp);
fprintf('   BF [MN-GLP / Flat]: ');  print_bf(lnML_MN_glp - lnML_flat_pMN_full);
fprintf('   BF [MN-GLP / MN-S1]: '); print_bf(lnML_MN_glp - lnML_MN_s1);
fprintf('\n CONDITIONING\n');
fprintf('   cond(X''X)          [Flat]:    %.4e\n', cond_flat);
fprintf('   cond(X''X+V^{-1})   [MN-GLP]: %.4e\n', cond_MN_glp);
fprintf('   Regularisation ratio:          %.2f\n', cond_flat/cond_MN_glp);
fprintf('\n ICA FEASIBILITY (LMS 2017)\n');
fprintf('   Hill alpha < 4: %d / %d variables\n', n_heavy_glp, M);
if n_heavy_glp >= M-1
    fprintf('   -> ICA identification FEASIBLE\n');
else
    fprintf('   -> ICA identification: CHECK\n');
end
fprintf(' ══════════════════════════════════════════════════════════════════\n');

% =========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function fprintf_corr(Corr, VAR_NAMES, M, label)
    fprintf('\n [%s]:\n', label);
    fprintf(' ');
    for j = 1:M, fprintf('%8s ', VAR_NAMES{j}); end
    fprintf('\n');
    for i = 1:M
        fprintf(' %-8s ', VAR_NAMES{i});
        for j = 1:M, fprintf('%8.4f ', Corr(i,j)); end
        fprintf('\n');
    end
end

function row = residual_tail_stats(u_j, HILL_FRAC, var_name)
    mu_j      = mean(u_j);
    sig_j     = std(u_j);
    u_std     = (u_j - mu_j) / sig_j;
    S_j       = mean(u_std.^3);
    K_j       = mean(u_std.^4);
    abs_u     = sort(abs(u_j), 'descend');
    m_h       = max(2, floor(HILL_FRAC * numel(u_j)));
    alpha_hat = 0;
    if abs_u(m_h+1) > 0
        alpha_hat = max(0, 1/(mean(log(abs_u(1:m_h))) - log(abs_u(m_h+1))));
    end
    fprintf(' %-10s %10.4f %10.4f %10.4f %12.4f %10.4f\n', ...
            var_name, mu_j, sig_j, S_j, K_j-3, alpha_hat);
    row = [mu_j, sig_j, S_j, K_j, alpha_hat];
end

function print_bf(delta)
    if abs(delta) > 700
        fprintf('Inf (overflow)\n');
    else
        bf = exp(delta);
        if     bf > 150, cat_str = 'decisive';
        elseif bf >  20, cat_str = 'strong';
        elseif bf >   3, cat_str = 'positive';
        else,            cat_str = 'negligible'; 
        end
        fprintf('%.4e  [%s]\n', bf, cat_str);
    end
end

