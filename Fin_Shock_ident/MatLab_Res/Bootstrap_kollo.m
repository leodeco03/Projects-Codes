% =========================================================================
%  Bootstrap_kollo.m — moving-block bootstrap of the contemporaneous
%                      independence diagnostics of the identified shocks.
%
%  INPUT   mori_3var.mat  (V_ref, M, T_eff)        [run Mori_3var.m first]
%  OUTPUT  bootstrap_kollo_results.mat  (point estimate, bootstrap standard
%          error and percentile interval for the three panels below; the
%          point estimates also appear as indep_diag in mori_3var.mat)
%
%  Panels, with the value implied by mutual independence:
%    (A) co-kurtosis      E[v_i^2 v_j^2] = 1   NOT robust to common stochastic
%                                              volatility -> a rejection here
%                                              is expected, not informative
%    (B) co-skewness      E[v_i^2 v_j]   = 0   SV-robust
%    (C) asymmetric 4th   E[v_i^3 v_j]   = 0   SV-robust
%  Blocks preserve the volatility clustering of the shocks, under which an
%  i.i.d. bootstrap would give intervals that are far too narrow.
% =========================================================================
clear; clc;
rng(7, 'twister');            % reproducible resampling

load('mori_3var.mat', 'V_ref', 'M', 'T_eff');

BLOCK_LEN    = 20;            % trading days (~one month of volatility persistence)
BLOCK_BOOT_N = 5000;          % replications
ALPHA        = 0.05;
q_lo = ALPHA/2;  q_hi = 1 - ALPHA/2;

Vsd  = (V_ref - mean(V_ref)) ./ std(V_ref);
nblk = ceil(T_eff / BLOCK_LEN);

boot_co_kurt  = zeros(M, M, BLOCK_BOOT_N);   % E[v_i^2 v_j^2]
boot_co_skew3 = zeros(M, M, BLOCK_BOOT_N);   % E[v_i^2 v_j]
boot_co_kurtA = zeros(M, M, BLOCK_BOOT_N);   % E[v_i^3 v_j]

fprintf('Moving-block bootstrap on V_ref (%d replications, block = %d days)...\n', ...
        BLOCK_BOOT_N, BLOCK_LEN);
tic;
for b = 1:BLOCK_BOOT_N
    starts = randi(T_eff - BLOCK_LEN + 1, nblk, 1);
    idx    = reshape((starts + (0:BLOCK_LEN-1))', [], 1);
    [boot_co_kurt(:,:,b), boot_co_skew3(:,:,b), boot_co_kurtA(:,:,b)] = ...
        co_moments(Vsd(idx(1:T_eff), :));
end
fprintf('Done in %.1f s.\n', toc);

% ---- point estimates, bootstrap standard errors and percentile intervals ----
[est_co_kurt, est_co_skew3, est_co_kurtA] = co_moments(Vsd);
[se_co_kurt,  ci_co_kurt ] = boot_summary(boot_co_kurt,  q_lo, q_hi);
[se_co_skew3, ci_co_skew3] = boot_summary(boot_co_skew3, q_lo, q_hi);
[se_co_kurtA, ci_co_kurtA] = boot_summary(boot_co_kurtA, q_lo, q_hi);

print_panel('(A) CO-KURTOSIS     E[v_i^2 v_j^2]  [not SV-robust]', 1, ...
            est_co_kurt,  se_co_kurt,  ci_co_kurt,  true);
print_panel('(B) CO-SKEWNESS     E[v_i^2 v_j]    [SV-robust]',     0, ...
            est_co_skew3, se_co_skew3, ci_co_skew3, false);
print_panel('(C) ASYMMETRIC 4th  E[v_i^3 v_j]    [SV-robust]',     0, ...
            est_co_kurtA, se_co_kurtA, ci_co_kurtA, false);

save('bootstrap_kollo_results.mat', ...
     'est_co_kurt','est_co_skew3','est_co_kurtA', ...
     'se_co_kurt','se_co_skew3','se_co_kurtA', ...
     'ci_co_kurt','ci_co_skew3','ci_co_kurtA', ...
     'BLOCK_LEN','BLOCK_BOOT_N','M','T_eff','-v7.3');
fprintf('\nSaved bootstrap_kollo_results.mat (estimates, SE, CI of the three panels).\n');

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function [c22, c21, c31] = co_moments(V)
% Co-moments of the standardised shocks, rows i and columns j:
%   c22(i,j) = E[v_i^2 v_j^2]   (symmetric)
%   c21(i,j) = E[v_i^2 v_j]     (ordered pairs)
%   c31(i,j) = E[v_i^3 v_j]     (ordered pairs)
    T   = size(V, 1);
    V2  = V.^2;
    c22 = (V2' * V2) / T;
    c21 = (V2' * V ) / T;
    c31 = ((V.^3)' * V) / T;
end

function [se, ci] = boot_summary(draws, q_lo, q_hi)
% Bootstrap standard error (M x M) and percentile interval (M x M x 2).
    se = std(draws, 0, 3);
    ci = quantile(draws, [q_lo q_hi], 3);
end

function print_panel(name, h0, est, se, ci, symmetric)
% One panel of the independence table.  symmetric = true reports the upper
% triangle only (the statistic is symmetric in i,j), false all ordered pairs.
    M = size(est, 1);
    fprintf('\n=== %s   (H0: independence => %g) ===\n', name, h0);
    fprintf('%-9s %-9s %10s %10s %20s %12s\n', ...
            'shock i','shock j','estimate','boot SE','95% boot CI','verdict');
    fprintf('%s\n', repmat('-', 1, 75));
    for i = 1:M
        for j = 1:M
            if symmetric && j <= i, continue; end
            if ~symmetric && j == i, continue; end
            lo = ci(i,j,1);  hi = ci(i,j,2);
            if h0 < lo || h0 > hi, verdict = '*** REJECT'; else, verdict = '    accept'; end
            fprintf('v_%-7d v_%-7d %10.4f %10.4f   [%6.4f, %6.4f] %12s\n', ...
                    i, j, est(i,j), se(i,j), lo, hi, verdict);
        end
    end
end
