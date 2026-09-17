% =========================================================================
%  proxy_svar1.m - STAGE A: from daily shocks to a monthly external
%                  instrument, plus its validation against uncertainty indices.
%
%  ORDER OF OPERATIONS
%    1. load the daily identified shocks and apply the sign convention
%    2. aggregate to monthly: signed mean = instrument; ABS/SQ/POS/NEG are
%       validation aggregates only
%    3. read the uncertainty indices (published monthly: no aggregation, only
%       calendar matching) and build their unforecastable component
%    4. correlate, on matched calendar months, level with level and
%       innovation with innovation
%
%  The shocks travel with NO economic label: v1..vM in their natural order.
%  Meaning is assigned ex post (IRF signature, own kurtosis, time series).
%
%  INPUT   mori_3var.mat  (V_ref, B_med, lam_med, VAR_NAMES, dates_vec)
%          US_Policy_Uncertainty_Data.xlsx, Global_Policy_Uncertainty_Data.xlsx,
%          data_gpr_export.xls
%  OUTPUT  shocks_monthly_<SRC>.mat          external instrument -> Stage B
%          corr_extIV_uncertainty_<SRC>.mat  correlation table, 9 measures
%          instrument_diagnostics_<SRC>.mat  mean-zero / no-autocorrelation tests
%          fig_p02A_corr_<SRC>.png, fig_p02A_corrALL_<SRC>.png,
%          fig_p02A_instr_<SRC>.png
%  NEXT    proxy_svar_02_cholesky.m  (Stage B: monthly macro IRFs)
%
%  Why the signed monthly mean is the instrument, why ABS/SQ/POS/NEG are not,
%  and what each correlation panel is for: see the thesis.
% =========================================================================

clear; clc; close all;

%% ---- CONFIGURATION ------------------------------------------------------
SRC           = 'mori_3var.mat';
[~, SRC_BASE] = fileparts(SRC);

% Sign convention, applied ONCE to the daily shocks before any aggregation and
% saved with them, so Stage B inherits it instead of redefining it.  Upstream
% each shock is anchored so that its FEVD-dominant variable loads positively
% (v1 -> CSPREAD, v2 -> SP500, v3 -> VIX); flipping v2 makes all three adverse,
% which puts the three IRF panels on a comparable footing.  ABS and SQ are even
% functions and are unaffected; POS and NEG swap.
SIGN_CONV = [+1, -1, +1];        % one entry per shock; [] = keep the upstream anchor

AR_Q   = 6;      % monthly lags of the AR filter used in STEP 3b
LB_H   = 12;     % Ljung-Box lags (one year) in STEP 4c
NTOP   = 6;      % months listed per shock in STEP 4b

banner(sprintf('proxy_svar1 - STAGE A: monthly instrument + uncertainty   SRC = %s', SRC));

%% =========================================================================
%  STEP 1 - DAILY SHOCKS  (all shocks, no selection)
% =========================================================================
S = load(SRC, 'V_ref', 'B_med', 'lam_med', 'VAR_NAMES', 'dates_vec');
V_ref     = S.V_ref;  [T_eff, M] = size(V_ref);
B_med     = S.B_med;
lam_med   = S.lam_med(:);
VAR_NAMES = cellstr(S.VAR_NAMES(:));
if ~isfield(S,'dates_vec') || ~isdatetime(S.dates_vec) || numel(S.dates_vec) ~= T_eff
    error('%s carries no usable dates_vec: re-run Mori_3var.m.', SRC);
end
dates_daily = S.dates_vec(:);
clear S

if isempty(SIGN_CONV), SIGN_CONV = ones(1,M); end
SIGN_CONV = SIGN_CONV(:)';
if numel(SIGN_CONV) ~= M
    error('SIGN_CONV has %d entries but there are %d shocks.', numel(SIGN_CONV), M);
end
if ~all(ismember(SIGN_CONV, [-1 1]))
    error('SIGN_CONV entries must be -1 or +1.');
end
V_ref = V_ref .* SIGN_CONV;       % daily shocks
B_med = B_med .* SIGN_CONV;       % impact loadings (B*eps stays invariant)

% Interpretation-only annotation: the variable each shock explains the most on
% impact, and the sign it loads with after SIGN_CONV.
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
shock_cols = 1:M;                 % kept for the files written below

fprintf('  M=%d  T_eff=%d  | all %d shocks carried to monthly, natural order\n', M, T_eff, M);
fprintf('  SIGN_CONV = [%s]  (applied to the DAILY shocks)\n', ...
        strjoin(arrayfun(@(s) sprintf('%+d',s), SIGN_CONV, 'UniformOutput', false), ' '));
for k = 1:M
    fprintf('    %-4s  kappa=%6.2f   [FEVD-dominant: %-8s loads %s on impact]\n', ...
            shock_tag{k}, lam_med(k), dom_daily{k}, tern(dom_sign(k)>0,'POSITIVELY','NEGATIVELY'));
end

%% =========================================================================
%  STEP 2 - MONTHLY AGGREGATION
%    instrument = MEAN (signed);  intensity = ABS, SQ;  asymmetry = POS, NEG.
%    This is the only place where the daily series is touched.
% =========================================================================
banner('STEP 2 - MONTHLY AGGREGATION (instrument = signed mean)');

ym_daily      = year(dates_daily)*100 + month(dates_daily);
[ym_u, ~, g]  = unique(ym_daily, 'sorted');
Tm            = numel(ym_u);
dates_monthly = datetime(floor(ym_u/100), mod(ym_u,100), 1);
nday_month    = accumarray(g, 1);
ym_monthly    = ym_u;

MEAN = zeros(Tm,M); SUM = zeros(Tm,M); POS = zeros(Tm,M);
NEG  = zeros(Tm,M); ABS = zeros(Tm,M); SQ  = zeros(Tm,M);
for k = 1:M
    v          = V_ref(:,k);
    SUM(:,k)   = accumarray(g, v);
    MEAN(:,k)  = SUM(:,k) ./ nday_month;
    POS(:,k)   = accumarray(g, max(v,0));
    NEG(:,k)   = accumarray(g, min(v,0));
    ABS(:,k)   = accumarray(g, abs(v));
    SQ(:,k)    = accumarray(g, v.^2);
end

save(sprintf('shocks_monthly_%s.mat', SRC_BASE), ...
     'dates_monthly','ym_monthly','nday_month','M','VAR_NAMES','shock_cols','shock_tag', ...
     'dom_daily','dom_sign','SIGN_CONV','MEAN','SUM','POS','NEG','ABS','SQ','SRC','-v7.3');
fprintf('  %d daily obs -> %d months (%s to %s).  Saved shocks_monthly_%s.mat\n', ...
        T_eff, Tm, string(dates_monthly(1),'yyyy-MM'), string(dates_monthly(end),'yyyy-MM'), SRC_BASE);

% Var(mean_m) = 1/D_m, so an unbalanced number of trading days makes the
% instrument conditionally heteroskedastic by construction.  Reported, not hidden.
[Dmin, iDmin] = min(nday_month);
[Dmax, iDmax] = max(nday_month);
fprintf('  Trading days/month: mean=%.1f  min=%d (%s)  max=%d (%s)  -> sd ratio %.2f\n', ...
        mean(nday_month), Dmin, string(dates_monthly(iDmin),'yyyy-MM'), ...
        Dmax, string(dates_monthly(iDmax),'yyyy-MM'), sqrt(Dmax/Dmin));
if Dmin < 15
    fprintf('  [!] %s has only %d trading days: its mean_k has ~%.0f%% larger sd.\n', ...
            string(dates_monthly(iDmin),'yyyy-MM'), Dmin, 100*(sqrt(mean(nday_month)/Dmin)-1));
end

%% =========================================================================
%  STEP 3 - UNCERTAINTY INDICES  (published monthly; only calendar matching)
% =========================================================================
banner('STEP 3 - UNCERTAINTY INDICES');

IDX = struct('name',{},'ym',{},'lvl',{});

Tu  = readtab('US_Policy_Uncertainty_Data.xlsx','Main News Index');
yU  = getcol(Tu,'Year'); mU = getcol(Tu,'Month'); vU = getcol(Tu,'News_Based');
okU = isfinite(yU) & isfinite(mU) & mU >= 1 & mU <= 12 & isfinite(vU);
IDX(end+1) = mk_idx('US_EPU', yU(okU)*100 + mU(okU), vU(okU));

Tg  = readtab('Global_Policy_Uncertainty_Data.xlsx','Sheet1');
yG  = getcol(Tg,'Year'); mG = getcol(Tg,'Month');
okG = isfinite(yG) & isfinite(mG) & mG >= 1 & mG <= 12;
for nm = {'GEPU_current','GEPU_ppp'}
    vG = getcol(Tg, nm{1});  ok = okG & isfinite(vG);
    IDX(end+1) = mk_idx(nm{1}, yG(ok)*100 + mG(ok), vG(ok)); %#ok<SAGROW>
end

Tr = readtab('data_gpr_export.xls','Sheet1');
dR = getcol(Tr,'month');
if ~isdatetime(dR)
    error('data_gpr_export.xls: the "month" column must be read as datetime.');
end
ymR = year(dR)*100 + month(dR);
for nm = {'GPR','GPRC_USA'}
    vR = getcol(Tr, nm{1});  ok = isfinite(ymR) & isfinite(vR);
    IDX(end+1) = mk_idx(nm{1}, ymR(ok), vR(ok)); %#ok<SAGROW>
end

for j = 1:numel(IDX)
    fprintf('    %-13s : %4d obs  %d-%d\n', IDX(j).name, numel(IDX(j).ym), ...
            min(IDX(j).ym), max(IDX(j).ym));
end

%% =========================================================================
%  STEP 3b - UNFORECASTABLE COMPONENT OF EACH INDEX
%    eta = residual of an AR(AR_Q) on the log level, i.e. the innovation with
%    the persistence ESTIMATED.  dlog imposes sum(rho) = 1 instead: the column
%    below is what the data say, and the comparison is itself a result.
%    The filter is univariate, on the index's own past only.
% =========================================================================
banner(sprintf('STEP 3b - UNFORECASTABLE COMPONENT (AR(%d) residual on log level)', AR_Q));

fprintf('  %-13s %10s %8s   %s\n','index','sum(rho)','n','reading');
fprintf('  %s\n', repmat('-',1,68));
for j = 1:numel(IDX)
    [IDX(j).eta, IDX(j).rho_sum, IDX(j).ar_n] = ...
        ar_innovation(log(IDX(j).lvl), IDX(j).ym, AR_Q);
    fprintf('  %-13s %10.3f %8d   %s\n', IDX(j).name, IDX(j).rho_sum, IDX(j).ar_n, ...
            tern(IDX(j).rho_sum < 0.95, 'mean-reverting -> dlog OVER-differences', ...
                                        'near unit root -> dlog ~ ok'));
end

%% =========================================================================
%  STEP 4 - MONTHLY CORRELATIONS  (instrument vs uncertainty)
%    Objects of the same nature only: level with level, innovation with
%    innovation.  level~mean_k is the deliberate mismatch control and must come
%    out around zero.  Which of pos_k / neg_k is "the bad side" flips with the
%    sign normalisation, so read (A2) as "is there asymmetry?" and not as
%    "the downside matters more".
% =========================================================================
banner('STEP 4 - MONTHLY CORRELATIONS (external instrument vs uncertainty)');

meas      = {'level~abs_k','level~sq_k','level~pos_k','level~|neg_k|', ...
             'dlog~mean_k','eta~mean_k','|dlog|~abs_k','|eta|~abs_k','level~mean_k'};
meas_note = {'turbulence, L1 (robust)','turbulence, L2 (tail-driven)', ...
             'one-sided activity: UP days','one-sided activity: DOWN days', ...
             'directional - persistence IMPOSED at 1', ...
             'directional - persistence ESTIMATED  [PREFERRED]', ...
             'two-sided, sign-free (dlog filter)','two-sided, sign-free (AR filter)', ...
             'MISMATCH CONTROL - must be ~0 by construction'};

rows = struct('index',{},'shock',{},'measure',{},'pearson_r',{},'pearson_p',{}, ...
              'spearman_r',{},'spearman_p',{},'n',{});
for j = 1:numel(IDX)
    [~, ia, ib] = intersect(IDX(j).ym, ym_monthly);
    if numel(ia) < 6, continue; end
    lvl  = IDX(j).lvl(ia);
    dlog = [NaN; diff(log(lvl))];
    % dlog is a true one-month change only where the matched months are
    % contiguous: 89 is the December -> January step in yyyymm.
    step = [NaN; diff(double(IDX(j).ym(ia)))];
    dlog(~(step == 1 | step == 89)) = NaN;
    et   = IDX(j).eta(ia);          % built in STEP 3b on the index's full sample

    for k = 1:M
        a  = ABS(ib,k);  s = SQ(ib,k);  mn = MEAN(ib,k);
        po = POS(ib,k);  ng = -NEG(ib,k);
        pairs = {lvl, a; lvl, s; lvl, po; lvl, ng; dlog, mn; et, mn; ...
                 abs(dlog), a; abs(et), a; lvl, mn};
        for mm = 1:numel(meas)
            rows(end+1) = mkrow(IDX(j).name, k, meas{mm}, pairs{mm,1}, pairs{mm,2}); %#ok<SAGROW>
        end
    end
end
CORR        = struct2table(rows);
idx_names   = {IDX.name};
rho_sum_all = [IDX.rho_sum];
idx_names_all = idx_names;
save(sprintf('corr_extIV_uncertainty_%s.mat', SRC_BASE), ...
     'CORR','SRC','shock_cols','shock_tag','AR_Q','rho_sum_all','idx_names_all','-v7.3');

for mm = 1:numel(meas)
    print_matrix(CORR, idx_names, shock_tag, dom_daily, meas{mm}, meas_note{mm});
end

% --- the instrument itself: the three filters of the index side by side ------
fprintf('\n  >>> THE INSTRUMENT (mean_k) against the three filters of each index\n');
fprintf('      eta  = AR(%d) innovation (persistence estimated)  <- preferred\n', AR_Q);
fprintf('      dlog = log-difference    (persistence imposed = 1)\n');
fprintf('      lvl  = raw level         (mismatch control, expected ~0)\n\n');
for k = 1:M
    fprintf('  %s\n', shock_tag{k});
    fprintf('  %-13s %9s %9s %9s %9s\n','index','eta~mn','dlog~mn','lvl~mn','rho_sum');
    fprintf('  %s\n', repmat('-',1,53));
    for j = 1:numel(idx_names)
        fprintf('  %-13s %9.3f %9.3f %9.3f %9.3f\n', idx_names{j}, ...
                getr(CORR, idx_names{j}, k, 'eta~mean_k'), ...
                getr(CORR, idx_names{j}, k, 'dlog~mean_k'), ...
                getr(CORR, idx_names{j}, k, 'level~mean_k'), IDX(j).rho_sum);
    end
    fprintf('\n');
end

% --- asymmetry: up-side vs down-side activity -------------------------------
fprintf('  >>> ASYMMETRY  (level~pos_k vs level~|neg_k|),  gap = r_neg - r_pos\n');
fprintf('  %-13s', 'index');
for k = 1:M
    fprintf(' %8s %8s %7s', [shock_tag{k} ' pos'], [shock_tag{k} ' neg'], 'gap');
end
fprintf('\n  %s\n', repmat('-',1,13+25*M));
for j = 1:numel(idx_names)
    fprintf('  %-13s', idx_names{j});
    for k = 1:M
        rp = getr(CORR, idx_names{j}, k, 'level~pos_k');
        rn = getr(CORR, idx_names{j}, k, 'level~|neg_k|');
        fprintf(' %8.3f %8.3f %7.3f', rp, rn, rn - rp);
    end
    fprintf('\n');
end

fprintf('\n  Strongest uncertainty association per instrument (|Pearson|, level~abs_k):\n');
for k = 1:M
    sub = CORR(strcmp(CORR.measure,'level~abs_k') & CORR.shock == k & isfinite(CORR.pearson_r), :);
    if isempty(sub), continue; end
    [~, o] = max(abs(sub.pearson_r));
    fprintf('    %-4s : %-13s r=%+.3f (p=%.3f)   [daily FEVD-dominant: %s]\n', ...
            shock_tag{k}, sub.index{o}, sub.pearson_r(o), sub.pearson_p(o), dom_daily{k});
end
fprintf('\n  The p-values assume i.i.d. sampling, so on the LEVEL panels (both series\n');
fprintf('  persistent) they are overstated: those panels are descriptive, the\n');
fprintf('  inferential weight sits on eta~mean_k, where both objects are innovations.\n');

%% =========================================================================
%  STEP 4b - THE INSTRUMENT, READ DIRECTLY
%    The months in which |mean_k| is largest should be recognisable events; if
%    they are not, the shock is a numerical artefact whatever the correlations say.
% =========================================================================
banner('STEP 4b - THE INSTRUMENT ITSELF (mean_k): largest months');

for k = 1:M
    fprintf('\n  %-4s  sd=%.4f  skew=%+.2f  kurt=%.2f  |  top-%d months by |mean_k|:\n', ...
            shock_tag{k}, std(MEAN(:,k)), skewness(MEAN(:,k)), kurtosis(MEAN(:,k)), NTOP);
    [~, o] = sort(abs(MEAN(:,k)), 'descend');
    fprintf('      %-9s %10s %10s %10s %10s\n','month','mean_k','pos_k','neg_k','abs_k');
    for t = o(1:min(NTOP,numel(o)))'
        fprintf('      %-9s %10.4f %10.3f %10.3f %10.3f\n', ...
                string(dates_monthly(t),'yyyy-MM'), MEAN(t,k), POS(t,k), NEG(t,k), ABS(t,k));
    end
end

%% =========================================================================
%  STEP 4c - INSTRUMENT VALIDITY DIAGNOSTICS
%    Relevance needs the macro residuals and belongs to Stage B.  What can be
%    tested here is what makes the aggregate a shock at all: mean zero and no
%    serial correlation.  Expected pattern: mean_k white, and mean_k^2, abs_k,
%    sq_k massively rejected - volatility clusters, which is why they are
%    validation objects and not instruments.
% =========================================================================
banner('STEP 4c - INSTRUMENT VALIDITY (mean zero + no serial correlation)');

fprintf('  Ljung-Box with %d lags; t-statistic on the sample mean.\n', LB_H);
fprintf('  %-14s %9s %8s %8s %10s %9s   %s\n', ...
        'series','mean','t(mean)','rho(1)','LB(12)','p-value','reading');
fprintf('  %s\n', repmat('-',1,86));
lb_tab = struct('series',{},'shock',{},'measure',{},'mean',{},'t_mean',{},'rho1',{},'LB',{},'p',{});
for k = 1:M
    series_set = {MEAN(:,k), MEAN(:,k).^2, ABS(:,k), SQ(:,k)};
    names_set  = {'mean_k','mean_k^2','abs_k','sq_k'};
    for si = 1:numel(series_set)
        x         = series_set{si};
        [Q,pv,r1] = ljungbox(x, LB_H);
        mu_x      = mean(x);
        t_x       = mu_x / (std(x)/sqrt(numel(x)));
        if si == 1
            verdict = tern(pv > 0.05, 'WHITE  ok', 'AUTOCORRELATED [!]');
        else
            verdict = tern(pv < 0.05, 'rejected (as expected)', 'not rejected [?]');
        end
        fprintf('  %-4s %-9s %9.4f %8.2f %8.3f %10.1f %9.4f   %s\n', ...
                shock_tag{k}, names_set{si}, mu_x, t_x, r1, Q, pv, verdict);
        lb_tab(end+1) = struct('series', sprintf('%s %s', shock_tag{k}, names_set{si}), ...
                               'shock', k, 'measure', names_set{si}, 'mean', mu_x, ...
                               't_mean', t_x, 'rho1', r1, 'LB', Q, 'p', pv); %#ok<SAGROW>
    end
    fprintf('  %s\n', repmat('.',1,86));
end
LB_TABLE = struct2table(lb_tab);
save(sprintf('instrument_diagnostics_%s.mat', SRC_BASE), 'LB_TABLE','LB_H','SIGN_CONV','-v7.3');

%% =========================================================================
%  STEP 5 - FIGURES
% =========================================================================
banner('STEP 5 - FIGURES');

nI = numel(IDX);
H  = corr_matrix(CORR, idx_names, M, 'level~abs_k');
f  = figure('Color','w','Position',[40 40 720 460]);
imagesc(H,[-1 1]); colormap(redbluemap_local()); colorbar;
set(gca,'XTick',1:M,'XTickLabel',shock_tag,'YTick',1:nI,'YTickLabel',idx_names, ...
        'TickLabelInterpreter','none');
title('External instrument (|shock|) vs uncertainty LEVEL - Pearson','Interpreter','none');
label_cells(H);
print(f, sprintf('fig_p02A_corr_%s.png', SRC_BASE), '-dpng','-r150');

ncolF = 3; nrowF = ceil(numel(meas)/ncolF);
f2 = figure('Color','w','Position',[20 20 400*ncolF 300*nrowF]);
for mm = 1:numel(meas)
    subplot(nrowF, ncolF, mm);
    Hm = corr_matrix(CORR, idx_names, M, meas{mm});
    imagesc(Hm,[-1 1]); colormap(redbluemap_local());
    set(gca,'XTick',1:M,'XTickLabel',shock_tag,'YTick',1:nI,'YTickLabel',idx_names, ...
            'TickLabelInterpreter','none','FontSize',7);
    title({meas{mm}, meas_note{mm}},'Interpreter','none','FontSize',8);
    label_cells(Hm, 7);
    if mm == numel(meas), cb = colorbar; cb.Label.String = 'Pearson r'; end
end
sgtitle('Uncertainty indices vs every monthly aggregate of the daily shocks (common scale)', ...
        'FontSize',11,'Interpreter','none');
print(f2, sprintf('fig_p02A_corrALL_%s.png', SRC_BASE), '-dpng','-r150');

f3 = figure('Color','w','Position',[20 20 1250 250*M]);
for k = 1:M
    subplot(M,2,2*k-1);
    plot(dates_monthly, MEAN(:,k), '-', 'Color',[0.85 0.33 0.10], 'LineWidth',0.8); hold on;
    yline(0,'k:'); grid on; axis tight;
    title(sprintf('%s - monthly instrument mean_k (signed monthly mean)', shock_tag{k}), ...
          'Interpreter','none','FontSize',9);
    ylabel('mean_k','Interpreter','none','FontSize',8); set(gca,'FontSize',7);

    subplot(M,2,2*k);
    bar(dates_monthly, POS(:,k), 1, 'FaceColor',[0.20 0.55 0.25],'EdgeColor','none'); hold on;
    bar(dates_monthly, NEG(:,k), 1, 'FaceColor',[0.75 0.20 0.15],'EdgeColor','none');
    plot(dates_monthly, SUM(:,k), 'k-','LineWidth',0.5);
    yline(0,'k:'); grid on; axis tight;
    title(sprintf('%s - pos_k (green) / neg_k (red); black = sum_k', shock_tag{k}), ...
          'Interpreter','none','FontSize',9);
    set(gca,'FontSize',7);
end
sgtitle('The monthly instrument and its one-sided decomposition','FontSize',11,'Interpreter','none');
print(f3, sprintf('fig_p02A_instr_%s.png', SRC_BASE), '-dpng','-r150');

banner(sprintf(['DONE - Stage A\n  instrument:   shocks_monthly_%s.mat (MEAN)\n' ...
                '  correlations: corr_extIV_uncertainty_%s.mat (%d measures, AR_Q=%d)\n' ...
                '  next:         proxy_svar_02_cholesky.m'], ...
                SRC_BASE, SRC_BASE, numel(meas), AR_Q));

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function T = readtab(file, sheet)
    T = readtable(file, 'Sheet', sheet, 'VariableNamingRule', 'preserve');
end

function v = getcol(T, key)
% First column whose name contains key (case-insensitive); text is converted
% to numbers, datetime and numeric columns are returned as they are.
    vn  = T.Properties.VariableNames;
    idx = find(contains(lower(vn), lower(key)), 1);
    if isempty(idx)
        error('Column "%s" not found in [%s].', key, strjoin(vn, ', '));
    end
    c = T.(vn{idx});
    if isnumeric(c) || isdatetime(c), v = c; else, v = str2double(string(c)); end
end

function s = mk_idx(name, ym, lvl)
% One uncertainty index: sorted by month, duplicates dropped.
    [ym, srt] = sort(ym(:));
    lvl       = lvl(:);  lvl = lvl(srt);
    [ym, ic]  = unique(ym, 'stable');
    s = struct('name', name, 'ym', ym, 'lvl', lvl(ic));
end

function [eta, rho_sum, n_used] = ar_innovation(y, ym, q)
% Residual of an AR(q) on y, formed only where the q preceding observations are
% calendar-contiguous, so no lag is ever built across a gap in the index.
% rho_sum is the estimated persistence, to be compared with the unit value that
% the log-difference imposes.
    y  = y(:);  ym = double(ym(:));  n = numel(y);
    X  = [ones(n,1), nan(n,q)];
    for L = 1:q
        X(L+1:end, 1+L) = y(1:end-L);
    end
    mnum   = 12*floor(ym/100) + mod(ym,100);            % months since year 0
    contig = false(n,1);
    contig(q+1:end) = (mnum(q+1:end) - mnum(1:end-q)) == q;
    ok     = contig & isfinite(y) & all(isfinite(X),2);
    eta    = nan(n,1);  rho_sum = NaN;  n_used = sum(ok);
    if n_used > 5*q
        b        = X(ok,:) \ y(ok);
        eta(ok)  = y(ok) - X(ok,:)*b;
        rho_sum  = sum(b(2:end));
    end
end

function r = mkrow(idxname, k, measure, x, y)
    o = corr_pair(x, y);
    r = struct('index', idxname, 'shock', k, 'measure', measure, ...
               'pearson_r', o.pr, 'pearson_p', o.pp, ...
               'spearman_r', o.sr, 'spearman_p', o.sp, 'n', o.n);
end

function out = corr_pair(x, y)
% Pearson and Spearman correlation on the finite pairs.
    x = x(:); y = y(:);
    ok = isfinite(x) & isfinite(y);
    x  = x(ok); y = y(ok);
    out = struct('pr',NaN,'pp',NaN,'sr',NaN,'sp',NaN,'n',numel(x));
    if numel(x) < 5 || std(x) == 0 || std(y) == 0, return; end
    [out.pr, out.pp] = corr(x, y);
    [out.sr, out.sp] = corr(x, y, 'Type', 'Spearman');
end

function r = getr(CORR, idxname, k, measure)
    v = CORR.pearson_r(strcmp(CORR.index,idxname) & CORR.shock == k & strcmp(CORR.measure,measure));
    if isempty(v), r = NaN; else, r = v(1); end
end

function [R, Pv] = corr_matrix(CORR, idx_names, M, measure)
% Index x shock matrix of Pearson correlations (and p-values) for one measure.
    R  = nan(numel(idx_names), M);
    Pv = nan(numel(idx_names), M);
    sub = CORR(strcmp(CORR.measure, measure), :);
    for j = 1:numel(idx_names)
        sj = sub(strcmp(sub.index, idx_names{j}), :);
        for k = 1:M
            hit = sj.shock == k;
            if any(hit)
                R(j,k)  = sj.pearson_r(find(hit,1));
                Pv(j,k) = sj.pearson_p(find(hit,1));
            end
        end
    end
end

function print_matrix(CORR, idx_names, shock_tag, dom_daily, measure, note)
    [R, Pv] = corr_matrix(CORR, idx_names, numel(shock_tag), measure);
    if all(isnan(R(:))), return; end
    fprintf('\n  >>> measure = %-14s  [%s]\n', measure, note);
    fprintf('  %-13s', 'index');
    for k = 1:numel(shock_tag)
        fprintf(' %14s', sprintf('%s(%s)', shock_tag{k}, dom_daily{k}(1:min(4,end))));
    end
    fprintf('\n  %s\n', repmat('-',1,13+15*numel(shock_tag)));
    for j = 1:numel(idx_names)
        fprintf('  %-13s', idx_names{j});
        for k = 1:numel(shock_tag)
            fprintf(' %9.3f%-5s', R(j,k), sig_star(Pv(j,k)));
        end
        fprintf('\n');
    end
    fprintf('  (*** p<.01  ** p<.05  * p<.10)\n');
end

function st = sig_star(p)
    if     isnan(p), st = '';
    elseif p < 0.01, st = '***';
    elseif p < 0.05, st = '**';
    elseif p < 0.10, st = '*';
    else,            st = '';
    end
end

function label_cells(A, fs)
% Write each correlation inside its heatmap cell.
    if nargin < 2, fs = 9; end
    for j = 1:size(A,1)
        for k = 1:size(A,2)
            if ~isnan(A(j,k))
                text(k, j, sprintf('%.2f',A(j,k)), 'HorizontalAlignment','center', ...
                     'FontSize', fs, 'Color', tern(abs(A(j,k))>0.5,'w','k'));
            end
        end
    end
end

function [Q, pv, r1] = ljungbox(x, h)
% Ljung-Box statistic and the first autocorrelation (no Econometrics Toolbox).
    x  = x(:); x = x(isfinite(x));
    T  = numel(x); x = x - mean(x);
    c0 = sum(x.^2);  Q = 0;  r1 = NaN;
    if T <= h+2 || c0 <= 0, Q = NaN; pv = NaN; return; end
    for k = 1:h
        rk = sum(x(1+k:end).*x(1:end-k)) / c0;
        if k == 1, r1 = rk; end
        Q = Q + rk^2/(T-k);
    end
    Q  = T*(T+2)*Q;
    pv = 1 - chi2cdf(Q, h);
end

function cmap = redbluemap_local()
    n   = 64;
    top = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1)];
    bot = [ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)'];
    cmap = [top; bot];
end

function banner(txt)
    line = repmat('=', 1, 70);
    fprintf('\n%s\n %s\n%s\n', line, txt, line);
end

function out = tern(cond, a, b)
    if cond, out = a; else, out = b; end
end
