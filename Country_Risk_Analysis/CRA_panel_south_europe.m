%% ========================================================================
%  CRA_panel_south_europe.m
%  ------------------------------------------------------------------------
%  "How the Sovereign Spread Shapes the Snowball Effect:
%   Evidence from Southern Europe"
%
%  STRUCTURE
%  =========
%  PART A — BASELINE PANEL: Italy, Portugal, Spain (IPS, N=72, 1996-2019)
%    The three countries are structurally comparable: high public debt,
%    macro-fiscal imbalances, but transparent public accounts.  Greece is
%    intentionally excluded from the baseline because its pre-crisis
%    statistics were fabricated, making it a structurally different
%    observation rather than a "high-stress" version of the same data
%    generating process.
%
%  PART B — EXTENSION INCLUDING GREECE (N=96)
%    Adding Greece allows us to show how and why results change when an
%    extreme, structurally anomalous country is included.  Two specific
%    problems with including Greece in the baseline:
%      (i)  Threshold identification: with IPS alone k* is found in the
%           interior of the spread distribution (~2-4 pp).  Adding Greece
%           pushes k* to the 90th-percentile boundary (5.71 pp), signalling
%           weak identification — the "high regime" contains almost only
%           Greek observations.
%      (ii) Variance dominance: Greek spreads (max 21 pp) and snowball
%           values (max 22 % GDP, incl. 2012 PSI) dwarf the IPS sample and
%           risk making the pooled coefficients primarily reflect the Greek
%           crisis rather than a common structural relationship.
%
%  ECONOMETRIC CHOICES
%  ====================
%  1. Sample size: IPS gives 72 obs — sufficient to rely on large-sample
%     theory for panel FE, unlike the Italy-only version (~25 obs) where the
%     Sims-Stock-Watson (1990) asymptotic argument for mixed I(1)/I(0) OLS
%     was not credible.
%
%  2. Estimator: Within / Fixed Effects by country (country dummies).
%     FE absorb time-invariant structural level differences and identification
%     comes from within-country variation.
%
%  3. Standard errors: Driscoll-Kraay (1998) — robust to heteroskedasticity,
%     serial correlation, AND cross-sectional dependence.  Euro-area shocks
%     (2008, 2010-12 sovereign stress, "whatever it takes") hit all countries
%     simultaneously; country-by-country HAC alone would understate SE.
%
%  4. Stationarity:
%     - GDP growth (g): already a percentage change of real GDP (AMECO OVGD).
%       It is I(0) by construction — NO transformation needed.
%     - Inflation (pie): percentage change of consumption deflator (PCPH).
%       Also I(0) by construction.
%     - Snowball (sb), spread, fiscal stance (pb): flow/rate variables
%       treated as I(0); confirmed by ADF + KPSS + IPS panel test.
%     - Debt/GDP: the one potentially I(1) variable.  Enters the model only
%       as a lagged control (predetermined).  Robustness R2 replaces it with
%       its first difference.
%     The automatic "difference-if-ADF-fails" rule from the original
%     Italy-only script is abandoned: at T=25 the ADF has very low power
%     and produced fragile, sample-sensitive transformations.
%
%  5. Slope heterogeneity: pooling imposes common coefficients.  Checked via
%     country-by-country regressions + Pesaran-Smith mean-group (R4).
%
%  DATA:  panel_south_europe.csv  (built by build_panel_data.py)
%         Source: AMECO (European Commission).
%         Spread = ILN(country) - ILN(Germany), 10-year nominal rate.
%
%  No Econometrics Toolbox required.
%% ========================================================================

clear; clc; close all;

%% ===================== CONFIGURATION ===================================
cfg.file        = fullfile(fileparts(mfilename('fullpath')), 'panel_south_europe.csv');
cfg.baseline    = {'ITA','PRT','ESP'};   % Part A: baseline countries
cfg.full        = {'ITA','GRC','PRT','ESP'};  % Part B: full panel
cfg.sampleStart = 1996;
cfg.sampleEnd   = 2019;   % Pre-COVID as in the paper.
                          % Set to 2024 to extend; a COVID dummy is added.
cfg.breakYear   = 2010;
cfg.outDir      = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(cfg.outDir,'dir'); mkdir(cfg.outDir); end

%% ========================================================================
%% ====  PART A — BASELINE PANEL: Italy, Portugal, Spain  ================
%% ========================================================================
fprintf('%s\n', repmat('=',1,72));
fprintf('PART A — BASELINE PANEL: ITALY, PORTUGAL, SPAIN\n');
fprintf('Sample: %d-%d  |  Countries: %s\n', cfg.sampleStart, cfg.sampleEnd, ...
    strjoin(cfg.baseline,', '));
fprintf('%s\n\n', repmat('=',1,72));

P = loadPanel(cfg, cfg.baseline);
nObs = numel(P.sb);
nC   = numel(cfg.baseline);
fprintf('Complete observations: %d (%d countries x ~%d years)\n\n', ...
    nObs, nC, round(nObs/nC));

%% --- A1. Descriptive Statistics ----------------------------------------
fprintf('%s\nDESCRIPTIVE STATISTICS\n%s\n', repmat('=',1,72), repmat('=',1,72));
descA = computeDescriptives(P,[]);
fprintf('\n--- Pooled (ITA+PRT+ESP) ---\n'); disp(descA);
for c = 1:nC
    fprintf('--- %s ---\n', cfg.baseline{c}); disp(computeDescriptives(P,c));
end
writetable(descA, fullfile(cfg.outDir,'A_descriptives_IPS.xlsx'),'WriteRowNames',true);

%% --- A2. Unit Root Tests -----------------------------------------------
fprintf('\n%s\nUNIT ROOT TESTS\n%s\n', repmat('=',1,72), repmat('=',1,72));
urVars   = {'sb','spread','debt','pb','g','pie'};
urLabels = {'Snowball','Spread','Debt/GDP','FiscalStance','Growth','Inflation'};
urTable  = runPanelUnitRoot(P, urVars, urLabels, cfg.baseline);
disp(urTable);
writetable(urTable, fullfile(cfg.outDir,'A_unitroot_IPS.xlsx'));
fprintf(['Interpretation: flow/rate variables (sb, spread, growth, inflation,\n' ...
         'fiscal stance) are treated as I(0) — ADF and IPS confirm this.\n' ...
         'GDP growth and inflation are I(0) by construction (percentage\n' ...
         'changes, not levels). Debt/GDP does not reject the unit root and\n' ...
         'enters as a lagged level control; R2 uses first differences.\n\n']);

%% --- A3. Common design matrices ----------------------------------------
FE      = countryDummies(P.cid, nC);
feNames = strcat('FE_', cfg.baseline(2:end));
D2010    = double(P.year >= cfg.breakYear);
spr_post = P.spread .* D2010;
covidCols  = []; covidNames = {};
if cfg.sampleEnd > 2019
    covidCols  = double(P.year >= 2020 & P.year <= 2021);
    covidNames = {'D_COVID'};
end
nYears  = numel(unique(P.year));
hacLags = max(1, floor(4*(nYears/100)^(2/9)));
fprintf('Driscoll-Kraay bandwidth: %d lag(s)  (T = %d years)\n\n', hacLags, nYears);

%% --- A.M1: Pooled OLS (benchmark, no FE) -------------------------------
fprintf('\n%s\nA.M1 — POOLED OLS (benchmark, no FE)\n%s\n', ...
    repmat('=',1,60), repmat('=',1,60));
X1 = [ones(nObs,1), P.spread, P.b_lag, P.pb, P.g, P.pie, covidCols];
n1 = [{'const','spread','debt_lag','fiscal_stance','growth','inflation'}, covidNames];
res_A1 = estimateDK(P.sb, X1, n1, P.year, hacLags);
printResults(res_A1, 'A.M1: Pooled OLS');

%% --- A.M2: Baseline FE + Driscoll-Kraay SE ----------------------------
fprintf('\n%s\nA.M2 — BASELINE: COUNTRY FE + DRISCOLL-KRAAY SE\n%s\n', ...
    repmat('=',1,60), repmat('=',1,60));
X2 = [ones(nObs,1), P.spread, P.b_lag, P.pb, P.g, P.pie, covidCols, FE];
n2 = [{'const','spread','debt_lag','fiscal_stance','growth','inflation'}, ...
      covidNames, feNames];
res_A2 = estimateDK(P.sb, X2, n2, P.year, hacLags);
printResults(res_A2, 'A.M2: Baseline FE');
exportResults(res_A2, fullfile(cfg.outDir,'A_baseline_FE_DK.xlsx'));

%% --- A.M3: Post-2010 Structural Break ----------------------------------
fprintf('\n%s\nA.M3 — POST-%d STRUCTURAL BREAK\n%s\n', ...
    repmat('=',1,60), cfg.breakYear, repmat('=',1,60));
X3 = [ones(nObs,1), P.spread, spr_post, D2010, P.b_lag, P.pb, P.g, P.pie, ...
      covidCols, FE];
n3 = [{'const','spread','spread_x_D2010','D2010','debt_lag','fiscal_stance', ...
       'growth','inflation'}, covidNames, feNames];
res_A3 = estimateDK(P.sb, X3, n3, P.year, hacLags);
printResults(res_A3, 'A.M3: Post-2010 Break FE');
exportResults(res_A3, fullfile(cfg.outDir,'A_post2010_FE_DK.xlsx'));
fprintf('\nMarginal effect of spread on snowball:\n');
fprintf('  Pre-%d:  %+.4f\n',   cfg.breakYear, res_A3.beta(2));
fprintf('  Post-%d: %+.4f\n\n', cfg.breakYear, res_A3.beta(2)+res_A3.beta(3));

%% --- A.M4: Threshold Effect --------------------------------------------
fprintf('\n%s\nA.M4 — THRESHOLD EFFECT (common k*, grid search on SSR)\n%s\n', ...
    repmat('=',1,60), repmat('=',1,60));
kstar_A = findThreshold(P.sb, P.spread, [P.b_lag, P.pb, P.g, P.pie, covidCols, FE]);
fprintf('Optimal threshold k* = %.2f pp  (%d-th percentile of IPS spread)\n\n', ...
    kstar_A, round(100*mean(P.spread <= kstar_A)));
spr_high_A = P.spread .* (P.spread > kstar_A);
X4 = [ones(nObs,1), P.spread, spr_high_A, P.b_lag, P.pb, P.g, P.pie, covidCols, FE];
n4 = [{'const','spread','spread_high','debt_lag','fiscal_stance','growth', ...
       'inflation'}, covidNames, feNames];
res_A4 = estimateDK(P.sb, X4, n4, P.year, hacLags);
printResults(res_A4, 'A.M4: Threshold FE');
exportResults(res_A4, fullfile(cfg.outDir,'A_threshold_FE_DK.xlsx'));

%% --- A.M5: Threshold + Post-2010 (Full) --------------------------------
fprintf('\n%s\nA.M5 — THRESHOLD + POST-%d (FULL MODEL)\n%s\n', ...
    repmat('=',1,60), cfg.breakYear, repmat('=',1,60));
spr_high_post_A = spr_high_A .* D2010;
X5 = [ones(nObs,1), P.spread, spr_high_A, spr_post, spr_high_post_A, D2010, ...
      P.b_lag, P.pb, P.g, P.pie, covidCols, FE];
n5 = [{'const','spread','spread_high','spread_x_D2010','spread_high_x_D2010', ...
       'D2010','debt_lag','fiscal_stance','growth','inflation'}, covidNames, feNames];
res_A5 = estimateDK(P.sb, X5, n5, P.year, hacLags);
printResults(res_A5, 'A.M5: Full Model FE');
b1=res_A5.beta(2); b2=res_A5.beta(3); th1=res_A5.beta(4); th2=res_A5.beta(5);
fprintf('\nMarginal effects (d snowball / d spread) by regime:\n');
fprintf('  Pre-%d,  spread <= k*: %+.4f\n',  cfg.breakYear, b1);
fprintf('  Pre-%d,  spread >  k*: %+.4f\n',  cfg.breakYear, b1+b2);
fprintf('  Post-%d, spread <= k*: %+.4f\n',  cfg.breakYear, b1+th1);
fprintf('  Post-%d, spread >  k*: %+.4f\n\n',cfg.breakYear, b1+b2+th1+th2);

%% --- A.R1: No Spread ---------------------------------------------------
fprintf('\n%s\nROBUSTNESS CHECKS (Baseline: ITA+PRT+ESP)\n%s\n', ...
    repmat('=',1,60), repmat('=',1,60));
fprintf('\n--- R1: No spread (relevance test) ---\n');
XR1 = [ones(nObs,1), P.b_lag, P.pb, P.g, P.pie, covidCols, FE];
nR1 = [{'const','debt_lag','fiscal_stance','growth','inflation'}, covidNames, feNames];
res_AR1 = estimateDK(P.sb, XR1, nR1, P.year, hacLags);
printResults(res_AR1, 'A.R1: No Spread');

%% --- A.R2: Debt in first differences -----------------------------------
fprintf('\n--- R2: Debt/GDP in first differences ---\n');
XR2 = [ones(nObs,1), P.spread, spr_post, D2010, P.d_debt, P.pb, P.g, P.pie, ...
       covidCols, FE];
nR2 = [{'const','spread','spread_x_D2010','D2010','d_debt','fiscal_stance', ...
        'growth','inflation'}, covidNames, feNames];
res_AR2 = estimateDK(P.sb, XR2, nR2, P.year, hacLags);
printResults(res_AR2, 'A.R2: Delta-debt');

%% --- A.R3: Two-way FE (conservative lower bound) -----------------------
fprintf('\n--- R3: Two-way FE (year dummies) — conservative lower bound ---\n');
fprintf(['NOTE: With N=3, year dummies absorb the common time component of\n' ...
         'spreads. Coefficient reflects only relative cross-country variation.\n\n']);
YD  = yearDummies(P.year);
XR3 = [ones(nObs,1), P.spread, P.b_lag, P.pb, P.g, P.pie, FE, YD.cols];
nR3 = [{'const','spread','debt_lag','fiscal_stance','growth','inflation'}, ...
       feNames, YD.names];
res_AR3 = estimateDK(P.sb, XR3, nR3, P.year, hacLags);
fprintf('Spread (two-way FE):  beta = %+.4f  t_DK = %+.2f  p = %.3f\n', ...
    res_AR3.beta(2), res_AR3.t_DK(2), res_AR3.p_DK(2));

%% --- A.R4: Country-by-country + mean group -----------------------------
fprintf('\n--- R4: Country-by-country regressions (slope heterogeneity) ---\n');
mgTable_A = meanGroup(P, cfg.baseline, hacLags, covidCols);
disp(mgTable_A);
writetable(mgTable_A, fullfile(cfg.outDir,'A_meangroup_IPS.xlsx'),'WriteRowNames',true);

%% --- A: Model Comparison -----------------------------------------------
fprintf('\n%s\nA — MODEL COMPARISON (ITA+PRT+ESP)\n%s\n\n', ...
    repmat('=',1,60), repmat('=',1,60));
modelsA = {res_A1, res_A2, res_A3, res_A4, res_A5, res_AR1, res_AR2};
namesA  = {'Pooled','BaselineFE','Post2010','Threshold','Full','NoSpread','DeltaDebt'};
compA   = compareModels(modelsA, namesA);
disp(compA);
writetable(compA, fullfile(cfg.outDir,'A_model_comparison_IPS.xlsx'),'WriteRowNames',true);

%% --- A: Figures --------------------------------------------------------
fprintf('\n=== Part A figures ===\n');
plotPanelSeries(P, cfg.baseline, cfg.breakYear, 'Baseline Panel: Italy, Portugal, Spain');
plotScatterByCountry(P, cfg.baseline, res_A2, 'IPS Baseline');
plotThresholdFig(P, kstar_A, res_A4, 'Threshold Effect — IPS Baseline');
plotMarginalEffects(res_A5, kstar_A, cfg.breakYear, 'Marginal Effects — IPS Full Model');

%% ========================================================================
%% ====  PART B — EXTENSION: Adding Greece  ==============================
%% ========================================================================
fprintf('\n%s\n', repmat('=',1,72));
fprintf('PART B — EXTENSION: ADDING GREECE (ITA+GRC+PRT+ESP, N~96)\n');
fprintf(['Purpose: show how and why results change when Greece is included.\n' ...
         'Greece ran falsified fiscal statistics pre-2009, inflating its debt\n' ...
         'and deficit figures; its crisis is structurally different from the\n' ...
         'IPS countries. Nevertheless including it reveals how the same spread-\n' ...
         'snowball mechanism operates under extreme stress conditions.\n']);
fprintf('%s\n\n', repmat('=',1,72));

Q = loadPanel(cfg, cfg.full);
nObs_Q = numel(Q.sb);
nC_Q   = numel(cfg.full);
fprintf('Complete observations: %d (%d countries x ~%d years)\n\n', ...
    nObs_Q, nC_Q, round(nObs_Q/nC_Q));

%% --- B: Design matrices ------------------------------------------------
FE_Q      = countryDummies(Q.cid, nC_Q);
feNames_Q = strcat('FE_', cfg.full(2:end));
D2010_Q    = double(Q.year >= cfg.breakYear);
spr_post_Q = Q.spread .* D2010_Q;
covidQ  = []; covidNQ = {};
if cfg.sampleEnd > 2019
    covidQ  = double(Q.year >= 2020 & Q.year <= 2021);
    covidNQ = {'D_COVID'};
end
nYears_Q  = numel(unique(Q.year));
hacLags_Q = max(1, floor(4*(nYears_Q/100)^(2/9)));

%% --- B.M2: Baseline FE -------------------------------------------------
fprintf('\n%s\nB.M2 — BASELINE FE (4 countries)\n%s\n', ...
    repmat('=',1,60), repmat('=',1,60));
X2Q = [ones(nObs_Q,1), Q.spread, Q.b_lag, Q.pb, Q.g, Q.pie, covidQ, FE_Q];
n2Q = [{'const','spread','debt_lag','fiscal_stance','growth','inflation'}, ...
       covidNQ, feNames_Q];
res_B2 = estimateDK(Q.sb, X2Q, n2Q, Q.year, hacLags_Q);
printResults(res_B2, 'B.M2: Baseline FE (with Greece)');
exportResults(res_B2, fullfile(cfg.outDir,'B_baseline_FE_DK.xlsx'));

%% --- B.M3: Post-2010 Break ---------------------------------------------
fprintf('\n%s\nB.M3 — POST-%d BREAK (4 countries)\n%s\n', ...
    repmat('=',1,60), cfg.breakYear, repmat('=',1,60));
X3Q = [ones(nObs_Q,1), Q.spread, spr_post_Q, D2010_Q, Q.b_lag, Q.pb, Q.g, Q.pie, ...
       covidQ, FE_Q];
n3Q = [{'const','spread','spread_x_D2010','D2010','debt_lag','fiscal_stance', ...
        'growth','inflation'}, covidNQ, feNames_Q];
res_B3 = estimateDK(Q.sb, X3Q, n3Q, Q.year, hacLags_Q);
printResults(res_B3, 'B.M3: Post-2010 Break FE (with Greece)');
exportResults(res_B3, fullfile(cfg.outDir,'B_post2010_FE_DK.xlsx'));
fprintf('\nMarginal effect of spread on snowball:\n');
fprintf('  Pre-%d:  %+.4f\n',   cfg.breakYear, res_B3.beta(2));
fprintf('  Post-%d: %+.4f\n\n', cfg.breakYear, res_B3.beta(2)+res_B3.beta(3));

%% --- B.M4: Threshold ---------------------------------------------------
fprintf('\n%s\nB.M4 — THRESHOLD (4 countries) — identification check\n%s\n', ...
    repmat('=',1,60), repmat('=',1,60));
kstar_B = findThreshold(Q.sb, Q.spread, [Q.b_lag, Q.pb, Q.g, Q.pie, covidQ, FE_Q]);
fprintf('Optimal threshold k* = %.2f pp  (%d-th percentile of IGPS spread)\n', ...
    kstar_B, round(100*mean(Q.spread <= kstar_B)));
fprintf(['NOTE: Compare with k* = %.2f pp (IPS baseline, %d-th percentile).\n' ...
         'Adding Greece pushes k* toward the boundary of the search grid,\n' ...
         'signalling weak identification: the "high regime" is almost\n' ...
         'exclusively Greek observations (2010-2012).\n\n'], ...
    kstar_A, round(100*mean(P.spread <= kstar_A)));
spr_high_B = Q.spread .* (Q.spread > kstar_B);
X4Q = [ones(nObs_Q,1), Q.spread, spr_high_B, Q.b_lag, Q.pb, Q.g, Q.pie, covidQ, FE_Q];
n4Q = [{'const','spread','spread_high','debt_lag','fiscal_stance','growth', ...
        'inflation'}, covidNQ, feNames_Q];
res_B4 = estimateDK(Q.sb, X4Q, n4Q, Q.year, hacLags_Q);
printResults(res_B4, 'B.M4: Threshold FE (with Greece)');

%% --- B.M5: Full Model --------------------------------------------------
fprintf('\n%s\nB.M5 — FULL MODEL (4 countries)\n%s\n', ...
    repmat('=',1,60), cfg.breakYear, repmat('=',1,60));
spr_high_post_B = spr_high_B .* D2010_Q;
X5Q = [ones(nObs_Q,1), Q.spread, spr_high_B, spr_post_Q, spr_high_post_B, D2010_Q, ...
       Q.b_lag, Q.pb, Q.g, Q.pie, covidQ, FE_Q];
n5Q = [{'const','spread','spread_high','spread_x_D2010','spread_high_x_D2010', ...
        'D2010','debt_lag','fiscal_stance','growth','inflation'}, covidNQ, feNames_Q];
res_B5 = estimateDK(Q.sb, X5Q, n5Q, Q.year, hacLags_Q);
printResults(res_B5, 'B.M5: Full Model FE (with Greece)');

%% --- B: Comparison table A vs B ----------------------------------------
fprintf('\n%s\nCOMPARISON: BASELINE (IPS) vs EXTENSION (with Greece)\n%s\n\n', ...
    repmat('=',1,60), repmat('=',1,60));
compAB = compareParts(res_A2, res_B2, res_A3, res_B3);
disp(compAB);
writetable(compAB, fullfile(cfg.outDir,'AB_comparison_spread_coeff.xlsx'), ...
    'WriteRowNames', true);

%% --- B: Figures --------------------------------------------------------
fprintf('\n=== Part B figures ===\n');
plotScatterByCountry(Q, cfg.full, res_B2, 'Extension Panel: ITA+GRC+PRT+ESP');
plotThresholdComparison(P, Q, kstar_A, kstar_B, cfg);
plotCoeffComparison(res_A2, res_A3, res_B2, res_B3, cfg);
fprintf('\nAll figures generated.\n');

%% ========================================================================
%% ===================== HELPER FUNCTIONS =================================
%% ========================================================================

%% --- loadPanel ----------------------------------------------------------
function P = loadPanel(cfg, countries)
    T = readtable(cfg.file);
    T = T(T.year >= cfg.sampleStart-1 & T.year <= cfg.sampleEnd, :);
    [tf, cid] = ismember(T.country, countries);
    T = T(tf,:); cid = cid(tf);

    nT = height(T);
    b_lag = NaN(nT,1); d_debt = NaN(nT,1);
    for c = 1:numel(countries)
        idx = find(cid == c);
        [~,ord] = sort(T.year(idx)); idx = idx(ord);
        yrs = T.year(idx); dbt = T.debt(idx);
        for j = 2:numel(idx)
            if yrs(j) == yrs(j-1)+1
                b_lag(idx(j))  = dbt(j-1);
                d_debt(idx(j)) = dbt(j)-dbt(j-1);
            end
        end
    end
    T.b_lag = b_lag; T.d_debt = d_debt;

    keep = T.year >= cfg.sampleStart & ...
           all(~ismissing(T(:,{'sb','spread','b_lag','d_debt','pb','g','pie'})),2);
    T = T(keep,:); cid = cid(keep);

    P.country = T.country; P.cid = cid;       P.year   = T.year;
    P.sb      = T.sb;      P.spread = T.spread; P.debt  = T.debt;
    P.b_lag   = T.b_lag;   P.d_debt = T.d_debt;
    P.pb      = T.pb;      P.pbal   = T.pbal;
    P.g       = T.g;       P.pie    = T.pie;

    fprintf('Loaded panel [%s]:\n', strjoin(countries,'+'));
    for c = 1:numel(countries)
        m = cid == c;
        fprintf('  %s: %d obs (%d-%d)\n', countries{c}, sum(m), ...
            min(P.year(m)), max(P.year(m)));
    end
end

%% --- countryDummies -----------------------------------------------------
function D = countryDummies(cid, nC)
    D = zeros(numel(cid), nC-1);
    for c = 2:nC; D(:,c-1) = double(cid==c); end
end

%% --- yearDummies --------------------------------------------------------
function YD = yearDummies(year)
    yrs = unique(year);
    YD.cols = zeros(numel(year), numel(yrs)-1);
    YD.names = cell(1, numel(yrs)-1);
    for j = 2:numel(yrs)
        YD.cols(:,j-1) = double(year==yrs(j));
        YD.names{j-1}  = sprintf('Y%d',yrs(j));
    end
end

%% --- computeDescriptives -----------------------------------------------
function d = computeDescriptives(P, c)
    if isempty(c); m = true(size(P.sb)); else; m = P.cid==c; end
    V = [P.sb(m), P.spread(m), P.debt(m), P.pb(m), P.g(m), P.pie(m)];
    s = [mean(V,'omitnan'); std(V,'omitnan'); min(V,[],'omitnan'); ...
         max(V,[],'omitnan'); median(V,'omitnan'); sum(~isnan(V))];
    d = array2table(s, 'VariableNames', ...
        {'Snowball','Spread','Debt_GDP','FiscalStance','Growth','Inflation'}, ...
        'RowNames', {'Mean','Std','Min','Max','Median','N'});
end

%% --- runPanelUnitRoot ---------------------------------------------------
function urTable = runPanelUnitRoot(P, vars, labels, countries)
    nV = numel(vars); nC = numel(countries); rows = {};
    for v = 1:nV
        tstats = NaN(nC,1);
        for c = 1:nC
            y = P.(vars{v})(P.cid==c);
            [tstats(c), lagc, cv5c] = adf_test(y);
            [~, kstat, kcv5] = kpss_test(y, false, max(1,floor(4*(numel(y)/100)^(2/9))));
            rows(end+1,:) = {labels{v}, countries{c}, tstats(c), lagc, cv5c, ...
                tstats(c)<cv5c, kstat, kcv5, kstat>kcv5}; %#ok<AGROW>
        end
        tbar = mean(tstats); cvIPS = -2.15;
        rows(end+1,:) = {labels{v},'IPS t-bar',tbar,NaN,cvIPS, ...
            tbar<cvIPS,NaN,NaN,NaN}; %#ok<AGROW>
    end
    urTable = cell2table(rows, 'VariableNames', ...
        {'Variable','Country','ADF_t','Lags','cv5pct','ADF_rejectUnitRoot', ...
         'KPSS_stat','KPSS_cv5','KPSS_rejectStationarity'});
end

%% --- adf_test -----------------------------------------------------------
function [tstat, bestLag, cv5] = adf_test(y)
    y = y(:); T0 = numel(y);
    maxLag = max(0, min(4,floor((T0-10)/3)));
    bestAIC = Inf; bestLag = 0; tstat = NaN;
    for p = 0:maxLag
        dy = diff(y); Teff = T0-1-p;
        Y  = dy(p+1:end); X = [ones(Teff,1), y(p+1:T0-1)];
        for j = 1:p; X = [X, dy(p+1-j:end-j)]; end %#ok<AGROW>
        b = X\Y; e = Y-X*b; k = size(X,2);
        aic = log(e'*e/Teff) + 2*k/Teff;
        if aic < bestAIC
            bestAIC = aic; bestLag = p;
            sig2 = (e'*e)/(Teff-k);
            Vb   = sig2*inv(X'*X); %#ok<MINV>
            tstat = b(2)/sqrt(Vb(2,2));
        end
    end
    Tef = T0-1-bestLag;
    cv5 = -2.8621 - 2.738/Tef - 8.36/Tef^2 - 4.85/Tef^3;
end

%% --- kpss_test ----------------------------------------------------------
function [h, stat, cv5] = kpss_test(y, trend, lags)
    y = y(:); T = numel(y);
    if trend; X=[ones(T,1),(1:T)']; cv5=0.146; else; X=ones(T,1); cv5=0.463; end
    e = y-X*(X\y); S = cumsum(e); s2 = (e'*e)/T;
    for j = 1:lags; w=1-j/(lags+1); s2=s2+2*w*((e(j+1:end)'*e(1:end-j))/T); end
    s2 = max(s2,1e-10); stat = (S'*S)/(T^2*s2); h = double(stat>cv5);
end

%% --- estimateDK ---------------------------------------------------------
function results = estimateDK(y, X, names, year, q)
    [T, k] = size(X);
    beta  = X\y; resid = y-X*beta; df = T-k;
    XXinv = inv(X'*X); %#ok<MINV>
    s2    = (resid'*resid)/df;
    se_OLS = sqrt(diag(s2*XXinv));

    yrs = unique(year); nY = numel(yrs);
    H = zeros(nY,k);
    for t = 1:nY
        m = (year==yrs(t));
        H(t,:) = sum(X(m,:).*resid(m),1);
    end
    S = H'*H;
    for j = 1:q
        w = 1-j/(q+1); G = H(j+1:end,:)'*H(1:end-j,:);
        S = S + w*(G+G');
    end
    S = S*(T/(T-k));
    V_DK  = XXinv*S*XXinv;
    se_DK = sqrt(max(diag(V_DK),0));

    t_OLS = beta./se_OLS; p_OLS = 2*tcdf_(-abs(t_OLS),df);
    t_DK  = beta./se_DK;  p_DK  = 2*tcdf_(-abs(t_DK), nY-1);
    SSR = sum(resid.^2); TSS = sum((y-mean(y)).^2);

    results.beta   = beta;    results.se_OLS = se_OLS; results.se_DK = se_DK;
    results.t_OLS  = t_OLS;   results.t_DK  = t_DK;
    results.p_OLS  = p_OLS;   results.p_DK  = p_DK;
    results.resid  = resid;   results.yhat  = X*beta;
    results.R2     = 1-SSR/TSS;
    results.adjR2  = 1-(1-results.R2)*(T-1)/df;
    results.AIC    = T*log(SSR/T)+2*k;
    results.BIC    = T*log(SSR/T)+k*log(T);
    results.DW     = sum(diff(resid).^2)/SSR;
    results.nObs   = T; results.nParams = k; results.names = names;
end

%% --- tcdf_ --------------------------------------------------------------
function p = tcdf_(t, v)
    p = 0.5*betainc(v./(v+t.^2), v/2, 0.5);
    p(t>0) = 1-p(t>0);
end

%% --- printResults -------------------------------------------------------
function printResults(r, name)
    fprintf('\n--- %s ---\n\n', name);
    tbl = table(r.beta, r.se_OLS, r.t_OLS, r.p_OLS, r.se_DK, r.t_DK, r.p_DK, ...
        'VariableNames',{'Coef','SE_OLS','t_OLS','p_OLS','SE_DK','t_DK','p_DK'}, ...
        'RowNames', r.names);
    disp(tbl);
    fprintf('R2=%.4f | AdjR2=%.4f | AIC=%.1f | BIC=%.1f | N=%d\n', ...
        r.R2, r.adjR2, r.AIC, r.BIC, r.nObs);
end

%% --- exportResults ------------------------------------------------------
function exportResults(r, filename)
    tbl = table(r.beta, r.se_OLS, r.t_OLS, r.p_OLS, r.se_DK, r.t_DK, r.p_DK, ...
        'VariableNames',{'Estimate','SE_OLS','t_OLS','p_OLS','SE_DK','t_DK','p_DK'}, ...
        'RowNames', r.names);
    writetable(tbl, filename, 'WriteRowNames', true);
    fprintf('Exported: %s\n', filename);
end

%% --- findThreshold ------------------------------------------------------
function kstar = findThreshold(sb, spr, Z)
    grid = linspace(prctile_(spr,10), prctile_(spr,90), 60);
    n = numel(sb); RSS = zeros(numel(grid),1);
    for i = 1:numel(grid)
        sh = spr.*(spr>grid(i));
        Xg = [ones(n,1), spr, sh, Z];
        b  = Xg\sb; RSS(i) = sum((sb-Xg*b).^2);
    end
    [~,ix] = min(RSS); kstar = grid(ix);
end

%% --- prctile_ -----------------------------------------------------------
function q = prctile_(x, p)
    x = sort(x(:)); n = numel(x);
    r = p/100*(n-1)+1; lo = floor(r); hi = ceil(r);
    q = x(lo)+(r-lo)*(x(hi)-x(lo));
end

%% --- meanGroup ----------------------------------------------------------
function mgTable = meanGroup(P, countries, q, covidCols)
    nC = numel(countries); out = NaN(nC+1,4);
    for c = 1:nC
        m = P.cid==c; n = sum(m);
        Xc = [ones(n,1), P.spread(m), P.b_lag(m), P.pb(m), P.g(m), P.pie(m)];
        if ~isempty(covidCols); Xc = [Xc, covidCols(m)]; end %#ok<AGROW>
        r = estimateDK(P.sb(m), Xc, {}, P.year(m), q);
        out(c,:) = [r.beta(2), r.se_DK(2), r.t_DK(2), r.p_DK(2)];
    end
    bMG = mean(out(1:nC,1)); seMG = std(out(1:nC,1))/sqrt(nC);
    out(nC+1,:) = [bMG, seMG, bMG/seMG, 2*tcdf_(-abs(bMG/seMG),nC-1)];
    mgTable = array2table(out, 'VariableNames',{'beta_spread','SE','t_stat','p_value'}, ...
        'RowNames',[countries,{'MeanGroup'}]);
end

%% --- compareModels ------------------------------------------------------
function compTable = compareModels(models, names)
    nM = numel(models); s = zeros(nM,5);
    for i = 1:nM
        s(i,:) = [models{i}.R2, models{i}.adjR2, models{i}.AIC, ...
                  models{i}.BIC, models{i}.nParams];
    end
    compTable = array2table(s, 'VariableNames',{'R2','AdjR2','AIC','BIC','k'}, ...
        'RowNames',names);
end

%% --- compareParts -------------------------------------------------------
function tbl = compareParts(rA2, rB2, rA3, rB3)
    % Summary table: spread coefficient across key models, IPS vs IGPS.
    rows = {'Baseline FE (spread)', 'Post-2010 FE (spread pre)', ...
            'Post-2010 FE (spread post)'};
    beta_IPS  = [rA2.beta(2);  rA3.beta(2); rA3.beta(2)+rA3.beta(3)];
    se_IPS    = [rA2.se_DK(2); rA3.se_DK(2); NaN];
    tDK_IPS   = [rA2.t_DK(2);  rA3.t_DK(2);  NaN];
    beta_IGPS = [rB2.beta(2);  rB3.beta(2); rB3.beta(2)+rB3.beta(3)];
    se_IGPS   = [rB2.se_DK(2); rB3.se_DK(2); NaN];
    tDK_IGPS  = [rB2.t_DK(2);  rB3.t_DK(2);  NaN];
    tbl = array2table([beta_IPS, se_IPS, tDK_IPS, beta_IGPS, se_IGPS, tDK_IGPS], ...
        'VariableNames', {'beta_IPS','SE_IPS','t_IPS', ...
                          'beta_IGPS','SE_IGPS','t_IGPS'}, ...
        'RowNames', rows);
end

%% ===================== FIGURES ==========================================

%% --- plotPanelSeries ----------------------------------------------------
function plotPanelSeries(P, countries, breakYear, suptitle)
    nC = numel(countries);
    figure('Name',suptitle,'Position',[60 60 1100 620],'Color','w');
    tiledlayout(2,nC,'TileSpacing','compact','Padding','compact');
    for c = 1:nC
        m = P.cid==c;
        nexttile(c);
        plot(P.year(m), P.sb(m),'-o','LineWidth',1.4,'MarkerSize',3);
        hold on; xline(breakYear,'--r'); yline(0,'k:');
        title(sprintf('%s — Snowball (%% GDP)',countries{c})); grid on;
        nexttile(nC+c);
        plot(P.year(m), P.spread(m),'-s','LineWidth',1.4,'MarkerSize',3, ...
            'Color',[0.85 0.33 0.10]);
        hold on; xline(breakYear,'--r'); yline(0,'k:');
        title(sprintf('%s — Spread vs Bund (pp)',countries{c}));
        grid on; xlabel('Year');
    end
    sgtitle(suptitle,'FontWeight','bold','FontSize',12);
end

%% --- plotScatterByCountry -----------------------------------------------
function plotScatterByCountry(P, countries, res, ttl)
    figure('Name',ttl,'Position',[100 100 720 500],'Color','w');
    cols = lines(numel(countries));
    for c = 1:numel(countries)
        m = P.cid==c;
        scatter(P.spread(m), P.sb(m), 46, cols(c,:),'filled','MarkerFaceAlpha',0.65);
        hold on;
    end
    xr = linspace(min(P.spread), max(P.spread), 50);
    plot(xr, res.beta(1)+res.beta(2)*xr,'k-','LineWidth',2);
    legend([cellfun(@(s){s},countries),{'FE pooled slope'}],'Location','northwest');
    xlabel('Spread vs Bund (pp)'); ylabel('Snowball Effect (% GDP)');
    title(ttl); grid on; box on;
end

%% --- plotThresholdFig ---------------------------------------------------
function plotThresholdFig(P, kstar, res, ttl)
    figure('Name',ttl,'Position',[140 140 720 500],'Color','w');
    lo = P.spread<=kstar; hi = ~lo;
    scatter(P.spread(lo),P.sb(lo),46,[0.2 0.6 0.3],'filled','MarkerFaceAlpha',0.6);
    hold on;
    scatter(P.spread(hi),P.sb(hi),46,[0.8 0.2 0.2],'filled','MarkerFaceAlpha',0.6);
    b1=res.beta(2); b2=res.beta(3); a0=res.beta(1);
    xl = linspace(min(P.spread),kstar,40);
    xh = linspace(kstar,max(P.spread),40);
    plot(xl, a0+b1*xl,       '-','Color',[0.2 0.6 0.3],'LineWidth',2);
    plot(xh, a0+(b1+b2)*xh,  '-','Color',[0.8 0.2 0.2],'LineWidth',2);
    xline(kstar,'--k',sprintf('k* = %.2f pp',kstar),'LineWidth',1.5);
    xlabel('Spread (pp)'); ylabel('Snowball Effect (% GDP)');
    title(ttl);
    legend({'Low-spread regime','High-spread regime'},'Location','northwest');
    grid on; box on;
end

%% --- plotMarginalEffects ------------------------------------------------
function plotMarginalEffects(res, kstar, breakYear, ttl)
    b1=res.beta(2); b2=res.beta(3); th1=res.beta(4); th2=res.beta(5);
    eff = [b1, b1+b2, b1+th1, b1+b2+th1+th2];
    lbl = {sprintf('Pre-%d\nspr<=k*',breakYear), ...
           sprintf('Pre-%d\nspr>k*', breakYear), ...
           sprintf('Post-%d\nspr<=k*',breakYear), ...
           sprintf('Post-%d\nspr>k*', breakYear)};
    figure('Name',ttl,'Position',[180 180 680 420],'Color','w');
    bar(eff,'FaceColor',[0.3 0.5 0.7]);
    set(gca,'XTickLabel',lbl,'XTickLabelRotation',0,'FontSize',10);
    ylabel('d Snowball / d Spread'); grid on; box on;
    title(sprintf('%s\n(k* = %.2f pp)',ttl,kstar),'FontSize',10);
    for i=1:4
        text(i,eff(i)+0.025*sign(eff(i)),sprintf('%.3f',eff(i)), ...
            'HorizontalAlignment','center','FontWeight','bold','FontSize',10);
    end
end

%% --- plotThresholdComparison --------------------------------------------
function plotThresholdComparison(P, Q, kstar_A, kstar_B, cfg)
    figure('Name','Threshold: IPS vs IGPS','Position',[160 160 1000 440],'Color','w');
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    datasets = {P, Q};
    kstars   = {kstar_A, kstar_B};
    subtitles = {sprintf('IPS baseline (k*=%.2f pp, p%d)', ...
                    kstar_A, round(100*mean(P.spread<=kstar_A))), ...
                 sprintf('IGPS with Greece (k*=%.2f pp, p%d)', ...
                    kstar_B, round(100*mean(Q.spread<=kstar_B)))};
    for s = 1:2
        D = datasets{s}; k = kstars{s};
        nexttile;
        lo = D.spread<=k; hi = ~lo;
        scatter(D.spread(lo),D.sb(lo),36,[0.2 0.6 0.3],'filled','MarkerFaceAlpha',0.55);
        hold on;
        scatter(D.spread(hi),D.sb(hi),36,[0.8 0.2 0.2],'filled','MarkerFaceAlpha',0.55);
        xline(k,'--k',sprintf('k*=%.2f',k),'LineWidth',1.5);
        xlabel('Spread (pp)'); ylabel('Snowball (% GDP)');
        title(subtitles{s},'FontSize',10); grid on; box on;
        legend({'Low regime','High regime'},'Location','northwest');
    end
    sgtitle('Threshold Identification: Excluding vs Including Greece', ...
        'FontWeight','bold','FontSize',12);
end

%% --- plotCoeffComparison ------------------------------------------------
function plotCoeffComparison(rA2, rA3, rB2, rB3, cfg)
    % Coefficient plot: spread beta + 95% DK CI across parts and specs.
    betas = [rA2.beta(2), rA3.beta(2), rA3.beta(2)+rA3.beta(3), ...
             rB2.beta(2), rB3.beta(2), rB3.beta(2)+rB3.beta(3)];
    ses   = [rA2.se_DK(2), rA3.se_DK(2), sqrt(rA3.se_DK(2)^2+rA3.se_DK(3)^2), ...
             rB2.se_DK(2), rB3.se_DK(2), sqrt(rB3.se_DK(2)^2+rB3.se_DK(3)^2)];
    labs  = {'IPS Baseline','IPS Pre-2010','IPS Post-2010', ...
             'IGPS Baseline','IGPS Pre-2010','IGPS Post-2010'};
    cols  = [repmat([0.2 0.5 0.8],3,1); repmat([0.8 0.3 0.1],3,1)];

    figure('Name','Spread Coefficient Comparison','Position',[200 200 760 440],'Color','w');
    hold on;
    for i = 1:6
        errorbar(i, betas(i), 1.96*ses(i), 1.96*ses(i), 'o', ...
            'Color',cols(i,:),'MarkerFaceColor',cols(i,:), ...
            'MarkerSize',8,'LineWidth',1.8,'CapSize',9);
    end
    yline(0,'--k','LineWidth',1);
    xline(3.5,'--','Color',[0.5 0.5 0.5],'LineWidth',1);
    set(gca,'XTick',1:6,'XTickLabel',labs,'XTickLabelRotation',20,'FontSize',10);
    ylabel('Spread coefficient (DK 95% CI)');
    title('Spread Effect on Snowball: IPS Baseline vs Extension with Greece');
    text(2, max(betas+1.96*ses)*0.92, 'IPS (ITA+PRT+ESP)', ...
        'Color',[0.2 0.5 0.8],'FontWeight','bold','HorizontalAlignment','center');
    text(5, max(betas+1.96*ses)*0.92, 'IGPS (incl. Greece)', ...
        'Color',[0.8 0.3 0.1],'FontWeight','bold','HorizontalAlignment','center');
    grid on; box on;
end
