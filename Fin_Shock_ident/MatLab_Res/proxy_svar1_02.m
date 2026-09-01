% =========================================================================
%  proxy_svar1_02.m  —  STAGE A: AGNOSTIC EXTERNAL INSTRUMENT
%                       (daily kurtosis shocks -> monthly)  +  uncertainty
%                       validation at MONTHLY frequency.
%
%  ORDER OF OPERATIONS (this file does nothing at daily frequency after STEP 1)
%    1. load the daily identified shocks and apply the sign convention
%    2. AGGREGATE them to monthly (signed mean = instrument; ABS/SQ/POS/NEG
%       = validation-only aggregates)
%    3. read the uncertainty indices, which are PUBLISHED monthly — no
%       aggregation is applied to them, only calendar matching
%    3b. build the UNFORECASTABLE component of each index (AR residual)
%    4. correlate, on matched calendar months
%
%  DIVISION OF LABOUR (kept separate on purpose)
%    proxy_svar1.m            : (legacy) aggregation + uncertainty correlation,
%                               with shocks PRE-LABELLED (credit/market/…).
%    THIS FILE (Stage A)      : the SAME idea but AGNOSTIC — the daily-identified
%                               structural shocks are aggregated to monthly as
%                               EXTERNAL INSTRUMENTS (signed monthly mean), with
%                               NO economic label and NO real proxy (EBP etc.),
%                               and their monthly correlation with the
%                               uncertainty indices (US-EPU, GEPU, GPR) is
%                               reported.  It does NOT touch the macro VAR.
%    proxy_svar_02_cholesky.m : STAGE B — reads the instrument written here and
%                               estimates the monthly MACRO effects (GLP-BVAR),
%                               using the shock as an INTERNAL instrument
%                               (Cholesky, ordered first).
%
%  WHY "external instrument".  The signed monthly mean (mean_k = sum/#trading-days)
%  is the aggregate that inherits the defining properties of a structural shock —
%  zero-mean and ~serially-uncorrelated — so it is a valid external instrument
%  in the Mertens-Ravn sense.  Averaging (not summing) is the correct HF->LF
%  filter for a linear VAR: it recovers the low-frequency structural shock and
%  avoids the temporal-aggregation bias of the sum (Bridge Proxy-SVAR; Foroni-
%  Marcellino 2014).  ABS/SQ are non-negative INTENSITY measures and POS/NEG are
%  one-sided accumulations: all four are for VALIDATION only (level-on-level),
%  NOT instruments — they are not zero-mean and, carrying the common stochastic
%  volatility, they load on the OTHER structural shocks too, which violates the
%  Mertens-Ravn exogeneity condition.
%
%  INPUT:  SRC .mat  (V_ref, Vt_draws, B_med, lam_med, dates_vec, M)
%          US_Policy_Uncertainty_Data.xlsx / Global_Policy_Uncertainty_Data.xlsx
%          data_gpr_export.xls
%  OUTPUT: shocks_monthly_<SRC>.mat            (external instrument -> Stage B)
%          corr_extIV_uncertainty_<SRC>.mat    (correlation table, 9 measures)
%          fig_p02A_corr_<SRC>.png             (main heatmap, level~abs_k)
%          fig_p02A_corrALL_<SRC>.png          (one heatmap per measure)
%          fig_p02A_instr_<SRC>.png            (the instrument mean_k itself +
%                                               its pos_k/neg_k decomposition)
%
%  REFERENCES
%    Mertens & Ravn (2013, AER)               — external instrument / proxy-SVAR
%    Bridge Proxy-SVAR (HF->LF aggregation)   — averaging = correct LF filter
%    Foroni & Marcellino (2014)               — temporal aggregation bias
%    Baker, Bloom & Davis (2016)              — EPU indices
%    Caldara & Iacoviello (2022)              — GPR indices
% =========================================================================

clear; clc; close all;

% ===================== CONFIG (set by hand) ==============================
SRC          = 'mori_3var.mat';
[~, SRC_BASE]= fileparts(SRC);
% ---- SHOCK SET ----------------------------------------------------------
%  ALL the shocks identified by the daily Mori step travel to the monthly
%  stage, in their natural order v1..vM.  There is NO selection step: the
%  identification returns the M columns of Q in one go, and every one of them
%  is carried forward.  Economic meaning is assigned ex post (IRF signature,
%  own excess kurtosis, time series of the shock), never upstream.
CSV_FALLBACK = 'financial_data_aligned_corrected.csv';
% ---- SIGN CONVENTION ----------------------------------------------------
%  The columns of B are identified only UP TO SIGN: replacing (B, eps) by
%  (B*S, S*eps) with S = diag(s_1..s_M), s_i in {-1,+1}, leaves B*eps and hence
%  the whole reduced form unchanged (S*S = I).  Each s_i is therefore a FREE and
%  INDEPENDENT normalisation — flipping one shock does not require flipping the
%  others.  It is a labelling choice, not an identifying restriction.
%
%  Upstream (Mori_3var.m, Lanne-Meitz-Saikkonen 2017) each shock is already
%  anchored so that its daily FEVD-dominant variable loads POSITIVELY:
%      v1 -> CSPREAD (+)      v2 -> SP500 (+)      v3 -> VIX (+)
%  That makes v1 and v3 adverse (credit stress up, volatility up) but v2
%  EXPANSIONARY (equity up).  SIGN_CONV re-anchors the shocks so that all three
%  are ADVERSE, which puts the three IRF panels on a comparable footing: every
%  panel should then show activity falling, and magnitudes can be read off
%  side by side instead of mentally inverting v2.
%
%  APPLIED ONCE, HERE, TO V_ref BEFORE ANY AGGREGATION, so MEAN/SUM/POS/NEG are
%  all consistent by construction (note POS and NEG SWAP under a flip; applying
%  the sign to the daily series makes that automatic).  It is then SAVED into
%  shocks_monthly_<SRC>.mat and READ BACK by Stage B, so the two stages cannot
%  drift apart.  ABS and SQ are even functions and are unaffected.
SIGN_CONV    = [+1, -1, +1];     % one entry per shock; [] or all +1 = upstream anchor
% ---- UNFORECASTABLE-COMPONENT FILTER ------------------------------------
AR_Q         = 6;                % monthly lags in the AR filter of STEP 3b
% =========================================================================

fprintf('\n%s\n proxy_svar1_02 — STAGE A: agnostic external instrument + uncertainty\n SRC = %s\n%s\n', ...
        repmat('=',1,70), SRC, repmat('=',1,70));

%% =========================================================================
%  STEP 1 — LOAD DAILY SHOCKS  (all shocks, no selection)
% =========================================================================
S = load(SRC);
V_ref     = S.V_ref;  [T_eff, M] = size(V_ref);
lam_med   = S.lam_med(:);
B_med     = S.B_med;
VAR_NAMES = cellstr(S.VAR_NAMES(:));
if isfield(S,'dates_vec') && isdatetime(S.dates_vec) && numel(S.dates_vec)==T_eff
    dates_daily = S.dates_vec(:);
else
    Traw = readtable(CSV_FALLBACK);
    d_all = datetime(Traw.Date,'InputFormat','yyyy-MM-dd');
    dates_daily = d_all(end-T_eff+1:end);
end

shock_cols = 1:M;                 % all shocks, natural order — no selection
N_shocks   = M;

% ---- SIGN CONVENTION: applied ONCE, to the daily series, before aggregation --
if isempty(SIGN_CONV), SIGN_CONV = ones(1,M); end
SIGN_CONV = SIGN_CONV(:)';
if numel(SIGN_CONV) ~= M
    error('SIGN_CONV has %d entries but there are %d shocks.', numel(SIGN_CONV), M);
end
if ~all(ismember(SIGN_CONV,[-1 1]))
    error('SIGN_CONV entries must be -1 or +1.');
end
V_ref = V_ref .* SIGN_CONV;        % daily shocks
B_med = B_med .* SIGN_CONV;        % impact loadings (keeps B*eps invariant)

% interpretation-only annotation (daily FEVD-dominant variable), NOT identifying
std_m = sqrt(sum(B_med.^2,2)); Bstd = abs(B_med)./max(std_m,1e-12);
dom_daily = cell(1,N_shocks); shock_tag = cell(1,N_shocks);
dom_sign  = zeros(1,N_shocks);     % sign of the loading AFTER SIGN_CONV
for ki=1:N_shocks
    [~,im]=max(Bstd(:,shock_cols(ki))); dom_daily{ki}=VAR_NAMES{im};
    dom_sign(ki) = sign(B_med(im,shock_cols(ki)));
    shock_tag{ki}=sprintf('v%d', shock_cols(ki));
end
fprintf('  M=%d  T_eff=%d  | all %d shocks carried to monthly, natural order\n', M, T_eff, N_shocks);
fprintf('  SIGN_CONV = [%s]  (applied to the DAILY shocks, before aggregation)\n', ...
        strjoin(arrayfun(@(s)sprintf('%+d',s),SIGN_CONV,'UniformOutput',false),' '));
for ki=1:N_shocks
    fprintf('    %-4s  kappa=%6.2f   [FEVD-dominant: %-8s loads %s on impact  <- sign anchor + ex-post reading]\n', ...
            shock_tag{ki}, lam_med(shock_cols(ki)), dom_daily{ki}, ...
            tern(dom_sign(ki)>0,'POSITIVELY','NEGATIVELY'));
end

%% =========================================================================
%  STEP 2 — MONTHLY AGGREGATION
%    external instrument = MEAN (signed);  intensity = ABS, SQ;  asym = POS,NEG
%
%  Everything from here on is MONTHLY.  This is the only place where the daily
%  series is touched, and it is touched once.
% =========================================================================
fprintf('\n%s\n STEP 2 — MONTHLY AGGREGATION (instrument = signed mean)\n%s\n', repmat('=',1,70), repmat('=',1,70));
ym_daily = year(dates_daily)*100 + month(dates_daily);
[ym_u,~,g] = unique(ym_daily,'sorted');  Tm = numel(ym_u);
dates_monthly = datetime(floor(ym_u/100), mod(ym_u,100), 1);
nday_month = accumarray(g,1);
% NB: MEAN/SUM/etc. are built over ALL shocks (columns 1..M) so downstream files
% can index any shock by its original column; instruments carried = shock_cols.
MEAN=zeros(Tm,M); SUM=zeros(Tm,M); POS=zeros(Tm,M); NEG=zeros(Tm,M); ABS=zeros(Tm,M); SQ=zeros(Tm,M);
for k=1:M
    v=V_ref(:,k);
    SUM(:,k)=accumarray(g,v); MEAN(:,k)=SUM(:,k)./nday_month;
    POS(:,k)=accumarray(g,max(v,0)); NEG(:,k)=accumarray(g,min(v,0));
    ABS(:,k)=accumarray(g,abs(v));   SQ(:,k)=accumarray(g,v.^2);
end
ym_monthly = ym_u;
save(sprintf('shocks_monthly_%s.mat',SRC_BASE), ...
     'dates_monthly','ym_monthly','nday_month','M','VAR_NAMES','shock_cols','shock_tag', ...
     'dom_daily','dom_sign','SIGN_CONV','MEAN','SUM','POS','NEG','ABS','SQ','SRC','-v7.3');
fprintf('  %d daily obs -> %d months.  Saved shocks_monthly_%s.mat (MEAN = external instrument)\n', ...
        T_eff, Tm, SRC_BASE);
fprintf('  Monthly span: %s — %s\n', string(dates_monthly(1),'yyyy-MM'), string(dates_monthly(end),'yyyy-MM'));
% --- trading-day diagnostic ---------------------------------------------
%  Var(mean_m) = 1/D_m, so an unbalanced D_m makes the instrument conditionally
%  heteroskedastic BY CONSTRUCTION.  The first month is additionally truncated
%  by the p lags consumed in the daily VAR.  Report it rather than hide it.
[Dmin,iDmin]=min(nday_month); [Dmax,iDmax]=max(nday_month);
fprintf('  Trading days/month: mean=%.1f  min=%d (%s)  max=%d (%s)  -> sd ratio %.2f\n', ...
        mean(nday_month), Dmin, string(dates_monthly(iDmin),'yyyy-MM'), ...
        Dmax, string(dates_monthly(iDmax),'yyyy-MM'), sqrt(Dmax/Dmin));
if Dmin < 15
    fprintf('  [!] %s has only %d trading days (partial month): its mean_k has ~%.0f%% larger sd.\n', ...
            string(dates_monthly(iDmin),'yyyy-MM'), Dmin, 100*(sqrt(mean(nday_month)/Dmin)-1));
end

%% =========================================================================
%  STEP 3 — UNCERTAINTY INDICES (US-EPU, GEPU, GPR)
%    Published MONTHLY.  No aggregation is applied to them: only the calendar
%    matching of STEP 4.
% =========================================================================
fprintf('\n%s\n STEP 3 — UNCERTAINTY INDICES\n%s\n', repmat('=',1,70), repmat('=',1,70));
IDX = struct('name',{},'ym',{},'lvl',{});
Tu = readtab('US_Policy_Uncertainty_Data.xlsx','Main News Index');
yU=getcol(Tu,'Year'); mU=getcol(Tu,'Month'); vU=getcol(Tu,'News_Based');
okU=isfinite(yU)&isfinite(mU)&mU>=1&mU<=12&isfinite(vU);
IDX(end+1)=mk_idx('US_EPU', yU(okU)*100+mU(okU), vU(okU));
Tg=readtab('Global_Policy_Uncertainty_Data.xlsx','Sheet1');
yG=getcol(Tg,'Year'); mG=getcol(Tg,'Month'); vGc=getcol(Tg,'GEPU_current'); vGp=getcol(Tg,'GEPU_ppp');
okG=isfinite(yG)&isfinite(mG)&mG>=1&mG<=12;
IDX(end+1)=mk_idx('GEPU_current', yG(okG&isfinite(vGc))*100+mG(okG&isfinite(vGc)), vGc(okG&isfinite(vGc)));
IDX(end+1)=mk_idx('GEPU_ppp',     yG(okG&isfinite(vGp))*100+mG(okG&isfinite(vGp)), vGp(okG&isfinite(vGp)));
Tr=readtab('data_gpr_export.xls','Sheet1');
dR=getcol(Tr,'month'); if ~isdatetime(dR), dR=datetime(string(dR)); end
ymR=year(dR)*100+month(dR); vRg=getcol(Tr,'GPR'); vRu=getcol(Tr,'GPRC_USA');
IDX(end+1)=mk_idx('GPR',      ymR(isfinite(ymR)&isfinite(vRg)), vRg(isfinite(ymR)&isfinite(vRg)));
IDX(end+1)=mk_idx('GPRC_USA', ymR(isfinite(ymR)&isfinite(vRu)), vRu(isfinite(ymR)&isfinite(vRu)));
for j=1:numel(IDX), fprintf('    %-13s : %4d obs  %d–%d\n', IDX(j).name, numel(IDX(j).ym), min(IDX(j).ym), max(IDX(j).ym)); end

%% =========================================================================
%  STEP 3b — UNFORECASTABLE COMPONENT OF EACH INDEX
%
%  A structural shock is unforecastable at t-1 BY CONSTRUCTION.  It therefore
%  has no reason to correlate with the PREDICTABLE part of an uncertainty
%  index — only with its innovation.  dlog is the residual of an AR(1) on the
%  log level with the coefficient IMPOSED at 1.  These indices are persistent
%  but mean-reverting, not unit-root, so dlog OVER-DIFFERENCES: it injects a
%  spurious MA(1), adds noise, and attenuates the correlation toward zero.
%  eta is the same object with the persistence ESTIMATED rather than imposed.
%
%  sum(rho) is printed precisely so the over-differencing can be SEEN: dlog
%  imposes 1.000, and the column below is what the data say instead.  Both
%  filters are carried to STEP 4 and reported side by side; the comparison
%  between the two rows is itself the result.
%
%  The filter is univariate — the index's own past only.  A multivariate
%  version (index lags + macro lags) would be closer to a "news" measure but
%  would condition on information a market participant need not have had.
% =========================================================================
fprintf('\n%s\n STEP 3b — UNFORECASTABLE COMPONENT (AR(%d) residual on log level)\n%s\n', ...
        repmat('=',1,70), AR_Q, repmat('=',1,70));
fprintf('  %-13s %10s %8s   %s\n','index','sum(rho)','n','reading');
fprintf('  %s\n', repmat('-',1,68));
for j = 1:numel(IDX)
    [IDX(j).eta, IDX(j).rho_sum, IDX(j).ar_n] = ...
        ar_innovation(log(IDX(j).lvl), IDX(j).ym, AR_Q);
    fprintf('  %-13s %10.3f %8d   %s\n', IDX(j).name, IDX(j).rho_sum, IDX(j).ar_n, ...
            tern(IDX(j).rho_sum < 0.95, 'mean-reverting -> dlog OVER-differences', ...
                                        'near unit root -> dlog ~ ok'));
end
fprintf('  dlog imposes sum(rho) = 1.000 by construction; compare with the column above.\n');

%% =========================================================================
%  STEP 4 — MONTHLY CORRELATIONS  (instrument vs uncertainty)
%
%  GOLDEN RULE.  A correlation is non-spurious only between objects of the SAME
%  statistical nature: level<->level, or innovation<->innovation.  Pairing a
%  persistent level with a zero-mean white series gives a mechanically-null
%  correlation — an artefact of the mismatch, not an economic result.  Hence:
%
%    (A)  index LEVEL   ~ intensity  abs_k, sq_k        turbulence (both are
%                                                       non-negative levels)
%    (A2) index LEVEL   ~ one-sided  pos_k, |neg_k|     ASYMMETRY: is uncertainty
%                                                       tied to the down-days or
%                                                       to the up-days?
%    (B)  index dlog    ~ signed     mean_k             directional, persistence
%                                                       IMPOSED at 1
%    (B') index eta     ~ signed     mean_k             directional, persistence
%                                                       ESTIMATED  [preferred]
%    (B2) index |dlog|  ~ abs_k                         two-sided, sign-free
%    (B2')index |eta|   ~ abs_k                         two-sided, AR filter
%    (C)  index LEVEL   ~ mean_k     [MISMATCH CONTROL] deliberately wrong pair;
%                                                       must come out ~0, and is
%                                                       reported to show WHY the
%                                                       index must be filtered
%
%  SIGN CAVEAT.  The daily shocks are identified UP TO SIGN, so which of pos_k /
%  neg_k is "the bad side" flips with the sign normalisation.  The GAP between
%  the two correlations is invariant (it only swaps); its direction is not.
%  Read (A2) as "is there asymmetry?", not as "the downside matters more".
% =========================================================================
fprintf('\n%s\n STEP 4 — MONTHLY CORRELATIONS (external instrument vs uncertainty)\n%s\n', repmat('=',1,70), repmat('=',1,70));
rows = struct('index',{},'shock',{},'measure',{},'pearson_r',{},'pearson_p',{}, ...
              'spearman_r',{},'spearman_p',{},'n',{});
for j=1:numel(IDX)
    [~,ia,ib]=intersect(IDX(j).ym, ym_monthly); if numel(ia)<6, continue; end
    lvl=IDX(j).lvl(ia); dlog=[NaN; diff(log(lvl))];
    % dlog is only a true one-month change where the matched months are
    % contiguous; break it wherever the overlap skips a month.
    ymj = IDX(j).ym(ia); step = [NaN; diff(double(ymj))];
    dlog(~(step==1 | step==89)) = NaN;     % 89 = December -> January (yyyymm)
    % eta was built in STEP 3b on the index's own FULL, contiguity-checked
    % sample, so it only has to be sliced onto the matched months here.
    et = IDX(j).eta(ia);
    for kk=1:N_shocks
        k=shock_cols(kk);
        a=ABS(ib,k); s=SQ(ib,k); mn=MEAN(ib,k); po=POS(ib,k); ng=-NEG(ib,k);
        rows(end+1)=mkrow(IDX(j).name,k,'level~abs_k',   lvl, a);       %#ok<AGROW>
        rows(end+1)=mkrow(IDX(j).name,k,'level~sq_k',    lvl, s);       %#ok<AGROW>
        rows(end+1)=mkrow(IDX(j).name,k,'level~pos_k',   lvl, po);      %#ok<AGROW>
        rows(end+1)=mkrow(IDX(j).name,k,'level~|neg_k|', lvl, ng);      %#ok<AGROW>
        rows(end+1)=mkrow(IDX(j).name,k,'dlog~mean_k',   dlog, mn);     %#ok<AGROW>
        rows(end+1)=mkrow(IDX(j).name,k,'eta~mean_k',    et,   mn);     %#ok<AGROW>
        rows(end+1)=mkrow(IDX(j).name,k,'|dlog|~abs_k',  abs(dlog), a); %#ok<AGROW>
        rows(end+1)=mkrow(IDX(j).name,k,'|eta|~abs_k',   abs(et),   a); %#ok<AGROW>
        rows(end+1)=mkrow(IDX(j).name,k,'level~mean_k',  lvl, mn);      %#ok<AGROW>
    end
end
CORR = struct2table(rows);
rho_sum_all = [IDX.rho_sum];  idx_names_all = {IDX.name};
save(sprintf('corr_extIV_uncertainty_%s.mat',SRC_BASE), ...
     'CORR','SRC','shock_cols','shock_tag','AR_Q','rho_sum_all','idx_names_all','-v7.3');

% Full matrices (index x shock) per measure
idx_names = {IDX.name};
meas = {'level~abs_k','level~sq_k','level~pos_k','level~|neg_k|', ...
        'dlog~mean_k','eta~mean_k','|dlog|~abs_k','|eta|~abs_k','level~mean_k'};
meas_note = {'turbulence, L1 (robust)','turbulence, L2 (tail-driven)', ...
             'one-sided activity: UP days','one-sided activity: DOWN days', ...
             'directional — persistence IMPOSED at 1', ...
             'directional — persistence ESTIMATED  [PREFERRED]', ...
             'two-sided, sign-free (dlog filter)','two-sided, sign-free (AR filter)', ...
             'MISMATCH CONTROL — must be ~0 by construction'};
for mm=1:numel(meas)
    print_matrix(CORR, idx_names, shock_cols, shock_tag, dom_daily, meas{mm}, meas_note{mm});
end

% --- the instrument itself: the three filters side by side ------------------
fprintf('\n  >>> THE INSTRUMENT (mean_k): the three filters of the index, side by side\n');
fprintf('      eta  = AR(%d) innovation  (persistence estimated)   <- preferred\n', AR_Q);
fprintf('      dlog = log-difference     (persistence imposed = 1)\n');
fprintf('      lvl  = raw level          (mismatch control, expected ~0)\n\n');
for kk=1:N_shocks
    fprintf('  %s\n', shock_tag{kk});
    fprintf('  %-13s %9s %9s %9s %9s\n','index','eta~mn','dlog~mn','lvl~mn','rho_sum');
    fprintf('  %s\n', repmat('-',1,53));
    for j=1:numel(idx_names)
        rE=getr(CORR,idx_names{j},shock_cols(kk),'eta~mean_k');
        rB=getr(CORR,idx_names{j},shock_cols(kk),'dlog~mean_k');
        rC=getr(CORR,idx_names{j},shock_cols(kk),'level~mean_k');
        fprintf('  %-13s %9.3f %9.3f %9.3f %9.3f\n', idx_names{j}, rE, rB, rC, IDX(j).rho_sum);
    end
    fprintf('\n');
end
fprintf('  Read: |eta~mn| should weakly dominate |dlog~mn| wherever rho_sum < 1, because\n');
fprintf('  dlog over-differences and the extra noise attenuates the correlation.  The third\n');
fprintf('  column is the mismatch control and is expected ~0 by construction.\n');

% --- ASYMMETRY: up-side vs down-side activity ------------------------------
fprintf('\n  >>> ASYMMETRY  (level~pos_k  vs  level~|neg_k|)   gap = r_neg - r_pos\n');
fprintf('  [sign convention arbitrary: the SIZE of the gap is invariant, its SIGN is not]\n');
fprintf('  %-13s', 'index');
for kk=1:N_shocks, fprintf(' %8s %8s %7s', sprintf('%s pos',shock_tag{kk}), sprintf('%s neg',shock_tag{kk}), 'gap'); end
fprintf('\n  %s\n', repmat('-',1,13+25*N_shocks));
for j=1:numel(idx_names)
    fprintf('  %-13s', idx_names{j});
    for kk=1:N_shocks
        rp=getr(CORR,idx_names{j},shock_cols(kk),'level~pos_k');
        rn=getr(CORR,idx_names{j},shock_cols(kk),'level~|neg_k|');
        fprintf(' %8.3f %8.3f %7.3f', rp, rn, rn-rp);
    end
    fprintf('\n');
end

% Agnostic reading: strongest uncertainty association per instrument
fprintf('\n  Strongest uncertainty association per instrument (|Pearson|, level~abs_k):\n');
for kk=1:N_shocks
    k=shock_cols(kk);
    sub=CORR(strcmp(CORR.measure,'level~abs_k') & CORR.shock==k & isfinite(CORR.pearson_r),:);
    if isempty(sub), continue; end
    [~,o]=max(abs(sub.pearson_r));
    fprintf('    %-4s : %-13s r=%+.3f (p=%.3f)   [interp: %s]\n', ...
            shock_tag{kk}, sub.index{o}, sub.pearson_r(o), sub.pearson_p(o), dom_daily{kk});
end
fprintf('\n  NOTE on the LEVEL panels: both series there are persistent, and the p-values\n');
fprintf('  above assume i.i.d. sampling, so they are overstated.  Those panels are\n');
fprintf('  descriptive; the inferential weight sits on the eta~mean_k panel, where both\n');
fprintf('  objects are innovations.\n');

%% =========================================================================
%  STEP 4b — THE INSTRUMENT, READ DIRECTLY  (narrative check on mean_k)
%    A correlation table says whether the instrument co-moves with something.
%    It does NOT say whether the instrument itself is sensible.  The months in
%    which |mean_k| is largest should be recognisable events; if they are not,
%    the shock is a numerical artefact whatever the correlations say.
% =========================================================================
fprintf('\n%s\n STEP 4b — THE INSTRUMENT ITSELF (mean_k): largest months\n%s\n', repmat('=',1,70), repmat('=',1,70));
NTOP=6;
for kk=1:N_shocks
    k=shock_cols(kk);
    fprintf('\n  %-4s  sd=%.4f  skew=%+.2f  kurt=%.2f  |  top-%d months by |mean_k|:\n', ...
            shock_tag{kk}, std(MEAN(:,k)), skew_local(MEAN(:,k)), kurt_local(MEAN(:,k)), NTOP);
    [~,o]=sort(abs(MEAN(:,k)),'descend');
    fprintf('      %-9s %10s %10s %10s %10s\n','month','mean_k','pos_k','neg_k','abs_k');
    for t=o(1:min(NTOP,numel(o)))'
        fprintf('      %-9s %10.4f %10.3f %10.3f %10.3f\n', string(dates_monthly(t),'yyyy-MM'), ...
                MEAN(t,k), POS(t,k), NEG(t,k), ABS(t,k));
    end
end

%% =========================================================================
%  STEP 4c — INSTRUMENT VALIDITY DIAGNOSTICS  (what makes z a SHOCK)
%
%  Stage A cannot test RELEVANCE: relevance is a covariance with the macro VAR
%  residuals, which only exist in Stage B.  What Stage A CAN test — without
%  touching the macro data — are the properties that make the aggregate a
%  structural shock at all:
%
%    (i)  MEAN ZERO.        A non-zero mean would be absorbed by the intercept
%         and would correlate mechanically with any persistent macro variable.
%    (ii) NO SERIAL CORRELATION.  A serially correlated "instrument" is
%         predictable from its own past, hence partly inside the information
%         set: it is a state variable, not a shock, and E[u_t z_t] would mix
%         contemporaneous with lagged effects.
%
%  The same table is what turns the ABS/SQ argument from a claim into a result:
%  mean_k should come out WHITE, abs_k and sq_k should be MASSIVELY rejected —
%  volatility clusters.  That is precisely why they are validation objects and
%  not instruments.  The test on mean_k^2 is the third result: it documents the
%  conditional heteroskedasticity that motivates the block bootstrap in Stage B.
%
%  NB a regression of mean_k on lagged SP500/VIX/CSPREAD would be close to
%  vacuous here: the shock is orthogonal to that information set BY
%  CONSTRUCTION (it comes from a VAR on those very series).  The informative
%  version is the test against information the daily VAR does NOT contain — the
%  macro panel — i.e. the Forni-Gambetti (2014) informational-sufficiency test,
%  which therefore lives in Stage B.
% =========================================================================
fprintf('\n%s\n STEP 4c — INSTRUMENT VALIDITY DIAGNOSTICS (mean zero + no serial correlation)\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
LB_H = 12;                              % 12 monthly lags = one year
fprintf('  Ljung-Box with %d lags; t-stat on the sample mean.\n', LB_H);
fprintf('  %-14s %9s %8s %8s %10s %9s   %s\n','series','mean','t(mean)','rho(1)','LB(12)','p-value','reading');
fprintf('  %s\n', repmat('-',1,86));
lb_tab = struct('series',{},'shock',{},'measure',{},'mean',{},'t_mean',{},'rho1',{},'LB',{},'p',{});
for kk=1:N_shocks
    k = shock_cols(kk);
    series_set = {MEAN(:,k), MEAN(:,k).^2, ABS(:,k), SQ(:,k)};
    names_set  = {'mean_k','mean_k^2','abs_k','sq_k'};
    for si=1:numel(series_set)
        x = series_set{si};
        [Q,pv,r1] = ljungbox(x, LB_H);
        mu_x = mean(x);  t_x = mu_x/(std(x)/sqrt(numel(x)));
        if si==1
            verdict = tern(pv>0.05, 'WHITE  ok', 'AUTOCORRELATED [!]');
        else
            verdict = tern(pv<0.05, 'rejected (as expected)', 'not rejected [?]');
        end
        fprintf('  %-4s %-9s %9.4f %8.2f %8.3f %10.1f %9.4f   %s\n', ...
                shock_tag{kk}, names_set{si}, mu_x, t_x, r1, Q, pv, verdict);
        lb_tab(end+1) = struct('series',sprintf('%s %s',shock_tag{kk},names_set{si}), ...
                               'shock',k,'measure',names_set{si},'mean',mu_x,'t_mean',t_x, ...
                               'rho1',r1,'LB',Q,'p',pv); %#ok<AGROW>
    end
    fprintf('  %s\n', repmat('.',1,86));
end
fprintf('  Expected pattern: mean_k white (p>0.05); mean_k^2, abs_k, sq_k all rejected (p~0).\n');
LB_TABLE = struct2table(lb_tab);
save(sprintf('instrument_diagnostics_%s.mat',SRC_BASE),'LB_TABLE','LB_H','SIGN_CONV','-v7.3');

%% =========================================================================
%  STEP 5 — FIGURES
%    fig_p02A_corr_<SRC>.png      main heatmap (level~abs_k)
%    fig_p02A_corrALL_<SRC>.png   one heatmap per measure
%    fig_p02A_instr_<SRC>.png     the instrument itself: mean_k
%                                 + its POS/NEG decomposition
% =========================================================================
fprintf('\n%s\n STEP 5 — FIGURES\n%s\n', repmat('=',1,70), repmat('=',1,70));
nI=numel(IDX); H=nan(nI,N_shocks);
for j=1:nI
    sub=CORR(strcmp(CORR.index,idx_names{j}) & strcmp(CORR.measure,'level~abs_k'),:);
    for kk=1:N_shocks
        r=sub.pearson_r(sub.shock==shock_cols(kk)); if ~isempty(r), H(j,kk)=r(1); end
    end
end
f=figure('Color','w','Position',[40 40 720 460]);
imagesc(H,[-1 1]); colormap(redbluemap_local()); colorbar;
set(gca,'XTick',1:N_shocks,'XTickLabel',shock_tag,'YTick',1:nI,'YTickLabel',idx_names,'TickLabelInterpreter','none');
title('External instrument (|shock|) vs uncertainty LEVEL — Pearson','Interpreter','none');
for j=1:nI, for kk=1:N_shocks
    if ~isnan(H(j,kk)), text(kk,j,sprintf('%.2f',H(j,kk)),'HorizontalAlignment','center','FontSize',9, ...
                             'Color', tern(abs(H(j,kk))>0.5,'w','k')); end
end, end
print(f, sprintf('fig_p02A_corr_%s.png',SRC_BASE),'-dpng','-r150');

% --- ALL measures on a common colour scale ---------------------------------
nM=numel(meas); ncolF=3; nrowF=ceil(nM/ncolF);
f2=figure('Color','w','Position',[20 20 400*ncolF 300*nrowF]);
for mm=1:nM
    Hm=nan(nI,N_shocks); sub=CORR(strcmp(CORR.measure,meas{mm}),:);
    for j=1:nI
        s2=sub(strcmp(sub.index,idx_names{j}),:);
        for kk=1:N_shocks
            r=s2.pearson_r(s2.shock==shock_cols(kk)); if ~isempty(r), Hm(j,kk)=r(1); end
        end
    end
    subplot(nrowF,ncolF,mm);
    imagesc(Hm,[-1 1]); colormap(redbluemap_local());
    set(gca,'XTick',1:N_shocks,'XTickLabel',shock_tag,'YTick',1:nI,'YTickLabel',idx_names, ...
            'TickLabelInterpreter','none','FontSize',7);
    title({meas{mm}, meas_note{mm}},'Interpreter','none','FontSize',8);
    for j=1:nI, for kk=1:N_shocks
        if ~isnan(Hm(j,kk)), text(kk,j,sprintf('%.2f',Hm(j,kk)),'HorizontalAlignment','center', ...
                                  'FontSize',7,'Color',tern(abs(Hm(j,kk))>0.5,'w','k')); end
    end, end
    if mm==nM, cb=colorbar; cb.Label.String='Pearson r'; end
end
sgtitle('Uncertainty indices vs every monthly aggregate of the daily shocks (Pearson, common scale)', ...
        'FontSize',11,'Interpreter','none');
print(f2, sprintf('fig_p02A_corrALL_%s.png',SRC_BASE),'-dpng','-r150');

% --- the instrument itself --------------------------------------------------
f3=figure('Color','w','Position',[20 20 1250 250*N_shocks]);
for kk=1:N_shocks
    k=shock_cols(kk);
    subplot(N_shocks,2,2*kk-1);
    plot(dates_monthly, MEAN(:,k),'-','Color',[0.85 0.33 0.10],'LineWidth',0.8); hold on;
    yline(0,'k:'); grid on; axis tight;
    title(sprintf('%s — monthly instrument  mean_k (signed monthly mean)', shock_tag{kk}), ...
          'Interpreter','none','FontSize',9);
    ylabel('mean_k','Interpreter','none','FontSize',8); set(gca,'FontSize',7);

    subplot(N_shocks,2,2*kk);
    bar(dates_monthly, POS(:,k), 1, 'FaceColor',[0.20 0.55 0.25],'EdgeColor','none'); hold on;
    bar(dates_monthly, NEG(:,k), 1, 'FaceColor',[0.75 0.20 0.15],'EdgeColor','none');
    plot(dates_monthly, SUM(:,k),'k-','LineWidth',0.5);
    yline(0,'k:'); grid on; axis tight;
    title(sprintf('%s — decomposition  pos_k (green) / neg_k (red);  black = sum_k', shock_tag{kk}), ...
          'Interpreter','none','FontSize',9);
    set(gca,'FontSize',7);
end
sgtitle('The monthly instrument and its one-sided decomposition (asymmetry is visible here, not in the mean)', ...
        'FontSize',11,'Interpreter','none');
print(f3, sprintf('fig_p02A_instr_%s.png',SRC_BASE),'-dpng','-r150');

fprintf('  fig_p02A_corr_%s.png      (main heatmap, level~abs_k)\n', SRC_BASE);
fprintf('  fig_p02A_corrALL_%s.png   (all %d measures, common scale)\n', SRC_BASE, nM);
fprintf('  fig_p02A_instr_%s.png     (mean_k + pos/neg decomposition)\n', SRC_BASE);

fprintf('\n%s\n DONE — Stage A\n  external instrument saved: shocks_monthly_%s.mat (MEAN)\n  correlations saved:        corr_extIV_uncertainty_%s.mat (%d measures, AR_Q=%d)\n  next: run proxy_svar_02_cholesky.m for the macro (Cholesky) stage.\n%s\n', ...
        repmat('=',1,70), SRC_BASE, SRC_BASE, nM, AR_Q, repmat('=',1,70));

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function T = readtab(file,sheet), T=readtable(file,'Sheet',sheet,'VariableNamingRule','preserve'); end

function v = getcol(T,key)
    vn=T.Properties.VariableNames; idx=find(contains(lower(vn),lower(key)),1);
    if isempty(idx), error('Column "%s" not found in [%s].', key, strjoin(vn,', ')); end
    c=T.(vn{idx});
    if isnumeric(c)||isdatetime(c), v=c; else, v=str2double(string(c)); end
end

function s = mk_idx(name,ym,lvl)
    [ym,srt]=sort(ym(:)); lvl=lvl(:); lvl=lvl(srt);
    [ym,ic]=unique(ym,'stable'); lvl=lvl(ic);
    s=struct('name',name,'ym',ym,'lvl',lvl);
end

function [eta, rho_sum, n_used] = ar_innovation(y, ym, q)
%  UNFORECASTABLE COMPONENT of y: residual of an AR(q) fitted on y, formed only
%  where the q preceding observations are calendar-contiguous, so that no lag is
%  ever built across a gap in the index.  rho_sum is the estimated persistence,
%  against which the unit value that the log-difference imposes can be compared:
%  rho_sum well below 1 means dlog over-differences.
%  Lags are built by hand — no Econometrics Toolbox dependency.
    y = y(:);  ym = double(ym(:));  n = numel(y);
    X = [ones(n,1), nan(n,q)];
    for L = 1:q
        X(L+1:end, 1+L) = y(1:end-L);
    end
    mnum   = 12*floor(ym/100) + mod(ym,100);          % months elapsed since year 0
    contig = false(n,1);
    contig(q+1:end) = (mnum(q+1:end) - mnum(1:end-q)) == q;
    ok  = contig & isfinite(y) & all(isfinite(X),2);
    eta = nan(n,1);  rho_sum = NaN;  n_used = sum(ok);
    if n_used > 5*q
        b = X(ok,:) \ y(ok);
        eta(ok) = y(ok) - X(ok,:)*b;
        rho_sum = sum(b(2:end));
    end
end

function r = mkrow(idxname,k,measure,x,y)
    o=corr_pair(x,y);
    r=struct('index',idxname,'shock',k,'measure',measure,'pearson_r',o.pr,'pearson_p',o.pp, ...
             'spearman_r',o.sr,'spearman_p',o.sp,'n',o.n);
end

function out = corr_pair(x,y)
    x=x(:); y=y(:); m=isfinite(x)&isfinite(y); x=x(m); y=y(m); n=numel(x);
    out=struct('pr',NaN,'pp',NaN,'sr',NaN,'sp',NaN,'n',n);
    if n<5||std(x)==0||std(y)==0, return; end
    [R,P]=corrcoef(x,y); out.pr=R(1,2); out.pp=P(1,2);
    [R2,P2]=corrcoef(rankavg(x),rankavg(y)); out.sr=R2(1,2); out.sp=P2(1,2);
end

function r = rankavg(x)
    x=x(:); n=numel(x); [~,ord]=sort(x); r=zeros(n,1); r(ord)=(1:n)';
    [u,~,iu]=unique(x); for i=1:numel(u), m=(iu==i); if sum(m)>1, r(m)=mean(r(m)); end, end
end

function r = getr(CORR, idxname, k, measure)
    v = CORR.pearson_r(strcmp(CORR.index,idxname) & CORR.shock==k & strcmp(CORR.measure,measure));
    if isempty(v), r = NaN; else, r = v(1); end
end

function [Q,pv,r1] = ljungbox(x,h)
    x=x(:); x=x(isfinite(x)); T=numel(x); x=x-mean(x);
    c0=sum(x.^2); Q=0; r1=NaN;
    if T<=h+2 || c0<=0, Q=NaN; pv=NaN; return; end
    for k=1:h
        rk=sum(x(1+k:end).*x(1:end-k))/c0;
        if k==1, r1=rk; end
        Q=Q+rk^2/(T-k);
    end
    Q=T*(T+2)*Q; pv=1-chi2cdf(Q,h);
end

function s = skew_local(x)
    x=x(:); x=x(isfinite(x)); x=x-mean(x); s=mean(x.^3)/max(mean(x.^2)^1.5,1e-18);
end

function k = kurt_local(x)
    x=x(:); x=x(isfinite(x)); x=x-mean(x); k=mean(x.^4)/max(mean(x.^2)^2,1e-18);
end

function print_matrix(CORR, idx_names, shock_cols, shock_tag, dom_daily, measure, note)
    if nargin<7, note=''; end
    sub=CORR(strcmp(CORR.measure,measure),:);
    if isempty(sub), return; end
    nI=numel(idx_names); Nk=numel(shock_cols); R=nan(nI,Nk); Pv=nan(nI,Nk);
    for j=1:nI
        s=sub(strcmp(sub.index,idx_names{j}),:);
        for kk=1:Nk
            r=s.pearson_r(s.shock==shock_cols(kk)); pp=s.pearson_p(s.shock==shock_cols(kk));
            if ~isempty(r), R(j,kk)=r(1); end; if ~isempty(pp), Pv(j,kk)=pp(1); end
        end
    end
    if isempty(note)
        fprintf('\n  >>> measure = %s   (cols: v_k [daily FEVD-dominant])\n', measure);
    else
        fprintf('\n  >>> measure = %-14s  [%s]   (cols: v_k [daily FEVD-dominant])\n', measure, note);
    end
    fprintf('  %-13s', 'index'); for kk=1:Nk, fprintf(' %14s', sprintf('%s(%s)',shock_tag{kk},dom_daily{kk}(1:min(4,end)))); end, fprintf('\n');
    fprintf('  %s\n', repmat('-',1,13+15*Nk));
    for j=1:nI
        fprintf('  %-13s', idx_names{j});
        for kk=1:Nk, fprintf(' %9.3f%-5s', R(j,kk), sig_star(Pv(j,kk))); end
        fprintf('\n');
    end
    fprintf('  (*** p<.01  ** p<.05  * p<.10)\n');
end

function st = sig_star(p)
    if isnan(p), st=''; elseif p<0.01, st='***'; elseif p<0.05, st='**'; elseif p<0.10, st='*'; else, st=''; end
end

function c = tern(cond,a,b), if cond, c=a; else, c=b; end, end

function cmap = redbluemap_local()
    n=64; top=[linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1)];
    bot=[ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)']; cmap=[top;bot];
end