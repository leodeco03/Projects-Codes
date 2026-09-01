%% SCRIPT: Bootstrap a blocchi per la significatività della co-curtosi (V_ref)
clear; clc;

load('mori_3var.mat', 'V_ref', 'M', 'T_eff');

% 2. Configurazione Bootstrap
BLOCK_LEN    = 20;   % 20 giorni catturano la persistenza tipica della volatilità (1 mese)
BLOCK_BOOT_N = 5000; % Più alto è, più stabile è il calcolo degli estremi (CI)

% Standardizziamo gli shock (come in Step 6c)
Vc  = V_ref - mean(V_ref);
Vsd = Vc ./ std(Vc);

nblk = ceil(T_eff / BLOCK_LEN);
boot_co_kurt  = zeros(M, M, BLOCK_BOOT_N);   % E[v_i^2 v_j^2]  (H0 indip. => 1)
boot_co_skew3 = zeros(M, M, BLOCK_BOOT_N);   % E[v_i^2 v_j]    (H0 indip. => 0) 3o ordine, SV-robusto
boot_co_kurtA = zeros(M, M, BLOCK_BOOT_N);   % E[v_i^3 v_j]    (H0 indip. => 0) 4o asimm., SV-robusto
%   NB: in mori_3var.mat questo oggetto e' indep_diag.co_kurt_robust

fprintf('Esecuzione Moving-Block Bootstrap su V_ref (%d replicazioni)...\n', BLOCK_BOOT_N);
tic;
for b = 1:BLOCK_BOOT_N
    % Estrazione indici a blocchi per preservare l'autocorrelazione (GARCH)
    starts = randi(T_eff - BLOCK_LEN + 1, nblk, 1);
    idx = zeros(nblk * BLOCK_LEN, 1);
    for q = 1:nblk
        idx((q-1)*BLOCK_LEN+1 : q*BLOCK_LEN) = starts(q) : starts(q)+BLOCK_LEN-1;
    end
    idx = idx(1:T_eff); % Taglia all'esatta lunghezza temporale

    % Campione bootstrap degli shock
    Vb = Vsd(idx, :);

    % Co-momenti per questo resample.
    %   co-curtosi   E[v_i^2 v_j^2]  -> contaminata da SV comune (vedi nota sotto)
    %   co-skewness  E[v_i^2 v_j]    -> robusta a SV (coppie ordinate i~=j)
    %   4o asimmetr. E[v_i^3 v_j]    -> robusta a SV (coppie ordinate i~=j)
    for i = 1:M
        for j = 1:M
            boot_co_kurt(i,j,b)  = mean(Vb(:,i).^2 .* Vb(:,j).^2);
            boot_co_skew3(i,j,b) = mean(Vb(:,i).^2 .* Vb(:,j));
            boot_co_kurtA(i,j,b) = mean(Vb(:,i).^3 .* Vb(:,j));
        end
    end
end
fprintf('Completato in %.2f secondi.\n\n', toc);

% 3. Inferenza Statistica (H0: indipendenza => E[v_i^2 v_j^2] = 1)
alpha  = 0.05;
q_low  = alpha / 2;
q_high = 1 - alpha / 2;

% -------------------------------------------------------------------------
% (A) CO-CURTOSI  E[v_i^2 v_j^2]  -- H0: indipendenza => 1
%     ATTENZIONE: questa condizione NON e' robusta alla volatilita' stocastica.
%     Sotto un fattore di volatilita' COMUNE omega_t (v_t = sqrt(omega_t) z_t,
%     z conditionatamente indip.):
%         E[v_i^2 v_j^2] = E[omega^2] E[z_i^2] E[z_j^2] = E[omega^2] = 1+Var(omega) > 1
%     anche se gli shock sono CONDIZIONATAMENTE INDIPENDENTI.
%     => un REJECT qui e' ATTESO con volatility clustering (LB(v^2) rifiuta) e
%        NON e' di per se' prova di dipendenza strutturale.
%     Riferimento: Drautzburg & Wright (2023, J.Econometrics), p.1832 (GMM1 vs
%     GMM2: la versione SV-robusta SCARTA proprio le condizioni E[v_i^2 v_j^2]=1).
% -------------------------------------------------------------------------
fprintf('=== (A) CO-CURTOSI (H0: E[v_i^2 v_j^2] = 1)  -- NON robusta a SV ===\n');
fprintf('    Nota: sotto volatilita'' comune E[v_i^2 v_j^2]=E[omega^2]>1 anche con\n');
fprintf('    shock condiz. indipendenti => un REJECT qui e'' atteso (Drautzburg-Wright 2023).\n');
fprintf('%-10s %-10s %10s %10s %22s %12s\n', 'Shock 1', 'Shock 2', 'Stima', 'SE Boot', '95% CI Boot', 'Esito (H0=1)');
fprintf('%s\n', repmat('-', 1, 80));

for i = 1:M
    for j = i+1:M  % co-curtosi e' simmetrica => solo triangolo superiore
        dist_b = squeeze(boot_co_kurt(i,j,:));
        estimate = mean(Vsd(:,i).^2 .* Vsd(:,j).^2);
        se       = std(dist_b);
        ci_low   = quantile(dist_b, q_low);
        ci_high  = quantile(dist_b, q_high);
        if 1 < ci_low || 1 > ci_high
            sig_flag = '*** REJECT';
        else
            sig_flag = '    Accetta';
        end
        fprintf('v_%-8d v_%-8d %10.4f %10.4f   [%6.4f, %6.4f] %12s\n', ...
            i, j, estimate, se, ci_low, ci_high, sig_flag);
    end
end

% -------------------------------------------------------------------------
% (B) CO-SKEWNESS  E[v_i^2 v_j]   e   (C) 4o ASIMMETRICO  E[v_i^3 v_j]
%     H0: indipendenza => 0.  Queste condizioni SONO robuste alla volatilita'
%     stocastica comune: sotto v_t = sqrt(omega_t) z_t con z indip. e simmetrico,
%         E[v_i^2 v_j]  = E[omega^{3/2}] E[z_i^2] E[z_j] = (...)*0 = 0
%         E[v_i^3 v_j]  = E[omega^2]     E[z_i^3] E[z_j] = (...)*0 = 0
%     => un REJECT qui e' evidenza GENUINA di dipendenza strutturale (es. leverage),
%        non un artefatto della volatilita' comune.  Sono le condizioni che il test
%        SV-robusto (GMM2) di Drautzburg-Wright MANTIENE.  Coppie ORDINATE i~=j
%        (i co-momenti NON sono simmetrici in (i,j)).
% -------------------------------------------------------------------------
test_specs = { 'CO-SKEWNESS  E[v_i^2 v_j]', boot_co_skew3, @(i,j) mean(Vsd(:,i).^2 .* Vsd(:,j)) ; ...
               '4o ASIMMETRICO E[v_i^3 v_j]', boot_co_kurtA, @(i,j) mean(Vsd(:,i).^3 .* Vsd(:,j)) };

for tt = 1:size(test_specs,1)
    tname = test_specs{tt,1};
    barr  = test_specs{tt,2};
    festi = test_specs{tt,3};
    fprintf('\n=== (%c) %s (H0 = 0)  -- ROBUSTA a SV ===\n', 'B'+tt-1, tname);
    fprintf('%-10s %-10s %10s %10s %22s %12s\n', 'Shock i', 'Shock j', 'Stima', 'SE Boot', '95% CI Boot', 'Esito (H0=0)');
    fprintf('%s\n', repmat('-', 1, 80));
    for i = 1:M
        for j = 1:M
            if i == j, continue; end   % coppie ordinate i~=j
            dist_b   = squeeze(barr(i,j,:));
            estimate = festi(i,j);
            se       = std(dist_b);
            ci_low   = quantile(dist_b, q_low);
            ci_high  = quantile(dist_b, q_high);
            if 0 < ci_low || 0 > ci_high   % 0 fuori dall'intervallo => REJECT
                sig_flag = '*** REJECT';
            else
                sig_flag = '    Accetta';
            end
            fprintf('v_%-8d v_%-8d %10.4f %10.4f   [%6.4f, %6.4f] %12s\n', ...
                i, j, estimate, se, ci_low, ci_high, sig_flag);
        end
    end
end



% =========================================================================
%  SAVE — le stime, gli SE bootstrap e gli intervalli sono i numeri riportati
%  nella tabella delle diagnostiche di indipendenza SV-robuste.  Senza questo
%  salvataggio esistevano solo sullo standard output, quindi non erano
%  riproducibili senza rilanciare 5.000 replicazioni.
% =========================================================================
est_co_kurt = zeros(M); est_co_skew3 = zeros(M); est_co_kurtA = zeros(M);
se_co_kurt  = zeros(M); se_co_skew3  = zeros(M); se_co_kurtA  = zeros(M);
ci_co_kurt  = zeros(M,M,2); ci_co_skew3 = zeros(M,M,2); ci_co_kurtA = zeros(M,M,2);
for i = 1:M
    for j = 1:M
        est_co_kurt(i,j)  = mean(Vsd(:,i).^2 .* Vsd(:,j).^2);
        est_co_skew3(i,j) = mean(Vsd(:,i).^2 .* Vsd(:,j));
        est_co_kurtA(i,j) = mean(Vsd(:,i).^3 .* Vsd(:,j));
        dk = squeeze(boot_co_kurt(i,j,:));  ds = squeeze(boot_co_skew3(i,j,:));
        da = squeeze(boot_co_kurtA(i,j,:));
        se_co_kurt(i,j)  = std(dk); se_co_skew3(i,j) = std(ds); se_co_kurtA(i,j) = std(da);
        ci_co_kurt(i,j,:)  = quantile(dk,[q_low q_high]);
        ci_co_skew3(i,j,:) = quantile(ds,[q_low q_high]);
        ci_co_kurtA(i,j,:) = quantile(da,[q_low q_high]);
    end
end
save('bootstrap_kollo_results.mat', ...
     'est_co_kurt','est_co_skew3','est_co_kurtA', ...
     'se_co_kurt','se_co_skew3','se_co_kurtA', ...
     'ci_co_kurt','ci_co_skew3','ci_co_kurtA', ...
     'BLOCK_LEN','BLOCK_BOOT_N','M','T_eff','-v7.3');
fprintf('\nSalvato bootstrap_kollo_results.mat (stime, SE, CI dei 3 pannelli).\n');
