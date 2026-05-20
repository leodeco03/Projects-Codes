
clear; close all; clc;

T  = 12;
tt = 0:T;                                           % t = 0..T (T+1 punti)

mods   = {'No_3b','No_3c','No_3d'};                 % float, managed, fixed (metti i nomi giusti)
labels = {'Float \psi_e=0','Managed \psi_e=0.80','Fixed e_c=0'};
vars   = {'y','c','ii','pie','ec','y_star','e_level','ca'};

pz = @(x) [0; reshape(x(1:T),[],1)];                % helper (nome diverso da mk!)

% storage: (T+1) x nMods
IRF = struct();
for k = 1:numel(vars)
    IRF.(vars{k}) = NaN(T+1, numel(mods));
end

for p = 1:numel(mods)
    dynare(mods{p}, 'noclearall');                  % nei .mod usa stoch_simul(..., nograph)

    for k = 1:numel(vars)
        v  = vars{k};
        f1 = [v '_eta_y_star'];
        f2 = [v '_eta_z'];

        if ~isfield(oo_.irfs, f1) || ~isfield(oo_.irfs, f2)
            warning('Nel modello %s mancano %s o %s -> salto %s', mods{p}, f1, f2, v);
            continue;
        end

        comb = -oo_.irfs.(f1) + -oo_.irfs.(f2);     % due shock negativi per linearità
        IRF.(v)(:,p) = pz(comb);                    % colonna + prepone 0 => T+1 punti
    end

    % pulizia tra run (ed eviti variabili che rimangono in workspace)
    clear M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info estim_options_;
    clear mk                                           % se qualche .mod ha definito mk, la rimuovi
end

% Plot 4x2 con le tre policy sovrapposte
figure('Name','Q3: Two-shock comparison across policies (b,c,d)','NumberTitle','off');
for k = 1:numel(vars)
    subplot(4,2,k); hold on;
    for p = 1:numel(mods)
        plot(tt, IRF.(vars{k})(:,p), 'LineWidth', 1.5);
    end
    yline(0); grid on; title(vars{k});
    if k == 1, legend(labels,'Location','best'); end
    hold off;
end
