%% NKmodel_lin2.mod
%
% DESCRIPTION: dynare.mod file to solve and simulate the  Final project of
% the course: "International Money and Finance"
%
% Suasday Chea and Leonardo Decorte
% Created: 10/10/2025
%
%

%% Small Open Economy NK – log/level deviations
%------------------------
%1) Variables
%// ------------------------
var c y i pi piH piF S q mc a ec e_level ca    // domestiche
    y_star pi_star i_star                      // estero (AR(1))
    tau g epsH;                                // premio rischio, preferenze, cost-push

varexo eps_m eps_tau eps_z eps_H eps_y_star eps_pi_star eps_i_star;

// ------------------------
// 2) Parametri (handout)
// ------------------------
parameters sigma alpha epsilon beta thetaH phi varphi h xi iotaH
          rho_i phi_pi phi_y phi_dy phi_e
          rho_tau rho_g rho_H rho_y_star rho_pi_star rho_i_star
          omega psi kappaH;

sigma   = 1;      // IES^-1
alpha   = 0.24;   // apertura
epsilon = 0.85;   // elast. sost. dom/import
beta    = 0.99;
thetaH  = 0.75;   // Calvo (prezzi domestici)
phi     = 0.01;   // coeff. premio rischio (UIP/CA)
varphi  = 1.26;   // Frisch^-1
h       = 0.25;   // habit
xi      = 0.15;   // elasticità consumo a shock di preferenze
iotaH   = 0.30;   // indicizzazione prezzi

rho_i   = 0.70;   // smoothing Taylor
phi_pi  = 1.90;
phi_y   = 0.05;
phi_dy  = 0.55;
phi_e   = 0.00;   // benchmark: niente risposta a FX

rho_tau   = 0.70;
rho_g     = 0.70;
rho_H     = 0.70;

rho_y_star  = 0.70;
rho_pi_star = 0.70;
rho_i_star  = 0.70;

// Coefficienti composti
kappaH = (1-thetaH)*(1-beta*thetaH)/thetaH;  // p-curve slope
omega  = alpha;                               // π = π_H + ω ΔS  (wedge CPI-PPI)
psi    = alpha*(epsilon-1)/(1-alpha);         // peso TOT in risorse (convenzione GM05)

// ------------------------
// 3) Modello (linearizzato)
// ------------------------
model(linear);

// (1) IS con habit e shock preferenze
c = (h/(1+h))*c(-1) + (1/(1+h))*c(+1) - (1/sigma)*((1-h)/(1+h))*( i - pi(+1) ) + xi*g;

// (2) Vincolo risorse (goods-market clearing)
y = (1-alpha)*c + alpha*y_star + psi*(2-alpha)*S;

// (3) Cambio reale e termini di scambio
q = (1-alpha)*S;

// (4) ΔS = π_F - π_H
S - S(-1) = piF - piH;

// (5) Phillips curve (prezzi domestici)
( piH - iotaH*piH(-1) ) = beta*( piH(+1) - iotaH*piH ) + kappaH*mc + epsH;

// (6) Costo marginale reale (spec. standard SOE)
mc = varphi*y + psi*S + sigma*c;

// (7) CPI vs PPI
pi = piH + omega*( S - S(-1) );

// (8) UIP con premio rischio e attese inflazione CPI
i - i_star = ec(+1) - phi*a + pi(+1) + tau;

// (9) Conto corrente / attività estere nette
y - c = a - (1/beta)*a(-1) + (phi/(1-alpha))*q;

// (10) Inflazione beni importati (LoP)
piF = ec + pi_star;

// (11) Regola di Taylor
i = rho_i*i(-1) + phi_pi*pi + phi_y*y + phi_dy*(y - y(-1)) + phi_e*ec - eps_m;

// (12) Premio al rischio AR(1)
tau = rho_tau*tau(-1) + eps_tau;

// (13) Livello preferenze AR(1)
g   = rho_g*g(-1) + eps_z;

// (14) Shock cost-push AR(1) (entra in (5))
epsH = rho_H*epsH(-1) + eps_H;

// (16.A)–(18.A) Estero: AR(1)
y_star  = rho_y_star*y_star(-1)   + eps_y_star;
pi_star = rho_pi_star*pi_star(-1) + eps_pi_star;
i_star  = rho_i_star*i_star(-1)   + eps_i_star;

// Book-keeping: livello del cambio nominale e conto corrente
e_level = e_level(-1) + ec;
ca      = y - c;

end;

// ------------------------
// 4) Steady state (0 per tutte le variabili log/level dev.)
// ------------------------
initval;
c=0; y=0; i=0; pi=0; piH=0; piF=0; S=0; q=0; mc=0; a=0; ec=0; e_level=0; ca=0;
y_star=0; pi_star=0; i_star=0; tau=0; g=0; epsH=0;
end;

steady; check;

// ------------------------
// 5) Shock (IRF a 1%: stderr = 0.01)
// ------------------------
shocks;
var eps_m;       stderr 0.01;
var eps_z;       stderr 0.01;
var eps_tau;     stderr 0.01;
var eps_i_star;  stderr 0.01;
var eps_y_star;  stderr 0.01;
var eps_pi_star; stderr 0.01;
var eps_H;       stderr 0.01;
end;

// ------------------------
// 6) IRF per Q1 (12 periodi), uno shock alla volta
//    + figure 4x2 automatiche
// ------------------------

// (a) Monetary policy shock
stoch_simul(order=1, irf=12, irf_shocks=(eps_m)) y c i pi ec i e_level ca;

// (b) Preference shock
stoch_simul(order=1, irf=12, irf_shocks=(eps_z)) y c i pi ec g e_level ca;

// (c) Risk premium shock
stoch_simul(order=1, irf=12, irf_shocks=(eps_tau)) y c i pi ec tau e_level ca;

// (d) Foreign interest-rate shock
stoch_simul(order=1, irf=12, irf_shocks=(eps_i_star)) y c i pi ec i_star e_level ca;

// ---- blocco MATLAB: figure 4x2 e salvataggio PNG ----
matlab;
T = 13; tt = 1:T;

figure('Name','Q1(a) – Monetary policy shock (1% eps_m)','Position',[100 100 900 900]);
subplot(4,2,1); plot(tt, oo_.irfs.y_eps_m(1:T));    title('Output y_t');
subplot(4,2,2); plot(tt, oo_.irfs.c_eps_m(1:T));    title('Consumption c_t');
subplot(4,2,3); plot(tt, oo_.irfs.i_eps_m(1:T));    title('Interest rate i_t');
subplot(4,2,4); plot(tt, oo_.irfs.pi_eps_m(1:T));   title('CPI inflation \pi_t');
subplot(4,2,5); plot(tt, oo_.irfs.ec_eps_m(1:T));   title('Depreciation e^c_t');
subplot(4,2,6); plot(tt, oo_.irfs.i_eps_m(1:T));    title('Shocked var: i_t');
subplot(4,2,7); plot(tt, oo_.irfs.e_level_eps_m(1:T)); title('Nominal FX level');
subplot(4,2,8); plot(tt, oo_.irfs.ca_eps_m(1:T));   title('Current account (y-c)');
print('-dpng','IRF_Q1a.png');

figure('Name','Q1(b) – Preference shock (1% eps_z)','Position',[100 100 900 900]);
subplot(4,2,1); plot(tt, oo_.irfs.y_eps_z(1:T));    title('Output y_t');
subplot(4,2,2); plot(tt, oo_.irfs.c_eps_z(1:T));    title('Consumption c_t');
subplot(4,2,3); plot(tt, oo_.irfs.i_eps_z(1:T));    title('Interest rate i_t');
subplot(4,2,4); plot(tt, oo_.irfs.pi_eps_z(1:T));   title('CPI inflation \pi_t');
subplot(4,2,5); plot(tt, oo_.irfs.ec_eps_z(1:T));   title('Depreciation e^c_t');
subplot(4,2,6); plot(tt, oo_.irfs.g_eps_z(1:T));    title('Shocked var: preference g_t');
subplot(4,2,7); plot(tt, oo_.irfs.e_level_eps_z(1:T)); title('Nominal FX level');
subplot(4,2,8); plot(tt, oo_.irfs.ca_eps_z(1:T));   title('Current account (y-c)');
print('-dpng','IRF_Q1b.png');

figure('Name','Q1(c) – Risk premium shock (1% eps_{tau})','Position',[100 100 900 900]);
subplot(4,2,1); plot(tt, oo_.irfs.y_eps_tau(1:T));  title('Output y_t');
subplot(4,2,2); plot(tt, oo_.irfs.c_eps_tau(1:T));  title('Consumption c_t');
subplot(4,2,3); plot(tt, oo_.irfs.i_eps_tau(1:T));  title('Interest rate i_t');
subplot(4,2,4); plot(tt, oo_.irfs.pi_eps_tau(1:T)); title('CPI inflation \pi_t');
subplot(4,2,5); plot(tt, oo_.irfs.ec_eps_tau(1:T)); title('Depreciation e^c_t');
subplot(4,2,6); plot(tt, oo_.irfs.tau_eps_tau(1:T));title('Shocked var: risk premium \tau_t');
subplot(4,2,7); plot(tt, oo_.irfs.e_level_eps_tau(1:T)); title('Nominal FX level');
subplot(4,2,8); plot(tt, oo_.irfs.ca_eps_tau(1:T)); title('Current account (y-c)');
print('-dpng','IRF_Q1c.png');

figure('Name','Q1(d) – Foreign interest-rate shock (1% eps_{i*})','Position',[100 100 900 900]);
subplot(4,2,1); plot(tt, oo_.irfs.y_eps_i_star(1:T));    title('Output y_t');
subplot(4,2,2); plot(tt, oo_.irfs.c_eps_i_star(1:T));    title('Consumption c_t');
subplot(4,2,3); plot(tt, oo_.irfs.i_eps_i_star(1:T));    title('Interest rate i_t');
subplot(4,2,4); plot(tt, oo_.irfs.pi_eps_i_star(1:T));   title('CPI inflation \pi_t');
subplot(4,2,5); plot(tt, oo_.irfs.ec_eps_i_star(1:T));   title('Depreciation e^c_t');
subplot(4,2,6); plot(tt, oo_.irfs.i_star_eps_i_star(1:T)); title('Shocked var: i^*_t');
subplot(4,2,7); plot(tt, oo_.irfs.e_level_eps_i_star(1:T)); title('Nominal FX level');
subplot(4,2,8); plot(tt, oo_.irfs.ca_eps_i_star(1:T));   title('Current account (y-c)');
print('-dpng','IRF_Q1d.png');
end;

