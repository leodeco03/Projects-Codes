%% %% GFC - NK closed model (Q3_c)
%
% DESCRIPTION: dynare.mod file to solve and simulate the  Final project of
% the course: "International Money and Finance"
%
% Suasday Chea and Leonardo Decorte
% Created: 10/10/2025
%
%

//------------------------
// 1) Variables
//------------------------
var c y ii pie piH piF S q mc a ec e_level ca      // domestic
    y_star pie_star i_star                        // foreign (AR(1))
    phi_hat eps_z epsH;
    
varexo eps_m eta_phi_hat eta_z eta_H eta_y_star eta_pie_star eta_i_star;


//------------------------
// 2) Parameters 
//------------------------
parameters sigma alpha etta betta thetaH chi varphi h gamma deltaH
          rho_i psi_pie psi_y psi_dy psi_e
          rho_phi_hat rho_z rho_H
          rho_y_star rho_pie_star rho_i_star
          kappaH;

sigma   = 1;      // inverse intertemporal elasticity of substitution
alpha   = 0.24;   // openness
etta    = 0.85;   // elasticity of substitution (dom/import)
betta   = 0.99;
thetaH  = 0.75;   // Calvo prices (domestic)
chi     = 0.01;   // coeff. risk premia 
varphi  = 1.26;   // Frisch^-1
h       = 0.25;   // habit
gamma   = 0.15;   // consumption elast. of preference shock
deltaH  = 0.30;   // price indexation

rho_i   = 0.70;   // smoothing della Taylor rule
psi_pie  = 1.90;  // response to inflation 
psi_y   = 0.05;   // response to output level
psi_dy  = 0.55;   // response to output change
psi_e   = 0.85;   // reponse to exchange rate fluctuations

rho_phi_hat = 0.70;
rho_z       = 0.70;
rho_H       = 0.70;

rho_y_star  = 0.70;
rho_pie_star = 0.70;
rho_i_star  = 0.70;

// composed coefficents
kappaH = (1-thetaH)*(1-betta*thetaH)/thetaH;    // p-curve slope

//------------------------
// 3) Modello (linearizzato)
//------------------------
model(linear);

    // (1) IS with habit and preference shock 
    c = (h/(1+h))*c(-1) + (1/(1+h))*c(+1) - (1/sigma)*((1-h)/(1+h))*( ii - pie(+1) ) + gamma*eps_z;

    // (2) Goods Market clearing
    y = (1-alpha)*c + alpha*y_star + alpha*etta*(2-alpha)*S;

    // (3) Link between Terms of Trade and Real exchange rate
    q = (1-alpha)*S;

    // (4) Chages of terms of trade
    S - S(-1) = piF - piH;

    // (5) real marginal cost
    mc = varphi*y + alpha*S + sigma*c;

    // (6) Phillips curve - domestic price inflation 
    ( piH - deltaH*piH(-1) ) = betta*( piH(+1) - deltaH*piH ) + kappaH*mc + epsH;

    // (7) CPI vs PPI
    pie = piH + alpha*( S - S(-1) );

    // (8) UIP 
    ii - i_star = ec(+1) - chi*(a + rho_phi_hat*phi_hat);

    // (9) net foreign asset position - C.A.
    y - c = a - (1/betta)*a(-1) + (alpha/(1-alpha))*q;

    // (10) Imported good inflation (LoP)
    piF = ec + pie_star;

    // (11) Monetary policy Rule
    ii = rho_i*ii(-1) + psi_pie*pie + psi_y*y + psi_dy*(y - y(-1)) + psi_e*ec - eps_m;

    // (12) risk premium shock AR(1)
    phi_hat = rho_phi_hat*phi_hat(-1) + eta_phi_hat;

    // (13) preference shock AR(1)
    eps_z   = rho_z*eps_z(-1) + eta_z;

    // (14) Cost-push shock AR(1)
    epsH = rho_H*epsH(-1) + eta_H;

    // (16.A)–(18.A) Estero: AR(1)
    y_star  = rho_y_star*y_star(-1)   + eta_y_star;
    pie_star = rho_pie_star*pie_star(-1) + eta_pie_star;
    i_star  = rho_i_star*i_star(-1)   + eta_i_star;

    // Book-keeping: 
    e_level = e_level(-1) + ec; // exchange rate level
    ca      = y - c;            // C.A.

end;


//------------------------
// 4) Shock (IRF a 1%)
//------------------------
shocks;
    var eps_m;          stderr 0.00;
    var eta_z;          stderr 0.03;
    var eta_phi_hat;    stderr 0.00;
    var eta_i_star;     stderr 0.00;
    var eta_y_star;     stderr 0.05;
    var eta_pie_star;   stderr 0.00;
    var eta_H;          stderr 0.00;
end;

//------------------------
// 5) IRF per Q1 (12 periodi) + figure 4×2
//------------------------

matlab;
T  = 12;
tt = 0:T;                                % x-axis labels t=0..T
mk = @(v,T) [0; reshape(v(1:T),[],1)];   % prepend 0, force column

stoch_simul(order=1, irf=12, irf_shocks=(eta_y_star, eta_z), nograph) y c ii pie ec y_star e_level ca;

% Build "negative" IRFs
oo_.irfs.y_eta_y_star_neg        = -oo_.irfs.y_eta_y_star;
oo_.irfs.y_eta_z_neg             = -oo_.irfs.y_eta_z;

oo_.irfs.c_eta_y_star_neg        = -oo_.irfs.c_eta_y_star;
oo_.irfs.c_eta_z_neg             = -oo_.irfs.c_eta_z;

oo_.irfs.ii_eta_y_star_neg       = -oo_.irfs.ii_eta_y_star;
oo_.irfs.ii_eta_z_neg            = -oo_.irfs.ii_eta_z;

oo_.irfs.pie_eta_y_star_neg      = -oo_.irfs.pie_eta_y_star;
oo_.irfs.pie_eta_z_neg           = -oo_.irfs.pie_eta_z;

oo_.irfs.ec_eta_y_star_neg       = -oo_.irfs.ec_eta_y_star;
oo_.irfs.ec_eta_z_neg            = -oo_.irfs.ec_eta_z;

oo_.irfs.y_star_eta_y_star_neg   = -oo_.irfs.y_star_eta_y_star;
oo_.irfs.y_star_eta_z_neg        = -oo_.irfs.y_star_eta_z;

oo_.irfs.e_level_eta_y_star_neg  = -oo_.irfs.e_level_eta_y_star;
oo_.irfs.e_level_eta_z_neg       = -oo_.irfs.e_level_eta_z;

oo_.irfs.ca_eta_y_star_neg       = -oo_.irfs.ca_eta_y_star;
oo_.irfs.ca_eta_z_neg            = -oo_.irfs.ca_eta_z;

% Linear combo (two negative shocks together)
oo_.irfs.y_combined        = oo_.irfs.y_eta_y_star_neg        + oo_.irfs.y_eta_z_neg;
oo_.irfs.c_combined        = oo_.irfs.c_eta_y_star_neg        + oo_.irfs.c_eta_z_neg;
oo_.irfs.ii_combined       = oo_.irfs.ii_eta_y_star_neg       + oo_.irfs.ii_eta_z_neg;
oo_.irfs.pie_combined      = oo_.irfs.pie_eta_y_star_neg      + oo_.irfs.pie_eta_z_neg;
oo_.irfs.ec_combined       = oo_.irfs.ec_eta_y_star_neg       + oo_.irfs.ec_eta_z_neg;
oo_.irfs.y_star_combined   = oo_.irfs.y_star_eta_y_star_neg   + oo_.irfs.y_star_eta_z_neg;
oo_.irfs.e_level_combined  = oo_.irfs.e_level_eta_y_star_neg  + oo_.irfs.e_level_eta_z_neg;
oo_.irfs.ca_combined       = oo_.irfs.ca_eta_y_star_neg       + oo_.irfs.ca_eta_z_neg;

% Single negative shock: eta_z
figure('Name','Negative Shock eta_z (-0.03), psi_e = 0.80','NumberTitle','off');
subplot(4,2,1); plot(tt, mk(oo_.irfs.y_eta_z_neg, T));         yline(0,'r','LineWidth',1); title('y');       grid on
subplot(4,2,2); plot(tt, mk(oo_.irfs.c_eta_z_neg, T));         yline(0,'r','LineWidth',1); title('c');       grid on
subplot(4,2,3); plot(tt, mk(oo_.irfs.ii_eta_z_neg, T));        yline(0,'r','LineWidth',1); title('i');       grid on
subplot(4,2,4); plot(tt, mk(oo_.irfs.pie_eta_z_neg, T));       yline(0,'r','LineWidth',1); title('\pi');     grid on
subplot(4,2,5); plot(tt, mk(oo_.irfs.ec_eta_z_neg, T));        yline(0,'r','LineWidth',1); title('e^c');     grid on
subplot(4,2,6); plot(tt, mk(oo_.irfs.y_star_eta_z_neg, T));    yline(0,'r','LineWidth',1); title('y^*');     grid on
subplot(4,2,7); plot(tt, mk(oo_.irfs.e_level_eta_z_neg, T));   yline(0,'r','LineWidth',1); title('e\_level'); grid on
subplot(4,2,8); plot(tt, mk(oo_.irfs.ca_eta_z_neg, T));        yline(0,'r','LineWidth',1); title('ca');      grid on

% Single negative shock: eta_y* 
figure('Name','Negative Shock eta_y* (-0.05), psi_e = 0.80','NumberTitle','off');
subplot(4,2,1); plot(tt, mk(oo_.irfs.y_eta_y_star_neg, T));        yline(0,'r','LineWidth',1); title('y');       grid on
subplot(4,2,2); plot(tt, mk(oo_.irfs.c_eta_y_star_neg, T));        yline(0,'r','LineWidth',1); title('c');       grid on
subplot(4,2,3); plot(tt, mk(oo_.irfs.ii_eta_y_star_neg, T));       yline(0,'r','LineWidth',1); title('i');       grid on
subplot(4,2,4); plot(tt, mk(oo_.irfs.pie_eta_y_star_neg, T));      yline(0,'r','LineWidth',1); title('\pi');     grid on
subplot(4,2,5); plot(tt, mk(oo_.irfs.ec_eta_y_star_neg, T));       yline(0,'r','LineWidth',1); title('e^c');     grid on
subplot(4,2,6); plot(tt, mk(oo_.irfs.y_star_eta_y_star_neg, T));   yline(0,'r','LineWidth',1); title('y^*');     grid on
subplot(4,2,7); plot(tt, mk(oo_.irfs.e_level_eta_y_star_neg, T));  yline(0,'r','LineWidth',1); title('e\_level'); grid on
subplot(4,2,8); plot(tt, mk(oo_.irfs.ca_eta_y_star_neg, T));       yline(0,'r','LineWidth',1); title('ca');      grid on

% Combined shocks
figure('Name','Combined (eta_y*  -0.05, eta_z -0.03), psi_e = 0.80','NumberTitle','off');
subplot(4,2,1); plot(tt, mk(oo_.irfs.y_combined, T));        yline(0,'r','LineWidth',1); title('y (combined)');       grid on
subplot(4,2,2); plot(tt, mk(oo_.irfs.c_combined, T));        yline(0,'r','LineWidth',1); title('c (combined)');       grid on
subplot(4,2,3); plot(tt, mk(oo_.irfs.ii_combined, T));       yline(0,'r','LineWidth',1); title('i (combined)');       grid on
subplot(4,2,4); plot(tt, mk(oo_.irfs.pie_combined, T));      yline(0,'r','LineWidth',1); title('\pi (combined)');     grid on
subplot(4,2,5); plot(tt, mk(oo_.irfs.ec_combined, T));       yline(0,'r','LineWidth',1); title('e^c (combined)');     grid on
subplot(4,2,6); plot(tt, mk(oo_.irfs.y_star_combined, T));   yline(0,'r','LineWidth',1); title('y^* (combined)');     grid on
subplot(4,2,7); plot(tt, mk(oo_.irfs.e_level_combined, T));  yline(0,'r','LineWidth',1); title('e\_level (combined)'); grid on
subplot(4,2,8); plot(tt, mk(oo_.irfs.ca_combined, T));       yline(0,'r','LineWidth',1); title('ca (combined)');      grid on
