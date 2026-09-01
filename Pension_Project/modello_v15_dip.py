#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MODELLO v15 — RIFORMA DEI LAVORATORI DIPENDENTI  ***MODELLO PRINCIPALE***
================================================================================
La biforcazione contributiva si applica ai soli LAVORATORI DIPENDENTI: 17,5 mln
fra privati (FPLD) e pubblici (CTPS). Autonomi, gestione separata e casse
professionali restano nel sistema a ripartizione, invariati.

Il 33% IVS e' l'aliquota dei dipendenti: gli autonomi versano il 24-26% e le
casse professionali sono enti privatizzati fuori dal perimetro INPS. Estendere
loro la biforcazione richiederebbe un intervento legislativo separato su enti
con autonomia costituzionalmente protetta. La riforma resta percio' circoscritta
al comparto dove la leva normativa esiste gia': "rivoluzionare tutto insieme e'
politicamente impossibile".

COERENZA CONTABILE DEL SOTTO-SISTEMA (correzione 2026-07).
Non basta restringere i contributi: se la spesa e i pensionati restassero quelli
di sistema, si sottrarrebbero contributi a una platea e si misurerebbero i
risparmi su un'altra. Qui TUTTE le grandezze del perimetro sono coerenti:

  - spesa pensionistica del perimetro = 73,9% del totale (Itinerari Previdenziali,
    XIII Rapporto, dati 2024: dipendenti privati 150,9 + pubblici 93,6 = 244,5
    €mld su 330,7 di spesa pensionistica complessiva);
  - pensionati e flussi di pensionamento = 68,5% del totale (INPS 2025: FPLD
    352.632 + Gestione Dipendenti Pubblici 126.591 + fondi speciali 63.630 su
    792.998 nuove pensioni). La quota di TESTE e' piu' bassa della quota di
    SPESA perche' le pensioni dei dipendenti — soprattutto pubblici — sono
    mediamente piu' alte: ne risulta una pensione media di perimetro di ~1.555
    €/mese contro ~1.441 di sistema, come dev'essere;
  - safety net assistenziale ridotto alla stessa quota.

Le leve PARAMETRICHE (perequazione, reversibilita', eta' pensionabile, chiusura
deroghe, recupero evasione) restano invece di SISTEMA: sono misure generali che
non dipendono dalla creazione del pilastro a capitalizzazione. La spesa delle
gestioni non riformate rientra invariata nel totale, cosi' il rapporto spesa/PIL
resta confrontabile con il sentiero RGS.

VARIANTE DI CONFRONTO: modello_v15_totale.py estende la biforcazione a tutti i
lavoratori (23,5 mln). Serve a quantificare cosa si lascia sul tavolo restando
sui soli dipendenti, non come ipotesi di lavoro.

RENDIMENTO STOCASTICO. Oltre allo scenario deterministico il run esegue una
simulazione Monte Carlo sul rendimento reale dei fondi (i.i.d. normale) e
produce i fan chart: foglio "Stocastico" nell'Excel e figura
6_fan_chart_montecarlo.png.

================================================================================
"""

from modello_v15_core import main

if __name__ == "__main__":
    main(perimetro="dipendenti",
         out_default="modello_dinamico_v15_dip.xlsx",
         figure_dir="figure_v15_dip",
         etichetta="MODELLO v15 — RIFORMA DEI DIPENDENTI (FPLD+CTPS) — PRINCIPALE")
