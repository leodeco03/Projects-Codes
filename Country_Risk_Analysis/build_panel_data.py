#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_panel_data.py
===================
Costruisce il dataset panel per l'analisi "Spread -> Snowball effect"
estesa al Sud Europa (Italia, Grecia, Portogallo, Spagna).

Fonte: CSV AMECO già scaricati in "../Dati EU2025/".

Serie estratte (codici AMECO):
  ADGGI  (1.0.0.0)    Snowball effect su debito lordo, % PIL  -> sb
  UDGG   (1.0.319.0)  Debito lordo consolidato PA, % PIL      -> debt
  UBLGAP (1.0.319.0)  Saldo PA aggiustato per il ciclo,
                      % PIL potenziale ("fiscal stance")      -> pb
  UBLGI  (1.0.319.0)  Saldo primario (netto interessi), % PIL -> pbal
  ILN    (1.1.0.0)    Tasso nominale a lungo termine (10y), % -> iln
  OVGD   (1.1.0.0)    PIL reale (livelli 2020)                -> g   (var. %)
  PCPH   (3.1.0.0)    Deflatore consumi privati (2020=100)    -> pie (var. %)

Spread: iln(paese) - iln(Germania), punti percentuali.

Output: panel_south_europe.csv (formato long: una riga per paese-anno).
Anni 2025-2026 esclusi (previsioni AMECO, non dati osservati).
"""

import csv
import os

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'Dati EU2025')
OUT_FILE = os.path.join(HERE, 'panel_south_europe.csv')

COUNTRIES  = ['ITA', 'GRC', 'PRT', 'ESP']   # panel
BENCHMARK  = 'DEU'                          # per lo spread
YEAR_MIN   = 1994                           # un anno extra per calcolare g e pie nel 1995
YEAR_MAX   = 2024                           # 2025-26 = previsioni -> escluse

# (nome_variabile, file, suffisso SERIES dopo il codice paese)
SPECS = [
    ('sb',   'Altri dati ameco_CSV/GOV_tot_snowball.CSV', '1.0.0.0.ADGGI'),
    ('debt', 'Altri dati ameco_CSV/GOV_tot_snowball.CSV', '1.0.319.0.UDGG'),
    ('pb',   'Lend_borr_gov.CSV',                         '1.0.319.0.UBLGAP'),
    ('pbal', 'Government_spending.CSV',                   '1.0.319.0.UBLGI'),
    ('iln',  'i_rate_ex_rate.CSV',                        '1.1.0.0.ILN'),
    ('rgdp', 'GDP.CSV',                                   '1.1.0.0.OVGD'),
    ('defl', 'Consumptio_and _CPI.CSV',                   '3.1.0.0.PCPH'),
]


def get_series(path, country, suffix):
    """Estrae {anno: valore} per una riga SERIES = '<country>.<suffix>'."""
    target = f'{country}.{suffix}'
    with open(path, encoding='utf-8', errors='replace') as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        years = hdr[11:]
        for row in rdr:
            if row and row[0].strip() == target:
                out = {}
                for y, v in zip(years, row[11:]):
                    v = v.strip()
                    if v not in ('NA', '', 'NaN'):
                        try:
                            out[int(y)] = float(v)
                        except ValueError:
                            pass
                return out
    raise ValueError(f'Serie non trovata: {target} in {os.path.basename(path)}')


def pct_change(series):
    """Variazione percentuale anno su anno: 100*(x_t/x_{t-1} - 1)."""
    return {y: 100.0 * (v / series[y - 1] - 1.0)
            for y, v in series.items() if (y - 1) in series and series[y - 1] != 0}


def main():
    # --- carica tutte le serie -------------------------------------------
    raw = {}
    for c in COUNTRIES + [BENCHMARK]:
        raw[c] = {}
        for name, fname, suffix in SPECS:
            raw[c][name] = get_series(os.path.join(DATA_DIR, fname), c, suffix)

    iln_deu = raw[BENCHMARK]['iln']

    # --- costruisci panel long -------------------------------------------
    rows = []
    for c in COUNTRIES:
        g   = pct_change(raw[c]['rgdp'])   # crescita PIL reale, %
        pie = pct_change(raw[c]['defl'])   # inflazione (deflatore consumi), %
        for y in range(YEAR_MIN + 1, YEAR_MAX + 1):
            def v(name, src=None):
                d = src if src is not None else raw[c][name]
                return d.get(y, '')
            spread = ''
            if y in raw[c]['iln'] and y in iln_deu:
                spread = raw[c]['iln'][y] - iln_deu[y]
            rows.append({
                'country': c,
                'year':    y,
                'sb':      v('sb'),
                'spread':  spread,
                'iln':     v('iln'),
                'iln_deu': iln_deu.get(y, ''),
                'debt':    v('debt'),
                'pb':      v('pb'),
                'pbal':    v('pbal'),
                'g':       g.get(y, ''),
                'pie':     pie.get(y, ''),
            })

    cols = ['country', 'year', 'sb', 'spread', 'iln', 'iln_deu',
            'debt', 'pb', 'pbal', 'g', 'pie']
    with open(OUT_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            for k in cols:
                if isinstance(r[k], float):
                    r[k] = f'{r[k]:.4f}'
            w.writerow(r)

    # --- report di copertura ---------------------------------------------
    print(f'Scritto: {OUT_FILE}  ({len(rows)} righe paese-anno)')
    core = ['sb', 'spread', 'debt', 'pb', 'g', 'pie']
    for c in COUNTRIES:
        sub = [r for r in rows if r['country'] == c
               and all(r[k] != '' for k in core)]
        yrs = [int(r['year']) for r in sub]
        print(f'  {c}: osservazioni complete {min(yrs)}-{max(yrs)} (n={len(yrs)})')
    n_pre2019 = sum(1 for r in rows
                    if 1996 <= int(r['year']) <= 2019
                    and all(r[k] != '' for k in core))
    print(f'  Totale obs complete 1996-2019 (campione paper): {n_pre2019}')


if __name__ == '__main__':
    main()
