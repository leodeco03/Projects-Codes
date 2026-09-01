#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modello a due generazioni — Mario e Luigi.

SCOPO. Il modello v15 e' aggregato: proietta miliardi, quote di PIL e coorti.
Questo file fa la cosa opposta e complementare: segue DUE persone, una per
generazione, e mostra dove va ogni punto di contribuzione. Serve a rendere
visibile il meccanismo che l'aggregato nasconde — perche' la riforma costa
subito e rende dopo, e chi paga quel disallineamento.

I DUE AGENTI
    MARIO  ha 50 anni nell'anno di avvio. Supera la soglia di adesione, quindi
           resta interamente nella ripartizione: versa, va in pensione e
           incassa una pensione pubblica piena. Non e' toccato dalla riforma.
    LUIGI  ha 25 anni nell'anno di avvio. Entra subito nella biforcazione: una
           parte crescente della sua aliquota smette di finanziare la pensione
           di Mario e va sul suo conto individuale.

L'ECONOMIA IN UNA RIGA. In un sistema a ripartizione i contributi di Luigi
pagano la pensione di Mario. Se una parte di quei contributi viene dirottata su
un conto individuale, la pensione di Mario va comunque pagata: lo Stato deve
trovare altrove i punti mancanti. Quello e' il costo di transizione, e questo
file lo misura per una singola coppia di individui.

RAPPORTO CON IL v15. I parametri sono importati dal modello aggregato, non
reinventati: aliquota, fasi di biforcazione, eta' pensionabile, coefficienti di
trasformazione, rendimenti, regola di ricalibrazione e pilastro zero sono gli
stessi. Cambia solo l'unita' di osservazione: qui una persona, li' una coorte.
Il confronto fra i due livelli e' quindi un controllo di coerenza, non un
esercizio separato.

Uso:  python3 modello_due_agenti.py
      python3 modello_due_agenti.py --eta-mario 50 --eta-luigi 25
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np

import modello_v15_core as core
from modello_v15_core import Params


# ------------------------------------------------------------------------------
# 1. UN AGENTE
# ------------------------------------------------------------------------------

@dataclasses.dataclass
class Agente:
    """Una persona seguita dall'ingresso al lavoro alla morte."""
    nome: str
    eta0: int                      # eta' nell'anno di avvio della riforma
    p: Params

    # --- anagrafica ------------------------------------------------------------
    @property
    def anno_nascita(self) -> int:
        return self.p.anno0 - self.eta0

    @property
    def anno_pens(self) -> int:
        """Anno di pensionamento: l'eta' pensionabile e' dinamica, quindi si
        cerca il primo anno in cui l'eta' raggiunta soddisfa il requisito."""
        for y in range(self.p.anno0, self.p.anno0 + 120):
            if y - self.anno_nascita >= core.eta_pensionabile(self.p, y):
                return y
        raise RuntimeError("nessun anno di pensionamento trovato")

    @property
    def eta_pens(self) -> int:
        return self.anno_pens - self.anno_nascita

    @property
    def anno_inizio(self) -> int:
        """Inizio carriera: tanti anni prima della pensione quanti la carriera."""
        return self.anno_pens - core.carriera_pens(self.p, self.anno_pens) + 1

    @property
    def biforca(self) -> bool:
        """Entra nella riforma solo chi rispetta la soglia di eta'."""
        return self.eta0 <= self.p.part_t0_eta

    def salario(self, anno: int) -> float:
        """Retribuzione dell'anno. Il livello di partenza della coorte cresce
        con l'anno di inizio carriera (stessa regola del modello aggregato)."""
        p = self.p
        w_iniziale = p.w0 * (1 + p.g_w) ** (self.anno_inizio - p.anno0)
        return w_iniziale * (1 + p.g_w) ** (anno - self.anno_inizio)

    def quota_biforcata(self, anno: int) -> tuple[float, float]:
        """(quota totale dirottata, di cui in titoli dedicati) nell'anno."""
        if not self.biforca or anno < self.p.anno0:
            return 0.0, 0.0
        t = anno - self.p.anno0
        if t >= self.p.anni:
            t = self.p.anni - 1
        return core.quota_bif(self.p, t)


# ------------------------------------------------------------------------------
# 2. LA CARRIERA DI UN AGENTE, ANNO PER ANNO
# ------------------------------------------------------------------------------

def carriera(a: Agente) -> dict:
    """Ricostruisce i flussi di un agente lungo tutta la vita lavorativa.

    Per ogni anno registra: retribuzione, punti versati alla ripartizione, punti
    dirottati al conto individuale, e le due componenti del montante — quello
    figurativo (che genera la pensione pubblica) e quello reale (che genera la
    rendita). Il primo si rivaluta alla crescita dell'economia, il secondo al
    rendimento dei mercati: la differenza fra i due tassi e' la ragione
    economica dell'intera riforma.
    """
    p = a.p
    anni = list(range(a.anno_inizio, a.anno_pens + 1))
    righe = []
    mont_fig_sq = mont_fig_rif = mont_fondi = mont_bond = 0.0
    for anno in anni:
        w = a.salario(anno)
        q_tot, q_bond = a.quota_biforcata(anno)
        q_fondi = q_tot - q_bond
        n = a.anno_pens - anno                     # anni di rivalutazione residui
        # montante figurativo: senza riforma tutta l'aliquota, con riforma quella
        # al netto dei punti dirottati. Rivalutazione alla crescita del PIL.
        mont_fig_sq += p.aliquota * w * (1 + p.g) ** n
        mont_fig_rif += (p.aliquota - q_tot) * w * (1 + p.g) ** n
        # montante reale: rendimento di mercato per i fondi, cedola per i titoli
        mont_fondi += q_fondi * w * (1 + p.rf) ** n
        mont_bond += q_bond * w * (1 + p.r_brdp) ** n
        righe.append(dict(anno=anno, eta=anno - a.anno_nascita, salario=w,
                          q_ripartizione=p.aliquota - q_tot, q_conto=q_tot,
                          q_fondi=q_fondi, q_bond=q_bond,
                          versato_ripartizione=(p.aliquota - q_tot) * w,
                          versato_conto=q_tot * w))
    return dict(agente=a, righe=righe,
                mont_fig_sq=mont_fig_sq, mont_fig_rif=mont_fig_rif,
                mont_fondi=mont_fondi, mont_bond=mont_bond,
                montante_reale=mont_fondi + mont_bond)


def pensione(a: Agente, c: dict, kappa: float) -> dict:
    """Converte i montanti in pensione mensile, applicando le stesse regole del
    modello aggregato: coefficiente di trasformazione all'eta' effettiva,
    conversione a due teste per la rendita, ricalibrazione della pensione
    pubblica con clausola di salvaguardia, pilastro zero con prova dei mezzi."""
    p = a.p
    cf = core.coeff_eta(p, a.eta_pens)
    cf2 = core.coeff_reversibile(p, cf)
    # kappa calibra la sola pensione PUBBLICA: serve a riportare il montante
    # figurativo di una carriera teorica sui livelli di pensione osservati, che
    # riflettono carriere reali piu' brevi e discontinue. Il montante REALE non
    # va scalato: e' denaro effettivamente versato e capitalizzato, e la rendita
    # che ne discende e' quella attuarialmente dovuta. Applicare kappa anche a
    # P2 comprimerebbe la rendita di un terzo senza alcuna ragione economica —
    # ed e' l'errore che il confronto con il modello aggregato ha fatto emergere.
    P1_sq = c["mont_fig_sq"] * cf / p.mensilita * kappa
    P1_ndc = c["mont_fig_rif"] * cf / p.mensilita * kappa
    P2 = c["montante_reale"] * cf2 / p.mensilita
    # ricalibrazione: prelievo sulla rendita, con salvaguardia
    P1 = max(0.0, P1_ndc - p.clawback_p2 * P2)
    if p.clawback_floor is not None:
        P1 = min(P1_ndc, max(P1, P1_sq + p.clawback_floor - P2))
    taper = p.p0_base / p.p0_cap
    P0 = max(0.0, p.p0_base - taper * (P1 + P2)) if a.biforca else 0.0
    return dict(P1_sq=P1_sq, P1_ndc=P1_ndc, P1=P1, P2=P2, P0=P0,
                totale=P1 + P2 + P0, cf=cf, cf2=cf2)


# ------------------------------------------------------------------------------
# 3. IL CONTO DELLO STATO
# ------------------------------------------------------------------------------

def conto_stato(mario: dict, luigi: dict, pens_m: dict, pens_l: dict,
                p: Params) -> dict:
    """Il bilancio pubblico visto attraverso i due agenti.

    IDEA. Finche' Luigi lavora e Mario e' in pensione, i contributi di Luigi
    finanziano la pensione di Mario. La riforma sottrae a quel finanziamento i
    punti dirottati: ogni anno lo Stato incassa meno senza pagare di meno,
    perche' la pensione di Mario e' gia' maturata e non si tocca. Il buco
    cumulato, capitalizzato al costo del debito, e' il costo di transizione.
    Quando Luigi va in pensione lo Stato inizia a rientrare, perche' gli paga
    una pensione pubblica piu' bassa di quella che gli avrebbe dovuto.
    """
    anno_p_m = mario["agente"].anno_pens
    anno_p_l = luigi["agente"].anno_pens
    # orizzonte: dall'avvio fino all'estinzione di Luigi
    fine = anno_p_l + int(2 * p.durata_media_pensione)
    righe = []
    debito = 0.0
    Hor = 2 * p.durata_media_pensione
    for anno in range(p.anno0, fine + 1):
        # --- contributi mancanti alla ripartizione per la biforcazione di Luigi
        mancato = 0.0
        for r in luigi["righe"]:
            if r["anno"] == anno:
                mancato = r["versato_conto"]
        # --- IRPEF incassata subito sui contributi dirottati (regime TEE)
        tee = mancato * p.irpef_contributi if p.regime_p2 == "TEE" else 0.0
        # --- minore pensione pubblica pagata a Luigi, quando e' in quiescenza
        risparmio = 0.0
        if anno >= anno_p_l:
            viva = max(0.0, 1 - (anno - anno_p_l) / Hor)
            risparmio = ((pens_l["P1_sq"] - pens_l["P1"]) * p.mensilita * viva)
            # il pilastro zero e' un costo che nello status quo non c'era
            risparmio -= pens_l["P0"] * p.mensilita * viva
        saldo = -mancato + tee + risparmio
        debito = debito * (1 + p.r_debito) - saldo
        righe.append(dict(anno=anno, mancato=mancato, tee=tee,
                          risparmio=risparmio, saldo=saldo, debito=debito))
    return dict(righe=righe, anno_pens_mario=anno_p_m, anno_pens_luigi=anno_p_l)


# ------------------------------------------------------------------------------
# 4. STAMPA
# ------------------------------------------------------------------------------

def nf(x, d=0):
    s = f"{x:,.{d}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def stampa(p: Params, eta_mario: int, eta_luigi: int) -> dict:
    kappa = core.run_v15(p)["kappa_c"]
    mario = Agente("Mario", eta_mario, p)
    luigi = Agente("Luigi", eta_luigi, p)
    cm, cl = carriera(mario), carriera(luigi)
    pm, pl = pensione(mario, cm, kappa), pensione(luigi, cl, kappa)
    st = conto_stato(cm, cl, pm, pl, p)

    print("=" * 78)
    print("  MODELLO A DUE GENERAZIONI — MARIO E LUIGI")
    print("=" * 78)
    for a, c, pe in ((mario, cm, pm), (luigi, cl, pl)):
        print(f"\n{a.nome.upper()} — {a.eta0} anni nel {p.anno0} "
              f"(nato nel {a.anno_nascita})")
        print(f"  carriera            {a.anno_inizio}–{a.anno_pens} "
              f"({len(c['righe'])} anni), pensione a {a.eta_pens} anni")
        print(f"  entra nella riforma {'SI' if a.biforca else 'NO — resta interamente nella ripartizione'}")
        print(f"  salario iniziale    {nf(a.salario(a.anno_inizio))} € → "
              f"finale {nf(a.salario(a.anno_pens))} €")
        if a.biforca:
            q0 = a.quota_biforcata(p.anno0)[0]
            qf = a.quota_biforcata(a.anno_pens)[0]
            print(f"  quota dirottata     dal {q0:.0%} al {qf:.0%} della retribuzione")
        print(f"  montante figurativo {nf(c['mont_fig_sq'])} € senza riforma, "
              f"{nf(c['mont_fig_rif'])} € con riforma")
        print(f"  montante reale      {nf(c['montante_reale'])} € "
              f"(fondi {nf(c['mont_fondi'])}, titoli {nf(c['mont_bond'])})")
        print(f"  PENSIONE            pubblica {nf(pe['P1'])} + rendita {nf(pe['P2'])}"
              f" + pilastro zero {nf(pe['P0'])} = {nf(pe['totale'])} €/mese")
        print(f"                      senza riforma avrebbe preso {nf(pe['P1_sq'])} €/mese"
              f"  →  {(pe['totale']/pe['P1_sq']-1)*100:+.1f}%")

    print("\n" + "-" * 78)
    print("  IL CONTO DELLO STATO (euro per la singola coppia)")
    print("-" * 78)
    r = st["righe"]
    tot_mancato = sum(x["mancato"] for x in r)
    tot_tee = sum(x["tee"] for x in r)
    tot_risp = sum(x["risparmio"] for x in r)
    print(f"  contributi che Luigi non versa piu' alla ripartizione  "
          f"{nf(tot_mancato):>12} €")
    print(f"  imposta incassata subito su quei contributi (TEE)      "
          f"{nf(tot_tee):>12} €")
    print(f"  minore pensione pubblica pagata a Luigi                "
          f"{nf(tot_risp):>12} €")
    print(f"  {'saldo non attualizzato':<53} {nf(tot_tee + tot_risp - tot_mancato):>12} €")
    picco = max(r, key=lambda x: x["debito"])
    print(f"\n  debito massimo         {nf(picco['debito']):>12} € nel {picco['anno']}")
    print(f"  debito residuo a fine  {nf(r[-1]['debito']):>12} € nel {r[-1]['anno']}")
    pos = [x for x in r if x["debito"] <= 0 and x["anno"] > st["anno_pens_luigi"]]
    print(f"  il debito si estingue  "
          f"{('nel ' + str(pos[0]['anno'])) if pos else 'oltre l orizzonte della coppia'}")

    print("\n  Anni chiave:")
    for anno in (p.anno0, st["anno_pens_mario"], st["anno_pens_luigi"],
                 st["anno_pens_luigi"] + 10):
        x = next((v for v in r if v["anno"] == anno), None)
        if x:
            ev = ("avvio" if anno == p.anno0 else
                  "Mario va in pensione" if anno == st["anno_pens_mario"] else
                  "Luigi va in pensione" if anno == st["anno_pens_luigi"] else "")
            print(f"    {anno}  saldo {nf(x['saldo']):>9} €   debito cumulato "
                  f"{nf(x['debito']):>10} €   {ev}")
    return dict(mario=cm, luigi=cl, pens_mario=pm, pens_luigi=pl, stato=st)


def verifica_coerenza(p: Params, etas=(25, 30, 35, 40, 45)) -> list[tuple]:
    """Controllo incrociato: la pensione di un agente calcolata individualmente
    deve coincidere con quella della sua coorte nel modello aggregato.

    E' il test che da' valore a questo file. Le due vie sono indipendenti — qui
    si sommano i versamenti di una persona, li' si distribuisce un flusso
    aggregato fra le coorti — quindi la coincidenza non e' garantita da alcuna
    identita': se i due livelli divergono, uno dei due sbaglia. E' cosi' che e'
    emerso che il fattore di calibrazione non va applicato alla rendita.
    """
    d = core.run_v15(p)
    esiti = []
    for eta in etas:
        a = Agente("X", eta, p)
        i = a.anno_pens - p.anno0
        if not (0 <= i < p.anni):
            continue
        c = carriera(a)
        pe = pensione(a, c, d["kappa_c"])
        tot15 = d["pensione_new"][i]
        scarto = pe["totale"] / tot15 - 1
        esiti.append((f"pensione a {eta} anni nel {p.anno0} (pens. {a.anno_pens})",
                      abs(scarto) < 0.02,
                      f"{pe['totale']:,.0f} vs {tot15:,.0f} €/mese "
                      f"({scarto:+.1%})"))
    return esiti


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta-mario", type=int, default=50)
    ap.add_argument("--eta-luigi", type=int, default=25)
    args = ap.parse_args()
    p = Params()
    stampa(p, args.eta_mario, args.eta_luigi)
    print("\n" + "-" * 78)
    print("  COERENZA CON IL MODELLO AGGREGATO")
    print("-" * 78)
    for nome, ok, det in verifica_coerenza(p):
        print(f"    [{'ok' if ok else '!!'}] {nome:44s} {det}")


if __name__ == "__main__":
    main()
