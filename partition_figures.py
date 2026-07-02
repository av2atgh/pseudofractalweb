#!/usr/bin/env python3
"""
Draw the pseudofractal (DGM) web colour-coded by the two partitions that the
paper contrasts:

  * the recursive BRANCH cut  -- favoured by the degree-corrected fork (Fork 2),
    and the closed-form partition whose evidence both forks evaluate;
  * the HUB-LEAF (core-periphery) cut -- favoured by the plain Bernoulli SBM
    (Fork 1), obtained by thresholding on degree.

Outputs two figures: partition_branch.pdf/png and partition_hubleaf.pdf/png.

Requires: networkx, matplotlib, mpmath.
Run:  python3 partition_figures.py [T]      (default generation T=4, N=123)
"""
import sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpmath import mp, mpf, loggamma, log as mlog

mp.dps = 60

# ------------------------------------------------------------------ web + labels
def build_dgm(T):
    """Pseudofractal web after T generations; also return the branch label of
    every vertex (0/1/2 for the three descent branches, -1 for the seed hubs)."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    branch = {0: -1, 1: -1, 2: -1}
    eb = {(0, 1): 0, (1, 0): 0, (1, 2): 1, (2, 1): 1, (0, 2): 2, (2, 0): 2}
    nxt = 3
    for _ in range(T):
        for (u, v) in list(G.edges()):
            b = eb[(u, v)]
            w = nxt; nxt += 1
            G.add_edge(w, u); G.add_edge(w, v)
            branch[w] = b
            for e in ((w, u), (u, w), (w, v), (v, w)):
                eb[e] = b
    return G, branch


# ------------------------------------------------------------------ evidence (for annotation)
def _lB(a, b):
    return loggamma(a) + loggamma(b) - loggamma(a + b)


def block_counts(G, part):
    nodes = list(G.nodes()); k = dict(G.degree())
    n1 = sum(part[v] == 0 for v in nodes); n2 = len(nodes) - n1
    e1 = e2 = e12 = 0
    for u, v in G.edges():
        if part[u] == part[v]:
            e1 += part[u] == 0; e2 += part[u] == 1
        else:
            e12 += 1
    k1 = sum(k[v] for v in nodes if part[v] == 0)
    k2 = sum(k[v] for v in nodes if part[v] == 1)
    return n1, n2, e1, e2, e12, k1, k2


def logR_plain(G, part, a=1.0):
    n1, n2, e1, e2, e12, _, _ = block_counts(G, part)
    a = mpf(a); N = n1 + n2
    m1 = n1 * (n1 - 1) // 2; m2 = n2 * (n2 - 1) // 2
    m12 = n1 * n2; M = N * (N - 1) // 2; E = e1 + e2 + e12
    num = (_lB(e1 + a, m1 - e1 + a) + _lB(e2 + a, m2 - e2 + a)
           + _lB(e12 + a, m12 - e12 + a) + _lB(n1 + a, n2 + a))
    den = 2 * _lB(a, a) + _lB(E + a, M - E + a) + _lB(a, N + a)
    return float((num - den).real)


def logR_dc(G, part, a=1.0):
    _, _, e1, e2, e12, k1, k2 = block_counts(G, part)
    a = mpf(a); m = e1 + e2 + e12; tm = k1 + k2
    O11 = mpf(k1 * k1) / (2 * tm); O22 = mpf(k2 * k2) / (2 * tm)
    O12 = mpf(k1 * k2) / tm; Ot = O11 + O22 + O12
    lZ = lambda e, O: loggamma(e + a) - (e + a) * mlog(O + a)
    return float((lZ(e1, O11) + lZ(e2, O22) + lZ(e12, O12) - lZ(m, Ot)
                  + 2 * (a * mlog(a) - loggamma(a))).real)


def logR_dc_K(G, part, a=1.0):
    """Degree-corrected evidence for a K-block partition vs the config null."""
    a = mpf(a); K = max(part.values()) + 1
    k = dict(G.degree()); e = {}
    for u, v in G.edges():
        r, s = part[u], part[v]
        if r > s: r, s = s, r
        e[(r, s)] = e.get((r, s), 0) + 1
    kap = [0] * K
    for v in G.nodes(): kap[part[v]] += k[v]
    m = G.number_of_edges(); twom = 2 * m
    def Om(r, s): return mpf(kap[r]*kap[s])/(2*twom) if r == s else mpf(kap[r]*kap[s])/twom
    lZ = lambda ev, O: loggamma(ev + a) - (ev + a) * mlog(O + a)
    num = sum(lZ(e.get((r, s), 0), Om(r, s)) for r in range(K) for s in range(r, K))
    den = lZ(m, mpf(m)); npairs = K * (K + 1) // 2
    return float((num - den + (npairs - 1) * (a * mlog(a) - loggamma(a))).real)


# ------------------------------------------------------------------ partitions
def branch_partition(G, branch):
    """Block 1 = one branch + the three seed hubs; block 2 = the other two."""
    return {v: (0 if branch[v] in (0, -1) else 1) for v in G.nodes()}


def hubleaf_partition(G):
    """Core/periphery by the degree threshold that maximises the plain evidence
    (matches the 'hub-leaf cut obtained by thresholding on degree' of the paper)."""
    k = dict(G.degree()); degs = sorted(set(k.values()))
    best, bestv = None, -1e18
    for th in degs[:-1]:
        part = {v: (1 if k[v] > th else 0) for v in G.nodes()}
        v = logR_plain(G, part)
        if v == v and v > bestv:
            bestv, best = v, part
    return best


def three_block_partition(G, branch):
    """Symmetric three-community partition: block b = branch b + seed hub b."""
    return {v: (branch[v] if branch[v] >= 0 else v) for v in G.nodes()}


# ------------------------------------------------------------------ layout
def layout(G):
    """Pin the three seed hubs at the corners of an equilateral triangle and
    relax the rest with a spring layout, so the three branches splay into lobes."""
    ang = {0: 90, 1: 210, 2: 330}
    init = {h: (np.cos(np.radians(a)), np.sin(np.radians(a))) for h, a in ang.items()}
    pos = nx.spring_layout(G, pos=init, fixed=[0, 1, 2],
                           k=1.6 / np.sqrt(len(G)), iterations=400, seed=7)
    return pos


# ------------------------------------------------------------------ drawing
def draw(G, part, pos, colors, title, fname, highlight_cross=False, legend=None):
    k = dict(G.degree())
    sizes = [10 + 26 * np.sqrt(k[v]) for v in G.nodes()]
    ncol = [colors[part[v]] for v in G.nodes()]

    fig, ax = plt.subplots(figsize=(5.2, 4.9), facecolor="white")
    # edges
    within = [(u, v) for u, v in G.edges() if part[u] == part[v]]
    cross = [(u, v) for u, v in G.edges() if part[u] != part[v]]
    nx.draw_networkx_edges(G, pos, edgelist=within, edge_color="#b9b9b9",
                           width=0.6, alpha=0.7, ax=ax)
    if highlight_cross:
        nx.draw_networkx_edges(G, pos, edgelist=cross, edge_color="#111111",
                               width=1.4, alpha=0.9, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, edgelist=cross, edge_color="#b9b9b9",
                               width=0.6, alpha=0.7, ax=ax)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=ncol,
                           edgecolors="white", linewidths=0.6, ax=ax)
    ax.set_title(title, fontsize=11, pad=6)
    ax.axis("off"); ax.set_aspect("equal")
    if legend:
        ax.legend(handles=legend, loc="lower center", ncol=len(legend),
                  frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02),
                  handletextpad=0.4, columnspacing=1.2)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{fname}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main(T=4):
    G, branch = build_dgm(T)
    N, E = G.number_of_nodes(), G.number_of_edges()
    pos = layout(G)
    print(f"Pseudofractal web T={T}:  N={N}, E={E}, "
          f"max degree={max(dict(G.degree()).values())}")

    # ---- Fork 2 / recursive branch cut (degree-corrected favourite) ----
    pb = branch_partition(G, branch)
    n1, n2, e1, e2, e12, k1, k2 = block_counts(G, pb)
    cB = {0: "#2f6db0", 1: "#e08a1e"}   # block1 (branch+hubs) / block2 (two branches)
    legB = [
        Line2D([], [], marker="o", ls="", mfc=cB[0], mec="white",
               ms=8, label=f"block 1: one branch + hubs  ($n_1={n1}$)"),
        Line2D([], [], marker="o", ls="", mfc=cB[1], mec="white",
               ms=8, label=f"block 2: two branches  ($n_2={n2}$)"),
    ]
    draw(G, pb, pos, cB,
         rf"Recursive branch cut  (Fork 2, degree-corrected)"
         f"\n$E_{{12}}={e12}$ crossing edges  |  "
         rf"$\log R_{{\rm dc}}={logR_dc(G,pb):.1f}$",
         "partition_branch", highlight_cross=True, legend=legB)
    print(f"  branch cut     : n1={n1} n2={n2} E1={e1} E2={e2} E12={e12}  "
          f"logR_plain={logR_plain(G,pb):.2f}  logR_dc={logR_dc(G,pb):.2f}")

    # ---- Fork 1 / hub-leaf cut (plain-SBM favourite) ----
    ph = hubleaf_partition(G)
    n1h, n2h, *_ = block_counts(G, ph)
    cH = {0: "#c9ccd1", 1: "#c0392b"}   # periphery (leaves) / core (hubs)
    # keep core = block 1 colour red regardless of label id
    core_label = 1
    legH = [
        Line2D([], [], marker="o", ls="", mfc="#c0392b", mec="white",
               ms=8, label=f"core (high degree)"),
        Line2D([], [], marker="o", ls="", mfc="#c9ccd1", mec="white",
               ms=8, label=f"periphery (leaves)"),
    ]
    # ensure colour map: label 1 (above threshold) -> red core, 0 -> grey
    cH_map = {0: "#c9ccd1", 1: "#c0392b"}
    draw(G, ph, pos, cH_map,
         rf"Hub–leaf cut  (Fork 1, plain Bernoulli SBM)"
         f"\ncore vs periphery  |  "
         rf"$\log R={logR_plain(G,ph):.1f}$",
         "partition_hubleaf", highlight_cross=False, legend=legH)
    print(f"  hub-leaf cut   : core={sum(ph[v]==1 for v in G)} "
          f"periphery={sum(ph[v]==0 for v in G)}  "
          f"logR_plain={logR_plain(G,ph):.2f}  logR_dc={logR_dc(G,ph):.2f}")

    # ---- symmetric three-community partition (hierarchy, level 1) ----
    p3 = three_block_partition(G, branch)
    c3 = {0: "#2f6db0", 1: "#e08a1e", 2: "#3a9a54"}   # three branches
    sizes3 = [sum(p3[v] == b for v in G.nodes()) for b in range(3)]
    leg3 = [
        Line2D([], [], marker="o", ls="", mfc=c3[b], mec="white", ms=8,
               label=f"branch {b}  ($n={sizes3[b]}$)") for b in range(3)
    ]
    draw(G, p3, pos, c3,
         rf"Three-community partition  (one branch each)"
         f"\nthree thin seams  |  "
         rf"$\log R_{{\rm dc}}={logR_dc_K(G,p3):.1f}$",
         "partition_3way", highlight_cross=True, legend=leg3)
    print(f"  3-community    : sizes={sizes3}  logR_dc_K={logR_dc_K(G,p3):.2f} "
          f"(vs 2-block {logR_dc(G,pb):.2f})")

    print("Saved: partition_branch.{pdf,png}, partition_hubleaf.{pdf,png}, "
          "partition_3way.{pdf,png}")


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(T)
