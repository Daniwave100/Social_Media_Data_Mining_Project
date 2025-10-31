"""
marvel_stats.py — size/density/centralities + optional plots for a GraphML network.

Usage (stats only):
  python marvel_stats.py "C:\\path\\to\\marvel-network.graphml" --top 10

Add plots (PNG files):
  python marvel_stats.py "C:\\path\\to\\marvel-network.graphml" --plot --k 20 --label-top 10 --prefix marvel_vis

Requirements:
  pip install networkx matplotlib
"""
import argparse, sys
from pathlib import Path
import networkx as nx

def top_n(d, k=10):
    return sorted(d.items(), key=lambda x: (-x[1], x[0]))[:k]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("graphml", nargs="?", default="marvel-network.graphml",
                   help="Path to the GraphML file (default: ./marvel-network.graphml)")
    p.add_argument("--top", type=int, default=5, help="How many top nodes to list (default: 5)")

    # plotting options
    p.add_argument("--plot", action="store_true", help="Generate PNG plots (full + k-core)")
    p.add_argument("--k", type=int, default=20, help="k for k-core plot (default: 20)")
    p.add_argument("--seed", type=int, default=42, help="Spring-layout seed (default: 42)")
    p.add_argument("--prefix", default="marvel_network", help="Output filename prefix for PNGs")
    p.add_argument("--label-top", type=int, default=0, help="Label top-N nodes by degree on plots")
    args = p.parse_args()

    gpath = Path(args.graphml)
    if not gpath.exists():
        sys.exit(f"ERROR: GraphML not found at {gpath.resolve()}")

    # Read and simplify to a simple undirected graph
    Graw = nx.read_graphml(gpath)
    G = nx.Graph()
    G.add_nodes_from(Graw.nodes(data=True))
    for u, v, data in Graw.edges(data=True):
        if u != v:
            G.add_edge(u, v, **data)

    print("=== Basic Statistics ===")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.6f}")
    print(f"Connected: {nx.is_connected(G)}")

    # Degree (interactions)
    degree = dict(G.degree())
    top_degree = top_n(degree, args.top)
    print("\n=== Top by Degree (most interactions) ===")
    for i, (n, val) in enumerate(top_degree, 1):
        print(f"{i}. {n}: {val}")

    # Centralities
    print("\n=== Centralities (popularity/influence) ===")
    deg_cent = nx.degree_centrality(G)
    print("\nTop Degree Centrality:")
    for i, (n, val) in enumerate(top_n(deg_cent, args.top), 1):
        print(f"{i}. {n}: {val:.6f}")

    close = nx.closeness_centrality(G)
    print("\nTop Closeness Centrality:")
    for i, (n, val) in enumerate(top_n(close, args.top), 1):
        print(f"{i}. {n}: {val:.6f}")

    bet = nx.betweenness_centrality(G, normalized=True)
    print("\nTop Betweenness Centrality:")
    for i, (n, val) in enumerate(top_n(bet, args.top), 1):
        print(f"{i}. {n}: {val:.6f}")

    pr = nx.pagerank(G, alpha=0.85)
    print("\nTop PageRank:")
    for i, (n, val) in enumerate(top_n(pr, args.top), 1):
        print(f"{i}. {n}: {val:.6f}")

    # Optional Eigenvector
    try:
        eig = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
        print("\nTop Eigenvector Centrality:")
        for i, (n, val) in enumerate(top_n(eig, args.top), 1):
            print(f"{i}. {n}: {val:.6f}")
    except nx.NetworkXException as e:
        print(f"\n[Eigenvector centrality skipped: {e}]")

    if args.plot:
        import matplotlib.pyplot as plt  # import only if plotting

        # layout once
        pos = nx.spring_layout(G, seed=args.seed)

        # full network
        sizes = [8 + 0.4 * degree[n] for n in G.nodes()]
        plt.figure(figsize=(10, 10))
        nx.draw_networkx_edges(G, pos, alpha=0.03, width=0.3)
        nx.draw_networkx_nodes(G, pos, node_size=sizes)
        if args.label_top > 0:
            top = sorted(degree.items(), key=lambda x: (-x[1], x[0]))[:args.label_top]
            for n, _ in top:
                x, y = pos[n]
                plt.text(x, y, str(n), fontsize=6)
        plt.axis("off")
        out_full = f"{args.prefix}_full.png"
        plt.tight_layout(); plt.savefig(out_full, dpi=200, bbox_inches="tight"); plt.close()
        print(f"Wrote {out_full}")

        # k-core
        try:
            core = nx.k_core(G, k=args.k)
        except nx.NetworkXError:
            kmax = nx.core_number(G)
            args.k = max(kmax.values())
            print(f"[Note] Requested k too high; using max k={args.k}")
            core = nx.k_core(G, k=args.k)

        deg_core = dict(core.degree())
        sizes_core = [12 + 0.7 * deg_core[n] for n in core.nodes()]
        plt.figure(figsize=(10, 10))
        nx.draw_networkx_edges(core, pos, alpha=0.08, width=0.5)
        nx.draw_networkx_nodes(core, pos, node_size=sizes_core)
        if args.label_top > 0:
            topc = sorted(deg_core.items(), key=lambda x: (-x[1], x[0]))[:args.label_top]
            for n, _ in topc:
                x, y = pos[n]
                plt.text(x, y, str(n), fontsize=7)
        plt.axis("off")
        out_core = f"{args.prefix}_kcore{args.k}.png"
        plt.tight_layout(); plt.savefig(out_core, dpi=220, bbox_inches="tight"); plt.close()
        print(f"Wrote {out_core}")

if __name__ == "__main__":
    main()
