import random

from tree import DomainTree
from scipy.stats import qmc
import altair as alt
import numpy as np
import pandas as pd
import webbrowser
import time

# (n+1)/2 leaves
DEPTH = 5
NODE_NAME = "node"
K = 1
VARIABLES = ["x" + str(i) for i in range(K)]
DOMAINS = {val: (0, 1) for i, val in enumerate(VARIABLES)}
x = {v: 0.5 for v in VARIABLES}
RANDOM = 0 # [0, 1]
MIN_SPLIT = 0.1 # [0, 0.5]
COEF_BOUNDS = (-5, 5)


def sanity_check1d(leaves):
    def correct_domain(leaf, next_leaf):
        problematic = ""
        correct = True
        for var in leaf.variables:
            # print(f"{leaf.domains[var][0]} == {next_leaf.domains[var][1]} or {leaf.domains[var][1]} == {next_leaf.domains[var][0]} or {leaf.domains[var]} == {next_leaf.domains[var]}?")
            if not (leaf.domains[var][0] == next_leaf.domains[var][
                1]):  # or leaf.domains[var][1] == next_leaf.domains[var][0] or leaf.domains[var] == next_leaf.domains[var]):
                correct = False
                problematic = var
                break

        return correct, problematic

    for i, leaf in enumerate(leaves):
        if i == len(leaves) - 1:
            break

        print(i)
        next_leaf = leaves[i + 1]

        correct, var = correct_domain(leaf, next_leaf)
        if not correct:
            raise Exception(
                f"Incorrect domains found. Leaf: {leaf.name} {leaf.domains} and Leaf: {next_leaf.name} {next_leaf.domains}.: . Mismatch on {var}")


def sobol(display=False, m=6):
    sampler = qmc.Sobol(d=K, scramble=False)
    X = sampler.random_base2(m=m)
    y = [tree.compute_f({var: x[i] for i, var in enumerate(VARIABLES)}) for x in X]

    if display:
        chart = tree.visualize_regressions(mode="none")
        df = pd.DataFrame(np.c_[X, y], columns=["x", "y"])
        fig = alt.Chart(df).mark_circle(size=20).encode(
            x=alt.X("x", scale=alt.Scale(zero=False), type="quantitative"),
            y=alt.Y("y", scale=alt.Scale(zero=False), type="quantitative"),
            color=alt.value("yellow"),
            tooltip=["x", "y"]
        )
        final = alt.layer(chart, fig).properties(width=600, height=500).interactive()
        # fname = "./imgs/final.html"
        # final.save(fname)
        #webbrowser.open_new_tab("C:\\_git\\ds_tests\\GlobalLime\\TreeDomain\\imgs\\final.html")
        final.show(open_browser=True)


def main():
    global tree
    tree = DomainTree(domains=DOMAINS,
                      name=NODE_NAME,
                      min_split=MIN_SPLIT,
                      depth_max=DEPTH,
                      random=RANDOM,
                      coef_bounds=COEF_BOUNDS)
    print("Tree built!")
    tree.visualize_all_domains()
    # tree.visualize_domains("x0", "x1")
    tree.print_tree()
    #tree.render_tree()
    #print(x)
    print(tree.compute_f(x))
    #print(tree.contains(x))
    sobol(display=True, m=6)


if __name__ == '__main__':
    main()
