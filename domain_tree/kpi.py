"""
-matching partitions con intervallo di confidenza -->
-R^2 totale --> check r2
-numero di partizioni --> check partition number
-grafico di comparazione
"""
import numpy as np
import copy
from tree import DomainTree, DomainNode
import altair as alt


def check_partition_num(original_m: DomainTree, approx_m):
    """
    Checks the number of partition in the models.
    :param original_m:
    :param approx_m:
    :return:
    """
    original_partitions = len(original_m.leaves)
    approx_partitions = len(approx_m.leaves)  # ???
    delta = -original_partitions + approx_partitions

    return original_partitions, approx_partitions, delta


def check_r2(approx_model):
    """
    Checks the R2 of the approx model.
    :param approx_model:
    :return:
    """
    leaves = approx_model.leaves
    # TODO FAKE R2
    for leaf in leaves:
        leaf.r2 = 0.1
    r2_list = [leaf.r2 for leaf in leaves]

    avg_r2 = np.average(r2_list)
    total_r2 = np.sum(r2_list)
    delta_r2 = len(r2_list) - total_r2
    r2_min = np.min(r2_list)
    r2_max = np.max(r2_list)

    return avg_r2, delta_r2, r2_min, r2_max

def matching_bounds(original, approx, conf: float) -> bool:
    """ Checks if 2 bounds match in respect to a confidence interval."""
    sx = original[0] - conf <= approx[0] <= original[0] + conf
    dx = original[1] - conf <= approx[1] <= original[1] + conf
    return sx and dx


def matching_intervals(original: DomainNode, approx: DomainNode, conf: float) -> bool:
    """ Checks if 2 intervals match in respect to a confidence interval."""
    # out_of_bounds = (not matching_bounds(original.domains[v], approx.domains[v], conf) for v in original.variables)
    # return not any(out_of_bounds)

    vars_in_bounds = (matching_bounds(original.domains[var], approx.domains[var], conf) for var in original.variables)
    return all(vars_in_bounds)


def compute_conf(n: int, d: int) -> float:
    return 0.5 * (d ** 0.5) / n


def check_matching_partitions(original_m: DomainTree, approx_model, n: int, d: int):
    """
    Checks the matching partitions between the original model and the approximated model, using a confidence interval
    proportional to the avg minimum distance between sobol points.
    :param original_m:
    :param approx_model:
    :param n:
    :param d:
    :return:
    """

    count = 0
    conf = compute_conf(n, d)
    # print(f"{n=}, {d=} {conf=}")
    for approx_leaf in approx_model.leaves:
        partition_match = (original_leaf for original_leaf in original_m.leaves if
                           matching_intervals(original_leaf, approx_leaf, conf))
        count += any(partition_match)

    return count


def summary(original_m, approx_model, n, d):
    original_partitions, approx_partitions, delta_p = check_partition_num(original_m, approx_model)
    avg_r2, delta_r2, r2_min, r2_max = check_r2(approx_model)
    matching_partitions_ = check_matching_partitions(original_m, approx_model, n, d)

    print(f"{14 * '-'}SUMMARY{14 * '-'}")
    print(f"PARTITIONS: {original_partitions=}, {approx_partitions=} , {delta_p=}")
    print(f"R2: {avg_r2=}, {delta_r2=}, {r2_min=}, {r2_max=}")
    print(f"MATCHING PARTITIONS: {matching_partitions_=}")
    print(f"{35 * '-'}")


def main():
    d = {"x0": (0, 1), "x1": (0, 1)}
    tree1 = DomainTree(domains=d, depth_max=3)
    tree2 = copy.deepcopy(tree1)
    tree3 = DomainTree(domains=d, depth_max=3)

    # tree1.print_tree()
    # tree1.visualize_all_domains()
    # tree3.print_tree()
    # tree3.visualize_all_domains(path="C:\\_git\\ds_tests\\GlobalLime\\TreeDomain\\imgs\\t3.html")

    # original_partitions, approx_partitions, delta = check_partition_num(tree1, tree2)
    # print(f"{original_partitions}, {approx_partitions}, {delta}")
    # print(check_matching_partitions(tree1, tree2, 10000, 4))

    summary(tree1, tree3, 10000, 4)
    comparison = alt.hconcat(tree1.visualize_all_domains(mode="none"), tree3.visualize_all_domains(mode="none"))
    comparison.show(embed_opt=True, open_browser=True)
    alt.layer(tree1.visualize_domains(var1="x0", var2="x1", mode="none"),
              tree3.visualize_domains(var1="x0", var2="x1", mode="none", color="red")).show(open_browser=True)


if __name__ == "__main__":
    main()
