import webbrowser

from anytree import NodeMixin, RenderTree
from anytree.search import find
from anytree.exporter import DotExporter

from functools import lru_cache, cached_property
from frozendict import frozendict
from deprecated import deprecated

import random as rand
from sklearn.linear_model import LinearRegression
import numpy as np

from collections import deque
import copy
import pandas as pd
from itertools import combinations, cycle

from graphviz import Source, render
import cv2
import altair as alt
from sequencer import Sequencer
from domain import Split, RealDomain

COLORS = cycle(['black', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'blue', 'orange',
                'burlywood', 'pink'])


# rand.seed(42)


class DomainNode(NodeMixin):
    """
    DomainNode represents the node of DomainTree.

    Attributes
    ----------
    :param name: name base to assign to the single nodes
    :type name: str

    :param domains: domains of the variable in the node, it is a dictionary e.g. {"x0": (0, 1), "x1": (-2, 5.44)}
    :type domains: dict, frozendict

    :param variables: list of the variables (domains.keys())
    :type variables: list

    :param parent: the parent of the node
    :type parent: DomainNode

    :param val: val on which we compute the stop criterion. Is the max depth reachable from that node
    :type val: float

    :param split_desc: describe the split occurred
    :type split_desc: dict

    Methods
    ----------
    dostuff(random):
        operates a subtraction on val, reducing the max depth reachable. random is the probability to randomly subtract
        2 instead of 1
    contains(x):
        Checks if the node contains the point x
    stopcrit():
        Checks if the node can't have other children
    generate_regression(a, b):
        generates a random linear regression in the node. (a, b) is the range of the parameters beta and the intercept
    kill():
        kills the node, instantly setting to 0 its max depth
    """

    def __init__(self, name, domains: RealDomain, parent=None, val=5, split_desc=Split(node_type="ROOT")):
        # super(self).__init__()
        self.val = val
        self.name = name
        self.parent = parent

        self.domains = domains  # {"x0": [-1, 2), "x1": [0, 12)...}

        self.split_desc = split_desc

        self.regression = None

    @cached_property
    def variables(self):
        return self.domains.get_variables()

    def __str__(self):
        return "Hi! I'm " + self.name

    def __repr__(self):
        return f"{self.name}: {self.domains.to_str()}___{self.split_desc}"

    # do something, simulating an operation
    def dostuff(self, random: float = 0):  # sourcery skip: assign-if-exp
        if rand.random() < random:
            self.val = self.val - 2
        else:
            self.val = self.val - 1

    # simulating a stop criterion
    def stopcrit(self) -> bool:
        return self.val <= 0

    def generate_regression(self, a: float = 0, b: float = 1):
        """
        Generates a random regression inside the node. (a, b) are the bounds for the random coefficients

        Parameters
        ----------
        :param a: left bound of (a, b)
        :type a: float

        :param b: right bound of (a, b)
        :type b: float
        :return:
        """

        if b < a:
            a, b = b, a

        lm = LinearRegression()
        coef = np.array([rand.uniform(a, b) for i in range(len(self.variables))])
        interc = rand.uniform(a, b)
        # coef = np.array([rand.random() * (b - a) + a for i in range(len(self.variables))])
        # interc = rand.random() * (b - a) + a
        lm.coef_ = coef
        lm.intercept_ = interc
        self.regression = lm

    # TODO, si può pensare una classe point/domain etc. invece dei dizionari...
    def contains(self, x) -> bool:
        # sourcery skip: inline-immediately-returned-variable
        """
        Check if a point x is contained in the node domains
        :param x: the point in format {"x0": 1, "x2": 1.59 ...}
        :type x: dict
        :return: True/False
        :rtype: bool
        """
        # Does a variable out of bound exists?
        # out_of_bounds = (not (self.domains[var][0] <= x[var] < self.domains[var][1]) for var in self.variables)
        # res = not any(out_of_bounds)

        # Are all variable in bounds?
        # in_bounds = (self.domains[var][0] <= x[var] < self.domains[var][1] for var in self.variables)
        # res = all(in_bounds)

        return self.domains.contains(x)

    def kill(self):
        self.val = 0


class NodeNotFoundException(Exception):
    """Exception if a node with certain characteristics is not found in the tree."""
    pass


class DomainTree:
    """
    DomainTree is a binary tree for which each node is a DomainNode. In each leaf of the tree, a random regression in
    fitted. Each node contains a domain for the variables. The tree splits randomly the domain of a non-terminal node in
    two ranges and creates two other nodes, until the depth_max is reached or a stop condition is met (e.g the domain is
    too small and cannot be splitted in two parts).

    Attributes
    ----------
    name: str
        base name for the nodes that will we sequenced in order using a Sequencer
    domains: dict
        contains the original domains of the variables that will be splitted e.g. {"x0": (0, 1), "x1": (-2, 5.44)}.
        The original domains are passed in the __init__() and are SORTED by the key. So, {"x2": (0, 1), "x1": (-2, 5)}
        becomes {"x1": (-2, 5), "x1": (0, 1)}
    variables: list
        contains the list of the variables that is nothing but domain.keys(), SORTED by the key. See the attribute
        "domains"
    depth_max: int
        maximum depth reachable by the tree
    random: int in [0, 1], is a probability
        gives a random behaviour to the tree construction, influencing the number of nephews of a node, cutting its depth.
        p=0 --> no random behaviour, always reduce by 1 the depth_max for the node
        p=(0, 1) --> randomly reduce the depth max of 1 (probability 1 - p) or 2 (probability p)
        p=1 --> always reduce by 2 the depth_max for the node
    min_split: float in [0, 0.5]
        represents the minimum part of the domain (in % from 0 to 50) to retain for the split. if min_split is > 0.5,
        obviously the split in unfeasible and the whole domain is kept.
    coef_bounds: tuple(float, float)
        the bounds for the random coefficients of the regressions in the nodes
    rounding: int
        the number of useful digits for the value of the split. round(value, round).
        e.g. round(0.587841263, 4) = 0.5878 retains 4 decimal digits
    tree: DomainNode
        represents the root of the tree. It is the tree structure de facto, without helper methods
    leaves: list(DomainNode)
        maintains the leaves of the tree for simplicity

    Methods
    ----------
    contains(x):
        Check if the tree contains x
    compute_f(x):
        computes the y value for the point x
    print_tree():
        prints the tree structure
    render_tree():
        renders graphically the tree
    visualize_regressions(mode):
        visualize the regressions and the splits in a chart with the specified mode. It works with 2 dimensions only.
    visualize_domains(var1, var2, mode):
        visualize the grid structure for the domains of a selected couple of variables var1 and var2
    visualize_all_domains(mode):
        visualize the grid structure for the domains of each combination of variables

    """

    def __init__(self, domains: RealDomain, min_split=0, name="NODE", depth_max=3, random=0, coef_bounds=(-5, 5),
                 rounding=4):
        self.name = name
        self.domains = domains
        # self.variables = sorted(list(domains.keys()))
        self.depth_max = depth_max
        self.random = random
        self.rounding = rounding
        self.min_split = min_split
        self.coef_bounds = coef_bounds

        self.tree, self.leaves = self.__generate_tree_domains(name=self.name, domains=self.domains, random=random,
                                                              val=self.depth_max, min_split=min_split)

    @cached_property
    def variables(self):
        return self.domains.get_variables()

    def __select_variable(self, variables):
        """
        It is the policy to select the variable for the split --> Random choice
        :param variables:
        :return:
        """
        return rand.choice(variables)

    def __generate_tree_domains(self, name, domains, random, val, min_split):
        """
        Generates the internal tree structure using DomainNodes
        :return: tree, leaves
        """
        s = Sequencer()
        tree = DomainNode(f"{name}{s.get_seq_num()}", domains=domains, val=val)

        """
        Short explaination:
        len_(bounds, split percentage): float
            computes the actual length of the minimum split given a range
        min_ranges:
            given the bounds of the tree, creates a dict wich contains the minimum absolute split
            for each variable
        range_length(bounds):
            computes the length of a range
        good_vars(node, min_ranges):
            returns the variables that respect the minimum split criterion
        """
        len_ = lambda bounds, split: abs(bounds[1] - bounds[0]) * split
        min_ranges = {d: len_(self.domains[d], min_split) for d in self.domains.keys()}
        range_length = lambda bound: abs(bound[1] - bound[0])
        good_vars = lambda n, min_ranges: [
            d
            for d in n.domains.keys()
            if (range_length(n.domains[d]) >= min_ranges[d] * 2)
        ]

        # stack of the open nodes
        stack = deque()
        stack.append(tree)

        # not necessary, node.leaves
        leaves = []
        while len(stack) > 0:
            node = stack.pop()
            node.dostuff(random)

            # continue to develop the tree
            candidate_variables = good_vars(node, min_ranges)

            if not node.stopcrit() and candidate_variables:
                # select a variable and obtain the bounds
                variable = self.__select_variable(candidate_variables)
                bounds = node.domains[variable]
                a = bounds[0]
                b = bounds[1]

                # w = (b - a) * min_split
                # split_value = a + w + ((b - w) - (a + w)) * rand.random()

                # scaled split value
                w = min_ranges[variable]
                split_value = a + w + ((b - w) - (a + w)) * rand.random()

                split_value = round(split_value, self.rounding)

                # changing bounds
                domains_sx, domains_dx = node.domains.split(split_value=split_value, var=variable)

                desc = {"node_type": "NODE", "split_var": variable, "split_value": split_value}
                node.split_desc = Split(node_type="NODE", split_var=variable, split_value=split_value)

                val = node.val
                node1 = DomainNode(self.name + str(s.get_seq_num()), domains=domains_sx, val=val,
                                   parent=node)
                node2 = DomainNode(self.name + str(s.get_seq_num()), domains=domains_dx, val=val,
                                   parent=node)

                stack.append(node1)
                stack.append(node2)

            # the node is a leaf
            else:
                node.kill()
                node.generate_regression(a=self.coef_bounds[0], b=self.coef_bounds[1])
                node.split_desc = Split(node_type="LEAF", intercept=node.regression.intercept_,
                                        coefficients=node.regression.coef_)
                leaves.append(node)

        # resetting sequencer
        s.reset()

        return tree, leaves

    # bugged and deprecated
    @deprecated("New method __recursive_search__")
    @lru_cache(maxsize=128, typed=False)
    def __recursive_search(self, node: DomainNode, x) -> DomainNode:
        """
        Recursive search on the tree in decision tree stile (but non as, every time each variable is checked...)
        :param node: node from which we want to start the search
        :param x: point in dictionary form
        :return: the node which contains the point
        :rtype: DomainNode
        """
        if node.is_leaf and node.contains(x):
            # print(f"__Found! {node.name}: {node.domains}")
            return node
        elif node.is_leaf:
            raise NodeNotFoundException(f"{x} is not in tree")
        else:
            # print(f"__Explored {node.name}")

            children = node.children
            if children[0].contains(x):
                return self.__recursive_search(children[0], x)
            else:
                return self.__recursive_search(children[1], x)

    # optimized decision tree search
    def greater(self, a, b, included):
        return a >= b if included[0] else a > b

    def less(self, a, b, included):
        return a <= b if included[1] else a < b

    @lru_cache(maxsize=128, typed=False)
    def __recursive_search__(self, node: DomainNode, x, verbose=False) -> DomainNode:
        """
        Recursive search on the tree in decision tree stile (but non as, every time each variable is checked...)
        :param node: node from which we want to start the search
        :param x: point in dictionary form
        :return: the node which contains the point
        :rtype: DomainNode
        """
        if node.is_leaf:
            verbose and print(f"__Found! {node.name}: {node.domains}")
            return node
        else:
            verbose and print(f"__Explored {node.name}")
            var = node.split_desc["split_var"]
            value = node.split_desc["split_value"]

            # if value <= x[var] < node.domains[var][1]:  # [, ) [, ) [, ) [, )
            #     rec_node = node.children[1]
            #     return self.__recursive_search__(rec_node, x, verbose)
            # elif node.domains[var][0] <= x[var] < value:
            #     rec_node = node.children[0]
            #     return self.__recursive_search__(rec_node, x, verbose)
            # else:
            #     raise NodeNotFoundException(f"{x} is non in tree")

            if self.greater(x[var], value, node.domains[var].included) and self.less(x[var], node.domains[var][1], node.domains[var].included):
                rec_node = node.children[1]
                return self.__recursive_search__(rec_node, x, verbose)
            elif self.greater(x[var], node.domains[var][0], node.domains[var].included) and self.less(x[var], value, node.domains[var].included):
                rec_node = node.children[0]
                return self.__recursive_search__(rec_node, x, verbose)
            else:
                raise NodeNotFoundException(f"{x} is non in tree")

    def contains(self, x) -> bool:
        """
        Check if the tree contains x
        :param x: is a point i dictionary form e.g. {"x0": 1.23, "x1: 3.6 ...}
        :return: True inf the tree contains x
        :rtype: bool
        """
        try:
            self.__recursive_search__(self.tree, frozendict(x), verbose=False)
            return True
        except NodeNotFoundException:
            return False

    def node_which_contains(self, x) -> DomainNode:
        """
        Returns the node which contains x
        :param x: is a point i dictionary form e.g. {"x0": 1.23, "x1: 3.6 ...}
        :return: True inf the tree contains x
        :rtype: bool
        """
        try:
            return self.__recursive_search__(self.tree, frozendict(x), verbose=False)
        except NodeNotFoundException:
            raise NodeNotFoundException(f"{x} not in tree")

    def compute_f(self, x):
        """
        Computes the value of node.regression.predict(x). So it checks which is the correct partition, obtains node
        and then the regression, so computes node.regression.predict(x).

        :param x: a point in form on dictionary e.g{"x0": 1, "x1": 32, "x2": 10}
        :type x: dict
        :return: node.regression.predict(x)
        :rtype: float
        """

        # node = next(node for node in self.leaves if node.contains(x))
        node = self.__recursive_search__(self.tree, frozendict(x), verbose=False)

        # sorting x
        values = [x[k] for k in sorted(x)]
        x_array = np.array(values).reshape(1, -1)
        return node.regression.predict(x_array)

    def print_tree(self):
        """
        Prints the tree on stdout
        :return:
        """
        print(RenderTree(self.tree))

    def render_tree(self, name="test", path="./imgs/"):
        """
        Graphically renders the tree structure
        :param name: name of the image
        :param path: path for saving the image
        :return:
        """
        DotExporter(self.tree).to_dotfile(path + name)
        Source.from_file(path + name)
        render('dot', 'png', path + name)
        fname = path + name + ".png"
        img = cv2.imread(fname)
        cv2.imshow("Render", img)
        cv2.waitKey(0)

    def visualize_regressions(self, var="x0", mode="html", path="C:\\_git\\ds_tests\\GlobalLime\\TreeDomain\\imgs"
                                                                "\\regression.html"):
        """
        Visualize the regressions and the splits in a chart with the specified mode.
        :param path: path where to save the image
        :param var: specify the variable for which we want to show the regressions and the splits
        :param mode: output mode
        :return: the figure
        :rtype: alt.Chart
        """

        figures = []
        for leaf in self.leaves:
            # searching the y value for x_n bounds
            x = leaf.domains[var].bounds
            y_a = leaf.regression.predict(np.array([[x[0]]]))
            y_b = leaf.regression.predict(np.array([[x[1]]]))
            y = (y_a, y_b)

            x_y = np.c_[np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)]
            df = pd.DataFrame(x_y, columns=["x", "y"])

            fig = alt.Chart(df).mark_line().encode(
                x=alt.X("x", scale=alt.Scale(zero=False), type="quantitative"),
                y=alt.Y("y", scale=alt.Scale(zero=False), type="quantitative"),
                color=alt.value("blue"),
                tooltip=["x", "y"]
            )

            # representing the splits like parallel lines
            ycut = np.array(self.coef_bounds) * 1.5
            xcut = [x[0], x[0]]
            xcut_ycut = np.c_[np.array(xcut).reshape(-1, 1), np.array(ycut).reshape(-1, 1)]
            df_cut = pd.DataFrame(xcut_ycut, columns=["x", "y"])
            cut = alt.Chart(df_cut).mark_line().encode(
                x=alt.X("x", scale=alt.Scale(zero=False), type="quantitative"),
                y=alt.Y("y", scale=alt.Scale(zero=False), type="quantitative"),
                color=alt.value("red")
            )
            figures.append(fig)
            figures.append(cut)

        chart = alt.layer(*figures).interactive()

        if mode == "html":
            chart.save(path)
            webbrowser.open_new_tab(path)

        return chart

    def visualize_domains(self, var1, var2, mode="html", color="black", path="C:\\_git\\ds_tests\\GlobalLime"
                                                                             "\\TreeDomain\\imgs\\domains.html"):

        def get_fig(x, y):
            xy = np.c_[np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)]
            df = pd.DataFrame(xy, columns=[var1, var2])
            fig = alt.Chart(df).mark_line().encode(
                x=alt.X(var1, scale=alt.Scale(zero=False), type="quantitative"),
                y=alt.Y(var2, scale=alt.Scale(zero=False), type="quantitative"),
                color=alt.value(color)
            )

            return fig

        figures = []
        for leaf in self.leaves:
            d_var1 = leaf.domains[var1]
            d_var2 = leaf.domains[var2]

            masks = [(0, 1, 0, 0), (0, 1, 1, 1), (0, 0, 0, 1), (1, 1, 0, 1)]
            for mask in masks:
                x = []
                y = []
                x.append(d_var1[mask[0]])
                x.append(d_var1[mask[1]])
                y.append(d_var2[mask[2]])
                y.append(d_var2[mask[3]])
                fig = get_fig(x, y)
                figures.append(fig)

        chart = alt.layer(*figures).interactive()

        if mode == "html":
            chart.interactive()
            chart.save(path)
            webbrowser.open_new_tab(path)

        return chart

    def visualize_all_domains(self, mode="html", path="C:\\_git\\ds_tests\\GlobalLime\\TreeDomain\\imgs\\alldomains"
                                                      ".html"):
        """
        Visualize the grid structure for the domains of each combination of variables
        :param path: path where to save the image
        :param mode: visualizazion mode
        :return: the charts of all the combinations of the domains
        """
        c = combinations(self.variables, 2)
        l = [self.visualize_domains(v[0], v[1], mode="none", color=next(COLORS)) for i, v in enumerate(c)]
        charts = alt.hconcat(*l)

        if mode == "html":
            charts.save(path)
            webbrowser.open_new_tab(path)

        return charts
