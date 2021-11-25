from abc import ABC, abstractmethod
import copy

"""
* For Node domains use RealDomain
* For the split use RealDomain.split(), which returns two non overlapped cut domain
* It is not mandatory to look at Real Interval logic because is intrisic in RealDomain
* Also AbstractDomain is more a defect of form that a strictly necessary hierarchy 
 (but if we would have needed categorical variables...)
* For usage look at the main which contains a simple demo
"""


# necessary?
class Point:

    # coordinates

    def __init__(self, coordinates: dict):
        self.coordinates = coordinates

    def __repr__(self):
        return str(self.coordinates)

    def __getitem__(self, item):
        return self.coordinates[item]

    def get(self, var):
        return self.coordinates[var]

    def set(self, var, value):
        self.coordinates[var] = value


class RealInterval:
    """
    Represents a real interval I, open or closed
    """

    def __init__(self, bounds: tuple, included: tuple):
        """

        :param bounds: bounds of the interval in tuple form. e.g. (0, 1)
        :param included: tells if the bounds are included e.g for [0, 1) is (True, False)
        """
        self.bounds = bounds
        self.included = included

    def contains(self, x: float):
        """
        Check if a float x is contained in the interval
        :param x: a float number to check
        :return: True or false
        """
        greater = lambda a, b: a >= b if self.included[0] else a > b
        less = lambda a, b: a <= b if self.included[1] else a < b
        res = greater(x, self.bounds[0]) and less(x, self.bounds[1])

        # res = False
        # if self.included[0] and self.included[1]:
        #     res = self.bounds[0] <= x <= self.bounds[1]
        # elif self.included[0]:
        #     res = self.bounds[0] <= x < self.bounds[1]
        # elif self.included[1]:
        #     res = self.bounds[0] < x <= self.bounds[1]
        # else:
        #     res = self.bounds[0] < x < self.bounds[1]

        return res

    def split_at(self, split: float):
        """
        Split the interval without any criteria for included
        :param split:
        :return:
        """
        if not self.contains(split):
            raise Exception("The split point is not present in the domain")

        bounds_sx = (self.bounds[0], split)
        bounds_dx = (split, self.bounds[1])

        return bounds_sx, bounds_dx

    def perfect_split(self, split):
        """
        Perform a split using the rule:
        on the left node we put open ')' the right bound.
        The rest remains unchanged e.g.

                       [ ]
                [ )          [ ]
            [ )   [ )     [ )   [ ]

        returns left and right interval. The original interval remains unchanged
        """
        bounds_sx, bounds_dx = self.split_at(split)
        included_sx = (self.included[0], False)
        included_dx = self.included

        interval_sx = RealInterval(bounds_sx, included_sx)
        interval_dx = RealInterval(bounds_dx, included_dx)

        return interval_sx, interval_dx

    @property
    def width(self):
        return self.bounds[1] - self.bounds[0]

    # dictionary/array like access
    def __getitem__(self, item):
        return self.bounds[item]

    def __repr__(self):
        left_par = lambda included: "[" if included else "("
        right_par = lambda included: "]" if included else ")"

        return f"{left_par(included=self.included[0])}{self.bounds[0]}, {self.bounds[1]}{right_par(included=self.included[1])}"


class AbstractDomain(ABC):
    """
    Abstract class for a generic domain
    """

    @abstractmethod
    def contains(self, x) -> bool:
        pass

    @abstractmethod
    def get_all(self):
        pass

    @abstractmethod
    def set_all(self, domains):
        pass

    # TODO
    # remove? insert?


class RealDomain(AbstractDomain, ABC):
    """
    domains: dict = {"x0": RealInterval0,
                     "x1": RealInterval1,
                     ...}

    Class for multi-variables continuous real domains
    """

    def __init__(self, domains: dict):
        """
        domains: dict = {"x0": RealInterval0,
                         "x1": RealInterval1,
                         ...}
        """
        self.domains = domains
        # TODO

    # def __init__(self, variables: list, intervals: RealInterval):
    #     domains = {var: i for var, i in zip(variables, intervals)}
    #     self.domains = domains

    def contains_single_var(self, x: float, var: str) -> bool:
        """
        Checks if the specific domain for var contains x
        :param x: the value
        :param var: the variable
        :return: True/False
        """
        return self.domains[var].contains(x)

    # def contains(self, x: dict) -> bool:
    #     """
    #     Checks if a point x is contained in a domain
    #     :param x: the point
    #     :return: True/False
    #     """
    #     vars_in_bounds = (self.contains_single_var(x[var], var) for var in self.domains)
    #     return all(vars_in_bounds)

    def contains(self, x) -> bool:
        """
        Checks if a point x is contained in a domain
        :param x: the point
        :return: True/False
        """
        if isinstance(x, Point):
            x = p.coordinates

        vars_in_bounds = (self.contains_single_var(x[var], var) for var in self.domains)
        return all(vars_in_bounds)

    # redundant
    def get_all(self):
        return self.domains

    def get(self, var):
        return self.domains[var]

    def get_variables(self):
        return list(self.domains.keys())

    #LEGACY
    def keys(self):
        return self.get_variables()

    def set_all(self, domains):
        self.domains = domains

    def set(self, var, interval):
        self.domains[var] = interval

    def copy(self):
        return RealDomain(copy.copy(self.domains))

    def split(self, split_value, var):
        """
        Perform a perfect split for a Real domain
        :param split_value: split value
        :param var: variable for split
        :return:
        """
        res_sx = self.copy()
        res_dx = self.copy()

        interval = self.domains[var]
        interval_sx, interval_dx = interval.perfect_split(split_value)

        res_sx.set(var, interval_sx)
        res_dx.set(var, interval_dx)

        return res_sx, res_dx

    def remove(self, var):
        del self.domains[var]

    def insert(self, var: str, interval: RealInterval):
        self.domains[var] = interval

    # dictionary/array like access
    def __getitem__(self, item) -> RealInterval:
        return self.domains[item]

    def __repr__(self):
        n = 33
        repr = ""
        # repr += "RealDomain: \n"
        repr += "-" * n
        repr += "\n"
        for var, interval in self.domains.items():
            repr += f"| {var: <10}|   "
            repr += f"{str(interval): <15} |"
            repr += "\n"
        repr += "-" * n

        return repr

    def to_str(self):
        return {x: str(val) for x, val in self.domains.items()}


class Split:
    def __init__(self, split_var="none", split_value="none", node_type="NODE", intercept=None, coefficients=None):
        self.node_type = node_type
        self.split_var = split_var
        self.split_value = split_value
        self.intercept = intercept
        self.coefficients = coefficients

    @property
    def dict_view(self):
        return {"node_type": self.node_type, "split_var": self.split_var, "split_value": self.split_value,
                "intercept": self.intercept, "coefficients": None}

    def __getitem__(self, item):
        return self.dict_view[item]

    def __repr__(self):
        return str(self.dict_view)


if __name__ == "__main__":
    i0 = RealInterval(bounds=(0, 1), included=(True, True))
    i1 = RealInterval(bounds=(1, 2), included=(True, False))

    d = {"x0": i0, "x1": i1}

    dom = RealDomain(d)

    # test
    print("1" + "*" * 40)
    print(i0)
    print(i1)
    print("")

    print(f"{i0} contains 0? {i0.contains(0)}")  # true
    print(f"{i0} contains 1? {i0.contains(1)}")  # true
    print(f"{i0} contains 2? {i0.contains(2)}")  # false
    print(f"{i1} contains 1? {i1.contains(1)}")  # true
    print(f"{i1} contains 2? {i1.contains(2)}")  # false
    print("")

    # split0
    sx, dx = i0.perfect_split(0.5)
    print("2" + "*" * 40)
    print(f"Split {i0=} in 0.5.")
    print(f"{sx=}")
    print(f"{dx=}")
    print("")

    # split1
    dx_1, dx_2 = dx.perfect_split(0.7)
    print("3" + "*" * 40)
    print(f"Split {dx=} from before in 0.7.")
    print(f"{dx_1=}")
    print(f"{dx_2=}")
    print("")

    # realdomain split
    d0, d1 = dom.split(0.5, "x0")
    print("4" + "*" * 40)
    print("DOM")
    print(dom)
    print("Split DOM in x0=0.5")
    print("D0")
    print(d0)
    print("D1")
    print(d1)
    print("")

    # realdomain contains
    # meglio prevedere classe point?
    print("5" + "*" * 40)
    print("DOM")
    print(dom)
    p = Point({"x0": 0.5, "x1": 1})
    # print(f"DOM contains {p.coordinates}? {dom.contains(p.coordinates)}")
    print(f"DOM contains {p}? {dom.contains(p)}")  # true
    p = Point({"x0": 1, "x1": 1})
    print(f"DOM contains {p}? {dom.contains(p)}")  # true
    p = Point({"x0": 1, "x1": 2})
    print(f"DOM contains {p}? {dom.contains(p)}")  # false

    print("")
    print("6" + "*" * 40)
    print("Test on random variables")

    import random as r


    def rand_bounds():
        a = round(r.random(), 3)
        b = round(r.random(), 3)
        return (a, b) if a < b else (b, a)


    i_s = {f"x{x}": RealInterval(bounds=(rand_bounds()), included=(bool(r.getrandbits(1)), bool(r.getrandbits(1)))) for
           x in range(15)}
    df = RealDomain(i_s)
    print(df)

    # playing with __getitem__
    print(df["x0"].bounds)
    print(df["x0"].included)
    print(df["x0"])
