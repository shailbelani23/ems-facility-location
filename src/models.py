"""Integer programs for ambulance station siting, solved with Gurobi.

Three models, all over the same binary coverage matrix ``a_ij``, which is 1 when
a station at candidate ``j`` reaches demand point ``i`` inside the response-time
threshold:

Set cover, "what is the cheapest fleet that leaves nobody outside the standard":

    min   sum_j y_j
    s.t.  sum_j a_ij y_j >= 1     for every demand point i
          y_j binary

Maximal covering (MCLP), "given only p stations, who do we reach":

    max   sum_i v_i z_i
    s.t.  sum_j y_j = p
          z_i <= sum_j a_ij y_j   for every demand point i
          y_j, z_i binary

The weight ``v_i`` is where the two MCLP variants diverge. With ``v_i = 1`` the
model maximises the number of block groups covered and treats a block group
that generates 200 calls exactly like one that generates 12,000. With
``v_i = call_count_i`` it maximises covered call volume. They select different
stations, and the gap between them is one of the reported results.

The LP relaxation of set cover is also exposed, because the integrality gap is
the honest way to say how hard this particular instance was.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as error:  # pragma: no cover - exercised only without Gurobi
    raise ImportError(
        "gurobipy is required. `pip install gurobipy` ships a size-limited licence "
        "that is more than large enough for these models (778 variables at most)."
    ) from error


@dataclass
class Solution:
    """Result of one solve, with everything needed to reproduce the numbers."""

    selected: list[int]
    covered: list[int]
    objective: float
    bound: float
    gap: float
    runtime: float
    n_variables: int
    n_constraints: int
    extra: dict = field(default_factory=dict)

    @property
    def n_facilities(self) -> int:
        return len(self.selected)


def _new_model(name: str, quiet: bool) -> gp.Model:
    environment = gp.Env(empty=True)
    environment.setParam("OutputFlag", 0 if quiet else 1)
    environment.start()
    return gp.Model(name, env=environment)


def solve_set_cover(coverage: np.ndarray, quiet: bool = True) -> Solution:
    """Minimum number of stations so that every demand point is covered.

    A demand point no candidate can reach would make the model infeasible, so
    those are detected first and reported rather than silently skipped, which
    would quietly turn "every demand point" into "every demand point we could
    manage".
    """
    n_demand, n_candidates = coverage.shape
    uncoverable = [i for i in range(n_demand) if coverage[i].sum() == 0]
    if uncoverable:
        raise ValueError(
            f"{len(uncoverable)} demand points are unreachable at this threshold, "
            "so set cover is infeasible. Raise the threshold or add candidates."
        )

    model = _new_model("set_cover", quiet)
    y = model.addVars(n_candidates, vtype=GRB.BINARY, name="y")
    model.setObjective(gp.quicksum(y[j] for j in range(n_candidates)), GRB.MINIMIZE)

    for i in range(n_demand):
        covering = np.flatnonzero(coverage[i]).tolist()
        model.addConstr(gp.quicksum(y[j] for j in covering) >= 1, name=f"cover_{i}")

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Set cover did not solve to optimality (status {model.Status}).")

    selected = [j for j in range(n_candidates) if y[j].X > 0.5]
    return Solution(
        selected=selected,
        covered=list(range(n_demand)),
        objective=model.ObjVal,
        bound=model.ObjBound,
        gap=model.MIPGap,
        runtime=model.Runtime,
        n_variables=model.NumVars,
        n_constraints=model.NumConstrs,
    )


def solve_set_cover_relaxation(coverage: np.ndarray, quiet: bool = True) -> float:
    """LP relaxation of set cover, for the integrality gap."""
    n_demand, n_candidates = coverage.shape

    model = _new_model("set_cover_lp", quiet)
    y = model.addVars(n_candidates, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
    model.setObjective(gp.quicksum(y[j] for j in range(n_candidates)), GRB.MINIMIZE)
    for i in range(n_demand):
        covering = np.flatnonzero(coverage[i]).tolist()
        if covering:
            model.addConstr(gp.quicksum(y[j] for j in covering) >= 1)

    model.optimize()
    return float(model.ObjVal)


def solve_max_cover(
    coverage: np.ndarray,
    p: int,
    weights: np.ndarray | None = None,
    quiet: bool = True,
) -> Solution:
    """Place exactly ``p`` stations to maximise covered weight.

    ``weights=None`` maximises the count of covered block groups. Passing call
    counts maximises covered call volume instead.
    """
    n_demand, n_candidates = coverage.shape
    v = np.ones(n_demand) if weights is None else np.asarray(weights, dtype=float)

    model = _new_model("max_cover", quiet)
    y = model.addVars(n_candidates, vtype=GRB.BINARY, name="y")
    z = model.addVars(n_demand, vtype=GRB.BINARY, name="z")

    model.setObjective(gp.quicksum(float(v[i]) * z[i] for i in range(n_demand)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(y[j] for j in range(n_candidates)) == p, name="cardinality")

    for i in range(n_demand):
        covering = np.flatnonzero(coverage[i]).tolist()
        if covering:
            model.addConstr(z[i] <= gp.quicksum(y[j] for j in covering), name=f"link_{i}")
        else:
            model.addConstr(z[i] == 0, name=f"unreachable_{i}")

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Max cover did not solve to optimality (status {model.Status}).")

    selected = [j for j in range(n_candidates) if y[j].X > 0.5]
    covered = [i for i in range(n_demand) if z[i].X > 0.5]
    return Solution(
        selected=selected,
        covered=covered,
        objective=model.ObjVal,
        bound=model.ObjBound,
        gap=model.MIPGap,
        runtime=model.Runtime,
        n_variables=model.NumVars,
        n_constraints=model.NumConstrs,
        extra={"p": p, "weighted": weights is not None},
    )


def assign_to_nearest(time_matrix: np.ndarray, selected: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest open station for every demand point.

    Returns ``(response_minutes, station_index)``. Assignment is to the nearest
    station, which is what an EMS dispatcher does, and is not the same thing as
    being covered: an uncovered demand point still has a nearest station, just
    one that arrives past the threshold.
    """
    if not selected:
        raise ValueError("No stations selected.")
    sub = time_matrix[:, selected]
    nearest = sub.argmin(axis=1)
    return sub[np.arange(len(sub)), nearest], np.asarray(selected)[nearest]
