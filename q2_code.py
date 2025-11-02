#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ruixu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2 — Realistic Capacity Expansion & Location
Data cleaning follows the *Q1 style*, plus a NY state lat/lon bounding-box filter:
  lat in [40.5, 45.0], lon in [-79.8, -71.8].

Key points
----------
- Cleaning matches Q1:
  * ZIP normalization: first 5 digits + zfill(5)
  * Population: Pop05 = '-5'; Pop012 = '-5' + '5-9' + 0.6*'10-14'
  * Income: numeric + mean imputation
  * Employment rate: convert % to [0,1] if needed + mean imputation
  * Regulated: capacity = total_capacity; exist_05 = infant + toddler + preschool
- Extra cleaning: drop facilities and candidate sites outside NY bounds.
- Modeling/constraints are *unchanged* from Q2:
  piecewise expansion bands, 0.06 mile spacing, 0–12 and 0–5 coverage, equipment cost, etc.
"""

import math
import time
import pandas as pd
import numpy as np
from itertools import combinations

# -----------------------------
# Paths & knobs
# -----------------------------
PATHS = {
    "population": "population.csv",
    "income": "avg_individual_income.csv",
    "employment": "employment_rate.csv",
    "regulated": "child_care_regulated.csv",
    "sites": "potential_locations.csv",
}


# Build types (assignment defaults; do not change constraints)
BUILD_TYPES = [
    {"name": "S", "S_k": 100, "U_k":  50, "C_k":  65000},
    {"name": "M", "S_k": 200, "U_k": 100, "C_k":  95000},
    {"name": "L", "S_k": 400, "U_k": 200, "C_k": 115000},
]

# Desert thresholds will come from Q1-style theta per ZIP:
# theta[z] ∈ {1/2 (high-demand), 1/3 (normal)}
TAU_NORMAL = 1/3
TAU_HIGH   = 1/2

# Spacing & costs (unchanged)
DELTA_MILES = 0.06
EQUIP_COST_UNDER5 = 100.0

# Solver parameters (unchanged)
MIP_GAP    = 0.005
TIME_LIMIT = 300       # seconds per ZIP
THREADS    = 4
SHOW_GUROBI_LOG = False

# -----------------------------
# Helpers
# -----------------------------
def zip5(s: pd.Series) -> pd.Series:
    """Q1-style ZIP normalization: take first 5 chars and zfill(5)."""
    return s.astype(str).str[:5].str.zfill(5)

def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.7613  # miles
    from math import radians, sin, cos, asin, sqrt
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dlmb/2)**2
    return 2*R*asin(sqrt(a))

def clip_to_ny_bounds(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Keep only points inside NY bounding box: lat [40.5, 45.0], lon [-79.8, -71.8]."""
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    m = (
        df[lat_col].between(40.5, 45.0, inclusive="both")
        & df[lon_col].between(-79.8, -71.8, inclusive="both")
    )
    return df.loc[m].dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

# -----------------------------
# Q1-style cleaning + NY bounds
# -----------------------------
def load_clean_population_q1(path: str) -> pd.DataFrame:
    """Return ZIP-level population with Q1 definitions."""
    pop = pd.read_csv(path)
    pop["zipcode"] = zip5(pop["zipcode"])
    # Q1 definitions:
    # pop_0_5 = '-5' ; pop_0_12 = '-5' + '5-9' + 0.6*'10-14'
    # (Assume these columns exist; if not, please align column names upstream.)
    pop["pop_0_5"]  = pd.to_numeric(pop["-5"], errors="coerce").fillna(0.0)
    five9           = pd.to_numeric(pop["5-9"], errors="coerce").fillna(0.0)
    ten14           = pd.to_numeric(pop["10-14"], errors="coerce").fillna(0.0)
    pop["pop_0_12"] = pop["pop_0_5"] + five9 + 0.6*ten14
    out = pop[["zipcode", "pop_0_5", "pop_0_12"]].copy()
    return out

def load_clean_income_q1(path: str) -> pd.DataFrame:
    """Q1: mean imputation, specific column names."""
    inc = pd.read_csv(path)
    # Standardize column names used in Q1 scripts
    if "ZIP code" in inc.columns:
        inc = inc.rename(columns={"ZIP code": "zipcode"})
    if "average income" in inc.columns:
        inc = inc.rename(columns={"average income": "avg_income"})
    inc["zipcode"] = zip5(inc["zipcode"])
    inc["avg_income"] = pd.to_numeric(inc["avg_income"], errors="coerce")
    mean_income = inc["avg_income"].mean(skipna=True)
    inc["avg_income"] = inc["avg_income"].fillna(mean_income)
    return inc[["zipcode", "avg_income"]].copy()

def load_clean_employment_q1(path: str) -> pd.DataFrame:
    """Q1: rate to [0,1] if needed + mean imputation."""
    emp = pd.read_csv(path)
    if "employment rate" in emp.columns:
        emp = emp.rename(columns={"employment rate": "employment_rate"})
    emp["zipcode"] = zip5(emp["zipcode"])
    emp["employment_rate"] = pd.to_numeric(emp["employment_rate"], errors="coerce")
    if emp["employment_rate"].max(skipna=True) > 1.0:
        emp["employment_rate"] = emp["employment_rate"] / 100.0
    mean_rate = emp["employment_rate"].mean(skipna=True)
    emp["employment_rate"] = emp["employment_rate"].fillna(mean_rate).clip(0, 1)
    return emp[["zipcode", "employment_rate"]].copy()

def load_clean_regulated_q1_ny(path: str) -> pd.DataFrame:
    """
    Q1-style facility cleaning + NY bounding box.
    - capacity = total_capacity
    - exist_05 = infant + toddler + preschool
    - keep only NY-bounded lat/lon
    """
    fac = pd.read_csv(path)
    if "zip_code" in fac.columns:
        fac = fac.rename(columns={"zip_code": "zipcode"})
    fac["zipcode"] = zip5(fac["zipcode"])

    # capacity
    if "total_capacity" in fac.columns:
        fac["capacity"] = fac["total_capacity"]
    fac["capacity"] = pd.to_numeric(fac["capacity"], errors="coerce").fillna(0.0)

    # ensure 0–5 age buckets exist
    for col in ["infant_capacity", "toddler_capacity", "preschool_capacity"]:
        if col not in fac.columns:
            fac[col] = 0.0
        fac[col] = pd.to_numeric(fac[col], errors="coerce").fillna(0.0)

    fac["exist_05"] = fac["infant_capacity"] + fac["toddler_capacity"] + fac["preschool_capacity"]

    # NY bounds filter
    fac = clip_to_ny_bounds(fac, "latitude", "longitude")

    # model expects: FacID, ZIP, lat, lon, n_f, u5_f
    out = fac.rename(columns={
        "facility_id": "FacID",
        "zipcode": "ZIP",
        "latitude": "lat",
        "longitude": "lon"
    })[["FacID", "ZIP", "lat", "lon", "capacity", "exist_05"]].copy()
    out = out.rename(columns={"capacity": "n_f", "exist_05": "u5_f"})
    return out

def load_clean_sites_q1_ny(path: str) -> pd.DataFrame:
    """
    Q1-style candidate sites + NY bounding box.
    - ZIP = first 5 digits
    - Keep lat/lon inside NY bounds
    - Generate SiteID for modeling
    """
    pot = pd.read_csv(path)
    pot["zipcode"] = zip5(pot["zipcode"])
    pot = clip_to_ny_bounds(pot, "latitude", "longitude")

    out = pot.rename(columns={
        "zipcode": "ZIP",
        "latitude": "lat",
        "longitude": "lon"
    })[["ZIP", "lat", "lon"]].copy()
    out = out.reset_index(drop=False).rename(columns={"index": "SiteID"})
    out["SiteID"] = out["SiteID"].apply(lambda i: f"SITE_{i+1:05d}")
    return out[["SiteID", "ZIP", "lat", "lon"]]

def clean_all_q1plusNY() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (zip_df, reg_df, site_df) cleaned per Q1 + NY bounds."""
    pop = load_clean_population_q1(PATHS["population"])
    inc = load_clean_income_q1(PATHS["income"])
    emp = load_clean_employment_q1(PATHS["employment"])
    reg = load_clean_regulated_q1_ny(PATHS["regulated"])
    site = load_clean_sites_q1_ny(PATHS["sites"])

    z_df = (pop.merge(inc, on="zipcode", how="left")
                .merge(emp, on="zipcode", how="left"))
    # Q1-style theta mapping:
    # high-demand if employment_rate>=0.6 OR avg_income<=60000
    z_df["theta"] = np.where(
        (z_df["employment_rate"] >= 0.60) | (z_df["avg_income"] <= 60000.0),
        TAU_HIGH, TAU_NORMAL
    )
    z_df = z_df.rename(columns={"zipcode": "ZIP",
                                "pop_0_5": "Pop05",
                                "pop_0_12": "Pop012"})
    return z_df[["ZIP", "Pop05", "Pop012", "avg_income", "employment_rate", "theta"]], reg, site

# -----------------------------
# Modeling (unchanged)
# -----------------------------
def solve_one_zip_gurobi(z, zip_row, Fz, Pz, reg_df, site_df, types, n_f_map, u5_f_map):
    """Build and solve the ZIP-level MILP (constraints unchanged)."""
    import gurobipy as gp
    from gurobipy import GRB

    Pop012 = float(zip_row["Pop012"])
    Pop05  = float(zip_row["Pop05"])
    tau_z  = float(zip_row["theta"])  # Q1-style: directly use theta

    # Coords
    coord_f = reg_df.loc[reg_df["FacID"].isin(Fz), ["FacID","lat","lon"]].set_index("FacID").to_dict(orient="index")
    coord_p_all = site_df.loc[site_df["SiteID"].isin(Pz), ["SiteID","lat","lon"]].set_index("SiteID").to_dict(orient="index")

    # Remove candidates too close to existing facilities (< DELTA_MILES)
    allowed_sites = []
    for p in Pz:
        cp = coord_p_all[p]
        ok = True
        for f in Fz:
            d = haversine_miles(cp["lat"], cp["lon"], coord_f[f]["lat"], coord_f[f]["lon"])
            if d < DELTA_MILES:
                ok = False
                break
        if ok:
            allowed_sites.append(p)
    Pz = allowed_sites
    coord_p = {pid: coord_p_all[pid] for pid in Pz}

    # Site-site conflicts (< DELTA_MILES)
    conflicts = []
    for p, q in combinations(Pz, 2):
        d = haversine_miles(coord_p[p]["lat"], coord_p[p]["lon"], coord_p[q]["lat"], coord_p[q]["lon"])
        if d < DELTA_MILES:
            conflicts.append((p, q))

    # Piecewise expansion parameters
    def band_caps(nf): return 0.10*nf, 0.05*nf, 0.05*nf
    def band_cost_coeffs(nf):
        base = 20000.0 / max(nf, 1.0)
        return base + 200.0, base + 400.0, base + 1000.0

    b1, b2, b3 = {}, {}, {}
    a1, a2, a3 = {}, {}, {}
    for f in Fz:
        nf = n_f_map[f]
        _b1,_b2,_b3 = band_caps(nf)
        b1[f], b2[f], b3[f] = _b1, _b2, _b3
        _a1,_a2,_a3 = band_cost_coeffs(nf)
        a1[f], a2[f], a3[f] = _a1, _a2, _a3

    # Build model
    m = gp.Model(f"Q2_ZIP_{z}")
    m.Params.OutputFlag = 1 if SHOW_GUROBI_LOG else 0
    m.Params.MIPGap     = MIP_GAP
    m.Params.TimeLimit  = TIME_LIMIT
    m.Params.Threads    = THREADS

    K = [t["name"] for t in types]
    S_k = {t["name"]: float(t["S_k"]) for t in types}
    U_k = {t["name"]: float(t["U_k"]) for t in types}
    C_k = {t["name"]: float(t["C_k"]) for t in types}

    # Variables
    y = m.addVars([(p,k) for p in Pz for k in K], vtype=GRB.BINARY, name="y")
    z1 = m.addVars(Fz, lb=0.0, ub={f: b1[f] for f in Fz}, name="z1")
    z2 = m.addVars(Fz, lb=0.0, ub={f: b2[f] for f in Fz}, name="z2")
    z3 = m.addVars(Fz, lb=0.0, ub={f: b3[f] for f in Fz}, name="z3")
    v_new = m.addVar(lb=0.0, name="v_new_zip")
    u_exp = m.addVars(Fz, lb=0.0, name="u_exp")

    # Objective (unchanged)
    build_cost  = gp.quicksum(C_k[k]*y[p,k] for p in Pz for k in K)
    expand_cost = gp.quicksum(a1[f]*z1[f] + a2[f]*z2[f] + a3[f]*z3[f] for f in Fz)
    equip_cost  = EQUIP_COST_UNDER5 * (v_new + gp.quicksum(u_exp[f] for f in Fz))
    m.setObjective(build_cost + expand_cost + equip_cost, GRB.MINIMIZE)

    # Constraints (unchanged)
    for p in Pz:
        m.addConstr(gp.quicksum(y[p,k] for k in K) <= 1, name=f"one_type[{p}]")
    for p,q in conflicts:
        m.addConstr(gp.quicksum(y[p,k] for k in K) + gp.quicksum(y[q,k] for k in K) <= 1,
                    name=f"conflict[{p},{q}]")

    m.addConstr(v_new <= gp.quicksum(U_k[k]*y[p,k] for p in Pz for k in K), name="vnew_link")

    for f in Fz:
        m.addConstr(u_exp[f] <= z1[f] + z2[f] + z3[f], name=f"u_exp_le_sumz[{f}]")

    new_builds = gp.quicksum(S_k[k]*y[p,k] for p in Pz for k in K)
    existing_plus_exp = gp.quicksum((n_f_map[f] + z1[f] + z2[f] + z3[f]) for f in Fz)
    m.addConstr(existing_plus_exp + new_builds >= tau_z * Pop012, name="cover_0_12")

    U5_exist = float(sum(u5_f_map.get(f, 0.0) for f in Fz))
    m.addConstr(U5_exist + v_new + gp.quicksum(u_exp[f] for f in Fz) >= (2.0/3.0) * Pop05, name="cover_0_5")

    # Solve
    m.optimize()
    status = int(m.Status)
    obj = float(m.ObjVal) if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) else None

    # Extract
    chosen = [(p,k) for p in Pz for k in K if (y[p,k].X if obj is not None else 0) > 0.5]
    exp_rows = []
    for f in Fz:
        exp_rows.append({
            "FacID": f,
            "z1": z1[f].X if obj is not None else 0.0,
            "z2": z2[f].X if obj is not None else 0.0,
            "z3": z3[f].X if obj is not None else 0.0,
            "u_exp": u_exp[f].X if obj is not None else 0.0
        })
    return {
        "zip": z, "status": status, "obj": obj,
        "builds": chosen,
        "expansion": pd.DataFrame(exp_rows),
        "U5_exist": U5_exist,
        "v_new": (v_new.X if obj is not None else 0.0)
    }

# -----------------------------
# Main
# -----------------------------
def main():
    # Q1-style cleaning + NY bounds (only change you requested)
    zip_df, reg_df, site_df = clean_all_q1plusNY()

    # Sets/maps for modeling (unchanged)
    n_f_map  = reg_df.set_index("FacID")["n_f"].to_dict()
    u5_f_map = reg_df.set_index("FacID")["u5_f"].to_dict()
    F_by_zip = reg_df.groupby("ZIP")["FacID"].apply(list).to_dict()
    P_by_zip = site_df.groupby("ZIP")["SiteID"].apply(list).to_dict()

    types = BUILD_TYPES

    solve_results = []
    total_obj = 0.0
    t0 = time.perf_counter()
    print(f"\nSolving {len(zip_df)} ZIPs | GAP={MIP_GAP}, TL={TIME_LIMIT}s, Threads={THREADS}")

    for i, row in zip_df.iterrows():
        z = row["ZIP"]
        Fz = F_by_zip.get(z, [])
        Pz = P_by_zip.get(z, [])

        # If a ZIP has neither facilities nor candidates, quick feasibility check
        if len(Fz) == 0 and len(Pz) == 0:
            feas = (0 >= float(row["theta"]) * row["Pop012"]) and (0 >= (2.0/3.0) * row["Pop05"])
            solve_results.append({
                "zip": z, "status": 2 if feas else 3, "obj": 0.0 if feas else None,
                "builds": [], "expansion": pd.DataFrame()
            })
            print(f"[{i+1}/{len(zip_df)}] ZIP {z}: skipped (no F/S), feas={feas}")
            continue

        tic = time.perf_counter()
        try:
            res = solve_one_zip_gurobi(z, row, Fz, Pz, reg_df, site_df, types, n_f_map, u5_f_map)
        except Exception as e:
            res = {"zip": z, "status": -1, "error": str(e), "obj": None, "builds": [], "expansion": pd.DataFrame()}
        dt = time.perf_counter() - tic

        print(f"[{i+1}/{len(zip_df)}] ZIP {z} | status={res.get('status')} | obj={res.get('obj')} | {dt:.1f}s")
        solve_results.append(res)
        if res.get("obj") is not None:
            total_obj += res["obj"]

    total_time = time.perf_counter() - t0
    print(f"\n=== Done. Total time: {total_time:.1f}s ===")
    print(f"Total objective (USD): {round(total_obj, 2)}")

    # Aggregate and save (unchanged)
    builds_all, exp_all = [], []
    for res in solve_results:
        for (site_id, k) in res.get("builds", []):
            builds_all.append({"ZIP": res["zip"], "SiteID": site_id, "Type": k})
        df = res.get("expansion")
        if isinstance(df, pd.DataFrame) and len(df):
            d = df.copy()
            d["ZIP"] = res["zip"]
            exp_all.append(d)

    builds_df = pd.DataFrame(builds_all)
    exp_df = pd.concat(exp_all, ignore_index=True) if exp_all else pd.DataFrame(columns=["FacID","z1","z2","z3","u_exp","ZIP"])

    builds_df.to_csv("q2_gurobi_builds.csv", index=False)
    exp_df.to_csv("q2_gurobi_expansion.csv", index=False)

    print("\n==== Summary ====")
    print("Chosen new facilities (#):", len(builds_df))
    if len(builds_df):
        print(builds_df.head(20).to_string(index=False))
    print("\nExpansions (first 10 rows):")
    print(exp_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
