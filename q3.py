import pandas as pd
from gurobipy import Model, GRB, quicksum
from geopy.distance import great_circle
import gurobipy as gp

# Input your gurobi license in params
env = gp.Env(params=params)


# --- Constants for the Fairness Problem ---
BUDGET_LIMIT = 100_000_000.0 # $100 million
FAIRNESS_GAP = 0.1           # Max allowed gap in coverage ratios
MIN_DISTANCE = 0.06          # 0.06 miles minimum distance between facilities

# --- Data Reading and Cleaning (reused from Q1) ---
print("Loading and cleaning data...")

pop = pd.read_csv('population.csv')
pop['zipcode'] = pop['zipcode'].astype(str).str[:5].str.zfill(5)
pop['pop_0_5'] = pop['-5']
pop['pop_0_12'] = pop['-5'] + pop['5-9'] + 0.6 * pop['10-14']
# NEW: Need pop_5_12 for the objective function
pop['pop_5_12'] = pop['pop_0_12'] - pop['pop_0_5']
pop = pop[['zipcode', 'pop_0_5', 'pop_5_12', 'pop_0_12']]

inc = pd.read_csv('avg_individual_income.csv')
inc.rename(columns={'ZIP code': 'zipcode', 'average income': 'avg_income'}, inplace=True)
inc['zipcode'] = inc['zipcode'].astype(str).str[:5].str.zfill(5)
inc['avg_income'] = pd.to_numeric(inc['avg_income'], errors='coerce')
mean_income = inc['avg_income'].mean(skipna=True)
inc['avg_income'] = inc['avg_income'].fillna(mean_income)

emp = pd.read_csv('employment_rate.csv')
emp.rename(columns={'employment rate': 'employment_rate'}, inplace=True)
emp['zipcode'] = emp['zipcode'].astype(str).str[:5].str.zfill(5)
emp['employment_rate'] = pd.to_numeric(emp['employment_rate'], errors='coerce')
if emp['employment_rate'].max(skipna=True) > 1.0:
    emp['employment_rate'] = emp['employment_rate'] / 100.0
mean_rate = emp['employment_rate'].mean(skipna=True)
emp['employment_rate'] = emp['employment_rate'].fillna(mean_rate).clip(0, 1)

fac = pd.read_csv('child_care_regulated.csv')
fac.rename(columns={'zip_code': 'zipcode'}, inplace=True)
# --- FIX: Drop rows with missing coordinates ---
fac = fac.dropna(subset=['latitude', 'longitude'])
fac['zipcode'] = fac['zipcode'].astype(str).str[:5].str.zfill(5)
fac['capacity'] = fac['total_capacity']
for col in ['infant_capacity', 'toddler_capacity', 'preschool_capacity']:
    if col not in fac.columns:
        fac[col] = 0.0
    fac[col] = fac[col].fillna(0.0)
fac['exist_05'] = (fac['infant_capacity'] + fac['toddler_capacity'] + fac['preschool_capacity'])
fac = fac[['facility_id', 'program_type', 'zipcode','capacity', 'latitude', 'longitude', 'exist_05']]

pot = pd.read_csv('potential_locations.csv')
# --- FIX: Drop rows with missing coordinates ---
pot = pot.dropna(subset=['latitude', 'longitude'])
pot['zipcode'] = pot['zipcode'].astype(str).str[:5].str.zfill(5)
pot = pot[['zipcode', 'latitude', 'longitude']].reset_index(drop=False).rename(columns={'index': 'loc_id'})

# Merge
z_df = (pop.merge(inc, on='zipcode', how='left')
       .merge(emp, on='zipcode', how='left'))
z_df['theta'] = ((z_df['employment_rate'] >= 0.6) |(z_df['avg_income'] <= 60000)).map({True: 0.5, False: 1/3})

print("Data processing complete.")

# --- Sets and Parameters ---
print("Setting up model parameters...")

Z = list(z_df['zipcode'])
F = list(fac['facility_id'])
L = list(pot['loc_id'])
S = ['S','M','L'] # Small, Medium, Large

# Population parameters
Pop05  = dict(zip(z_df['zipcode'], z_df['pop_0_5'].astype(float)))
Pop512 = dict(zip(z_df['zipcode'], z_df['pop_5_12'].astype(float)))
Pop012 = dict(zip(z_df['zipcode'], z_df['pop_0_12'].astype(float)))
theta  = dict(zip(z_df['zipcode'], z_df['theta'].astype(float))) # Desert threshold

# Facility parameters
z_of_f = dict(zip(fac['facility_id'], fac['zipcode']))
n_f    = dict(zip(fac['facility_id'], fac['capacity'].astype(float)))
eub = {f: 0.2 * n_f[f] for f in F if n_f[f] > 0} # Expansion Upper Bound (20%)

# Potential location parameters
z_of_l = dict(zip(pot['loc_id'], pot['zipcode']))
Cap_tot = {'S':100.0, 'M':200.0, 'L':400.0}
Cap_05  = {'S':50.0,  'M':100.0, 'L':200.0}
C_build = {'S':65000.0, 'M':95000.0, 'L':115000.0} # Build cost

# Cost parameters
alpha = 2.0/3.0 # 0-5 coverage constraint
C_expand_fixed = 20000.0
# Marginal cost coefficients per tier
C_tier_coeff = {1: 200.0, 2: 400.0, 3: 1000.0}
# Tier limits as a fraction of n_f
C_tier_limits = {1: 0.10, 2: 0.15, 3: 0.20}

# Helper lookups
F_by_z = {z: [f for f in F if z_of_f[f]==z] for z in Z}
L_by_z = {z: [l for l in L if z_of_l[l]==z] for z in Z}
exist05_by_z    = fac.groupby('zipcode')['exist_05'].sum().to_dict()
existTotal_by_z = fac.groupby('zipcode')['capacity'].sum().to_dict()
for z in Z:
    exist05_by_z.setdefault(z, 0.0)
    existTotal_by_z.setdefault(z, 0.0)

# Coordinates for distance constraint
fac_coords = {row.facility_id: (row.latitude, row.longitude) for row in fac.itertuples()}
pot_coords = {row.loc_id: (row.latitude, row.longitude) for row in pot.itertuples()}

print("Parameter setup complete.")

# --- Initialize Model ---
m = Model('Problem_of_Fairness',env=env)
m.Params.OutputFlag = 1

# --- Decision Variables ---
print("Defining variables...")

# New facilities
y = m.addVars(L, S, vtype=GRB.BINARY, name='y')
x_new_05  = m.addVars(L, S, lb=0.0, vtype=GRB.CONTINUOUS, name='x_new_05')
x_new_512 = m.addVars(L, S, lb=0.0, vtype=GRB.CONTINUOUS, name='x_new_512')

# Existing facilities (expansion)
e = m.addVars(F, lb=0.0, vtype=GRB.CONTINUOUS, name='e') # total added slots
x_ext_05  = m.addVars(F, lb=0.0, vtype=GRB.CONTINUOUS, name='x_ext_05')
x_ext_512 = m.addVars(F, lb=0.0, vtype=GRB.CONTINUOUS, name='x_ext_512')

# 3-Tier piecewise expansion variables
e_tier1 = m.addVars(F, lb=0.0, vtype=GRB.CONTINUOUS, name='e_tier1') # (0, 0.1]
e_tier2 = m.addVars(F, lb=0.0, vtype=GRB.CONTINUOUS, name='e_tier2') # (0.1, 0.15]
e_tier3 = m.addVars(F, lb=0.0, vtype=GRB.CONTINUOUS, name='e_tier3') # (0.15, 0.2]

# Fairness variables
R = m.addVars(Z, lb=0.0, vtype=GRB.CONTINUOUS, name='R') # R_z = TotalSlots_z / Pop012_z
R_min = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='R_min')
R_max = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='R_max')

# Helper variable for distance constraint
y_loc = m.addVars(L, vtype=GRB.BINARY, name='y_loc') # 1 if any facility built at l

# --- Helper Expressions for Slot Totals ---
TotalSlots_05 = {}
TotalSlots_512 = {}
TotalSlots_All = {}

for z in Z:
    # Total 0-5 slots in zip z
    TotalSlots_05[z] = exist05_by_z[z] \
        + quicksum(x_ext_05[f] for f in F_by_z[z]) \
        + quicksum(x_new_05[l,s] for l in L_by_z[z] for s in S)

    # Total 5-12 slots in zip z
    TotalSlots_512[z] = (existTotal_by_z[z] - exist05_by_z[z]) \
        + quicksum(x_ext_512[f] for f in F_by_z[z]) \
        + quicksum(x_new_512[l,s] for l in L_by_z[z] for s in S)

    # Total 0-12 slots in zip z
    TotalSlots_All[z] = TotalSlots_05[z] + TotalSlots_512[z]

# --- Objective Function: Maximize Social Coverage Index ---
print("Defining objective function...")
objective_expr = quicksum(
    (2/3) * (TotalSlots_05[z] / Pop05[z])    # 2/3 weight for 0-5
  + (1/3) * (TotalSlots_512[z] / Pop512[z])  # 1/3 weight for 5-12
    for z in Z if Pop05[z] > 0 and Pop512[z] > 0
)
m.setObjective(objective_expr, GRB.MAXIMIZE)

# --- Define Cost Expressions for Budget Constraint ---
build_cost = quicksum(C_build[s] * y[l,s] for l in L for s in S)

# Marginal cost for each tier
expansion_cost = quicksum(
    ((C_expand_fixed / n_f[f]) + C_tier_coeff[1]) * e_tier1[f] +
    ((C_expand_fixed / n_f[f]) + C_tier_coeff[2]) * e_tier2[f] +
    ((C_expand_fixed / n_f[f]) + C_tier_coeff[3]) * e_tier3[f]
    for f in F if n_f[f] > 0
)

# --- Constraints ---
print("Adding constraints...")

# 1) Budget Constraint
m.addConstr(build_cost + expansion_cost <= BUDGET_LIMIT, name="budget")

# 2) New Facility Constraints
for l in L:
    # Can only build one size at each location
    m.addConstr(quicksum(y[l,s] for s in S) <= 1, name=f'one_size[{l}]')
    # Link y_loc helper variable
    m.addConstr(y_loc[l] == quicksum(y[l,s] for s in S), name=f'y_loc_link[{l}]')

    for s in S:
        # Total capacity of new build
        m.addConstr(x_new_05[l,s] + x_new_512[l,s] <= Cap_tot[s] * y[l,s], name=f'new_tot[{l},{s}]')
        # 0-5 capacity of new build
        m.addConstr(x_new_05[l,s] <= Cap_05[s] * y[l,s], name=f'new_05_cap[{l},{s}]')

# 3) Expansion Constraints
for f in F:
    # Link total expansion to its 0-5 and 5-12 components
    m.addConstr(x_ext_05[f] + x_ext_512[f] == e[f], name=f'ext_link[{f}]')
    # Enforce 20% expansion cap
    m.addConstr(e[f] <= eub.get(f, 0.0), name=f'ext_cap[{f}]')

    # Link 3-tier piecewise variables
    if n_f[f] > 0:
        m.addConstr(e[f] == e_tier1[f] + e_tier2[f] + e_tier3[f], name=f'piece_link[{f}]')
        # Tier 1 cap: 0% to 10%
        m.addConstr(e_tier1[f] <= (C_tier_limits[1] - 0) * n_f[f], name=f'piece_cap1[{f}]')
        # Tier 2 cap: 10% to 15%
        m.addConstr(e_tier2[f] <= (C_tier_limits[2] - C_tier_limits[1]) * n_f[f], name=f'piece_cap2[{f}]')
        # Tier 3 cap: 15% to 20%
        m.addConstr(e_tier3[f] <= (C_tier_limits[3] - C_tier_limits[2]) * n_f[f], name=f'piece_cap3[{f}]')
    else:
        # Cannot expand a facility with 0 capacity
        m.addConstr(e[f] == 0)
        m.addConstr(e_tier1[f] == 0)
        m.addConstr(e_tier2[f] == 0)
        m.addConstr(e_tier3[f] == 0)

# 4) Eliminate Deserts (0-12)
for z in Z:
    m.addConstr(TotalSlots_All[z] >= theta[z] * Pop012[z], name=f'desert[{z}]')

# 5) 0-5 Coverage
for z in Z:
    m.addConstr(TotalSlots_05[z] >= alpha * Pop05[z], name=f'cov05[{z}]')

# 6) Fairness Constraint
for z in Z:
    # Define the ratio R_z
    if Pop012[z] > 0:
        m.addConstr(R[z] * Pop012[z] == TotalSlots_All[z], name=f'ratio_def[{z}]')
    else:
        m.addConstr(R[z] == 0) # Define ratio as 0 if no population

    # Link R_z to the min/max variables
    m.addConstr(R[z] >= R_min, name=f'rmin_link[{z}]')
    m.addConstr(R[z] <= R_max, name=f'rmax_link[{z}]')

# Enforce the maximum gap
m.addConstr(R_max - R_min <= FAIRNESS_GAP, name="fairness_gap")

# 7) Distance Constraint
print("Adding distance constraints...")
for z in Z:
    F_z = F_by_z.get(z, []) # Existing facilities in zip
    L_z = L_by_z.get(z, []) # Potential locations in zip

    # A) (Existing vs New)
    for f in F_z:
        for l in L_z:
            coord_f = fac_coords.get(f)
            coord_l = pot_coords.get(l)
            # Check if lookups were successful (they should be, but good practice)
            if coord_f and coord_l:
                dist = great_circle(coord_f, coord_l).miles
                if dist < MIN_DISTANCE:
                    # If f and l are too close, cannot build at l
                    m.addConstr(y_loc[l] == 0, name=f'dist_exist_new[{f},{l}]')

    # B) (New vs New)
    for i in range(len(L_z)):
        for j in range(i + 1, len(L_z)):
            l1 = L_z[i]
            l2 = L_z[j]
            coord_l1 = pot_coords.get(l1)
            coord_l2 = pot_coords.get(l2)
            # Check if lookups were successful
            if coord_l1 and coord_l2:
                dist = great_circle(coord_l1, coord_l2).miles
                if dist < MIN_DISTANCE:
                    # If l1 and l2 are too close, can build at most one
                    m.addConstr(y_loc[l1] + y_loc[l2] <= 1, name=f'dist_new_new[{l1},{l2}]')

# --- Solve ---
print("All constraints added. Starting optimization...")
m.optimize()

# --- Report Results ---
if m.status == GRB.OPTIMAL:
    social_index = m.objVal
    total_cost = build_cost.getValue() + expansion_cost.getValue()

    print(f"\n" + "="*30)
    print(f"   OPTIMAL SOLUTION FOUND")
    print(f"   Maximum Social Coverage Index: {social_index:,.4f}")
    print(f"="*30)

    print(f"\nBUDGET & COST:")
    print(f"  Total Cost:                ${total_cost:,.2f}")
    print(f"  Budget Limit:              ${BUDGET_LIMIT:,.2f}")
    print(f"  Construction Cost:         ${build_cost.getValue():,.2f}")
    print(f"  Expansion Cost:            ${expansion_cost.getValue():,.2f}")

    min_ratio = R_min.X
    max_ratio = R_max.X
    print(f"\nFAIRNESS RATIO (Slots / 0-12 Pop):")
    print(f"  Min Ratio (R_min):         {min_ratio:.4f}")
    print(f"  Max Ratio (R_max):         {max_ratio:.4f}")
    print(f"  Gap (R_max - R_min):       {(max_ratio - min_ratio):.4f} (Constraint: <= {FAIRNESS_GAP})")

    chosen_new = [(l, s) for l in L for s in S if y[l, s].X > 0.5]
    print(f"\nNEW FACILITIES BUILT: {len(chosen_new)}")
    print(f"  Sites (first 15): {chosen_new[:15]}{' ...' if len(chosen_new) > 15 else ''}")

elif m.status == GRB.INFEASIBLE:
    print(f"\n" + "="*30)
    print(f"   MODEL IS INFEASIBLE")
    print(f"   It is impossible to satisfy all constraints simultaneously.")
    print(f"="*30)
    print("Per the project instructions, this is the reported issue:")
    print("It is impossible to satisfy the fairness requirement (max gap of 0.1) "
          "while also eliminating all deserts and meeting the 0-5 coverage quota "
          f"within the ${BUDGET_LIMIT:,.0f} budget and distance constraints.")

    # Compute and write IIS (Irreducible Inconsistent Subsystem) to help debug
    print("\nComputing IIS to find conflicting constraints...")
    m.computeIIS()
    m.write("fairness_model_conflict.ilp")
    print("IIS written to 'fairGness_model_conflict.ilp'")

else:
    print(f"\n--- Optimization Error ---")
    print(f"Model status code: {m.status}")


