#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprint Planning MILP (PuLP)
- Maximize delivered value minus lambda * active_people
- Enforce: capacities, role coverage, <=13 points per dev-owner, >=1 release for active devs,
  dependencies, bug reservations, QA WIP buffer.
Data: data/*.csv + data/config.yaml
Outputs: results/*.csv + results/summary.txt
"""
import os, csv, yaml, math, sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

try:
    import pulp as pl
except Exception as e:
    print("PuLP no está instalado en este entorno.\n"
          "Para ejecutar localmente:\n"
          "  pip install pulp\n"
          "Luego corre: python solver/solve_sprint.py\n")
    sys.exit(0)

# ---------- Load data ----------
with open(os.path.join(DATA,"people.csv"),encoding="utf-8") as f:
    people_rows = list(csv.DictReader(f))
with open(os.path.join(DATA,"stories.csv"),encoding="utf-8") as f:
    story_rows = list(csv.DictReader(f))
with open(os.path.join(DATA,"roles.csv"),encoding="utf-8") as f:
    role_rows = list(csv.DictReader(f))
with open(os.path.join(DATA,"config.yaml"),encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

hours_per_point = float(CFG["hours_per_point"])
bugs_per_sprint = int(CFG["bugs_per_sprint"])
max_points_per_dev = int(CFG["max_points_per_dev"])
lambda_people = float(CFG["lambda_people_penalty"])
require_release_roles = set(CFG["require_release_for_roles"])
theta_release_hours = float(CFG["min_hours_to_count_release"])
qa_cov_factor = float(CFG["qa_coverage_factor"])
wip_factor_QA = float(CFG["wip_factor_QA_capacity"])
forbid_points = set(int(x) for x in CFG.get("forbid_points",[]))

I = [p["person"] for p in people_rows]
role_of = {p["person"]:p["role"] for p in people_rows}
cap_i = {p["person"]: float(p["capacity_hours"]) for p in people_rows}
I_by_role = defaultdict(list)
for i in I:
    I_by_role[ role_of[i] ].append(i)

S_all = [s["story_id"] for s in story_rows]
S = [s["story_id"] for s in story_rows if int(s["points"]) not in forbid_points]
points = {s["story_id"]: int(s["points"]) for s in story_rows if int(s["points"]) not in forbid_points}
value  = {s["story_id"]: float(s["value"]) for s in story_rows if int(s["points"]) not in forbid_points}

deps = []
for s in story_rows:
    sid = s["story_id"]
    if int(s["points"]) in forbid_points: 
        continue
    dep = (s.get("depends_on") or "").strip()
    if dep:
        deps.append((sid, dep))

R = [r["role"] for r in role_rows]
role_share = {r["role"]: float(r["share_of_hours"]) for r in role_rows}
mtg_per_story = {r["role"]: float(r["meeting_load_per_story_hours"]) for r in role_rows}
bug_hours_per_bug = {r["role"]: float(r["bug_hours_per_bug"]) for r in role_rows}

# Per-story required hours by role
hrs_tot = {s: points[s]*hours_per_point for s in S}
req = {(s,"BE"): role_share.get("BE",0.0)*hrs_tot[s] for s in S}
req.update({(s,"FE"): role_share.get("FE",0.0)*hrs_tot[s] for s in S})
req.update({(s,"QA"): role_share.get("QA",0.0)*hrs_tot[s]*qa_cov_factor for s in S})
req.update({(s,"TL"): role_share.get("TL",0.0)*hrs_tot[s] for s in S})
req.update({(s,"ARQ"):role_share.get("ARQ",0.0)*hrs_tot[s] for s in S})

# Add meeting load per story (we charge QA by default)
for s in S:
    for r in ("QA","TL"):
        extra = mtg_per_story.get(r,0.0)
        if extra>0:
            req[(s,r)] += extra

# Aggregate role capacities and bug reservations
role_cap = {r: sum(cap_i[i] for i in I_by_role[r]) for r in R}
role_bug_reserve = {r: bugs_per_sprint * bug_hours_per_bug.get(r,0.0) for r in R}
role_cap_eff = {r: role_cap.get(r,0.0) - role_bug_reserve.get(r,0.0) for r in R}
# QA WIP buffer
role_cap_eff["QA"] = min(role_cap_eff.get("QA",0.0), role_cap.get("QA",0.0)*wip_factor_QA)

# ---------- Model ----------
m = pl.LpProblem("SprintPlanningMILP", pl.LpMaximize)

# Decision variables
x = pl.LpVariable.dicts("x", [(i,s) for i in I for s in S], lowBound=0, cat=pl.LpContinuous)
z = pl.LpVariable.dicts("z", S, lowBound=0, upBound=1, cat=pl.LpBinary)
y = pl.LpVariable.dicts("y", I, lowBound=0, upBound=1, cat=pl.LpBinary)
owner = pl.LpVariable.dicts("owner", [(i,s) for i in I for s in S], lowBound=0, upBound=1, cat=pl.LpBinary)
rel = pl.LpVariable.dicts("rel", I, lowBound=0, upBound=1, cat=pl.LpBinary)

# Objective: maximize value - lambda * active people
m += pl.lpSum(value[s]*z[s] for s in S) - lambda_people * pl.lpSum(y[i] for i in I)

# Capacity per person
for i in I:
    m += pl.lpSum(x[(i,s)] for s in S) <= cap_i[i] * y[i], f"Cap_{i}"

# Role coverage per story
for s in S:
    for r in ("BE","FE","QA","TL","ARQ"):
        if r in I_by_role:  # only if we have people in that role
            m += pl.lpSum(x[(i,s)] for i in I_by_role[r]) >= req[(s,r)] * z[s], f"Req_{r}_{s}"

# Aggregate role capacity (bugs + buffer for QA)
for r in ("BE","FE","QA","TL","ARQ"):
    if r in I_by_role:
        m += pl.lpSum(x[(i,s)] for s in S for i in I_by_role[r]) <= role_cap_eff[r], f"RoleCap_{r}"

# Owners: exactly one owner among devs if story selected
devs = [i for i in I if role_of[i] in ("BE","FE")]
for s in S:
    m += pl.lpSum(owner[(i,s)] for i in devs) == z[s], f"OwnerOne_{s}"
    # Link owner to hours (if owner=1 must invest at least theta hours)
    for i in devs:
        m += pl.lpSum(x[(i,s)]) >= theta_release_hours * owner[(i,s)], f"OwnerHours_{i}_{s}"

# Points cap per dev (sum of points of owned stories <= 13)
for i in devs:
    m += pl.lpSum(points[s]*owner[(i,s)] for s in S) <= max_points_per_dev, f"PointsCap_{i}"

# Releases: every active dev must release at least one story
for i in devs:
    # rel_i <= sum owners; and rel_i >= y_i (if active -> must release at least one)
    m += rel[i] <= pl.lpSum(owner[(i,s)] for s in S), f"RelLink_{i}"
    m += rel[i] >= y[i], f"RelActive_{i}"

# Dependencies: z_s <= z_p if s depends on p
for (s,p) in deps:
    if p in S:  # only enforce if predecessor not forbidden
        m += z[s] <= z[p], f"Dep_{s}_on_{p}"

# Solve
solver = pl.PULP_CBC_CMD(msg=False)
res = m.solve(solver)

status = pl.LpStatus[m.status]
obj = pl.value(m.objective)

# ---------- Outputs ----------
# Selected stories
sel = []
for s in S:
    if pl.value(z[s]) > 0.5:
        # find owner and role
        owner_i = None
        for i in devs:
            if pl.value(owner[(i,s)]) > 0.5:
                owner_i = i
                break
        sel.append([s, points[s], value[s], owner_i])

with open(os.path.join(OUT,"selected_stories.csv"),"w",encoding="utf-8",newline="") as f:
    w = csv.writer(f); w.writerow(["story_id","points","value","owner"])
    for row in sel:
        w.writerow(row)

# Assignments
assign_rows = []
for i in I:
    for s in S:
        hrs = pl.value(x[(i,s)])
        if hrs and hrs>1e-4:
            assign_rows.append([i, role_of[i], s, round(hrs,2)])
with open(os.path.join(OUT,"assignments.csv"),"w",encoding="utf-8",newline="") as f:
    w = csv.writer(f); w.writerow(["person","role","story_id","hours"])
    for row in assign_rows:
        w.writerow(row)

# Utilization per person
util = []
for i in I:
    used = sum(pl.value(x[(i,s)]) for s in S)
    util.append([i, role_of[i], round(used or 0.0,2), cap_i[i], round((used or 0.0)/cap_i[i],3), int(pl.value(y[i])>0.5), int(pl.value(rel[i] or 0.0)>0.5)])
with open(os.path.join(OUT,"person_utilization.csv"),"w",encoding="utf-8",newline="") as f:
    w = csv.writer(f); w.writerow(["person","role","hours_used","capacity","utilization","active(y)","release(rel)"])
    for row in util:
        w.writerow(row)

# Summary
with open(os.path.join(OUT,"summary.txt"),"w",encoding="utf-8") as f:
    f.write(f"Status: {status}\n")
    f.write(f"Objective (value - lambda*people): {obj:.3f}\n")
    val_total = sum(value[s] for s in S if pl.value(z[s])>0.5)
    active_people = sum(1 for i in I if pl.value(y[i])>0.5)
    f.write(f"Delivered value: {val_total:.3f}\n")
    f.write(f"Active people: {active_people}\n")
    f.write(f"Stories selected: {len(sel)} / {len(S)}\n")
    f.write(f"Dependencies respected: {len(deps)}\n")
    f.write("\nParameters:\n")
    f.write(f"  hours_per_point={hours_per_point:.4f}\n")
    f.write(f"  lambda_people_penalty={lambda_people}\n")
    f.write(f"  bugs_per_sprint={bugs_per_sprint}\n")
    f.write(f"  max_points_per_dev={max_points_per_dev}\n")
    f.write(f"  qa_coverage_factor={qa_cov_factor}\n")
    f.write(f"  wip_factor_QA_capacity={wip_factor_QA}\n")

print("Solved.", "Status:", status, "Objective:", round(obj,3))
print("Outputs saved in 'results' folder.")
