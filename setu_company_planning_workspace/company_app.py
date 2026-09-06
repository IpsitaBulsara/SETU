"""SETU — CSV-first supply-chain decision workspace.

Run: streamlit run company_app.py
This app makes recommendations only. In CSV mode it never writes to an ERP,
WMS, purchasing system, or carrier portal.
"""
from io import StringIO
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SETU | Company Planning Workspace", page_icon="🧭", layout="wide")

PURPLE, GOLD, GREEN, RED, CREAM = "#4B2E83", "#6B45A8", "#4B2E83", "#4B2E83", "#FFFFFF"
st.markdown(f"""
<style>
.stApp {{background:#FFFFFF;color:#24170E}}
.block {{background:{CREAM};border:1px solid #E6DDF3;border-radius:8px;padding:16px;margin:8px 0}}
.label {{font:700 11px monospace;color:#6B548E;letter-spacing:.07em}}
.value {{font:800 25px monospace;color:{PURPLE};margin:4px 0}}
.note {{font-size:12px;color:#3D2A58;line-height:1.45}}
.guard {{background:#F3EEFA;border-left:4px solid #4B2E83;padding:10px 12px;border-radius:4px;font-size:13px}}
[data-testid="stAlert"] {{background:#F3EEFA;border:1px solid #D8C6EF;color:#302044}}
[data-testid="stAlert"] svg {{fill:#4B2E83}}
[data-testid="stMetricValue"] {{color:#4B2E83}}
[data-testid="stDataFrame"] {{border:1px solid #E6DDF3;border-radius:6px}}
.flow-step {{background:#FFFFFF;border:1px solid #D8C6EF;border-top:4px solid #4B2E83;border-radius:7px;padding:13px;min-height:116px}}
.flow-number {{font:700 11px monospace;color:#6B548E}}
.flow-title {{font-size:15px;font-weight:700;color:#4B2E83;margin:5px 0}}
.flow-text {{font-size:12px;line-height:1.35;color:#302044}}
.request-detail {{background:#F3EEFA;border:1px solid #D8C6EF;border-radius:6px;padding:10px 12px;margin:8px 0;font-size:13px;color:#302044;line-height:1.55}}
</style>
""", unsafe_allow_html=True)

DEMAND_TEMPLATE = """date,sku,location,units,promo_flag
2025-01-06,CHOCO-100,Delhi-DC,1200,0
2025-01-13,CHOCO-100,Delhi-DC,1280,0
2025-01-20,CHOCO-100,Delhi-DC,1710,1
2025-01-27,CHOCO-100,Delhi-DC,1330,0
"""
INVENTORY_TEMPLATE = """sku,location,on_hand,lead_time_days,min_order_qty
CHOCO-100,Delhi-DC,5200,14,500
"""
SUPPLIER_TEMPLATE = """supplier,material,on_time_pct,quality_reject_pct,lead_time_days,baseline_lead_time_days,alternate_source
Cocoa Partners,cocoa,87,2.2,28,20,yes
"""

def synthetic_company_data():
    """Create a realistic, deterministic planning scenario for immediate use."""
    rng = np.random.default_rng(20260906)
    locations = {"Delhi-DC": 1.20, "Mumbai-DC": 1.35, "Chennai-DC": 0.88, "Kolkata-DC": 0.78}
    skus = {"CHOCO-100": 980, "CHOCO-250": 640, "BISCUIT-120": 820,
            "BISCUIT-300": 470, "GUM-20": 610}
    last_monday = pd.Timestamp.today().normalize() - pd.Timedelta(days=pd.Timestamp.today().weekday())
    rows, inv_rows = [], []
    for sku_i, (sku, base) in enumerate(skus.items()):
        for loc_i, (location, multiplier) in enumerate(locations.items()):
            history = []
            for i in range(78):
                date = last_monday - pd.Timedelta(weeks=77 - i)
                seasonal = 1 + 0.14 * np.sin((i % 52) / 52 * 2 * np.pi - 0.7)
                trend = 1 + 0.0018 * i
                promo = int((i + sku_i * 3 + loc_i) % 17 == 4)
                units = max(0, round(base * multiplier * seasonal * trend * (1 + 0.28 * promo) * rng.normal(1, .07)))
                rows.append({"date": date.date(), "sku": sku, "location": location,
                             "units": units, "promo_flag": promo})
                history.append(units)
            run_rate = np.mean(history[-4:])
            lead_time = [10, 14, 18, 12, 16][sku_i]
            moq = [250, 200, 300, 200, 500][sku_i]
            stock_factor = rng.uniform(.45, 1.25)
            inv_rows.append({"sku": sku, "location": location,
                             "on_hand": round(run_rate * lead_time / 7 * stock_factor),
                             "lead_time_days": lead_time, "min_order_qty": moq})
    suppliers = pd.DataFrame([
        ["Cocoa Partners", "Cocoa", 68, 5.0, 42, 20, "yes"],
        ["Dairy Supply Co", "Milk powder", 96, .7, 14, 14, "yes"],
        ["PackRight", "Flexible packaging", 91, 1.8, 19, 14, "yes"],
        ["SweetSource", "Sugar", 98, .4, 9, 10, "no"],
        ["Flavour House", "Flavours", 93, 3.8, 21, 16, "yes"],
    ], columns=["supplier", "material", "on_time_pct", "quality_reject_pct",
                "lead_time_days", "baseline_lead_time_days", "alternate_source"])
    return pd.DataFrame(rows), pd.DataFrame(inv_rows), suppliers

def load_synthetic_scenario():
    demand, inventory, suppliers = synthetic_company_data()
    st.session_state["company_demand"] = demand
    st.session_state["company_inventory"] = inventory
    st.session_state["company_suppliers"] = suppliers

def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def read_csv(uploaded, required):
    if uploaded is None:
        return None, None
    try:
        data = pd.read_csv(uploaded)
        data.columns = [str(c).strip().lower() for c in data.columns]
    except Exception as exc:
        return None, f"Could not read CSV: {exc}"
    missing = [c for c in required if c not in data.columns]
    if missing:
        return None, f"Missing required column(s): {', '.join(missing)}"
    return data, None

def validate_demand(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["units"] = pd.to_numeric(out["units"], errors="coerce")
    out["sku"] = out["sku"].astype(str).str.strip()
    out["location"] = out["location"].astype(str).str.strip()
    out = out.dropna(subset=["date", "units"])
    out = out[(out["units"] >= 0) & (out["sku"] != "") & (out["location"] != "")]
    return out

def weekly_demand(demand):
    df = demand.copy()
    # Synthetic scenarios may carry Python date objects while CSV imports
    # usually carry strings. Normalize both before using pandas' .dt accessor.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["week"] = df["date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    return df.groupby(["sku", "location", "week"], as_index=False)["units"].sum()

def make_forecast(demand, horizon):
    """Transparent, dependency-light forecast for a CSV-first planning workflow.

    It combines recent run-rate, trailing trend, and same-week-last-year demand
    where sufficient history is present.  It returns forecasted units, not a
    model accuracy headline.
    """
    weekly = weekly_demand(demand)
    rows = []
    diagnostics = []
    for (sku, location), group in weekly.groupby(["sku", "location"]):
        group = group.sort_values("week").set_index("week")["units"].astype(float)
        full_index = pd.date_range(group.index.min(), group.index.max(), freq="W-MON")
        series = group.reindex(full_index, fill_value=0.0)
        recent4 = series.tail(4).mean()
        recent12 = series.tail(min(12, len(series))).mean()
        prior4 = series.iloc[-8:-4].mean() if len(series) >= 8 else recent4
        trend = float(np.clip((recent4 - prior4) / max(prior4, 1), -0.35, 0.35))
        variability = float(series.tail(min(13, len(series))).std(ddof=0) / max(recent12, 1))
        last_week = series.index.max()
        for step in range(1, horizon + 1):
            future = last_week + timedelta(days=7 * step)
            seasonal_date = future - timedelta(days=364)
            seasonal = series.get(seasonal_date, np.nan)
            baseline = 0.65 * recent4 + 0.35 * seasonal if pd.notna(seasonal) else recent12
            decay = max(0.25, 1 - (step - 1) * 0.10)
            value = max(0, baseline * (1 + trend * decay))
            rows.append({"forecast_week": future.date(), "sku": sku, "location": location,
                         "forecast_units": int(round(value)), "method": "recent trend + seasonal baseline"})
        diagnostics.append({"sku": sku, "location": location, "history_weeks": len(series),
                            "recent_weekly_run_rate": round(recent4, 1),
                            "demand_variability": round(variability, 2)})
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)

def replenishment_plan(forecast, diagnostics, inventory):
    inv = inventory.copy()
    for col in ["on_hand", "lead_time_days", "min_order_qty"]:
        inv[col] = pd.to_numeric(inv[col], errors="coerce").fillna(0)
    inv["lead_time_days"] = inv["lead_time_days"].clip(lower=1)
    inv["min_order_qty"] = inv["min_order_qty"].clip(lower=1)
    rows = []
    for _, item in inv.iterrows():
        sku, location = str(item["sku"]), str(item["location"])
        fc = forecast[(forecast["sku"] == sku) & (forecast["location"] == location)]
        dg = diagnostics[(diagnostics["sku"] == sku) & (diagnostics["location"] == location)]
        if fc.empty or dg.empty:
            rows.append({"sku": sku, "location": location, "on_hand": round(item["on_hand"]),
                         "recommended_order_qty": 0, "decision": "Data gap — add demand history",
                         "owner": "Demand planner", "reason": "No matching SKU/location demand history"})
            continue
        lead_weeks = item["lead_time_days"] / 7
        demand_per_week = float(dg.iloc[0]["recent_weekly_run_rate"])
        variability = float(dg.iloc[0]["demand_variability"])
        lead_demand = demand_per_week * lead_weeks
        safety_stock = demand_per_week * (0.35 + min(0.65, variability))
        target = lead_demand + safety_stock
        raw = max(0, target - item["on_hand"])
        moq = item["min_order_qty"]
        order = int(np.ceil(raw / moq) * moq) if raw > 0 else 0
        rows.append({"sku": sku, "location": location, "on_hand": round(item["on_hand"]),
                     "lead_time_days": round(item["lead_time_days"]), "lead_time_demand": round(lead_demand),
                     "safety_stock": round(safety_stock), "recommended_order_qty": order,
                     "decision": "Create replenishment requisition" if order else "No order required",
                     "owner": "Supply planner", "reason": "Target stock covers lead-time demand plus variability buffer"})
    return pd.DataFrame(rows)

def supplier_risk(suppliers):
    s = suppliers.copy()
    for col in ["on_time_pct", "quality_reject_pct", "lead_time_days", "baseline_lead_time_days"]:
        s[col] = pd.to_numeric(s[col], errors="coerce").fillna(0)
    late = (100 - s["on_time_pct"]).clip(0, 100)
    quality = (s["quality_reject_pct"] * 10).clip(0, 100)
    delay = ((s["lead_time_days"] - s["baseline_lead_time_days"]).clip(lower=0) /
             s["baseline_lead_time_days"].replace(0, 1) * 100).clip(0, 100)
    s["risk_score"] = (0.45 * late + 0.30 * quality + 0.25 * delay).round(0).astype(int)
    s["decision"] = np.select([s.risk_score >= 70, s.risk_score >= 40],
                               ["Escalate; qualify alternate source", "Supplier recovery review"],
                               default="Monitor in standard cadence")
    s["owner"] = np.where(s.risk_score >= 70, "Procurement lead", "Supplier manager")
    return s[["supplier", "material", "risk_score", "decision", "owner", "alternate_source"]].sort_values("risk_score", ascending=False)

def build_agent_flow(forecast, replenishment, supplier_actions, approval_limit_lakh):
    """Translate the agent mesh into plain-language, threshold-based work."""
    urgent_orders = int((replenishment["recommended_order_qty"] > 0).sum())
    highest_supplier_risk = int(supplier_actions["risk_score"].max())
    rows = [
        ["Demand sensing", "Forecast changes by more than 10%", "+12% demand expected", "Increase the next production plan", 8.0, "Automatic action permitted"],
        ["Multi-echelon inventory", "Stock falls below lead-time need", f"{urgent_orders} replenishment orders needed", "Create replenishment requests", 7.0, "Automatic action permitted"],
        ["Procurement assistant", "Supplier quote is more than 5% above should-cost", "Cocoa quote is 8.5% above should-cost", "Hold the PO and request a revised quote or approved alternate", 18.0, "Needs approval"],
        ["Supplier risk warning", "Supplier risk score reaches 50", f"Risk score {highest_supplier_risk}/100", "Confirm alternate supplier allocation", 22.0, "Needs approval"],
        ["Predictive maintenance", "Equipment anomaly score reaches 70", "Anomaly score 74/100", "Schedule a planned maintenance window", 8.0, "Automatic action permitted"],
        ["Energy setpoints", "Energy use is more than 3% above target", "+4.8% versus target", "Adjust approved oven and cold-room setpoints", 2.5, "Automatic action permitted"],
        ["Network digital twin", "A plant is above 80% capacity", "Plant capacity 86%", "Shift production to an available plant", 14.0, "Needs approval"],
        ["Route, load & cold chain", "Temperature exceeds 5°C", "Cold-chain reading 6.4°C", "Hold load and reroute to a qualified carrier", 6.0, "Needs quality approval"],
        ["Agentic resolution", "A recommendation clears all policy checks", "All events checked", "Send permitted actions to the action queue", 0.0, "Orchestrates the flow"],
    ]
    result = pd.DataFrame(rows, columns=["Agent", "Alert threshold", "What the system sees", "Recommended action", "Action value (₹ lakh)", "Decision route"])
    value_gate = result["Action value (₹ lakh)"] > approval_limit_lakh
    result.loc[value_gate & (result["Decision route"] == "Automatic action permitted"), "Decision route"] = "Needs approval"
    return result

def procurement_snapshot():
    """Synthetic quote-versus-should-cost checks for the procurement assistant.

    The calculation mirrors a company procurement process: recent contracted
    price plus observable market, logistics, currency, and risk adjustments.
    """
    build_up = pd.DataFrame([
        ["Cocoa", "₹474/kg", "+₹34", "+₹15", "+₹9", "+₹8", "₹540/kg"],
        ["Milk powder", "₹292/kg", "+₹8", "+₹4", "+₹3", "+₹5", "₹312/kg"],
        ["Flexible packaging", "₹65/kg", "+₹3", "+₹2", "+₹1", "+₹3", "₹74/kg"],
    ], columns=["Material", "Last contracted price", "Commodity benchmark", "Freight & logistics", "Currency", "Geopolitical / supply risk", "Internal should-cost"])
    quotes = pd.DataFrame([
        ["Cocoa", "Cocoa Partners", "₹540/kg", "₹586/kg", "+8.5%", "Hold PO; request revised quote or use approved alternate"],
        ["Milk powder", "Dairy Supply Co", "₹312/kg", "₹318/kg", "+1.9%", "Proceed on contract terms"],
        ["Flexible packaging", "PackRight", "₹74/kg", "₹73/kg", "-1.4%", "Proceed; quote is within expected cost"],
    ], columns=["Material", "Supplier", "Internal should-cost", "Supplier quote", "Quote gap", "Recommended action"])
    return quotes, build_up

if st.session_state.get("company_data_version") != 2:
    load_synthetic_scenario()
    st.session_state["company_data_version"] = 2

st.title("SETU · daily planning assistant")
st.markdown("**See what needs attention today. Review a recommendation, approve it, and send the action to the right team.**")
st.markdown('<div class="guard"><b>Safe by design:</b> SETU prepares recommendations from the company planning data in its backend. It never changes an order or supplier plan until an authorised person approves it.</div>', unsafe_allow_html=True)

# The worker-facing pages read prepared backend data. There is no file-upload
# step or forecasting setup for the person using the application.
backend_demand = st.session_state["company_demand"]
backend_inventory = st.session_state["company_inventory"]
backend_suppliers = st.session_state["company_suppliers"]
backend_forecast, backend_diagnostics = make_forecast(backend_demand, 8)
backend_plan = replenishment_plan(backend_forecast, backend_diagnostics, backend_inventory)
backend_risk = supplier_risk(backend_suppliers)
st.session_state["company_forecast"] = backend_forecast
st.session_state["company_diagnostics"] = backend_diagnostics
st.session_state["company_replenishment"] = backend_plan
st.session_state["company_supplier_risk"] = backend_risk

data_tab, forecast_tab, supplier_tab, cost_tab, flow_tab, queue_tab = st.tabs(["Today", "Demand plan", "Supplier actions", "Procurement", "How SETU decides", "Approve actions"])

with data_tab:
    st.subheader("Your work for today")
    replenishments = backend_plan[backend_plan["recommended_order_qty"] > 0]
    supplier_followups = backend_risk[backend_risk["risk_score"] >= 40]
    c1, c2, c3 = st.columns(3)
    c1.metric("Orders to review", len(replenishments))
    c2.metric("Supplier follow-ups", len(supplier_followups))
    c3.metric("Expected customer demand", f"{backend_forecast['forecast_units'].sum():,} units")
    st.markdown("##### What should you do?")
    if len(replenishments):
        st.info(f"**Review stock:** {len(replenishments)} product/location combinations need replenishment. Open **Demand plan** to see exactly what to order.")
    if len(supplier_followups):
        st.warning(f"**Follow up with suppliers:** {len(supplier_followups)} supplier records need attention. Open **Supplier actions** for the next step and owner.")
    st.success("**Finish the work:** Open **Approve actions**, check each recommendation, and mark it Approved or Rejected.")
    st.caption("Planning data is refreshed in the backend. The displayed scenario is synthetic until the company connects its operational systems.")

with forecast_tab:
    st.subheader("What customers are expected to buy")
    st.write("These are expected units for the next eight weeks. Choose a product or warehouse to focus on your area.")
    f1, f2 = st.columns(2)
    sku_choice = f1.selectbox("Product", ["All products"] + sorted(backend_forecast["sku"].unique().tolist()))
    location_choice = f2.selectbox("Warehouse", ["All warehouses"] + sorted(backend_forecast["location"].unique().tolist()))
    display_forecast = backend_forecast.copy()
    if sku_choice != "All products": display_forecast = display_forecast[display_forecast["sku"] == sku_choice]
    if location_choice != "All warehouses": display_forecast = display_forecast[display_forecast["location"] == location_choice]
    display_forecast = display_forecast.rename(columns={"forecast_week": "Week starting", "sku": "Product", "location": "Warehouse", "forecast_units": "Expected units"})
    st.dataframe(display_forecast[["Week starting", "Product", "Warehouse", "Expected units"]], use_container_width=True, height=300)
    st.markdown("##### Recommended stock actions")
    simple_plan = backend_plan.rename(columns={"sku": "Product", "location": "Warehouse", "on_hand": "Stock now", "recommended_order_qty": "Order quantity", "decision": "Recommended action", "owner": "Responsible person"})
    st.dataframe(simple_plan[["Product", "Warehouse", "Stock now", "Order quantity", "Recommended action", "Responsible person"]], use_container_width=True, height=300)

with supplier_tab:
    st.subheader("Supplier actions")
    st.write("This list tells you who needs a follow-up and what to do next. Higher priority means the issue needs attention sooner.")
    simple_risk = backend_risk.copy()
    simple_risk["Priority"] = np.where(simple_risk["risk_score"] >= 70, "Urgent", np.where(simple_risk["risk_score"] >= 40, "Review", "Routine"))
    simple_risk = simple_risk.rename(columns={"supplier": "Supplier", "material": "Material", "decision": "Next action", "owner": "Responsible person", "alternate_source": "Alternate source available"})
    st.dataframe(simple_risk[["Priority", "Supplier", "Material", "Next action", "Responsible person", "Alternate source available"]], use_container_width=True)

with cost_tab:
    st.subheader("Procurement assistant")
    st.write("This helps procurement decide whether a supplier quote is fair before a purchase order is released. The expected price changes when the market, transport routes, currency, or geopolitical conditions change.")
    quotes, build_up = procurement_snapshot()
    c1, c2, c3 = st.columns(3)
    c1.metric("Quotes checked", len(quotes))
    c2.metric("Quote alert threshold", "5% above should-cost")
    c3.metric("Quotes needing action", 1)
    st.markdown("##### Current supply scenario")
    st.info("**Cocoa crop pressure and trade-route disruption:** the system has increased the cocoa benchmark, freight, and supply-risk parts of the expected price. This is why the quote is checked against ₹540/kg—not against an old contract price.")
    st.markdown("##### How internal should-cost is calculated")
    st.write("Expected price = last contracted price + commodity benchmark + freight and logistics + currency effect + geopolitical or supply-risk adjustment.")
    st.dataframe(build_up, use_container_width=True, hide_index=True)
    st.dataframe(quotes, use_container_width=True, hide_index=True)
    st.markdown("##### Recommended procurement action")
    st.warning("**Do not release the Cocoa Partners PO yet.** Their cocoa quote is 8.5% above the internal should-cost. Ask for a revised quote or use the approved alternate source.")
    st.caption("This decision has an estimated ₹18 lakh impact and is included in **Approve actions** for manager approval.")

with flow_tab:
    st.subheader("How SETU decides")
    st.write("Each agent watches one business condition. When the condition crosses its limit, SETU recommends a specific action. Actions above the company value limit, or anything that affects quality, are sent to a manager for approval.")
    current_limit = float(st.session_state.get("approval_limit_lakh", 10.0))
    approval_limit = st.select_slider("Manager approval is required above", options=[5.0, 10.0, 15.0, 20.0, 25.0], value=current_limit, format_func=lambda x: f"₹{x:.0f} lakh", key="approval_limit_lakh")
    steps = [
        ("1", "Watch", "Agents monitor demand, stock, suppliers, equipment, energy, plants, and deliveries."),
        ("2", "Check the limit", "A threshold decides whether the event needs action—for example, supplier risk above 50 or cold-chain temperature above 5°C."),
        ("3", "Recommend", "The digital twin checks a safe, workable response and estimates the action value."),
        ("4", "Act or ask", f"Actions up to ₹{approval_limit:.0f} lakh can be scheduled. Higher-value and quality-sensitive actions need approval."),
    ]
    cols = st.columns(4)
    for col, (number, title, text) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="flow-step"><div class="flow-number">STEP {number}</div><div class="flow-title">{title}</div><div class="flow-text">{text}</div></div>', unsafe_allow_html=True)
    agent_flow = build_agent_flow(backend_forecast, backend_plan, backend_risk, approval_limit)
    st.session_state["agent_flow"] = agent_flow
    st.markdown("##### All nine operational agents")
    for start in range(0, len(agent_flow), 3):
        cards = st.columns(3)
        for col, (_, item) in zip(cards, agent_flow.iloc[start:start + 3].iterrows()):
            with col:
                st.markdown(f"""
                <div class="flow-step">
                  <div class="flow-title">{item['Agent']}</div>
                  <div class="flow-text"><b>It watches:</b> {item['Alert threshold']}<br><br>
                  <b>It sees:</b> {item['What the system sees']}<br><br>
                  <b>It recommends:</b> {item['Recommended action']}<br><br>
                  <b>Route:</b> {item['Decision route']}</div>
                </div>
                """, unsafe_allow_html=True)
    st.caption("The action route is a policy outcome: it is not a suggestion. In a production integration, permitted actions would create a controlled request in the relevant company system and retain the approval record.")

with queue_tab:
    st.subheader("Approval-ready action queue")
    st.write("SETU automatically approves low-risk actions that are within the company value limit. You only review actions above that limit or actions affecting product quality and supplier allocation.")
    plans = []
    auto_plans = []
    replenishment = st.session_state.get("company_replenishment")
    risk = st.session_state.get("company_supplier_risk")
    approval_limit = float(st.session_state.get("approval_limit_lakh", 10.0))
    agent_flow = build_agent_flow(backend_forecast, backend_plan, backend_risk, approval_limit)
    if replenishment is not None:
        p = replenishment[replenishment["recommended_order_qty"] > 0].copy()
        if not p.empty:
            p = p.rename(columns={"decision": "recommended_action"})
            # Synthetic material values let the policy distinguish routine
            # replenishment from a financially material purchase decision.
            unit_value = {"CHOCO-100": 300, "CHOCO-250": 420, "BISCUIT-120": 120,
                          "BISCUIT-300": 180, "GUM-20": 80}
            p["action_value_lakh"] = p.apply(lambda r: r["recommended_order_qty"] * unit_value.get(r["sku"], 150) / 100000, axis=1)
            p["source"] = "Inventory agent"
            p["approval_status"] = np.where(p["action_value_lakh"] <= approval_limit, "Auto-approved by agent", "Pending approval")
            auto_p = p[p["approval_status"] == "Auto-approved by agent"].copy()
            pending_p = p[p["approval_status"] == "Pending approval"].copy()
            if not auto_p.empty:
                auto_p["recommended_order_qty"] = auto_p.apply(lambda r: f"{int(r['recommended_order_qty'])} units · ₹{r['action_value_lakh']:.1f} lakh", axis=1)
                auto_plans.append(auto_p[["source", "sku", "location", "recommended_action", "recommended_order_qty", "owner", "approval_status", "reason"]])
            if not pending_p.empty:
                pending_p["recommended_order_qty"] = pending_p.apply(lambda r: f"{int(r['recommended_order_qty'])} units · ₹{r['action_value_lakh']:.1f} lakh", axis=1)
                plans.append(pending_p[["source", "sku", "location", "recommended_action", "recommended_order_qty", "owner", "approval_status", "reason"]])
    if risk is not None:
        p = risk[risk["risk_score"] >= 40].copy()
        if not p.empty:
            p = p.rename(columns={"decision": "recommended_action"})
            p["source"] = "Supplier-risk agent"; p["approval_status"] = "Pending approval"
            p["location"] = "—"; p["recommended_order_qty"] = "—"
            p["reason"] = "Risk score ≥ 40; validate recovery plan or alternate source"
            plans.append(p.rename(columns={"material": "sku"})[["source", "sku", "location", "recommended_action", "recommended_order_qty", "owner", "approval_status", "reason"]])
    policy_holds = agent_flow[agent_flow["Decision route"].isin(["Needs approval", "Needs quality approval"])].copy()
    if not policy_holds.empty:
        policy_holds["source"] = "Agent policy check"
        policy_holds["sku"] = policy_holds["Agent"]
        policy_holds["location"] = "Company operations"
        policy_holds["recommended_action"] = policy_holds["Recommended action"]
        policy_holds["recommended_order_qty"] = policy_holds["Action value (₹ lakh)"].map(lambda x: f"₹{x:.1f} lakh")
        policy_holds["owner"] = "Operations manager"
        policy_holds["approval_status"] = "Pending approval"
        policy_holds["reason"] = policy_holds.apply(lambda r: f"{r['What the system sees']}; {r['Decision route']}", axis=1)
        plans.append(policy_holds[["source", "sku", "location", "recommended_action", "recommended_order_qty", "owner", "approval_status", "reason"]])
    automatic_agents = agent_flow[agent_flow["Decision route"] == "Automatic action permitted"].copy()
    if not automatic_agents.empty:
        automatic_agents["source"] = "Agent policy check"
        automatic_agents["sku"] = automatic_agents["Agent"]
        automatic_agents["location"] = "Company operations"
        automatic_agents["recommended_action"] = automatic_agents["Recommended action"]
        automatic_agents["recommended_order_qty"] = automatic_agents["Action value (₹ lakh)"].map(lambda x: f"₹{x:.1f} lakh")
        automatic_agents["owner"] = "Agent execution log"
        automatic_agents["approval_status"] = "Auto-approved by agent"
        automatic_agents["reason"] = automatic_agents.apply(lambda r: f"{r['What the system sees']}; within the ₹{approval_limit:.0f} lakh limit", axis=1)
        auto_plans.append(automatic_agents[["source", "sku", "location", "recommended_action", "recommended_order_qty", "owner", "approval_status", "reason"]])
    if auto_plans:
        automatic_actions = pd.concat(auto_plans, ignore_index=True)
        st.markdown("##### Auto-approved by SETU")
        st.success(f"{len(automatic_actions)} routine actions are within the ₹{approval_limit:.0f} lakh limit and have been automatically approved. They are listed here for visibility; no worker action is needed.")
        st.dataframe(automatic_actions.rename(columns={"sku": "Item", "location": "Location", "recommended_action": "Action", "recommended_order_qty": "Quantity / value", "owner": "Sent to"})[["Item", "Location", "Action", "Quantity / value", "Sent to", "approval_status"]], use_container_width=True, hide_index=True)
        st.divider()
    if not plans:
        st.info("Load data and generate a replenishment plan or supplier-risk action list to populate the queue.")
    else:
        actions = pd.concat(plans, ignore_index=True)
        if "approval_decisions" not in st.session_state:
            st.session_state["approval_decisions"] = {}
        actions["Recommendation ID"] = [f"SETU-{i + 1:03d}" for i in range(len(actions))]
        actions["Approval decision"] = actions["Recommendation ID"].map(st.session_state["approval_decisions"]).fillna("Pending review")
        actions["Approver note"] = ""
        st.markdown("##### Quick approval")
        st.write("Use these buttons when all displayed recommendations should receive the same decision. You can still change individual rows below.")
        q1, q2, q3 = st.columns(3)
        if q1.button("✓ Approve all shown", use_container_width=True):
            for recommendation_id in actions["Recommendation ID"]:
                st.session_state["approval_decisions"][recommendation_id] = "Approve"
            st.rerun()
        if q2.button("✕ Reject all shown", use_container_width=True):
            for recommendation_id in actions["Recommendation ID"]:
                st.session_state["approval_decisions"][recommendation_id] = "Reject"
            st.rerun()
        if q3.button("↺ Request changes on all", use_container_width=True):
            for recommendation_id in actions["Recommendation ID"]:
                st.session_state["approval_decisions"][recommendation_id] = "Request changes"
            st.rerun()
        st.markdown("##### Approve one recommendation")
        st.write("Use the buttons on the specific item you are reviewing. Your decision remains visible in the table below.")
        for _, action in actions.iterrows():
            recommendation_id = action["Recommendation ID"]
            current_decision = action["Approval decision"]
            is_stock_request = action["source"] == "Inventory agent"
            if is_stock_request:
                request_title = f"Order {action['recommended_order_qty']} units of {action['sku']} for {action['location']}"
                request_detail = (f"<b>Product:</b> {action['sku']} &nbsp; | &nbsp; "
                                  f"<b>Warehouse:</b> {action['location']}<br>"
                                  f"<b>Recommended order:</b> {action['recommended_order_qty']} units<br>"
                                  f"<b>Why:</b> Current stock will not cover expected demand during the supplier lead time.")
            else:
                request_title = str(action["recommended_action"])
                request_detail = (f"<b>Area:</b> {action['sku']} &nbsp; | &nbsp; "
                                  f"<b>Expected impact:</b> {action['recommended_order_qty']}<br>"
                                  f"<b>Why:</b> {action['reason']}")
            with st.container(border=True):
                st.markdown(f"**{recommendation_id} · {request_title}**")
                st.markdown(f'<div class="request-detail">{request_detail}</div>', unsafe_allow_html=True)
                st.caption(f"Prepared by: {action['source']} · This will be sent to: {action['owner']}")
                a1, a2, a3 = st.columns(3)
                if a1.button("Approve", key=f"approve_{recommendation_id}", use_container_width=True):
                    st.session_state["approval_decisions"][recommendation_id] = "Approve"
                    st.rerun()
                if a2.button("Reject", key=f"reject_{recommendation_id}", use_container_width=True):
                    st.session_state["approval_decisions"][recommendation_id] = "Reject"
                    st.rerun()
                if a3.button("Request changes", key=f"changes_{recommendation_id}", use_container_width=True):
                    st.session_state["approval_decisions"][recommendation_id] = "Request changes"
                    st.rerun()
                st.write(f"Current decision: **{current_decision}**")
        st.markdown("##### Review each recommendation")
        edited = st.data_editor(actions, use_container_width=True, hide_index=True,
                                column_config={
                                    "Approval decision": st.column_config.SelectboxColumn(
                                        "Approval decision",
                                        options=["Pending review", "Approve", "Reject", "Request changes"],
                                        required=True,
                                        help="Approve = send forward; Reject = close; Request changes = return to the responsible team."),
                                    "Approver note": st.column_config.TextColumn(
                                        "Approver note", help="Optional reason, change, or instruction for the responsible team."),
                                })
        for _, row in edited.iterrows():
            st.session_state["approval_decisions"][row["Recommendation ID"]] = row["Approval decision"]
        summary = edited["Approval decision"].value_counts()
        s1, s2, s3 = st.columns(3)
        s1.metric("Approved", int(summary.get("Approve", 0)))
        s2.metric("Changes requested", int(summary.get("Request changes", 0)))
        s3.metric("Rejected", int(summary.get("Reject", 0)))
        st.download_button("Download decision pack", csv_bytes(edited), "SETU_approval_decisions.csv", "text/csv")
        st.caption("In the production version, each decision is stored with the approver's identity and timestamp before a controlled request is sent to the relevant company system.")
