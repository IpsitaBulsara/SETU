# SETU Company Planning Workspace

## Run locally

```powershell
pip install -r requirements.txt
streamlit run company_app.py
```

Start with the `company_app.py` CSV-first workspace. It accepts a demand-history CSV, inventory snapshot CSV, and optional supplier scorecard CSV; it returns forecasted values by SKU/location, replenishment quantities, supplier actions, and an approval-ready action pack.

The legacy `app.py` remains available as a visual scenario prototype. It is not the company workflow.

Before operational use, connect approved data sources and configure identity, action approval, monitoring, audit retention, and secure ERP/WMS/procurement APIs.
