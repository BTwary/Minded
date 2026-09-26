import httpx
import json
import uuid

client = httpx.Client(timeout=30.0)

BASE = "http://127.0.0.1:8000/api/v1"

# The /analysis endpoint requires a real authenticated user (see
# apps/api/src/core/security.get_current_user / verify_project_ownership).
# Register a throwaway account through the real /auth/register endpoint so
# this script exercises the actual auth path end-to-end instead of bypassing
# it -- this was previously missing, which made every call 401.
reg = client.post(
    f"{BASE}/auth/register",
    json={
        "email": f"pipeline-test-{uuid.uuid4().hex[:8]}@example.com",
        "password": "TestPass123!",
        "full_name": "Pipeline Test User",
    },
)
reg.raise_for_status()
token = reg.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

print("=== EXECUTING AUTONOMOUS INVESTIGATION ===")
res = client.post(
    f"{BASE}/analysis",
    json={"question": "Why did revenue fall in March?", "project_id": "proj-default"},
    headers=headers,
)

print("HTTP Status:", res.status_code)
d = res.json()

print("\n--- 1. PHASE 1: DISCOVERY & GRAIN ---")
disc = d.get("discovery") or {}
print("Grain:", disc.get("grain"))
print("Primary Metrics:", disc.get("primary_metrics"))
print("Dimensions:", disc.get("primary_dimensions"))
print("Worthy Inquiries:", disc.get("worthy_inquiries"))

print("\n--- 2. PHASE 2: DATA AUDIT ---")
aud = d.get("audit") or {}
print("Data Quality Score:", aud.get("data_quality_score"), "/ 100")
print("Audited Rows:", aud.get("total_rows"))

print("\n--- 3. PHASE 3 & 4: HYPOTHESES & RE-PLANNING CYCLE ---")
for h in d.get("hypotheses", []):
    print(f"[{h.get('id')}] {h.get('statement')[:60]}... -> STATUS: {h.get('status').upper()}")
    if h.get("reason_for_rejection"):
        print(f"   Reason for Rejection: {h.get('reason_for_rejection')}")
    if h.get("confirmed_findings"):
        print(f"   Confirmed Finding: {h.get('confirmed_findings')}")

print("\n--- 4. MULTI-VECTOR CONFIDENCE ---")
print("Confidence Breakdown:", json.dumps(d.get("confidence_breakdown"), indent=2))

print("\n--- 5. EVIDENCE GRAPH & PROVENANCE ---")
for node in d.get("evidence_graph", []):
    print(f"Node: {node.get('finding_id')} -> {node.get('evidence_id')} -> Tool: {node.get('tool_used')} -> Table: {node.get('table_name')} (v{node.get('dataset_version')})")
    print(f"Query: {node.get('query_executed')}")

print("\n--- 6. DETERMINISTIC WHAT-IF SCENARIOS ---")
for sc in d.get("what_if_scenarios", []):
    print(f"Scenario: {sc.get('scenario')} -> Impact: {sc.get('projected_impact')}")
    print(f"Uncertainty Note: {sc.get('uncertainty_note')}")
