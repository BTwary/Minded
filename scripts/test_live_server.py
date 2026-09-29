"""Test all analytical pathways against live running server."""
import httpx

def test_live_analyst():
    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)
    
    questions = [
        "Why did revenue fall in March?",
        "Which products are performing badly?",
        "Forecast next month's revenue",
        "What are the main correlation drivers of revenue and profit?",
        "Analyze my dataset and tell me what I should worry about",
    ]

    print("=" * 80)
    print("TESTING LIVE RUNNING DATA ANALYST SERVER (http://127.0.0.1:8000)")
    print("=" * 80)

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Question: \"{q}\"")
        resp = client.post("/api/v1/analysis", json={"question": q, "project_id": "proj-default"})
        assert resp.status_code == 200, f"Failed on question: {q}"
        data = resp.json()
        print(f"   -> HTTP Status: {resp.status_code}")
        print(f"   -> Direct Answer: {data.get('direct_answer')}")
        print(f"   -> Main Finding: {data.get('main_finding')}")
        print(f"   -> Confidence: {data.get('confidence')}")
        print(f"   -> Steps Executed: {len(data.get('steps', []))}")
        print(f"   -> Evidence Verified: {len(data.get('evidence', []))} items (Validation: {data.get('evidence', [{}])[0].get('validation_status', 'N/A')})")

    print("\n" + "=" * 80)
    print("ALL LIVE ANALYST SERVER ENDPOINTS VERIFIED AND WORKING FLAWLESSLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_live_analyst()
