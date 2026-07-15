import httpx
import asyncio
import time
import json


async def main():
    async with httpx.AsyncClient(timeout=180) as client:
        # Test 1: Basic Chinese medical query
        print("=" * 60)
        print("TEST 1: Basic Medical Query (RAG)")
        t0 = time.time()
        r = await client.post(
            "http://localhost:8000/api/query",
            json={"query": "高血压患者痛风发作，能吃布洛芬吗？", "user_id": "test_u1"},
        )
        data = r.json()
        t1 = time.time()
        print(f"  Time: {t1 - t0:.1f}s")
        print(f"  Route: {data.get('route_path')}")
        print(f"  Confidence: {data.get('confidence')}")
        print(f"  Sources: {len(data.get('sources', []))}")
        print(f"  Fallback: {data.get('is_safe_fallback')}")
        ans = data.get("answer", "")
        print(f"  Answer ({len(ans)} chars): {ans[:300]}")
        print()

        # Test 2: Second query to test multi-turn context
        print("=" * 60)
        print("TEST 2: Second Query (Multi-turn)")
        r2 = await client.post(
            "http://localhost:8000/api/query",
            json={"query": "那秋水仙碱可以吗？", "user_id": "test_u1"},
        )
        data2 = r2.json()
        print(f"  Route: {data2.get('route_path')}")
        print(f"  Confidence: {data2.get('confidence')}")
        print(f"  Answer[:200]: {data2.get('answer', '')[:200]}")
        print()

        # Test 3: MDT query (multi-department)
        print("=" * 60)
        print("TEST 3: MDT Query (Multi-department)")
        r3 = await client.post(
            "http://localhost:8000/api/query",
            json={
                "query": "患者同时有高血压、胃溃疡和痛风，正在服用氯吡格雷，膝盖疼痛该用什么止痛药？",
                "user_id": "test_u2",
            },
        )
        data3 = r3.json()
        print(f"  Route: {data3.get('route_path')}")
        print(f"  Departments: {data3.get('departments')}")
        print(f"  Confidence: {data3.get('confidence')}")
        print(f"  Sources: {len(data3.get('sources', []))}")
        ans3 = data3.get("answer", "")
        print(f"  Answer ({len(ans3)} chars): {ans3[:300]}")
        print()

        # Test 4: All Harness endpoints
        print("=" * 60)
        print("TEST 4: Harness API Endpoints")
        endpoints = [
            "/api/metrics",
            "/api/harness/traces?n=5",
            "/api/harness/evaluate",
            "/api/harness/safety",
            "/api/harness/events",
            "/api/harness/runs",
            "/api/harness/context/snapshot",
            "/api/health",
        ]
        for ep in endpoints:
            try:
                r4 = await client.get(f"http://localhost:8000{ep}")
                d = r4.json()
                keys = list(d.keys())[:5]
                print(f"  [OK] {ep} -> keys: {keys}")
            except Exception as e:
                print(f"  [FAIL] {ep}: {e}")
        print()

        # Test 5: User feedback
        print("=" * 60)
        print("TEST 5: User Feedback API")
        r5 = await client.post(
            "http://localhost:8000/api/feedback",
            json={"query": "test query", "rating": 0.9, "comment": "excellent answer", "feedback_type": "rating"},
        )
        print(f"  Feedback: {r5.json()}")
        r5b = await client.get("http://localhost:8000/api/feedback?n=5")
        fb_data = r5b.json()
        print(f"  Feedback count: {len(fb_data.get('feedback', []))}")
        print()

        print("=" * 60)
        print("ALL TESTS COMPLETE")


asyncio.run(main())
