import httpx
import asyncio
import time


async def test():
    async with httpx.AsyncClient(timeout=180) as c:
        t0 = time.time()
        r = await c.post(
            "http://localhost:8000/api/query",
            json={"query": "高血压患者能用布洛芬吗", "user_id": "u99"},
        )
        d = r.json()
        print(f"Time: {time.time() - t0:.0f}s")
        print(f"Route: {d['route_path']}")
        print(f"Conf: {d['confidence']}")
        print(f"Sources: {len(d.get('sources', []))}")
        print(f"Fallback: {d.get('is_safe_fallback')}")
        print(f"Answer[:300]: {d['answer'][:300]}")

        print()
        print("--- Harness Endpoints ---")
        for ep in [
            "/api/metrics",
            "/api/harness/events",
            "/api/harness/runs",
            "/api/harness/context/snapshot",
            "/api/harness/safety",
            "/api/harness/traces?n=3",
            "/api/feedback?n=3",
        ]:
            r2 = await c.get(f"http://localhost:8000{ep}")
            d2 = r2.json()
            print(f"  [OK] {ep}: keys={list(d2.keys())[:5]}")


asyncio.run(test())
