"""Start the app, time it to a fully rendered page, and save the report screenshots (needs Playwright + Chromium).

  python demo/screenshots.py [--time-only]
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"


def main():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "demo" / "app.py"), "--server.port",
                             str(port), "--server.headless", "true"], cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        while True:
            try:
                socket.create_connection(("localhost", port), timeout=1).close()
                break
            except OSError:
                if time.time() - t0 > 60:
                    raise SystemExit("the app did not start within 60 s")
                time.sleep(0.2)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 2300}, device_scale_factor=1.5)
            base = f"http://localhost:{port}"

            def settle():
                page.wait_for_timeout(1500)
                if page.get_by_test_id("stException").count():
                    raise SystemExit("the app showed an exception")

            page.goto(f"{base}/student", wait_until="networkidle")
            page.get_by_text("Forgetting over time").first.wait_for(timeout=60_000)
            page.locator('[data-testid="stVegaLiteChart"] canvas, [data-testid="stVegaLiteChart"] svg').first.wait_for(timeout=60_000)
            print(f"startup to rendered student view: {time.time() - t0:.1f} s")
            if "--time-only" in sys.argv:
                return
            settle()
            FIG.mkdir(parents=True, exist_ok=True)
            top = page.get_by_text("Forgetting over time").first.bounding_box()["y"]
            page.screenshot(path=FIG / "demo_student.png", clip={"x": 0, "y": 0, "width": 1280, "height": top - 12})
            page.screenshot(path=FIG / "demo_forgetting.png",
                            clip={"x": 300, "y": top - 8, "width": 980, "height": min(2300 - top, 900)})
            page.get_by_role("link", name="Pre-exam planner").click()
            page.get_by_text("Pre-exam revision plan").first.wait_for(timeout=30_000)
            settle()
            page.screenshot(path=FIG / "demo_planner.png", clip={"x": 0, "y": 0, "width": 1280, "height": 1500})
            for name, heading in (("How it works", "The score"), ("Results", "Known defects")):
                page.get_by_role("link", name=name, exact=True).click()
                page.get_by_text(heading, exact=True).first.wait_for(timeout=30_000)
                settle()
                if len(sys.argv) > 1:
                    page.screenshot(path=Path(sys.argv[1]) / f"check_{name.split()[0].lower()}.png")
            browser.close()
            print("saved report/figures/demo_{student,forgetting,planner}.png")
    finally:
        proc.terminate()
        proc.wait(timeout=15)


if __name__ == "__main__":
    main()
