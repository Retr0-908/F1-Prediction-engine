import subprocess
import sys
import time
from playwright.sync_api import sync_playwright
import os

os.makedirs('Screenshots', exist_ok=True)

print("Starting server...")
server_process = subprocess.Popen([sys.executable, "-m", "engine.serving.server"])
time.sleep(4) # Wait for server to start

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        # Desktop
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        try:
            page.goto('http://127.0.0.1:5762', timeout=20000)
            page.wait_for_timeout(8000) # wait for data to load and spinner to vanish
            page.screenshot(path='Screenshots/desktop.png', full_page=True)
            
            # Tablet
            page.set_viewport_size({"width": 768, "height": 1024})
            page.wait_for_timeout(1000)
            page.screenshot(path='Screenshots/tablet.png', full_page=True)
            
            # Mobile
            page.set_viewport_size({"width": 375, "height": 812})
            page.wait_for_timeout(1000)
            page.screenshot(path='Screenshots/mobile.png', full_page=True)
            print("Screenshots captured successfully.")
        except Exception as e:
            print(f"Error capturing screenshots: {e}")
        finally:
            browser.close()
finally:
    print("Terminating server...")
    server_process.terminate()


