import unittest

try:
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError):
    TestClient = None

from engine.serving.server import app

class UISmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TestClient is None:
            raise unittest.SkipTest("fastapi TestClient requires httpx/httpx2")
        cls.client = TestClient(app)

    def test_root_index_html(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("F1 FANTASY PREDICTOR", res.text)

    def test_static_assets_exist(self):
        res_css = self.client.get("/static/style.css")
        self.assertEqual(res_css.status_code, 200)
        res_js = self.client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        res_springs = self.client.get("/static/apple-springs.js")
        self.assertEqual(res_springs.status_code, 200)

    def test_sepang_track_map_asset(self):
        res = self.client.get("/static/images/tracks/sepang.svg")
        self.assertEqual(res.status_code, 200)
        self.assertIn("<svg", res.text.lower())
        self.assertIn("</svg>", res.text.lower())

        res_js = self.client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("renderCircuitMap", res_js.text)
        self.assertIn("/static/images/tracks/sepang.svg", res_js.text)
        self.assertNotIn("[ Map Unavailable ]", res_js.text)

    def test_api_status(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json().get("status"), "ok")

    def test_api_system_health(self):
        res = self.client.get("/api/system/health")
        self.assertEqual(res.status_code, 200)
        self.assertIn("status", res.json())

    def test_api_races_and_next_race_completion(self):
        res_next = self.client.get("/api/race/next")
        self.assertEqual(res_next.status_code, 200)
        data_next = res_next.json()
        self.assertEqual(data_next.get("round"), 16)
        self.assertEqual(data_next.get("latest_completed_round"), 15)

        res_completed = self.client.get("/api/race/latest-completed")
        self.assertEqual(res_completed.status_code, 200)
        self.assertTrue(res_completed.json().get("completed"))
        self.assertEqual(res_completed.json().get("round"), 15)

        res_races = self.client.get("/api/races")
        self.assertEqual(res_races.status_code, 200)
        races = res_races.json()
        r15 = next(r for r in races if r["round"] == 15)
        self.assertTrue(r15.get("is_completed"))
        r16 = next(r for r in races if r["round"] == 16)
        self.assertFalse(r16.get("is_completed"))

    def test_ui_telemetry_tokens_and_canvas_css(self):
        """Verify style.css includes new telemetry tokens and waveform canvas."""
        res = self.client.get("/static/style.css")
        self.assertEqual(res.status_code, 200)
        css = res.text
        self.assertIn("--telemetry-cyan", css)
        self.assertIn("#telemetry-waveform-canvas", css)
        self.assertNotIn("f1-drive", css)

    def test_ui_analysis_screen_canvas_and_hud(self):
        """Verify index.html contains telemetry canvas and segmented ensemble container."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.text
        self.assertIn('id="telemetry-waveform-canvas"', html)
        self.assertIn('id="mc-iterations-counter"', html)
        self.assertIn('id="ensemble-bar-wrap"', html)
        self.assertNotIn('class="f1-svg-loader"', html)

    def test_ui_app_js_telemetry_controller(self):
        """Verify app.js includes canvas telemetry loop and segmented HUD logic."""
        res = self.client.get("/static/app.js")
        self.assertEqual(res.status_code, 200)
        js = res.text
        self.assertIn("initTelemetryWaveform", js)
        self.assertIn("updateEnsembleSegments", js)
        self.assertIn("animateMonteCarloCounter", js)

    def test_ui_adaptive_leaderboard_classes(self):
        """Verify CSS contains .ro-col-telemetry and responsive race columns breakpoint."""
        res = self.client.get("/static/style.css")
        self.assertEqual(res.status_code, 200)
        css = res.text
        self.assertIn(".ro-col-telemetry", css)
        self.assertIn("@media (max-width: 1360px)", css)

    def test_ui_table_responsive_container(self):
        """Verify style.css includes .table-responsive-container for bias and accuracy grids."""
        res = self.client.get("/static/style.css")
        self.assertEqual(res.status_code, 200)
        css = res.text
        self.assertIn(".table-responsive-container", css)
        self.assertIn(".card-title", css)

    def test_f1_sync_ui_elements(self):
        """Verify index.html and app.js include F1 sync button and modal handlers."""
        res_html = self.client.get("/")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("btn-sync-f1", res_html.text)
        self.assertIn("f1-sync-modal", res_html.text)

        res_js = self.client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("openF1SyncModal", res_js.text)
        self.assertIn("importF1Team", res_js.text)

    def test_f1_sync_api_endpoints(self):
        """Verify /api/f1-sync/teams and /api/f1-sync/import endpoints function properly."""
        res_teams = self.client.get("/api/f1-sync/teams")
        self.assertEqual(res_teams.status_code, 200)
        data = res_teams.json()
        self.assertEqual(data.get("status"), "ok")
        teams = data.get("teams", [])
        self.assertGreaterEqual(len(teams), 1)

        # Test importing Team 1
        res_import = self.client.post("/api/f1-sync/import", json={"team_no": 1})
        self.assertEqual(res_import.status_code, 200)
        import_data = res_import.json()
        self.assertEqual(import_data.get("status"), "ok")
        self.assertIn("Bobby_Racing", import_data.get("imported", {}).get("team_name", ""))

# Alias for compatibility with test runners targeting TestUISmoke
TestUISmoke = UISmokeTests


