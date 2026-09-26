import unittest
from fastapi.testclient import TestClient
from engine.serving.server import app

class UISmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

