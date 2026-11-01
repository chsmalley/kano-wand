import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import app


class EventLogTests(unittest.TestCase):
    def test_records_timestamped_events_and_appends_without_rewriting_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "events.csv"
            with patch.object(app, "EVENT_LOG_PATH", log_path):
                app.record_event(
                    "spell",
                    spell="LUMOS",
                    confidence=0.94,
                    wand_id="wand-1",
                )
                app.record_event("treat", source="button")

            with log_path.open(newline="", encoding="utf-8") as log_file:
                rows = list(csv.DictReader(log_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["event"], "spell")
        self.assertEqual(rows[0]["spell"], "LUMOS")
        self.assertEqual(rows[0]["confidence"], "0.94")
        self.assertEqual(rows[0]["wand_id"], "wand-1")
        self.assertEqual(rows[1]["event"], "treat")
        self.assertEqual(rows[1]["source"], "button")
        for row in rows:
            self.assertTrue(row["timestamp"].endswith("+00:00"))

    def test_activity_endpoint_aggregates_mapped_spells_and_treats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "events.csv"
            now = datetime.now(timezone.utc).isoformat()
            with log_path.open("w", newline="", encoding="utf-8") as log_file:
                writer = csv.DictWriter(
                    log_file,
                    fieldnames=("timestamp", "event", "spell", "confidence", "wand_id", "source"),
                )
                writer.writeheader()
                writer.writerows([
                    {"timestamp": now, "event": "spell", "spell": "LUMOS"},
                    {"timestamp": now, "event": "spell", "spell": "UNKNOWN"},
                    {"timestamp": now, "event": "treat", "source": "button"},
                ])

            with patch.object(app, "EVENT_LOG_PATH", log_path):
                response = app.app.test_client().get("/api/activity")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["interval_minutes"], 15)
        self.assertEqual(data["window_hours"], 24)
        self.assertEqual(len(data["labels"]), 97)
        self.assertEqual(sum(data["series"]["LIGHTS"]), 1)
        self.assertEqual(sum(data["series"]["Treats"]), 1)
        self.assertEqual(sum(sum(counts) for counts in data["series"].values()), 2)


if __name__ == "__main__":
    unittest.main()
