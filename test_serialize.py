import json
import asyncio
from main_logic import run_full_pipeline

def dummy_callback(run_id, stage, status, message, data=None):
    if stage == "COMPLETE":
        try:
            print("Trying to serialize COMPLETE event...")
            json.dumps(data)
            print("Serialization successful!")
        except TypeError as e:
            print(f"Serialization failed: {e}")

res = run_full_pipeline("test1234", [], [], 100.0, 0.0, 1, {}, dummy_callback)
