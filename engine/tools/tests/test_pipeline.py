from engine.serving.pipeline import run_full_pipeline
import json

def progress_cb(run_id, stage, status, msg, data=None):
    print(f'[{stage}] {status}: {msg}')

result = run_full_pipeline(
    run_id='test_run',
    my_drivers=['Max Verstappen', 'Lando Norris'],
    my_constructors=['Red Bull', 'McLaren'],
    budget=100.0,
    points=0.0,
    transfers=3,
    options={'race_round': 1},
    progress_callback=progress_cb
)

if result:
    race_order = result['predictions']['race_order']
    print(f'Race predictions: {len(race_order)} drivers')
    print(f'Winner: {race_order[0]["driver"]}')
else:
    print('Pipeline failed')
