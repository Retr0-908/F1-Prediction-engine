import pandas as pd

df = pd.read_csv('output/backtest_2024_2025.csv')

for year in [2024, 2025]:
    d = df[df.year == year]
    best_idx = d.race_rho.idxmax()
    worst_idx = d.race_rho.idxmin()
    print(f'--- {year} ({len(d)} races) ---')
    print(f'  Race MAE:       {d.race_mae.mean():.3f}  (std {d.race_mae.std():.3f})')
    print(f'  Race RMSE:      {d.race_rmse.mean():.3f}')
    print(f'  Spearman rho:   {d.race_rho.mean():.3f}  (std {d.race_rho.std():.3f})')
    print(f'  Quali MAE:      {d.quali_mae.mean():.3f}')
    print(f'  Quali rho:      {d.quali_rho.mean():.3f}')
    print(f'  Winner correct: {d.winner_correct.mean()*100:.1f}%')
    print(f'  Top-3 hits:     {d.top3_hits.mean():.2f}/3  ({d.top3_hits.mean()/3*100:.0f}%)')
    print(f'  Top-5 hits:     {d.top5_hits.mean():.2f}/5  ({d.top5_hits.mean()/5*100:.0f}%)')
    print(f'  Best  rho:  {d.loc[best_idx, "race_name"]} = {d.race_rho.max():.3f}')
    print(f'  Worst rho:  {d.loc[worst_idx, "race_name"]} = {d.race_rho.min():.3f}')
    print()

print(f'--- ALL ({len(df)} races) ---')
print(f'  Race MAE:       {df.race_mae.mean():.3f}')
print(f'  Race RMSE:      {df.race_rmse.mean():.3f}')
print(f'  Spearman rho:   {df.race_rho.mean():.3f}')
print(f'  Quali rho:      {df.quali_rho.mean():.3f}')
print(f'  Winner correct: {df.winner_correct.mean()*100:.1f}%')
print(f'  Top-3/race:     {df.top3_hits.mean():.2f}/3')
print(f'  Top-5/race:     {df.top5_hits.mean():.2f}/5')

print()
print('--- BASELINES (N=20 drivers) ---')
print('  Random MAE:     ~6.67  (random permutation avg)')
print('  Random rho:     ~0.00')
print(f'  MAE improvement over random: {6.67 - df.race_mae.mean():.2f} positions')
print(f'  Rho lift over random:        {df.race_rho.mean() - 0.0:.3f}')

# Distribution of rho
print()
print('--- rho DISTRIBUTION ---')
bins = [(-1,0), (0,0.3), (0.3,0.5), (0.5,0.7), (0.7,1.01)]
for lo, hi in bins:
    count = ((df.race_rho >= lo) & (df.race_rho < hi)).sum()
    pct = count / len(df) * 100
    print(f'  {lo:+.1f} to {hi:+.1f}: {count} races ({pct:.0f}%)')

print()
print('--- PER-RACE TABLE ---')
cols = ['year','round','race_name','race_mae','race_rho','winner_correct','top3_hits','top5_hits']
print(df[cols].to_string(index=False))
