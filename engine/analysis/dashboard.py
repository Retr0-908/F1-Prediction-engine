"""
dashboard.py — Generates a premium standalone HTML dashboard from prediction results.
Called at the end of main.py to replace the plain .txt recommendations file.
"""
import datetime
from pathlib import Path


# ── Team colour hex map (matches config.py TEAM_COLORS intent) ──────────────
_TEAM_HEX = {
    "McLaren":         "#FF8000",
    "Ferrari":         "#E8002D",
    "Red Bull":        "#3671C6",
    "Mercedes":        "#27F4D2",
    "Aston Martin":    "#229971",
    "Alpine":          "#FF87BC",
    "Williams":        "#64C4FF",
    "Racing Bulls":    "#6692FF",
    "Kick Sauber":     "#52E252",
    "Haas":            "#B6BABD",
}

def _tc(team: str) -> str:
    """Return hex colour for a team, defaulting to white."""
    for k, v in _TEAM_HEX.items():
        if k.lower() in team.lower():
            return v
    return "#CCCCCC"


def _weather_icon(condition: str) -> str:
    c = (condition or "").lower()
    if "rain" in c or "wet" in c or "storm" in c:
        return "🌧️"
    if "cloud" in c or "overcast" in c:
        return "⛅"
    if "mixed" in c:
        return "🌤️"
    return "☀️"


def generate_html_dashboard(
    race: dict,
    weather: dict,
    race_order: list,
    quali_order: list,
    driver_pts: list,
    ctor_pts: list,
    suggestions: dict,
    optimal: dict,
    best_ctors: list,
    my_drivers: list,
    my_constructors: list,
    my_points: float,
    driver_prices: dict,
    constructor_prices: dict,
    output_dir: Path,
    mc_results: dict = None,
) -> Path:
    """
    Build a self-contained, styled HTML file with all race weekend recommendations.
    Returns the path to the saved file.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    race_name  = race.get("name", "Unknown Race")
    round_num  = race.get("round", "?")
    race_date  = race.get("date", "")
    locality   = race.get("locality", "")
    country    = race.get("country", "")
    circuit    = race.get("circuit", "")

    race_session = weather.get("sessions", {}).get("Race", {})
    wx_cond    = weather.get("summary_condition", weather.get("condition", "Unknown"))
    wx_temp    = race_session.get("temp_day_c") or weather.get("temp_c") or "?"
    wx_rain    = weather.get("rain_risk", weather.get("rain_chance_pct", "?"))
    wx_wind    = race_session.get("wind_speed_kph") or weather.get("wind_kmh") or "?"
    wx_icon    = _weather_icon(str(wx_cond))

    # ── Sorted driver fantasy pts ────────────────────────────────────────────
    sorted_driver_pts = sorted(driver_pts, key=lambda x: x.get("total_pts", 0), reverse=True)

    # ── Transfers ────────────────────────────────────────────────────────────
    changes = suggestions.get("suggested_changes", [])
    proj_current = suggestions.get("projected_pts_current", 0)
    proj_new     = suggestions.get("projected_pts_new", 0)
    pts_gain     = suggestions.get("points_gain", 0)

    # ════════════════════════════════════════════════════════════════════════
    # Helper builders
    # ════════════════════════════════════════════════════════════════════════

    def _race_rows() -> str:
        rows = ""
        for d in race_order[:10]:
            pos  = d.get("predicted_rank", "?")
            name = d.get("driver", "")
            team = d.get("team", "")
            dnf  = d.get("dnf_prob_pct", 0)
            conf = d.get("confidence_pct", 50)
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, f"P{pos}")
            col  = _tc(team)
            dnf_cls = "badge-red" if dnf > 15 else "badge-yellow" if dnf > 8 else "badge-green"
            rows += f"""
            <tr>
              <td class="pos-cell">{medal}</td>
              <td><span class="dot" style="background:{col}"></span>{name}</td>
              <td class="dim-cell">{team}</td>
              <td><span class="badge {dnf_cls}">{dnf:.0f}%</span></td>
              <td class="dim-cell">{conf:.0f}%</td>
            </tr>"""
        return rows

    def _quali_rows() -> str:
        rows = ""
        for q in quali_order[:10]:
            pos  = q.get("predicted_grid", "?")
            name = q.get("driver", "")
            team = q.get("team", "")
            q3   = "✓" if q.get("q3_likely") else ""
            col  = _tc(team)
            rows += f"""
            <tr>
              <td class="pos-cell">P{pos}</td>
              <td><span class="dot" style="background:{col}"></span>{name}</td>
              <td class="dim-cell">{team}</td>
              <td class="accent-cell">{q3}</td>
            </tr>"""
        return rows

    def _pts_rows() -> str:
        rows = ""
        for d in sorted_driver_pts[:15]:
            name = d.get("driver", "")
            team = d.get("team", "")
            pts  = d.get("total_pts", 0)
            col  = _tc(team)
            pts_cls = "badge-green" if pts >= 25 else "badge-yellow" if pts >= 12 else ""
            rows += f"""
            <tr>
              <td><span class="dot" style="background:{col}"></span>{name}</td>
              <td class="dim-cell">{team}</td>
              <td><span class="badge {pts_cls}">{pts:.1f}</span></td>
            </tr>"""
        return rows

    def _transfer_cards() -> str:
        if not changes:
            return '<p class="no-changes">✅ No beneficial changes found — your team is well positioned!</p>'
        cards = ""
        for i, c in enumerate(changes, 1):
            gain   = c.get("pts_gain", 0)
            cdiff  = c.get("cost_diff", 0)
            cost_cl = "text-red" if cdiff > 0 else "text-green"
            cost_s  = f"+{cdiff:.1f}M" if cdiff > 0 else f"{cdiff:.1f}M"
            cards += f"""
            <div class="transfer-card">
              <div class="transfer-num">#{i}</div>
              <div class="transfer-body">
                <div class="transfer-out">
                  <span class="label-out">OUT</span>
                  <strong>{c.get("out","")}</strong>
                  <span class="dim-cell">${c.get("out_price",0)}M · {c.get("out_pts",0):.0f}pts</span>
                </div>
                <div class="transfer-arrow">➜</div>
                <div class="transfer-in">
                  <span class="label-in">IN</span>
                  <strong>{c.get("in","")}</strong>
                  <span class="dim-cell">${c.get("in_price",0)}M · {c.get("in_pts",0):.0f}pts</span>
                </div>
                <div class="transfer-meta">
                  <span class="badge badge-green">+{gain:.1f} pts</span>
                  <span class="{cost_cl}">{cost_s}</span>
                </div>
              </div>
            </div>"""
        return cards

    def _dream_rows() -> str:
        rows = ""
        turbo = optimal.get("turbo_driver")
        for d in optimal.get("drivers", []):
            col = _tc(d.get("team", ""))
            # Plan 9-M10: turbo driver scores 2x — mark and double the row so
            # TOTAL == sum(parts) and the assignment is visible
            is_turbo = d.get("name") == turbo
            pts_disp = d.get("pts", 0) * 2.0 if is_turbo else d.get("pts", 0)
            badge = f'{pts_disp:.1f} <span class="badge">TURBO ×2</span>' if is_turbo \
                else f'{pts_disp:.1f}'
            rows += f"""
            <tr>
              <td class="type-cell">Driver</td>
              <td><span class="dot" style="background:{col}"></span>{d.get("name","")}</td>
              <td class="dim-cell">{d.get("team","")}</td>
              <td><span class="badge badge-green">{badge}</span></td>
              <td class="dim-cell">${d.get("price",0):.1f}M</td>
            </tr>"""
        for c in optimal.get("constructors", []):
            col = _tc(c.get("name", ""))
            rows += f"""
            <tr>
              <td class="type-cell">Constructor</td>
              <td><span class="dot" style="background:{col}"></span>{c.get("name","")}</td>
              <td></td>
              <td><span class="badge badge-green">{c.get("pts",0):.1f}</span></td>
              <td class="dim-cell">${c.get("price",0):.1f}M</td>
            </tr>"""
        return rows

    def _my_team_tags(names: list, prices_dict: dict) -> str:
        tags = ""
        for n in names:
            info = prices_dict.get(n, {})
            # For constructors, `n` IS the team name — entries carry no "team" key
            col = _tc(info.get("team", "") or n)
            tags += f'<span class="player-tag" style="border-left:4px solid {col}">{n}<span class="player-price">${info.get("price",0):.1f}M</span></span>'
        return tags

    def _ctor_rows() -> str:
        rows = ""
        for i, c in enumerate(best_ctors[:5], 1):
            col = _tc(c.get("constructor", ""))
            rows += f"""
            <tr>
              <td class="pos-cell">#{i}</td>
              <td><span class="dot" style="background:{col}"></span>{c.get("constructor","")}</td>
              <td><span class="badge badge-green">{c.get("total_pts",0):.1f}</span></td>
              <td class="dim-cell">${c.get("price",0):.1f}M</td>
              <td class="dim-cell">{c.get("value","")}</td>
            </tr>"""
        return rows

    def _mc_rows() -> str:
        if not mc_results:
            return "<tr><td colspan='5' class='dim-cell'>No MC data available</td></tr>"
        rows = ""
        # Sort by Top-3 probability
        sorted_mc = sorted(mc_results.values(), key=lambda x: getattr(x, "p_top3", 0), reverse=True)
        for stats in sorted_mc[:12]:
            col = _tc(stats.team)
            p_top3 = getattr(stats, "p_top3", 0)
            p_pts  = getattr(stats, "p_points_finish", 0)
            p_dnf  = getattr(stats, "p_dnf", 0)
            rows += f"""
            <tr>
              <td><span class="dot" style="background:{col}"></span>{stats.driver}</td>
              <td class="accent-cell" style="color:var(--yellow)">{p_top3:.1f}%</td>
              <td class="accent-cell">{p_pts:.1f}%</td>
              <td class="dim-cell">{stats.p90_pts:.1f}</td>
              <td><span class="badge {'badge-red' if p_dnf > 15 else 'dim-cell'}">{p_dnf:.1f}%</span></td>
            </tr>"""
        return rows

    # ════════════════════════════════════════════════════════════════════════
    # HTML Template
    # ════════════════════════════════════════════════════════════════════════
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>F1 Fantasy — R{round_num} {race_name}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:        #050507;
    --surface:   rgba(22, 22, 26, 0.7);
    --surface2:  rgba(30, 30, 36, 0.8);
    --border:    rgba(255, 255, 255, 0.1);
    --red:       #ff4d6a;
    --green:     #00d084;
    --yellow:    #ffcc00;
    --blue:      #3671c6;
    --text:      #e8e8f0;
    --dim:       #9494ad;
    --radius:    16px;
    --glass:     blur(12px);
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    background: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #050507 100%);
    color:var(--text);
    font-family:'Outfit', 'Segoe UI', system-ui, sans-serif;
    font-size:14px; line-height:1.6;
    min-height: 100vh;
  }}
  
  /* ── Layout ── */
  .page-wrap {{ max-width:1200px; margin:0 auto; padding:40px 20px 80px; }}

  /* ── Hero header ── */
  .hero {{
    background: linear-gradient(135deg, rgba(232,0,45,0.1) 0%, rgba(54,113,198,0.05) 100%);
    backdrop-filter: var(--glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 40px;
    margin-bottom: 32px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content:''; position:absolute; top:-100px; right:-100px;
    width:300px; height:300px;
    background:radial-gradient(circle, rgba(232,0,45,0.15), transparent 70%);
    border-radius:50%;
  }}
  .hero-badge {{ font-size:11px; letter-spacing:3px; color:var(--red); text-transform:uppercase; margin-bottom:12px; font-weight:700; }}
  .hero-title {{ font-size:36px; font-weight:800; color:#fff; margin-bottom:8px; letter-spacing:-0.5px; }}
  .hero-sub   {{ color:var(--dim); font-size:15px; }}
  .hero-meta  {{ display:flex; gap:32px; margin-top:24px; flex-wrap:wrap; }}
  .hero-meta-item {{ display:flex; flex-direction:column; }}
  .hero-meta-label {{ font-size:10px; letter-spacing:2px; text-transform:uppercase; color:var(--dim); }}
  .hero-meta-value {{ font-size:16px; font-weight:600; color:#fff; }}

  /* ── Weather strip ── */
  .weather-strip {{
    display:flex; gap:20px; align-items:center;
    background:var(--surface); border:1px solid var(--border);
    border-radius:var(--radius); padding:14px 20px;
    margin-bottom:28px; flex-wrap:wrap;
  }}
  .wx-icon {{ font-size:32px; }}
  .wx-info  {{ display:flex; gap:32px; flex-wrap:wrap; }}
  .wx-item  {{ display:flex; flex-direction:column; }}
  .wx-label {{ font-size:10px; letter-spacing:1.5px; text-transform:uppercase; color:var(--dim); }}
  .wx-val   {{ font-size:16px; font-weight:600; }}

  /* ── Section headers ── */
  .section {{ margin-bottom:32px; }}
  .section-title {{
    font-size:11px; letter-spacing:2.5px; text-transform:uppercase;
    color:var(--dim); margin-bottom:14px;
    padding-bottom:8px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; gap:10px;
  }}
  .section-title span {{ color:var(--red); font-size:16px; }}

  /* ── Cards grid ── */
  .card-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; }}
  .card {{
    background:var(--surface); border:1px solid var(--border);
    border-radius:var(--radius); padding:20px;
  }}
  .card-title {{ font-size:12px; letter-spacing:1.5px; text-transform:uppercase; color:var(--dim); margin-bottom:14px; }}

  /* ── Tables ── */
  table {{ width:100%; border-collapse:collapse; }}
  th    {{ text-align:left; padding:6px 10px; font-size:11px; letter-spacing:1px; text-transform:uppercase; color:var(--dim); border-bottom:1px solid var(--border); }}
  td    {{ padding:8px 10px; border-bottom:1px solid #1c1c24; vertical-align:middle; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:rgba(255,255,255,.02); }}

  .pos-cell   {{ font-weight:700; width:44px; text-align:center; }}
  .dim-cell   {{ color:var(--dim); font-size:13px; }}
  .accent-cell{{ color:var(--green); font-weight:600; text-align:center; }}
  .type-cell  {{ color:var(--dim); font-size:12px; letter-spacing:1px; text-transform:uppercase; }}
  .text-red   {{ color:#ff4d6a; }}
  .text-green {{ color:var(--green); }}

  /* ── Dot ── */
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; vertical-align:middle; }}

  /* ── Badges ── */
  .badge {{ display:inline-block; padding:2px 8px; border-radius:20px; font-size:12px; font-weight:600; }}
  .badge-green  {{ background:rgba(0,208,132,.15); color:var(--green); }}
  .badge-yellow {{ background:rgba(255,204,0,.15);  color:var(--yellow); }}
  .badge-red    {{ background:rgba(232,0,45,.15);   color:#ff4d6a; }}

  /* ── Projection banner ── */
  .proj-banner {{
    display:flex; gap:0; background:var(--surface2);
    border:1px solid var(--border); border-radius:var(--radius);
    overflow:hidden; margin-bottom:28px;
  }}
  .proj-item {{
    flex:1; padding:20px 24px; text-align:center;
    border-right:1px solid var(--border);
  }}
  .proj-item:last-child {{ border-right:none; }}
  .proj-label {{ font-size:11px; letter-spacing:1.5px; text-transform:uppercase; color:var(--dim); margin-bottom:6px; }}
  .proj-value {{ font-size:26px; font-weight:700; }}
  .proj-gain  {{ color:var(--green); }}
  .proj-base  {{ color:var(--yellow); }}
  .proj-start {{ color:var(--text); }}

  /* ── Transfer cards ── */
  .transfer-card {{
    background:var(--surface2); border:1px solid var(--border);
    border-radius:var(--radius); padding:16px 20px;
    margin-bottom:12px; display:flex; align-items:center; gap:16px;
  }}
  .transfer-num {{ font-size:22px; font-weight:700; color:var(--border); width:30px; flex-shrink:0; }}
  .transfer-body {{ display:flex; align-items:center; gap:12px; flex:1; flex-wrap:wrap; }}
  .transfer-out, .transfer-in {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
  .transfer-arrow {{ font-size:20px; color:var(--dim); }}
  .label-out {{ background:rgba(232,0,45,.2); color:#ff4d6a; font-size:10px; letter-spacing:1px; text-transform:uppercase; padding:2px 6px; border-radius:4px; }}
  .label-in  {{ background:rgba(0,208,132,.2); color:var(--green); font-size:10px; letter-spacing:1px; text-transform:uppercase; padding:2px 6px; border-radius:4px; }}
  .transfer-meta {{ display:flex; align-items:center; gap:10px; margin-left:auto; }}
  .no-changes {{
    color:var(--green); background:rgba(0,208,132,.08);
    border:1px solid rgba(0,208,132,.25); border-radius:var(--radius);
    padding:16px 20px;
  }}

  /* ── My team tags ── */
  .player-tags {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:4px; }}
  .player-tag  {{
    background:var(--surface2); border:1px solid var(--border);
    border-radius:8px; padding:6px 12px;
    font-size:13px; font-weight:600;
    display:flex; align-items:center; gap:8px;
  }}
  .player-price {{ color:var(--dim); font-size:12px; font-weight:400; }}

  /* ── Footer ── */
  .footer {{
    margin-top:48px; text-align:center;
    color:var(--dim); font-size:12px;
    border-top:1px solid var(--border); padding-top:20px;
  }}
</style>
</head>
<body>
<div class="page-wrap">

  <!-- HERO -->
  <div class="hero">
    <div class="hero-badge">F1 Fantasy Prediction Tool · {ts}</div>
    <div class="hero-title">Round {round_num} — {race_name}</div>
    <div class="hero-sub">{locality}, {country} · {circuit}</div>
    <div class="hero-meta">
      <div class="hero-meta-item">
        <span class="hero-meta-label">Race Date</span>
        <span class="hero-meta-value">{race_date}</span>
      </div>
      <div class="hero-meta-item">
        <span class="hero-meta-label">Your Points</span>
        <span class="hero-meta-value">{my_points:.0f} pts</span>
      </div>
      <div class="hero-meta-item">
        <span class="hero-meta-label">Projected After Best Moves</span>
        <span class="hero-meta-value" style="color:var(--green)">{my_points + proj_new:.0f} pts</span>
      </div>
    </div>
  </div>

  <!-- WEATHER -->
  <div class="weather-strip">
    <div class="wx-icon">{wx_icon}</div>
    <div class="wx-info">
      <div class="wx-item"><span class="wx-label">Condition</span><span class="wx-val">{wx_cond}</span></div>
      <div class="wx-item"><span class="wx-label">Temperature</span><span class="wx-val">{wx_temp}°C</span></div>
      <div class="wx-item"><span class="wx-label">Rain Risk</span><span class="wx-val">{wx_rain}</span></div>
      <div class="wx-item"><span class="wx-label">Wind</span><span class="wx-val">{wx_wind} km/h</span></div>
    </div>
  </div>

  <!-- PROJECTION BANNER -->
  <div class="proj-banner">
    <div class="proj-item">
      <div class="proj-label">Current Points</div>
      <div class="proj-value proj-start">{my_points:.0f}</div>
    </div>
    <div class="proj-item">
      <div class="proj-label">Projected (No Changes)</div>
      <div class="proj-value proj-base">{my_points + proj_current:.0f}</div>
    </div>
    <div class="proj-item">
      <div class="proj-label">Projected (Best Moves)</div>
      <div class="proj-value proj-gain">{my_points + proj_new:.0f}</div>
    </div>
    <div class="proj-item">
      <div class="proj-label">Gain From Transfers</div>
      <div class="proj-value proj-gain">+{pts_gain:.1f}</div>
    </div>
  </div>

  <!-- TRANSFER SUGGESTIONS -->
  <div class="section">
    <div class="section-title"><span>⇄</span> Recommended Transfers</div>
    {_transfer_cards()}
  </div>

  <!-- MY TEAM -->
  <div class="section">
    <div class="section-title"><span>👤</span> Your Current Team</div>
    <div class="card">
      <div class="card-title">Drivers</div>
      <div class="player-tags">{_my_team_tags(my_drivers, driver_prices)}</div>
      <div class="card-title" style="margin-top:16px">Constructors</div>
      <div class="player-tags">{_my_team_tags(my_constructors, constructor_prices)}</div>
    </div>
  </div>

  <!-- DREAM TEAM -->
  <div class="section">
    <div class="section-title"><span>⭐</span> Dream Team — Best $100M Lineup</div>
    <div class="card">
      <table>
        <thead><tr><th>Type</th><th>Player</th><th>Team</th><th>Proj Pts</th><th>Price</th></tr></thead>
        <tbody>{_dream_rows()}</tbody>
      </table>
      <div style="margin-top:14px; display:flex; gap:20px; font-size:13px;">
        <span>Total: <strong style="color:var(--green)">{optimal.get("total_pts",0):.1f} pts</strong></span>
        <span>Cost: <strong style="color:var(--yellow)">${optimal.get("total_price",0):.1f}M</strong></span>
        <span>Budget left: <strong style="color:var(--green)">${optimal.get("budget_left",0):.1f}M</strong></span>
      </div>
    </div>
  </div>

  <!-- PREDICTIONS -->
  <div class="card-grid section">
    <div class="card">
      <div class="card-title">🏁 Predicted Race Order (Top 10)</div>
      <table>
        <thead><tr><th>Pos</th><th>Driver</th><th>Team</th><th>DNF%</th><th>Conf</th></tr></thead>
        <tbody>{_race_rows()}</tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-title">⏱️ Predicted Qualifying (Top 10)</div>
      <table>
        <thead><tr><th>Grid</th><th>Driver</th><th>Team</th><th>Q3</th></tr></thead>
        <tbody>{_quali_rows()}</tbody>
      </table>
    </div>
  </div>

  <!-- FANTASY POINTS + CONSTRUCTORS -->
  <div class="card-grid section">
    <div class="card">
      <div class="card-title">💎 Fantasy Points Projection (Top 15)</div>
      <table>
        <thead><tr><th>Driver</th><th>Team</th><th>Proj Pts</th></tr></thead>
        <tbody>{_pts_rows()}</tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-title">🔬 Monte Carlo Probabilities (Top 12)</div>
      <table>
        <thead><tr><th>Driver</th><th>Top 3%</th><th>Points%</th><th>Ceiling</th><th>DNF%</th></tr></thead>
        <tbody>{_mc_rows()}</tbody>
      </table>
    </div>
  </div>
  
  <div class="card-grid section">
    <div class="card">
      <div class="card-title">🔧 Constructor Rankings</div>
      <table>
        <thead><tr><th>Rank</th><th>Constructor</th><th>Proj Pts</th><th>Price</th><th>Value</th></tr></thead>
        <tbody>{_ctor_rows()}</tbody>
      </table>
    </div>
  </div>

  <div class="footer">
    Generated by F1 Fantasy Prediction Tool · {ts}<br>
    Powered by Jolpica API · ML Ensemble · Monte Carlo Simulation
  </div>

</div>
</body>
</html>
"""

    fname = output_dir / f"dashboard_R{round_num}_{race_name.replace(' ', '_')}.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html)
    return fname
