let currentTeam = { drivers: [], constructors: [], budget: 100.0, points: 0.0, transfers: 1 };
let marketPrices = { drivers: {}, constructors: {} };
let lastResults = null;
let chipState = { used: {}, used_on_round: {}, used_on_circuit: {}, season: 2026 };
let _currentRaceInfo = null;  // populated after pipeline completes
let _charts = { practice: null, radar: null, comparison: null, telemetry: null, accuracyTrend: null };

document.addEventListener('DOMContentLoaded', () => {
    // Heartbeat starts IMMEDIATELY — if it waited for full dashboard init, a
    // slow /api/prices fetch could delay the first beat past the watchdog's
    // 20s timeout and shut the server down mid-load.
    setupHeartbeat();
    initApp();
    setupNavigation();
    setupTabs();
});

// Boot welcome sequence — runs only before server responds.
// Stops automatically when waitForServer() calls setLaunchStatus().
let _welcomeAnimRunning = true;
(function animateLaunchStatus() {
    const msgs = [
        'Powering up the pit wall...',
        'Loading racing line data...',
        'Spinning up prediction algorithms...',
        'Connecting to telemetry engine...',
    ];
    let step = 0;
    const el = document.getElementById('launch-status');
    if (!el) return;
    const interval = setInterval(() => {
        if (!_welcomeAnimRunning) { clearInterval(interval); return; }
        el.textContent = msgs[step % msgs.length];
        step++;
    }, 600);
})();

async function initApp() {
    document.querySelector('.app-container').style.visibility = 'hidden';
    await waitForServer();
    await loadDashboardData();
    await loadChipState();

    document.querySelector('.app-container').style.visibility = 'visible';

    // Phase 3/9: Cinematic spring transition from launch to dashboard
    const launchEl  = document.getElementById('launch-screen');
    const navEl     = document.getElementById('main-nav');
    const dashEl    = document.getElementById('dashboard-screen');

    if (typeof gsap !== 'undefined' && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        // Prepare dashboard: off-screen above, invisible
        gsap.set(dashEl, { opacity: 0, y: 30 });
        dashEl.classList.add('active');
        navEl.classList.remove('hidden');

        // Spring transition: launch fades out while dashboard springs in
        const tl = gsap.timeline();
        tl.to(launchEl, { opacity: 0, y: -20, duration: 0.35, ease: 'power2.in',
            onComplete: () => {
                launchEl.classList.remove('active');
                gsap.set(launchEl, { clearProps: 'all' });
                launchEl.style.display = 'none';
            }
        });
        tl.to(dashEl, { opacity: 1, y: 0, duration: 0.45, ease: 'power2.out',
            onComplete: () => gsap.set(dashEl, { clearProps: 'all' })
        }, '-=0.15');
    } else {
        // Reduced-motion fallback: instant swap
        launchEl.classList.remove('active');
        launchEl.style.display = 'none';
        navEl.classList.remove('hidden');
        dashEl.classList.add('active');
    }
}

// ---- NAVIGATION ----
function setupNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            
            const targetId = e.target.getAttribute('data-target');
            document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');
            
            if (targetId === 'results-screen') {
                clearNavBadge('nav-results');
            } else if (targetId === 'standings-screen') {
                fetchStandings();
            } else if (targetId === 'past-archive-screen') {
                initPastRaces();
                initPastPredictions();
            } else if (targetId === 'pred-analysis-screen') {
                initPredictionAnalysis();
            }
        });
    });
}

function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            const parent = btn.closest('.screen');
            
            // Toggle buttons
            parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle content
            parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(targetTab).classList.add('active');
            
            // Force chart resize if needed
            window.dispatchEvent(new Event('resize'));
        });
    });
}

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
}

function activateNavBtn(targetId) {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    const btn = document.querySelector(`.nav-btn[data-target="${targetId}"]`);
    if (btn) btn.classList.add('active');
}

function setNavBadge(navId, text, type) {
    // type: 'live' | 'new'
    const btn = document.getElementById(navId);
    if (!btn) return;
    clearNavBadge(navId);
    const badge = document.createElement('span');
    badge.className = 'nav-badge badge-' + type;
    badge.id = navId + '-badge';
    badge.textContent = text;
    
    btn.appendChild(badge);
}

function clearNavBadge(navId) {
    const existing = document.getElementById(navId + '-badge');
    if (existing) existing.remove();
}

// ---- DATA LOADING ----
async function waitForServer() {
    let attempts = 0;
    markBootStepActive('boot-step-server');
    setLaunchSubStatus('The prediction engine starts up in the background when you launch the app.');
    while (true) {
        try {
            const res = await fetch('/api/status');
            if (res.ok) {
                _welcomeAnimRunning = false;
                setLaunchStatus('Engine online. Loading your dashboard...');
                setLaunchSubStatus('Fetching live race data, weather, and market prices...');
                markBootStepDone('boot-step-server');
                break;
            }
        } catch (e) {
            attempts++;
            if (attempts === 3) {
                setLaunchStatus('Starting prediction engine...');
                setLaunchSubStatus('First launch may take 5-10 seconds while Python initialises.');
            } else if (attempts > 10) {
                setLaunchStatus('Engine is taking longer than usual...');
                setLaunchSubStatus('Still starting up. This is normal on first run or slow machines.');
            }
        }
        await new Promise(r => setTimeout(r, 1000));
    }
}

const trackMaps = {
    'bahrain': 'Bahrain', 'jeddah': 'Saudi%20Arabia', 'albert_park': 'Australia',
    'suzuka': 'Japan', 'shanghai': 'China', 'miami': 'Miami', 'imola': 'Emilia%20Romagna',
    'monaco': 'Monaco', 'catalunya': 'Spain', 'villeneuve': 'Canada',
    'red_bull_ring': 'Austria', 'silverstone': 'Great%20Britain', 'hungaroring': 'Hungary',
    'spa': 'Belgium', 'zandvoort': 'Netherlands', 'monza': 'Italy', 'baku': 'Azerbaijan',
    'marina_bay': 'Singapore', 'americas': 'USA', 'rodriguez': 'Mexico',
    'interlagos': 'Brazil', 'vegas': 'Las%20Vegas', 'losail': 'Qatar', 'yas_marina': 'Abu%20Dhabi',
    'madring': 'Spain'
};

// Team accent colours (2026 Season)
const teamColors = {
    'Red Bull':     '#3671C6', 'McLaren':      '#FF8000',
    'Ferrari':      '#E8002D', 'Mercedes':     '#27F4D2',
    'Aston Martin': '#358C75', 'Alpine':       '#FF87BC',
    'Williams':     '#64C4FF', 'Racing Bulls': '#6692FF',
    'Audi':         '#BB0000', 'Haas':         '#B6BABD',
    'Cadillac':     '#D6E4FF', 'VCARB':        '#6692FF',
    'Red Bull Racing': '#3671C6'
};

// Driver -> Constructor team mapping (2026 grid)
const driverTeamMap = {
    'Max Verstappen':    'Red Bull',     'Isack Hadjar':      'Red Bull',
    'Lando Norris':      'McLaren',      'Oscar Piastri':     'McLaren',
    'Charles Leclerc':   'Ferrari',      'Lewis Hamilton':    'Ferrari',
    'George Russell':    'Mercedes',     'Kimi Antonelli':    'Mercedes',
    'Fernando Alonso':   'Aston Martin', 'Lance Stroll':      'Aston Martin',
    'Pierre Gasly':      'Alpine',       'Franco Colapinto':  'Alpine',
    'Liam Lawson':       'Racing Bulls', 'Arvid Lindblad':    'Racing Bulls',
    'Carlos Sainz':      'Williams',     'Alexander Albon':   'Williams',
    'Nico Hulkenberg':   'Audi',         'Gabriel Bortoleto': 'Audi',
    'Esteban Ocon':      'Haas',         'Oliver Bearman':    'Haas',
    'Sergio Perez':      'Cadillac',     'Valtteri Bottas':   'Cadillac'
};

// Official Driver Acronyms (TV Broadcast Codes)
const driverCodes = {
    'Max Verstappen': 'VER', 'Isack Hadjar': 'HAD',
    'Lando Norris': 'NOR',   'Oscar Piastri': 'PIA',
    'Charles Leclerc': 'LEC', 'Lewis Hamilton': 'HAM',
    'George Russell': 'RUS', 'Kimi Antonelli': 'ANT',
    'Fernando Alonso': 'ALO', 'Lance Stroll': 'STR',
    'Pierre Gasly': 'GAS',   'Franco Colapinto': 'COL',
    'Liam Lawson': 'LAW',    'Arvid Lindblad': 'LIN',
    'Carlos Sainz': 'SAI',   'Alexander Albon': 'ALB',
    'Nico Hulkenberg': 'HUL', 'Gabriel Bortoleto': 'BOR',
    'Esteban Ocon': 'OCO',   'Oliver Bearman': 'BEA',
    'Sergio Perez': 'PER',   'Valtteri Bottas': 'BOT'
};

// Driver headshots — local files in ui/images/drivers/
const driverHeadshots = {
    'Max Verstappen':    '/static/images/drivers/verstappen.png',
    'Isack Hadjar':      '/static/images/drivers/hadjar.png',
    'Lando Norris':      '/static/images/drivers/norris.png',
    'Oscar Piastri':     '/static/images/drivers/piastri.png',
    'Charles Leclerc':   '/static/images/drivers/leclerc.png',
    'Lewis Hamilton':    '/static/images/drivers/hamilton.png',
    'George Russell':    '/static/images/drivers/russell.png',
    'Kimi Antonelli':    '/static/images/drivers/antonelli.png',
    'Fernando Alonso':   '/static/images/drivers/alonso.png',
    'Lance Stroll':      '/static/images/drivers/stroll.png',
    'Pierre Gasly':      '/static/images/drivers/gasly.png',
    'Franco Colapinto':  '/static/images/drivers/colapinto.png',
    'Liam Lawson':       '/static/images/drivers/lawson.png',
    'Arvid Lindblad':    '/static/images/drivers/lindblad.png',
    'Carlos Sainz':      '/static/images/drivers/sainz.png',
    'Alexander Albon':   '/static/images/drivers/albon.png',
    'Nico Hulkenberg':   '/static/images/drivers/hulkenberg.png',
    'Gabriel Bortoleto': '/static/images/drivers/bortoleto.png',
    'Esteban Ocon':      '/static/images/drivers/ocon.png',
    'Oliver Bearman':    '/static/images/drivers/bearman.png',
    'Sergio Perez':      '/static/images/drivers/perez.png',
    'Valtteri Bottas':   '/static/images/drivers/bottas.png',
    'Jack Doohan':       '/static/images/drivers/doohan.png',
    'Yuki Tsunoda':      '/static/images/drivers/tsunoda.png'
};

// Constructor logos — local files in ui/images/teams/
const constructorLogos = {
    'Red Bull':        '/static/images/teams/red_bull.png',
    'Red Bull Racing': '/static/images/teams/red_bull.png',
    'Ferrari':         '/static/images/teams/ferrari.png',
    'Mercedes':        '/static/images/teams/mercedes.png',
    'McLaren':         '/static/images/teams/mclaren.png',
    'Aston Martin':    '/static/images/teams/aston_martin.png',
    'Alpine':          '/static/images/teams/alpine.png',
    'Williams':        '/static/images/teams/williams.png',
    'Haas':            '/static/images/teams/haas.png',
    'Haas F1 Team':    '/static/images/teams/haas.png',
    'Racing Bulls':    '/static/images/teams/racing_bulls.png',
    'VCARB':           '/static/images/teams/racing_bulls.png',
    'Audi':            '/static/images/teams/audi.png',
    'Cadillac':        '/static/images/teams/cadillac.png',
    'Kick Sauber':     '/static/images/teams/kick_sauber.png',
    'Sauber':          '/static/images/teams/kick_sauber.png'
};

function setLaunchStatus(msg) {
    const el = document.getElementById('launch-status');
    if (el) el.innerText = msg;
}

function setLaunchSubStatus(msg) {
    const el = document.getElementById('launch-sub-status');
    if (el) el.innerText = msg;
}

function markBootStepActive(stepId) {
    const el = document.getElementById(stepId);
    if (!el) return;
    document.querySelectorAll('.boot-step.active').forEach(s => s.classList.remove('active'));
    el.classList.add('active');
}

function markBootStepDone(stepId) {
    const el = document.getElementById(stepId);
    if (!el) return;
    el.classList.remove('active');
    el.classList.add('done');
}

async function loadDashboardData() {
    let raceName = "", raceDate = "";

    // Race
    setLaunchStatus('Loading next race data...');
    setLaunchSubStatus('Fetching race schedule and circuit configuration from Jolpica API...');
    markBootStepActive('boot-step-race');
    updateStatusRow('status-race', 'yellow', 'Fetching...');
    try {
        const raceRes = await fetch('/api/race/next');
        const race = await raceRes.json();
        raceName = race.name; raceDate = race.date;
        document.getElementById('dash-race-name').innerText = race.name;
        let dateSubtitle = `Round ${race.round} \u00b7 ${race.date}`;
        if (race.latest_completed_round && race.latest_completed_name) {
            dateSubtitle += `  (Latest Completed: R${race.latest_completed_round} ${race.latest_completed_name})`;
        }
        document.getElementById('dash-race-date').innerText = dateSubtitle;
        updateStatusRow('status-race', 'green', `${race.name} \u00b7 R${race.round}`);
        const cId = race.circuit_id || '';
        if (cId && trackMaps[cId]) {
            const mapUrl = `https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Track%20icons%204x3/${trackMaps[cId]}.png`;
            document.getElementById('dash-track-map').innerHTML = `<img src="${mapUrl}" style="width:100%;height:100%;object-fit:contain;filter:drop-shadow(0 0 5px rgba(255,255,255,0.2));" alt="Track Map">`;
        } else {
            document.getElementById('dash-track-map').innerHTML = `<div style="height:200px;display:flex;align-items:center;justify-content:center;opacity:0.2;">[ Map Unavailable ]</div>`;
        }
        setupTimezoneSelector(race);
        renderSessionTimes(race);
        markBootStepDone('boot-step-race');
    } catch (e) {
        updateStatusRow('status-race', 'red', 'Error');
        markBootStepDone('boot-step-race');
    }

    // Weather
    if (raceName) {
        setLaunchStatus('Fetching race weekend weather forecast...');
        setLaunchSubStatus('Checking forecast for ' + raceName + ' race weekend via Open-Meteo...');
        markBootStepActive('boot-step-weather');
        updateStatusRow('status-weather', 'yellow', 'Fetching...');
        try {
            const weatherRes = await fetch(`/api/weather?race_name=${encodeURIComponent(raceName)}&date=${raceDate}`);
            const weather = await weatherRes.json();
            
            // Determine risk based on session data or flat risk
            let riskRaw = weather.rain_risk || 'unknown';
            let temp = weather.temp_c || '—';
            let precip = weather.precip_mm || '0';
            let humidity = weather.humidity || '—';
            let wind = weather.wind_kph || '—';

            // Try to pull from sessions.Race if available (Phase 2 fix)
            if (weather.sessions && weather.sessions.Race) {
                const raceW = weather.sessions.Race;
                riskRaw = raceW.rain_risk || riskRaw;
                temp = raceW.temp_day_c || temp;
                precip = (raceW.precip_mm || 0); // OpenF1 uses mm
                humidity = raceW.humidity_avg_pct || humidity;
                wind = raceW.wind_speed_max_kph || wind;
            }

            const riskMap = { low: '\uD83D\uDFE2 Low', medium: '\uD83D\uDFE1 Medium', high: '\uD83D\uDD34 High', unknown: '\u2753 Unknown' };
            const iconMap = { low: '☀️', medium: '⛅', high: '🌧️', unknown: '❓' };
            
            document.getElementById('hero-weather-icon').innerText = iconMap[riskRaw] || '❓';
            updateStatusRow('status-weather', riskRaw === 'low' ? 'green' : riskRaw === 'medium' ? 'yellow' : 'red', `${riskMap[riskRaw] || riskRaw} risk`);
            
            // Update Hero Mini Stats
            const tempEl = document.querySelector('#weather-temp div:last-child');
            if (tempEl) tempEl.innerText = `${temp}°C`;
            const precipEl = document.querySelector('#weather-precip div:last-child');
            if (precipEl) precipEl.innerText = `${precip}${weather.sessions ? 'mm' : '%'}`; // Handle unit difference
            const humEl = document.querySelector('#weather-humidity div:last-child');
            if (humEl) humEl.innerText = `${humidity}%`;
            const windEl = document.querySelector('#weather-wind div:last-child');
            if (windEl) windEl.innerText = `${wind}kph`;

            markBootStepDone('boot-step-weather');
        } catch (e) {
            updateStatusRow('status-weather', 'red', 'Error');
            markBootStepDone('boot-step-weather');
        }
    }

    // System Health (Phase 2)
    setLaunchStatus('Checking system health...');
    setLaunchSubStatus('Verifying ML engine, FastF1 cache, and Julia compiler status...');
    markBootStepActive('boot-step-health');
    try {
        const healthRes = await fetch('/api/system/health');
        const health = await healthRes.json();
        updateStatusRow('status-engine', health.compiler === 'Detected' ? 'green' : 'yellow', health.engine);
        updateStatusRow('status-cache', 'green', health.cache_size);
        markBootStepDone('boot-step-health');
    } catch (e) {
        markBootStepDone('boot-step-health');
    }

    // Prices
    setLaunchStatus('Scraping F1 Fantasy market prices...');
    markBootStepActive('boot-step-prices');
    
    let priceTimerSecs = 0;
    const priceTimer = setInterval(() => {
        priceTimerSecs++;
        setLaunchSubStatus(`Connecting to F1 Fantasy servers... (${priceTimerSecs}s elapsed)`);
    }, 1000);

    try {
        const pricesRes = await fetch('/api/prices');
        marketPrices = await pricesRes.json();
        clearInterval(priceTimer);
        updateStatusRow('status-team', 'grey', 'Scanned Market');
        renderPriceLists();
        markBootStepDone('boot-step-prices');
        setLaunchSubStatus(`Market prices loaded in ${priceTimerSecs}s ✓`);
    } catch (e) {
        clearInterval(priceTimer);
        setLaunchSubStatus('Using cached prices (live fetch failed)');
        markBootStepDone('boot-step-prices');
    }

    // Team
    setLaunchStatus('Restoring your saved team...');
    setLaunchSubStatus('Loading your driver and constructor selections from local storage...');
    markBootStepActive('boot-step-team');
    try {
        const teamRes = await fetch('/api/team');
        currentTeam = await teamRes.json();
        currentTeam.transfers = currentTeam.transfers || 1;
        document.getElementById('team-budget').value = currentTeam.budget_remaining || 100.0;
        document.getElementById('team-points').value = currentTeam.current_points || 0.0;
        if (currentTeam.drivers && currentTeam.drivers.length > 0) {
            updateStatusRow('status-team', 'green', `${currentTeam.drivers.length} drivers selected`);
        } else {
            updateStatusRow('status-team', 'grey', 'Not set');
            currentTeam.drivers = [];
            currentTeam.constructors = [];
        }
        renderTeamPickers();
        markBootStepDone('boot-step-team');
    } catch (e) {
        updateStatusRow('status-team', 'grey', 'Error');
        markBootStepDone('boot-step-team');
    }

    setLaunchStatus('All systems ready. Welcome back! 🏎️');
    setLaunchSubStatus('Dashboard loading now...');
    await new Promise(resolve => setTimeout(resolve, 700));
}

function updateStatusRow(rowId, colorClass, valueText) {
    const row = document.getElementById(rowId);
    if (!row) return;
    row.querySelector('.status-indicator').className = `status-indicator ${colorClass}`;
    row.querySelector('.status-val').innerText = valueText;
}

// ---- TEAM SETUP (SCREEN 3) ----
function buildDriverCard(name, data, isSelected) {
    const code = driverCodes[name] || name.split(' ').map(w => w[0]).join('').toUpperCase();
    const team = driverTeamMap[name] || '';
    const teamColor = teamColors[team] || '#FF003C';
    const imgUrl = driverHeadshots[name];
    const photoHtml = imgUrl
        ? `<img class="card-headshot" src="${imgUrl}" alt="${name}" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';">
           <div class="card-headshot-fallback" style="display:none;background:${teamColor}20;border-color:${teamColor}">${code}</div>`
        : `<div class="card-headshot-fallback" style="background:${teamColor}20;border-color:${teamColor}">${code}</div>`;
    return `
        <div class="card-team-bar" style="background:${teamColor}"></div>
        ${photoHtml}
        <div class="card-code">${code}</div>
        <div class="card-title">${name}</div>
        <div class="card-team-name dim-text">${team}</div>
        <div class="card-price">$${data.price.toFixed(1)}M</div>
    `;
}

function buildConstructorCard(name, data, isSelected) {
    const color = teamColors[name] || '#FF003C';
    const logoUrl = constructorLogos[name];
    const logoHtml = logoUrl
        ? `<img class="card-logo" src="${logoUrl}" alt="${name}" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';">
           <div class="card-logo-fallback" style="display:none;color:${color}">${name}</div>`
        : `<div class="card-logo-fallback" style="color:${color}">${name}</div>`;
    return `
        <div class="card-team-bar" style="background:${color}"></div>
        ${logoHtml}
        <div class="card-title">${name}</div>
        <div class="card-price">$${data.price.toFixed(1)}M</div>
    `;
}

function renderTeamPickers() {
    const dGrid = document.getElementById('driver-grid');
    const cGrid = document.getElementById('ctor-grid');
    dGrid.innerHTML = ''; cGrid.innerHTML = '';

    Object.entries(marketPrices.drivers).forEach(([name, data]) => {
        const isSel = currentTeam.drivers.includes(name);
        const el = document.createElement('div');
        el.className = `card card-player ${isSel ? 'selected' : ''}`;
        el.innerHTML = buildDriverCard(name, data, isSel);
        el.onclick = () => toggleTeamSelection('drivers', name, el);
        dGrid.appendChild(el);
    });

    Object.entries(marketPrices.constructors).forEach(([name, data]) => {
        const isSel = currentTeam.constructors.includes(name);
        const el = document.createElement('div');
        el.className = `card card-player card-ctor ${isSel ? 'selected' : ''}`;
        el.innerHTML = buildConstructorCard(name, data, isSel);
        el.onclick = () => toggleTeamSelection('constructors', name, el);
        cGrid.appendChild(el);
    });

    updateTeamCounts();
}

function toggleTeamSelection(type, name, el) {
    const list = currentTeam[type];
    const max = type === 'drivers' ? 5 : 2;
    if (list.includes(name)) {
        list.splice(list.indexOf(name), 1);
        el.classList.remove('selected');
    } else {
        if (list.length >= max) return;
        list.push(name);
        el.classList.add('selected');
    }
    updateTeamCounts();
}

function updateTeamCounts() {
    document.getElementById('driver-count').innerText = `${currentTeam.drivers.length}/5`;
    document.getElementById('ctor-count').innerText = `${currentTeam.constructors.length}/2`;
}

async function saveTeam() {
    currentTeam.budget_remaining = parseFloat(document.getElementById('team-budget').value);
    currentTeam.current_points = parseFloat(document.getElementById('team-points').value);
    currentTeam.transfers = parseInt(document.getElementById('team-transfers').value);
    
    await fetch('/api/team', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentTeam)
    });
    
    updateStatusRow('status-team', 'green', `${currentTeam.drivers.length} drivers selected`);
    showToast('Team saved successfully!', 'success');
}

// ---- PRICES (SCREEN 4) ----
function renderPriceLists() {
    const dList = document.getElementById('prices-driver-list');
    const cList = document.getElementById('prices-ctor-list');
    dList.innerHTML = ''; cList.innerHTML = '';

    Object.entries(marketPrices.drivers).sort((a,b) => b[1].price - a[1].price).forEach(([name, data]) => {
        const imgUrl = driverHeadshots[name];
        const team = driverTeamMap[name] || '';
        const teamColor = teamColors[team] || '#FF003C';
        const code = driverCodes[name] || name.split(' ').pop().slice(0,3).toUpperCase();
        const thumbHtml = imgUrl
            ? `<img class="price-thumb" src="${imgUrl}" alt="" onerror="this.style.display='none'">`
            : `<div class="price-thumb-fallback" style="background:${teamColor}22;color:${teamColor}">${code}</div>`;
        dList.innerHTML += `<div class="price-item">
            ${thumbHtml}
            <span class="price-name">${name}</span>
            <span class="editable-price" onclick="editPrice('driver', '${name}')" title="Click to edit">$${data.price.toFixed(1)}M</span>
        </div>`;
    });

    Object.entries(marketPrices.constructors).sort((a,b) => b[1].price - a[1].price).forEach(([name, data]) => {
        const logoUrl = constructorLogos[name];
        const color = teamColors[name] || '#FF003C';
        const thumbHtml = logoUrl
            ? `<img class="price-thumb" src="${logoUrl}" alt="" onerror="this.style.display='none'">`
            : `<div class="price-thumb-fallback" style="background:${color}22;color:${color}">${name.slice(0,3).toUpperCase()}</div>`;
        cList.innerHTML += `<div class="price-item">
            ${thumbHtml}
            <span class="price-name">${name}</span>
            <span class="editable-price" onclick="editPrice('ctor', '${name}')" title="Click to edit">$${data.price.toFixed(1)}M</span>
        </div>`;
    });
}

function editPrice(type, name) {
    const pDict = type === 'driver' ? marketPrices.drivers : marketPrices.constructors;
    const newPrice = prompt(`Enter new price for ${name} (current: $${pDict[name].price.toFixed(1)}M):`);
    if (newPrice && !isNaN(newPrice)) {
        pDict[name].price = parseFloat(newPrice);
        renderPriceLists();
        renderTeamPickers();
    }
}

// ── Live F1 Pit Wall Telemetry Waveform Engine ────────────────────────
let _telemetryAnimFrame = null;
let _telemetryPhase = 0;
let _telemetrySpeedMultiplier = 1.0;
let _mcCounterTimer = null;

function initTelemetryWaveform() {
    const canvas = document.getElementById('telemetry-waveform-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width > 0 ? rect.width : 600;
    const height = rect.height > 0 ? rect.height : 110;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    function renderFrame() {
        ctx.fillStyle = 'rgba(6, 10, 18, 0.35)'; // Slight trail fade
        ctx.fillRect(0, 0, width, height);

        // Draw background grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1;
        for (let x = 0; x < width; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y < height; y += 22) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        _telemetryPhase += 0.04 * _telemetrySpeedMultiplier;

        // Channel 1: Speed km/h Trace (Cyan)
        ctx.strokeStyle = '#00E5FF';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let x = 0; x < width; x += 3) {
            const t = (x / 60) + _telemetryPhase;
            const y = height * 0.45 + Math.sin(t) * 22 + Math.cos(t * 2.3) * 12 + (Math.sin(t * 0.5) * 8);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Channel 2: Throttle % Trace (Teal/Green)
        ctx.strokeStyle = '#00D2BE';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let x = 0; x < width; x += 4) {
            const t = (x / 45) + _telemetryPhase * 1.2;
            const rawThrottle = Math.sin(t * 1.5) > -0.2 ? (Math.sin(t * 1.5) * 25) : -20;
            const y = height * 0.70 + rawThrottle;
            if (x === 0) ctx.moveTo(x, Math.max(10, Math.min(height - 10, y)));
            else ctx.lineTo(x, Math.max(10, Math.min(height - 10, y)));
        }
        ctx.stroke();

        // Channel 3: Gear Shift Impulses (Purple vertical pulses)
        ctx.strokeStyle = 'rgba(177, 56, 221, 0.6)';
        ctx.lineWidth = 1.5;
        const pulseSpacing = 120;
        const shiftOffset = (_telemetryPhase * 35) % pulseSpacing;
        for (let sx = shiftOffset; sx < width; sx += pulseSpacing) {
            ctx.beginPath();
            ctx.moveTo(sx, height - 15);
            ctx.lineTo(sx, height - 35);
            ctx.stroke();
        }

        if (_isPipelineRunning) {
            _telemetryAnimFrame = requestAnimationFrame(renderFrame);
        }
    }

    if (_telemetryAnimFrame) cancelAnimationFrame(_telemetryAnimFrame);
    _telemetryAnimFrame = requestAnimationFrame(renderFrame);
}

function updateEnsembleSegments(rfWeight, xgbWeight, lgbWeight) {
    const rf = Math.round((rfWeight !== undefined ? rfWeight : 0.25) * 100);
    const xgb = Math.round((xgbWeight !== undefined ? xgbWeight : 0.35) * 100);
    const lgb = Math.round((lgbWeight !== undefined ? lgbWeight : 0.40) * 100);

    const segLgb = document.getElementById('ens-seg-lgb');
    const segXgb = document.getElementById('ens-seg-xgb');
    const segRf  = document.getElementById('ens-seg-rf');

    if (segLgb) segLgb.style.width = `${lgb}%`;
    if (segXgb) segXgb.style.width = `${xgb}%`;
    if (segRf)  segRf.style.width  = `${rf}%`;

    const lblLgb = document.getElementById('ens-lbl-lgb');
    const lblXgb = document.getElementById('ens-lbl-xgb');
    const lblRf  = document.getElementById('ens-lbl-rf');

    if (lblLgb) lblLgb.textContent = `LGBM: ${lgb}%`;
    if (lblXgb) lblXgb.textContent = `XGBoost: ${xgb}%`;
    if (lblRf)  lblRf.textContent  = `RF: ${rf}%`;
}

function animateMonteCarloCounter(current, target) {
    const el = document.getElementById('mc-iterations-counter');
    if (!el) return;
    if (_mcCounterTimer) {
        clearInterval(_mcCounterTimer);
        _mcCounterTimer = null;
    }
    let val = current;
    const step = Math.max(1, Math.round((target - current) / 10));
    _mcCounterTimer = setInterval(() => {
        val = Math.min(target, val + step);
        el.textContent = `${val.toLocaleString()} / 10,000 SIMS`;
        if (val >= target) {
            clearInterval(_mcCounterTimer);
            _mcCounterTimer = null;
        }
    }, 25);
}

// ---- ANALYSIS / SSE (SCREEN 5) ----
const stages = ["NEXT_RACE", "WEATHER", "PRICES", "ML_MODEL", "PREDICTIONS", "ANALYSIS", "COMPLETE"];
let activeOverrides = {};
let gridOverrides = {};
let _isPipelineRunning = false;

async function startAnalysis() {
    if (_isPipelineRunning) {
        showToast("Pipeline is already running!", "warning");
        return;
    }
    _isPipelineRunning = true;
    
    const runBtn = document.getElementById('btn-run-analysis');
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.classList.add('disabled');
        runBtn.textContent = '⏳ Analysing...';
    }

    // Add LIVE badge to Analysis nav item
    setNavBadge('nav-analysis', '● LIVE', 'live');

    // Only auto-navigate to analysis screen if user is on dashboard or already on analysis.
    // Otherwise leave them on their current tab and notify via toast.
    const _activeScreenNow = document.querySelector('.screen.active');
    const _activeScreenId = _activeScreenNow ? _activeScreenNow.id : '';
    if (_activeScreenId === 'dashboard-screen' || _activeScreenId === 'analysis-screen') {
        showScreen('analysis-screen');
        activateNavBtn('analysis-screen');
    } else {
        showToast('⚡ Analysis running in background — visit the Analysis tab to watch progress', 'info');
    }

    const logEl = document.getElementById('analysis-log');
    logEl.innerHTML = '';
    
    // Plan I4: Engine Telemetry panel — key/value grid fed by SSE payloads
    let telPanel = document.getElementById('engine-telemetry');
    if (!telPanel) {
        telPanel = document.createElement('div');
        telPanel.id = 'engine-telemetry';
        telPanel.style.cssText = 'margin-top:12px;padding:10px;background:rgba(0,0,0,0.3);border-radius:8px;font-size:11px;color:var(--text-secondary);';
        telPanel.innerHTML = '<div style="font-weight:bold;margin-bottom:6px;color:var(--text-primary);">⚙ ENGINE TELEMETRY</div><div id="telemetry-kv"></div>';
        logEl.parentElement.appendChild(telPanel);
    }
    const telKv = document.getElementById('telemetry-kv');
    telKv.innerHTML = '';
    
    // Track stage arrival times for duration badges
    window._stageTimes = {};
    
    const anim = document.getElementById('analysis-animation-container');
    if (anim) anim.style.display = 'flex';
    
    // Live Telemetry Waveform Engine
    _telemetrySpeedMultiplier = 1.0;
    initTelemetryWaveform();
    
    // Reset HUD displays to clean pending placeholders
    const roundEl = document.getElementById('hud-round');
    const circuitEl = document.getElementById('hud-circuit');
    const weatherEl = document.getElementById('hud-weather');
    const trackEl = document.getElementById('hud-track');
    const ensembleEl = document.getElementById('hud-ensemble');
    const simsEl = document.getElementById('hud-sims');
    
    if (roundEl) roundEl.innerHTML = `Round — <span class="hud-unit">—</span>`;
    if (circuitEl) circuitEl.innerHTML = `Detecting... <span class="hud-unit">—</span>`;
    if (weatherEl) weatherEl.innerHTML = `— <span class="hud-unit">—</span>`;
    if (trackEl) trackEl.innerHTML = `— <span class="hud-unit">—</span>`;
    if (ensembleEl) ensembleEl.innerHTML = `Fitting... <span class="hud-unit">—</span>`;
    if (simsEl) {
        if (simsEl.dataset.interval) {
            clearInterval(simsEl.dataset.interval);
            delete simsEl.dataset.interval;
        }
        simsEl.innerHTML = `Pending... <span class="hud-unit">—</span>`;
    }
    
    // Render progress track
    const track = document.getElementById('analysis-progress');
    track.innerHTML = stages.map(s => `
        <div class="track-node" id="node-${s}">
            <div class="node-circle"></div>
            <div class="node-label">${s.replace('_', ' ')}</div>
        </div>
    `).join('');
    
    // Collect overrides
    Object.entries(marketPrices.drivers).forEach(([name, d]) => { activeOverrides[name] = { price: d.price }; });
    Object.entries(marketPrices.constructors).forEach(([name, d]) => { activeOverrides[name] = { price: d.price }; });

    // Trigger backend
    let result_json;
    try {
        const res = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                drivers: currentTeam.drivers,
                constructors: currentTeam.constructors,
                budget: currentTeam.budget_remaining || 100.0,
                points: currentTeam.current_points || 0.0,
                transfers: currentTeam.transfers || 1,
                options: {
                    overrides: activeOverrides,
                    grid_overrides: gridOverrides,
                    mode: document.getElementById('mode-selector') ? document.getElementById('mode-selector').value : "auto"
                }
            })
        });
        result_json = await res.json();
    } catch (err) {
        showToast('Failed to start analysis: ' + err.message, "error");
        _resetRunButton();
        _isPipelineRunning = false;
        return;
    }

    if (result_json.status === "error") {
        showToast(result_json.message, "error");
        _resetRunButton();
        _isPipelineRunning = false;
        return;
    }
    
    const run_id = result_json.run_id;
    connectStream(run_id);
}

function _resetRunButton() {
    if (_telemetryAnimFrame) {
        cancelAnimationFrame(_telemetryAnimFrame);
        _telemetryAnimFrame = null;
    }
    _telemetrySpeedMultiplier = 1.0;
    if (_mcCounterTimer) {
        clearInterval(_mcCounterTimer);
        _mcCounterTimer = null;
    }
    const runBtn = document.getElementById('btn-run-analysis');
    if (runBtn) {
        runBtn.disabled = false;
        runBtn.classList.remove('disabled');
        runBtn.innerHTML = '&#9654; Run Analysis';
    }
}

// Poll GET /api/results/{run_id} as a fallback when the SSE stream drops.
// The server sanitizes the payload, so this recovers from transient stream
// errors instead of leaving the UI permanently locked.
function _pollResultsFallback(runId, attempts = 0) {
    const MAX_ATTEMPTS = 120;   // 10 min at 5s intervals
    if (attempts > MAX_ATTEMPTS) {
        showToast('Lost connection to the analysis stream and results are not ready yet.', "error");
        _resetRunButton();
        _isPipelineRunning = false;
        return;
    }
    fetch(`/api/results/${runId}`)
        .then(r => r.json())
        .then(data => {
            if (data.status === "ok" && data.data) {
                showToast('Recovered analysis results after stream interruption.', "info");
                displayResults(data.data);
                _resetRunButton();
                _isPipelineRunning = false;
            } else if (data.status === "not_found") {
                // Result not stored yet — either still running or pruned; keep waiting
                setTimeout(() => _pollResultsFallback(runId, attempts + 1), 5000);
            }
        })
        .catch(() => setTimeout(() => _pollResultsFallback(runId, attempts + 1), 5000));
}

function connectStream(runId) {
    const evtSource = new EventSource(`/api/run/stream/${runId}`);
    const logEl = document.getElementById('analysis-log');
    
    evtSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Update Log
        const line = document.createElement('div');
        const now = new Date();
        const ts = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
        
        // Plan I4: stage duration badge
        if (window._stageTimes && window._stageTimes[data.stage]) {
            const dur = ((Date.now() - window._stageTimes[data.stage]) / 1000).toFixed(1);
            data._dur = dur;
        }
        window._stageTimes[data.stage] = Date.now();
        
        line.className = `log-line ${data.status}`;
        line.innerText = `[${ts} ${data.stage}] ${data.message}`;
        
        // Plan I4: collapsible payload details
        if (data.data && Object.keys(data.data).length > 0) {
            try {
                const details = document.createElement('details');
                details.style.marginLeft = '12px';
                const summary = document.createElement('summary');
                summary.innerText = '▸ payload';
                summary.style.cursor = 'pointer';
                summary.style.fontSize = '10px';
                summary.style.color = 'var(--text-secondary)';
                const pre = document.createElement('pre');
                pre.style.cssText = 'font-size:9px;margin:2px 0;white-space:pre-wrap;color:var(--text-secondary);';
                pre.innerText = JSON.stringify(data.data, null, 1);
                details.appendChild(summary);
                details.appendChild(pre);
                line.appendChild(details);
            } catch (e) { /* non-critical */ }
        }
        
        logEl.appendChild(line);
        logEl.scrollTop = logEl.scrollHeight;
        
        // Plan I4: telemetry panel — merge any arriving payload keys into kv grid
        const telKvEl = document.getElementById('telemetry-kv');
        if (telKvEl && data.data) {
            for (const [k, v] of Object.entries(data.data)) {
                let el = telKvEl.querySelector(`[data-tk="${k}"]`);
                if (!el) {
                    el = document.createElement('div');
                    el.dataset.tk = k;
                    el.style.cssText = 'display:flex;justify-content:space-between;';
                    el.innerHTML = `<span>${k.replace(/_/g,' ')}</span><span style="color:var(--text-primary)" data-tv></span>`;
                    telKvEl.appendChild(el);
                }
                el.querySelector('[data-tv]').innerText =
                    typeof v === 'number' ? v.toFixed(2).replace(/\.00$/,'') : String(v);
            }
        }
        // Stage duration badge
        const stageBadge = telKvEl?.querySelector('[data-tk="stage_duration"]');
        if (stageBadge && data._dur) {
            stageBadge.querySelector('[data-tv]').innerText = `${data._dur}s`;
        }
        
        const statusTextEl = document.getElementById('animation-status-text');
        if (statusTextEl && data.stage) {
            statusTextEl.innerText = data.stage.replace('_', ' ');
        }
        
        const animContainer = document.getElementById('analysis-animation-container');
        if (animContainer && data.stage) {
            if (['ML_MODEL', 'PREDICTIONS', 'ANALYSIS'].includes(data.stage)) {
                animContainer.classList.add('engine-running');
            } else {
                animContainer.classList.remove('engine-running');
            }
        }
        
        // Update Track
        stages.forEach(s => {
            const node = document.getElementById(`node-${s}`);
            if (!node) return;
            if (s === data.stage) {
                if (data.status === 'done') {
                    node.classList.remove('active'); node.classList.add('done');
                } else {
                    node.classList.add('active');
                }
            } else if (stages.indexOf(s) < stages.indexOf(data.stage)) {
                node.classList.remove('active'); node.classList.add('done');
            }
        });
        
        // Update Diagnostics HUD with actual calculations data
        if (data.data) {
            const payload = data.data;
            if (data.stage === 'NEXT_RACE') {
                const roundEl = document.getElementById('hud-round');
                const circuitEl = document.getElementById('hud-circuit');
                const trackEl = document.getElementById('hud-track');
                
                if (roundEl) roundEl.innerHTML = `Round ${payload.round} <span class="hud-unit">${payload.season}</span>`;
                if (circuitEl) circuitEl.innerHTML = `${payload.race_name.replace(' Grand Prix', '')} <span class="hud-unit">${payload.circuit_type}</span>`;
                if (trackEl) trackEl.innerHTML = `${payload.downforce} DF <span class="hud-unit">${payload.overtaking} OVERTAKING</span>`;

                const bannerPost = document.getElementById('mode-banner-post');
                const bannerPre = document.getElementById('mode-banner-pre');
                if (payload.mode === 'post-quali' || payload.mode === 'race-day') {
                    if (bannerPost) bannerPost.style.display = 'block';
                    if (bannerPre) bannerPre.style.display = 'none';
                } else if (payload.mode === 'pre-quali') {
                    if (bannerPost) bannerPost.style.display = 'none';
                    if (bannerPre) bannerPre.style.display = 'block';
                } else {
                    if (bannerPost) bannerPost.style.display = 'none';
                    if (bannerPre) bannerPre.style.display = 'none';
                }
            }
            if (data.stage === 'WEATHER') {
                const weatherEl = document.getElementById('hud-weather');
                if (weatherEl) weatherEl.innerHTML = `${payload.temp}°C <span class="hud-unit">${payload.summary} (${payload.rain_risk} RISK)</span>`;
            }
            if (data.stage === 'ML_MODEL') {
                const ensembleEl = document.getElementById('hud-ensemble');
                if (ensembleEl) {
                    if (payload.training_status) {
                        ensembleEl.innerHTML = `${payload.training_status} <span class="hud-unit">TRAINING ENSEMBLE</span>`;
                    } else if (payload.rf_weight !== undefined) {
                        ensembleEl.innerHTML = `RF:${Math.round(payload.rf_weight*100)}% | XGB:${Math.round(payload.xgb_weight*100)}% | LGB:${Math.round(payload.lgb_weight*100)}% <span class="hud-unit">WEIGHTS</span>`;
                    }
                }
                if (payload.rf_weight !== undefined) {
                    updateEnsembleSegments(payload.rf_weight, payload.xgb_weight, payload.lgb_weight);
                }
            }
            if (data.stage === 'PREDICTIONS') {
                _telemetrySpeedMultiplier = 2.2;
                animateMonteCarloCounter(0, payload.sims || payload.sims_done || 10000);
                const simsEl = document.getElementById('hud-sims');
                if (simsEl) {
                    if (data.status === 'loading' && payload.sims_done) {
                        simsEl.innerHTML = `${payload.sims_done} / ${payload.sims_total} <span class="hud-unit">TOP: ${payload.top_ev}</span>`;
                    } else if (data.status === 'done' && payload.sims) {
                        simsEl.innerHTML = `${payload.sims.toLocaleString()} runs <span class="hud-unit">SC RISK: ${payload.sc_prob}%</span>`;
                    }
                }
            }
        } else if (data.stage === 'PREDICTIONS' && data.status === 'loading') {
            // General predicted pole/winner or status messages during predictions phase
            _telemetrySpeedMultiplier = 2.2;
            animateMonteCarloCounter(0, 10000);
            const simsEl = document.getElementById('hud-sims');
            if (simsEl) {
                simsEl.innerHTML = `${data.message} <span class="hud-unit">PREDICTION INTERMEDIATE</span>`;
            }
        }
        
        if (data.stage === 'COMPLETE' && data.data) {
            evtSource.close();
            if (_telemetryAnimFrame) {
                cancelAnimationFrame(_telemetryAnimFrame);
                _telemetryAnimFrame = null;
            }
            _telemetrySpeedMultiplier = 1.0;
            if (_mcCounterTimer) {
                clearInterval(_mcCounterTimer);
                _mcCounterTimer = null;
            }
            const anim = document.getElementById('analysis-animation-container');
            if (anim) anim.style.display = 'none';
            lastResults = data.data;
            _currentRaceInfo = data.data.race || null;
            setTimeout(() => displayResults(data.data), 800);
            _isPipelineRunning = false;
            const runBtn = document.getElementById('btn-run-analysis');
            if (runBtn) {
                runBtn.disabled = false;
                runBtn.classList.remove('disabled');
                runBtn.innerHTML = '&#9654; Run Analysis';
            }
        } else if (data.stage === 'ERROR') {
            evtSource.close();
            if (_telemetryAnimFrame) {
                cancelAnimationFrame(_telemetryAnimFrame);
                _telemetryAnimFrame = null;
            }
            _telemetrySpeedMultiplier = 1.0;
            if (_mcCounterTimer) {
                clearInterval(_mcCounterTimer);
                _mcCounterTimer = null;
            }
            const anim = document.getElementById('analysis-animation-container');
            if (anim) anim.style.display = 'none';
            showToast('Analysis failed: ' + data.message, 'error');
            _isPipelineRunning = false;
            const runBtn = document.getElementById('btn-run-analysis');
            if (runBtn) {
                runBtn.disabled = false;
                runBtn.classList.remove('disabled');
                runBtn.innerHTML = '&#9654; Run Analysis';
            }
        }
    };
    
    evtSource.onerror = () => {
        if (_telemetryAnimFrame) {
            cancelAnimationFrame(_telemetryAnimFrame);
            _telemetryAnimFrame = null;
        }
        _telemetrySpeedMultiplier = 1.0;
        const simsEl = document.getElementById('hud-sims');
        if (simsEl && simsEl.dataset.interval) {
            clearInterval(simsEl.dataset.interval);
            delete simsEl.dataset.interval;
        }
        evtSource.close();
        // Don't leave the UI locked — fall back to polling for the result.
        // (Laptop sleep / proxy timeout can drop SSE while the pipeline runs.)
        showToast('Stream interrupted — switching to polling…', "warning");
        _pollResultsFallback(runId);
    };
}

// ---- RESULTS (SCREEN 6) ----
function displayResults(data) {
    // Show results nav item and add NEW badge
    document.getElementById('nav-results').style.display = 'inline-block';
    clearNavBadge('nav-analysis');
    setNavBadge('nav-results', '● NEW', 'new');

    // Auto-navigate only if user is currently watching the analysis screen.
    // Otherwise notify via toast and let them come to results at their own pace.
    const _activeNow = document.querySelector('.screen.active');
    const _isOnAnalysis = _activeNow && _activeNow.id === 'analysis-screen';
    if (_isOnAnalysis) {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('nav-results').classList.add('active');
        showScreen('results-screen');
    } else {
        showToast('✅ Analysis complete! Click Results in the sidebar to view your predictions.', 'success');
    }
    
    const sugg = data.suggestions || {};
    const opt = data.optimal || { drivers: [], constructors: [], total_price: 0, total_pts: 0 };
    const chips = data.chips || [];

    // ── Score ──
    const basePts = currentTeam.current_points || 0.0;
    const projGain = sugg.projected_pts_new || 0.0;
    document.getElementById('res-score').innerText = `${projGain.toFixed(1)}`;

    // ── Transfers ──
    const tList = document.getElementById('res-transfers');
    tList.innerHTML = '';
    // Backend uses key "in" not "in_"
    const transfers = sugg.suggested_changes || [];
    const usingMC = sugg.using_mc_pts ? '🎲 Monte Carlo EV' : '📊 Deterministic';
    const hasPractice = data.practice_data && data.practice_data.session && data.practice_data.session !== 'N/A';
    const basisNote = `<div style="font-size:0.78rem; color: var(--text-dim); margin-top: 10px; padding: 8px 12px; background: rgba(255,255,255,0.03); border-radius: 0; border-left: 2px solid rgba(255,255,255,0.1);">
        <strong style="color: var(--text-main);">Basis:</strong> ${usingMC} scoring · EWMA 5-race form · Glicko-2 ratings · Circuit history
        ${hasPractice ? ` · <span style="color:var(--color-green)">${data.practice_data.session} pace</span>` : ' · <span style="color: #f0c420">⚠ No practice data yet (pre-race)</span>'}
        <br><span style="font-size:0.73rem; opacity:0.65;">These are model projections, not guarantees. Always apply your own judgement.</span>
    </div>`;

    if (sugg.locked) {
        tList.innerHTML = `<div class="res-card hold-card" style="border-left: 4px solid var(--f1-red);">
            <div class="res-card-icon" style="color: var(--f1-red);">🔒</div>
            <div>
                <div style="font-weight:700; margin-bottom:4px;">Transfers Locked</div>
                <div class="dim-text">Qualifying has happened. Transfer recommendations are not available post-qualifying.</div>
            </div>
        </div>${basisNote}`;
    } else if (transfers.length === 0) {
        tList.innerHTML = `<div class="res-card hold-card">
            <div class="res-card-icon">✓</div>
            <div>
                <div style="font-weight:700; margin-bottom:4px;">Hold current team</div>
                <div class="dim-text">No profitable moves found within budget constraints.</div>
            </div>
        </div>${basisNote}`;
    } else {
        transfers.forEach((t, i) => {
            const gain = t.pts_gain || 0;
            tList.innerHTML += `
                <div class="res-card transfer-card">
                    <div class="transfer-badge">#${i+1}</div>
                    <div class="transfer-body">
                        <div class="transfer-players">
                            <span class="player-out">▼ ${t.out}</span>
                            <span class="transfer-arrow">→</span>
                            <span class="player-in">▲ ${t.in}</span>
                        </div>
                        <div class="transfer-meta dim-text">
                            Budget impact: ${t.cost_diff > 0 ? '+' : ''}$${(t.cost_diff || 0).toFixed(1)}M &nbsp;·&nbsp;
                            <span style="color:var(--color-green)">+${gain.toFixed(1)} pts projected gain</span>
                        </div>
                    </div>
                    <button class="btn btn-primary btn-sm" style="font-size:0.8rem; padding:4px 12px;" onclick="showToast('Auto-transfer feature coming soon!', 'info')">QUICK APPLY</button>
                </div>
            `;
        });
        tList.innerHTML += basisNote;
    }

    // ── Race & Sprint Projections View ──
    const preds = data.predictions || {};
    const normalView = document.getElementById('normal-race-view');
    const sprintView = document.getElementById('sprint-race-view');
    const sprintBanner = document.getElementById('sprint-weekend-banner');

    if (data.is_sprint && preds.sprint_order && preds.sprint_order.length > 0) {
        // Toggle views
        if (normalView) normalView.style.display = 'none';
        if (sprintView) sprintView.style.display = 'block';
        if (sprintBanner) {
            sprintBanner.style.display = 'block';
            renderWeekendTimeline(data.race, 'sprint-weekend-timeline');
        }
        
        // Render sprint layout using compact cards
        renderSprintWeekendView(preds.sprint_order, 'res-sprint-cards', true);
        renderSprintWeekendView(preds.race_order, 'res-race-cards', false);
    } else {
        // Toggle views back to normal
        if (normalView) normalView.style.display = 'block';
        if (sprintView) sprintView.style.display = 'none';
        if (sprintBanner) sprintBanner.style.display = 'none';
        
        // Render standard table split into exactly half left and half right
        const raceOrder = [...(preds.race_order || [])].sort((a, b) => (a.predicted_rank || 99) - (b.predicted_rank || 99));
        const half = Math.ceil(raceOrder.length / 2);
        const leftSide = raceOrder.slice(0, half);
        const rightSide = raceOrder.slice(half);
        renderRaceOrder(leftSide, 'res-race-order-left');
        renderRaceOrder(rightSide, 'res-race-order-right');
    }

    // ── Qualifying Projections View ──
    const qualiOrder = [...(preds.quali_order || [])].sort((a, b) => (a.predicted_grid || 99) - (b.predicted_grid || 99));
    const qualiHalf = Math.ceil(qualiOrder.length / 2);
    renderQualiOrder(qualiOrder.slice(0, qualiHalf), 'res-quali-order-left');
    renderQualiOrder(qualiOrder.slice(qualiHalf), 'res-quali-order-right');

    // ── Practice Data ──
    renderPracticeData(data.practice_data || {});
    
    // ── Radar Comparison ──
    renderDriverRadar(data.predictions || {});
    
    // ── Comparison Tool Setup ──
    setupComparisonTool(data.predictions || {});

    // ── Chip Strategy ──
    renderChipCards(chips);
    
    // ── AI Insights Panel ──
    renderInsightsPanel(data);

    // ── Dream Team ──
    document.getElementById('dt-budget').innerText = opt.total_price ? opt.total_price.toFixed(1) : '0.0';
    document.getElementById('dt-pts').innerText = opt.total_pts ? opt.total_pts.toFixed(1) : '0.0';
    const dtGrid = document.getElementById('res-dream-team');
    dtGrid.innerHTML = '';

    const allDtPlayers = [
        ...(opt.drivers || []).map((d, i) => ({ ...d, role: 'driver', rank: i + 1 })),
        ...(opt.constructors || []).map(c => ({ ...c, role: 'constructor' }))
    ];

    allDtPlayers.forEach(p => {
        const isDriver = p.role === 'driver';
        const rankBadge = isDriver ? `<div class="dt-rank">#${p.rank}</div>` : `<div class="dt-rank ctor-badge">CTOR</div>`;
        const imgUrl = isDriver ? driverHeadshots[p.name] : constructorLogos[p.name];
        const thumbHtml = imgUrl ? `<img class="dt-thumb" src="${imgUrl}" alt="" onerror="this.style.display='none'">` : '';
        const explText = isDriver && p.explanation ? `<div class="dt-explanation dim-text">${p.explanation}</div>` : '';
        dtGrid.innerHTML += `
            <div class="dt-card ${isDriver ? '' : 'dt-ctor'}">
                ${rankBadge}
                ${thumbHtml}
                <div class="dt-name">${p.name}</div>
                <div class="dt-stats">
                    <span class="dt-pts-val">${(p.pts||0).toFixed(1)} pts</span>
                    <span class="dt-price-val">$${(p.price||0).toFixed ? (p.price||0).toFixed(1) : (p.price||0)}M</span>
                </div>
                ${explText}
            </div>
        `;
    });

    // ── Update Quick Summary on Dashboard ──
    updateQuickSummary(data);
    showToast('Analysis complete! Results are ready.', 'success');
}

function renderChipCards(chips) {
    const container = document.getElementById('res-chips');
    if (!chips || chips.length === 0) {
        container.innerHTML = `<div class="dim-text" style="padding:16px;">No chip data available.</div>`;
        return;
    }
    container.innerHTML = '';
    const raceInfo = _currentRaceInfo || {};
    chips.forEach(cs => {
        let statusClass = 'chip-hold', statusLabel = '⏳ HOLD';
        if (cs.already_used) {
            statusClass = 'chip-used';
            statusLabel = `✓ USED${cs.used_on ? ' · ' + cs.used_on : ''}`;
        } else if (cs.use_now) {
            statusClass = 'chip-play';
            statusLabel = '🔥 PLAY NOW';
        }
        const urgencyDot = cs.urgency === 'high' ? '🔴' : cs.urgency === 'medium' ? '🟡' : '⚪';
        const evText = cs.expected_gain > 0 ? `+${cs.expected_gain.toFixed(0)} pts EV` : '';
        const holdText = cs.hold_until ? `Hold until: <span class="chip-hold-target">${cs.hold_until}</span>` : '';
        const scoreBar = Array.from({length:10},(_,i)=>`<span class="score-pip ${i<Math.round(cs.raw_score)?'active':''}"></span>`).join('');
        const markBtn = (!cs.already_used)
            ? `<button class="btn-chip-mark" onclick="markChipUsed('${cs.chip}', ${raceInfo.round||0}, '${raceInfo.name||'Unknown'}')" title="Mark as used this race">✓ I Used This</button>`
            : '';
        container.innerHTML += `
            <div class="chip-card ${statusClass}">
                <div class="chip-header">
                    <div class="chip-name">${cs.chip}</div>
                    <div class="chip-status-badge">${statusLabel}</div>
                </div>
                <div class="chip-score-bar">${scoreBar}</div>
                <div class="chip-reason">${urgencyDot} ${cs.reason}</div>
                ${holdText ? `<div class="chip-hold-text dim-text">${holdText}</div>` : ''}
                ${evText ? `<div class="chip-ev">${evText}</div>` : ''}
                ${markBtn}
            </div>
        `;
    });
}

function renderRaceOrder(order, containerId, isSprint = false) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (!order || order.length === 0) {
        el.innerHTML = '<div class="dim-text" style="padding:12px;">No prediction data available.</div>';
        return;
    }

    const rankKey  = isSprint ? 'predicted_sprint_rank' : 'predicted_rank';
    const posKey   = isSprint ? 'predicted_pos'         : 'predicted_pos';

    // Sort by rank
    const sorted = [...order].sort((a, b) => (a[rankKey] || a.predicted_rank || 99) - (b[rankKey] || b.predicted_rank || 99));

    el.innerHTML = `
        <div class="race-order-header">
            <span class="ro-col-pos">POS</span>
            <span class="ro-col-driver">DRIVER</span>
            <span class="ro-col-team">TEAM</span>
            <span class="ro-col-telemetry">STATUS</span>
            <span class="ro-col-pred">PRED</span>
            <span class="ro-col-conf">CONFIDENCE</span>
        </div>
    ` + sorted.map(d => {
        const rank       = d[rankKey] || d.predicted_rank || '?';
        const name       = d.driver || d.name || '?';
        const team       = d.team || '';
        const pred       = (d[posKey] || d.predicted_pos || 0).toFixed(1);
        const conf       = d.confidence_pct || 0;
        const confW      = Math.round(conf);
        const confColor  = conf >= 70 ? 'var(--color-green)' : conf >= 45 ? 'var(--color-yellow)' : 'var(--accent-danger)';
        const teamColor  = teamColors[team] || '#FF003C';
        const imgUrl     = driverHeadshots[name];
        const thumbHtml  = imgUrl ? `<img class="ro-thumb" src="${imgUrl}" alt="" onerror="this.style.visibility='hidden'">` : '';
        const isRookie   = d.is_rookie ? '<span class="telemetry-mini-pill pill-rookie" title="Rookie Driver">ROOKIE</span>' : '';
        const dnf        = d.dnf_prob_pct || 0;
        const dnfHtml    = dnf >= 15 ? `<span class="telemetry-mini-pill pill-dnf" title="DNF Risk: ${dnf.toFixed(0)}%">⚠ ${dnf.toFixed(0)}%</span>` : '';
        const momentum   = d.momentum || 0;
        const mClass     = momentum > 0.5 ? 'pill-momentum-up' : momentum < -0.5 ? 'pill-momentum-down' : 'pill-momentum-flat';
        const mArrow     = momentum > 0.5 ? '▲' : momentum < -0.5 ? '▼' : '–';
        const momentumHtml = `<span class="telemetry-mini-pill ${mClass}" title="Momentum: ${momentum > 0 ? '+' : ''}${momentum.toFixed(1)}">${mArrow}</span>`;
        const posClass   = rank <= 3 ? 'ro-pos-top3' : rank <= 10 ? 'ro-pos-points' : 'ro-pos-out';

        return `<div class="race-order-row">
            <span class="ro-col-pos ${posClass}">${rank}</span>
            <span class="ro-col-driver">
                ${thumbHtml}
                <span class="ro-name" title="${name}">${name}</span>
            </span>
            <span class="ro-col-team" style="color:${teamColor}" title="${team}">${team}</span>
            <span class="ro-col-telemetry">
                ${isRookie}
                ${dnfHtml}
                ${momentumHtml}
            </span>
            <span class="ro-col-pred">P${pred}</span>
            <span class="ro-col-conf">
                <div class="ro-conf-bar">
                    <div class="ro-conf-fill" style="width:${confW}%;background:${confColor}"></div>
                </div>
                <span class="ro-conf-pct" style="color:${confColor}">${conf.toFixed(0)}%</span>
            </span>
        </div>`;
    }).join('');
}

function renderQualiOrder(order, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    el.innerHTML = `
        <div class="race-order-header">
            <span class="ro-col-pos">GRID</span>
            <span class="ro-col-driver">DRIVER</span>
            <span class="ro-col-team">TEAM</span>
            <span class="ro-col-pred">Q-TIME</span>
            <span class="ro-col-conf">STATUS</span>
        </div>
    ` + order.map(d => {
        const grid       = d.predicted_grid || '?';
        const name       = d.driver || d.name || '?';
        const team       = d.team || '';
        const teamColor  = teamColors[team] || '#FF003C';
        const isActual   = d.is_actual;
        
        let timeStr = "N/A";
        if (isActual) {
            timeStr = d.q3_time || d.q2_time || d.q1_time || 'No Time';
        } else {
            timeStr = `P${(d.predicted_pos || 0).toFixed(1)}`;
        }
        
        const statusHtml = isActual ? 
            `<span class="legend-badge" style="background: rgba(0,208,132,0.15); color: var(--color-green); border: 1px solid var(--color-green);">LOCKED</span>` : 
            `<span class="legend-badge" style="background: rgba(255,255,255,0.1); color: #ccc; border: 1px solid rgba(255,255,255,0.2);">PREDICTED</span>`;
        
        const penaltyHtml = d.grid_penalty ? `<span style="color:var(--color-yellow); font-size:0.7rem; margin-left:4px;" title="Grid Penalty">(-${d.grid_penalty})</span>` : '';

        return `<div class="race-order-row" draggable="true" ondragstart="dragQuali(event, '${name}')">
            <span class="ro-col-pos" style="cursor: grab;">☰ ${grid}</span>
            <span class="ro-col-driver">
                <span class="ro-name">${name}</span> ${penaltyHtml}
            </span>
            <span class="ro-col-team" style="color:${teamColor}">${team}</span>
            <span class="ro-col-pred" style="font-family: monospace;">${timeStr}</span>
            <span class="ro-col-conf">${statusHtml}</span>
        </div>`;
    }).join('');
}

let draggedQualiDriver = null;

function dragQuali(ev, driverName) {
    draggedQualiDriver = driverName;
}

function allowDrop(ev) {
    ev.preventDefault();
}

function dropQuali(ev) {
    ev.preventDefault();
    if (!draggedQualiDriver) return;
    
    let target = ev.target;
    while (target && !target.classList.contains('race-order-row')) {
        target = target.parentElement;
    }
    
    if (target) {
        const targetDriver = target.querySelector('.ro-name').innerText;
        if (targetDriver && targetDriver !== draggedQualiDriver) {
            const targetPos = parseInt(target.querySelector('.ro-col-pos').innerText.replace(/[^0-9]/g, ''));
            gridOverrides[draggedQualiDriver] = targetPos;
            
            document.getElementById('quali-order-status').innerHTML = `
                <span style="color: var(--color-yellow); background: rgba(255,255,0,0.1); padding: 5px 10px; border-radius: 0; display: inline-block;">
                    Override saved: <strong>${draggedQualiDriver}</strong> to P${targetPos}. 
                    <a href="#" onclick="startAnalysis()" style="color:white;text-decoration:underline;margin-left:10px;">▶ Re-run Analysis to apply</a>
                    <a href="#" onclick="clearQualiOverrides()" style="color:red;text-decoration:underline;margin-left:10px;">✖ Clear</a>
                </span>
            `;
        }
    }
    draggedQualiDriver = null;
}

function clearQualiOverrides() {
    gridOverrides = {};
    document.getElementById('quali-order-status').innerHTML = '';
}


function renderWeekendTimeline(race, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (!race) {
        el.innerHTML = '';
        return;
    }
    
    // We have: FP1, Sprint Quali (fp2), Sprint, Quali, Race
    const sessions = [
        { label: 'FP1', date: race.fp1_date, time: race.fp1_time, class: 'completed' },
        { label: 'Sprint Quali', date: race.fp2_date, time: race.fp2_time, class: 'completed' },
        { label: 'Sprint Race', date: race.sprint_date, time: race.sprint_time, class: 'sprint-active active' },
        { label: 'Qualifying', date: race.quali_date, time: race.quali_time, class: 'active' },
        { label: 'Grand Prix', date: race.date, time: race.time, class: 'race-active' }
    ];
    
    // Determine active steps based on current local time
    const now = new Date();
    sessions.forEach(s => {
        if (!s.date) return;
        const sTime = s.time ? s.time.replace('Z', '') : '12:00:00';
        const sDateTime = new Date(`${s.date}T${sTime}`);
        if (now > sDateTime) {
            s.status = 'completed';
        } else {
            s.status = s.label.includes('Sprint') ? 'sprint-active' : s.label.includes('Quali') ? 'active' : 'race-active';
        }
    });

    el.innerHTML = sessions.map(s => {
        if (!s.date) return '';
        // Format time to HH:MM (strip seconds and Z)
        let formattedTime = 'N/A';
        if (s.time) {
            const parts = s.time.split(':');
            if (parts.length >= 2) {
                formattedTime = `${parts[0]}:${parts[1]}`;
            }
        }
        
        // Format date to "MMM DD" (e.g. "Jun 06")
        const dateParts = s.date.split('-');
        let formattedDate = s.date;
        if (dateParts.length === 3) {
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            const mIdx = parseInt(dateParts[1], 10) - 1;
            if (mIdx >= 0 && mIdx < 12) {
                formattedDate = `${months[mIdx]} ${dateParts[2]}`;
            }
        }

        const activeClass = s.status === 'completed' ? 'completed' : s.class;

        return `
            <div class="timeline-step ${activeClass}">
                <div class="timeline-dot"></div>
                <div class="timeline-label">${s.label}</div>
                <div class="timeline-time">${formattedDate} &middot; ${formattedTime}</div>
            </div>
        `;
    }).join('');
}


function renderSprintWeekendView(order, containerId, isSprint = false) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (!order || order.length === 0) {
        el.innerHTML = '<div class="dim-text" style="padding:12px;">No prediction data available.</div>';
        return;
    }

    const rankKey  = isSprint ? 'predicted_sprint_rank' : 'predicted_rank';
    const posKey   = isSprint ? 'predicted_pos'         : 'predicted_pos';

    // Sort by rank
    const sorted = [...order].sort((a, b) => (a[rankKey] || a.predicted_rank || 99) - (b[rankKey] || b.predicted_rank || 99));

    el.innerHTML = sorted.map(d => {
        const rank       = d[rankKey] || d.predicted_rank || '?';
        const name       = d.driver || d.name || '?';
        const team       = d.team || '';
        const pred       = (d[posKey] || d.predicted_pos || 0).toFixed(1);
        const conf       = d.confidence_pct || 0;
        const confW      = Math.round(conf);
        const confColor  = conf >= 70 ? 'var(--color-green)' : conf >= 45 ? 'var(--color-yellow)' : 'var(--accent-danger)';
        const teamColor  = teamColors[team] || '#FF003C';
        const imgUrl     = driverHeadshots[name];
        
        // Driver thumbnail
        const thumbHtml = imgUrl 
            ? `<img class="cc-thumb" src="${imgUrl}" alt="${name}" onerror="this.style.visibility='hidden'">` 
            : `<div class="price-thumb-fallback" style="color: ${teamColor}; border-radius: 50%; width: 32px; height: 32px; font-size: 0.65rem; border-color: ${teamColor}; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-weight: 700;">${name.split(' ').map(n=>n[0]).join('')}</div>`;

        // Badges: Rookie badge
        const isRookie   = d.is_rookie ? '<span class="cc-rookie-badge">ROOKIE</span>' : '';
        
        // DNF Risk
        const dnf        = d.dnf_prob_pct || 0;
        const dnfHtml    = dnf >= 15 ? `<span class="cc-dnf-risk" title="DNF risk: ${dnf.toFixed(0)}%">⚠ ${dnf.toFixed(0)}%</span>` : '';
        
        // Momentum
        const momentum   = d.momentum || 0;
        const mArrow     = momentum > 0.5 ? '▲' : momentum < -0.5 ? '▼' : '–';
        const mColor     = momentum > 0.5 ? 'var(--color-green)' : momentum < -0.5 ? 'var(--accent-danger)' : 'var(--text-dim)';
        const momentumHtml = `<span class="cc-momentum" style="color:${mColor}" title="Form Momentum">${mArrow}</span>`;

        return `
            <div class="driver-compact-card pos-${rank}" style="--team-color: ${teamColor};">
                <div class="cc-pos">${rank}</div>
                <div class="cc-thumb-wrap">
                    ${thumbHtml}
                </div>
                <div class="cc-info-block">
                    <div class="cc-name-row">
                        <span class="cc-name">${name}</span>
                        <div class="cc-badges">
                            ${isRookie}
                            ${dnfHtml}
                            ${momentumHtml}
                        </div>
                    </div>
                    <div class="cc-team" style="color: ${teamColor};">${team}</div>
                </div>
                <div class="cc-stats-block">
                    <div class="cc-pred-val">P${pred}</div>
                    <div class="cc-conf-bar-wrap">
                        <div class="cc-conf-bar-fill" style="width:${confW}%; background:${confColor}"></div>
                    </div>
                    <div class="cc-conf-text" style="color:${confColor}">${conf.toFixed(0)}% CONF</div>
                </div>
            </div>
        `;
    }).join('');
}


function renderPracticeData(practiceData) {
    const panel = document.getElementById('res-practice-panel');
    const labelEl = document.getElementById('res-practice-label');
    const listEl  = document.getElementById('res-practice-deltas');
    if (!panel || !labelEl || !listEl) return;

    const session = practiceData.session || 'N/A';
    const deltas  = practiceData.deltas || {};
    const entries = Object.entries(deltas);

    if (session === 'N/A' || entries.length === 0) {
        panel.style.display = '';
        labelEl.innerHTML = '';
        listEl.innerHTML = `
            <div style="text-align:center; padding: 40px 20px; color: var(--text-dim);">
                <div style="font-size: 2.5rem; opacity: 0.2; margin-bottom: 14px;">🏎</div>
                <div style="font-weight: 600; font-size: 1rem; margin-bottom: 8px; color: var(--text-main);">No Practice Session Data Available</div>
                <div style="font-size: 0.85rem; line-height: 1.6; max-width: 400px; margin: 0 auto;">
                    Telemetry from FP1/FP2/FP3 will appear here once practice sessions for this race weekend have taken place.
                    <br><br>
                    The model is currently using <span style="color: var(--color-green);">historical form, Glicko-2 ratings, and EWMA momentum trends</span> for predictions.
                </div>
            </div>
        `;
        if (_charts && _charts.practice) {
            _charts.practice.destroy();
            _charts.practice = null;
        }
        const chartContainer = panel.querySelector('.chart-container');
        if (chartContainer) chartContainer.style.display = 'none';
        return;
    }

    panel.style.display = '';
    const chartContainer = panel.querySelector('.chart-container');
    if (chartContainer) chartContainer.style.display = '';
    labelEl.innerHTML = `
        <strong>${session} long-run race pace</strong> — calculated average of 3+ consecutive laps (not the fastest single lap).
        <br><span style="color:var(--color-green)"> Faster drivers listed first</span> (negative delta = faster than field average).
    `;

    // Sort fastest first (most negative delta first)
    const sorted = entries.sort((a, b) => a[1] - b[1]);
    const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.001);

    listEl.innerHTML = sorted.map(([name, delta], idx) => {
        const isFaster  = delta <= 0;
        const color     = isFaster ? 'var(--color-green)' : 'var(--accent-danger)';
        const sign      = isFaster ? '' : '+';
        const barW      = Math.min(100, (Math.abs(delta) / maxAbs) * 100);
        const team      = driverTeamMap[name] || '';
        const teamColor = teamColors[team] || '#888';
        const imgUrl    = driverHeadshots[name];
        const thumb     = imgUrl ? `<img class="ro-thumb" src="${imgUrl}" alt="" onerror="this.style.visibility='hidden'">` : '<div class="ro-thumb"></div>';
        const pos       = idx + 1;
        return `<div class="practice-row">
            ${thumb}
            <span class="practice-pos">${pos}</span>
            <div class="practice-name-block">
                <span class="practice-name">${name}</span>
                <span class="practice-team" style="color:${teamColor}">${team}</span>
            </div>
            <div class="practice-bar-wrap">
                <div class="practice-bar ${isFaster ? 'bar-fast' : 'bar-slow'}" style="width:${barW}%"></div>
            </div>
            <span class="practice-delta" style="color:${color}">${sign}${delta.toFixed(3)}s</span>
        </div>`;
    }).join('');

    renderPracticeChart(sorted);
}

function updateQuickSummary(data) {
    const panel = document.querySelector('#dashboard-screen .summary-content');
    if (!panel) return;

    const sugg = data.suggestions || {};
    const opt = data.optimal || {};
    const chips = data.chips || [];
    const race = data.race || {};

    const projPts = sugg.projected_pts_new || 0;
    const transfers = sugg.suggested_changes || [];
    const bestMove = transfers.length > 0 ? transfers[0] : null;

    // Best chip to play now
    const playNowChip = chips.find(c => c.use_now && !c.already_used);
    const chipText = playNowChip
        ? `<div class="summary-alert play-now">🔥 Play <strong>${playNowChip.chip}</strong> this week</div>`
        : `<div class="summary-alert hold">✓ Hold all chips this race</div>`;

    const dreamDrivers = (opt.drivers || []).slice(0, 3).map(d => d.name).join(', ');

    panel.classList.remove('empty-state');
    panel.innerHTML = `
        <div class="summary-stat">
            <div class="summary-stat-label">Projected Points</div>
            <div class="summary-stat-value highlight-text">${projPts.toFixed(1)}</div>
        </div>
        <div class="summary-stat">
            <div class="summary-stat-label">Race</div>
            <div class="summary-stat-value" style="font-size:0.9rem;">${race.name || '—'}</div>
        </div>
        ${bestMove ? `
        <div class="summary-stat">
            <div class="summary-stat-label">Top Transfer</div>
            <div class="summary-stat-value" style="font-size:0.85rem;">
                <span style="color:var(--accent-danger)">▼ ${bestMove.out}</span> → <span style="color:#00FF41">▲ ${bestMove.in}</span>
            </div>
        </div>` : `
        <div class="summary-stat">
            <div class="summary-stat-label">Transfers</div>
            <div class="summary-stat-value dim-text" style="font-size:0.85rem;">Hold team</div>
        </div>`}
        <div class="summary-stat">
            <div class="summary-stat-label">Top Dream Pick</div>
            <div class="summary-stat-value" style="font-size:0.85rem;">${dreamDrivers || '—'}</div>
        </div>
        ${chipText}
        <button class="btn btn-primary" style="width:100%; margin-top:12px;" onclick="
            document.getElementById('nav-results').click()
        ">View Full Results →</button>
    `;
}

// ---- TOAST NOTIFICATIONS ----
function showToast(message, type = 'info') {
    const existing = document.getElementById('toast-container');
    if (!existing) {
        const tc = document.createElement('div');
        tc.id = 'toast-container';
        tc.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px;';
        document.body.appendChild(tc);
    }
    const toast = document.createElement('div');
    const colors = { success: '#00FF41', error: '#ff4444', info: '#748ffc' };
    toast.style.cssText = `
        background: #1a1a2e; border-left: 3px solid ${colors[type] || colors.info};
        color: #e8e8f0; padding: 12px 18px; border-radius: 0;
        font-size: 0.85rem; font-family: inherit;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        animation: slideIn 0.3s ease; max-width: 320px;
    `;
    toast.innerText = message;
    document.getElementById('toast-container').appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

function setupHeartbeat() {
    if (setupHeartbeat._installed) return;   // idempotent
    setupHeartbeat._installed = true;
    setInterval(() => {
        fetch('/api/heartbeat', { method: 'POST' }).catch(() => {});
    }, 5000);
}

// ---- CHIP STATE ----
async function loadChipState() {
    try {
        const res = await fetch('/api/chips');
        chipState = await res.json();
        renderSettingsChips();
    } catch(e) {}
}

async function markChipUsed(chipName, roundNum, circuitName) {
    if (!confirm(`Mark "${chipName}" as used at ${circuitName} (R${roundNum})?`)) return;
    await fetch('/api/chips/mark', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ chip_name: chipName, round_used: roundNum, circuit_name: circuitName })
    });
    await loadChipState();
    // Re-render chips in results if visible
    if (lastResults && lastResults.chips) {
        const res = await fetch('/api/chips');
        chipState = await res.json();
        // Sync already_used flag into lastResults.chips
        lastResults.chips.forEach(c => {
            c.already_used = chipState.used[c.chip] || false;
            c.used_on = chipState.used_on_circuit[c.chip] || null;
        });
        renderChipCards(lastResults.chips);
    }
    showToast(`${chipName} marked as used!`, 'success');
}

async function resetChips() {
    if (!confirm('Reset all chips? This marks them all as unused (use at season start).')) return;
    await fetch('/api/chips/reset', { method: 'POST' });
    await loadChipState();
    showToast('All chips reset.', 'success');
    document.getElementById('settings-chip-status').innerText = '✓ All chips reset.';
}

function renderSettingsChips() {
    const el = document.getElementById('settings-chip-list');
    if (!el) return;
    const chips = ['Limitless', 'No Negative', '3x Boost', 'Wildcard', 'Final Fix'];
    el.innerHTML = chips.map(chip => {
        const used = chipState.used && chipState.used[chip];
        const usedOn = chipState.used_on_circuit && chipState.used_on_circuit[chip];
        const rnd = chipState.used_on_round && chipState.used_on_round[chip];
        const statusTxt = used ? `✓ Used — ${usedOn || '?'} (R${rnd || '?'})` : '⬜ Available';
        const badgeCls = used ? 'chip-badge-used' : 'chip-badge-avail';
        return `<div class="settings-chip-row">
            <span class="settings-chip-name">${chip}</span>
            <span class="chip-badge ${badgeCls}">${statusTxt}</span>
            ${!used ? `<button class="btn-chip-mark" onclick="markChipUsedManual('${chip}')">Mark Used</button>` : ''}
        </div>`;
    }).join('');
}

async function markChipUsedManual(chipName) {
    const rnd = prompt(`Round number when you used ${chipName}?`);
    if (!rnd || isNaN(rnd)) return;
    const circuit = prompt('Circuit name?') || 'Unknown';
    await markChipUsed(chipName, parseInt(rnd), circuit);
    renderSettingsChips();
}

// ---- SETTINGS ----
async function clearCache(type) {
    const labels = {
        api: 'API data', models: 'ML models', weather: 'weather',
        fastf1: 'FastF1 telemetry', elo: 'Elo ratings', predictions: 'past predictions', all: 'ALL caches'
    };
    if (!confirm(`Clear ${labels[type] || type} cache? This cannot be undone.`)) return;
    const res = await fetch(`/api/cache/${type}`, { method: 'DELETE' });
    const json = await res.json();
    const statusEl = document.getElementById('settings-cache-status');
    if (json.status === 'ok') {
        const msg = `✓ Cleared: ${Array.isArray(json.cleared) ? json.cleared.join(', ') : json.cleared}`;
        if (statusEl) statusEl.innerText = msg;
        showToast(msg, 'success');
    } else {
        showToast('Error: ' + json.message, 'error');
    }
}

async function resetMemory() {
    if (!confirm('WARNING: This will wipe your saved team, chip history, and prices. Continue?')) return;
    const res = await fetch('/api/memory/reset', { method: 'POST' });
    const json = await res.json();
    if (json.status === 'ok') {
        showToast('Memory wiped. Restarting...', 'info');
        setTimeout(() => location.reload(), 1500);
    } else {
        showToast('Error resetting memory', 'error');
    }
}

// ── NEW PHASE 2 FUNCTIONS ─────────────────────────────────────

function renderPracticeChart(sortedData) {
    const ctx = document.getElementById('chart-practice-pace');
    if (!ctx) return;
    
    if (_charts.practice) _charts.practice.destroy();
    
    const labels = sortedData.map(d => d[0].split(' ').pop());
    const deltas = sortedData.map(d => d[1]);
    const colors = deltas.map(v => v <= 0 ? 'rgba(0, 210, 190, 0.6)' : 'rgba(232, 0, 45, 0.6)');
    
    _charts.practice = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Lap Time Delta (s)',
                data: deltas,
                backgroundColor: colors,
                borderWidth: 0,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Delta: ${ctx.raw > 0 ? '+' : ''}${ctx.raw.toFixed(3)}s`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(28, 32, 45, 0.5)' },
                    ticks: { color: '#5c647b', font: { family: 'JetBrains Mono' } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f0f2f5', font: { family: 'JetBrains Mono', size: 12 } }
                }
            }
        }
    });
}

function renderDriverRadar(preds) {
    const ctx = document.getElementById('chart-driver-radar');
    if (!ctx || !preds.race_order || preds.race_order.length === 0) return;

    if (_charts.radar) _charts.radar.destroy();

    const topDrivers = [...preds.race_order]
        .sort((a,b) => (a.predicted_rank || 99) - (b.predicted_rank || 99))
        .slice(0, 3);

    const datasets = topDrivers.map((d, i) => {
        const team = d.team || '';
        const color = teamColors[team] || '#FF003C';
        return {
            label: d.driver,
            data: [
                (d.momentum || 0) * 50 + 50,
                d.confidence_pct || 50,
                100 - (d.dnf_prob_pct || 10),
                (21 - (d.predicted_rank || 20)) * 5,
                (d.is_rookie ? 40 : 90)
            ],
            backgroundColor: `${color}22`,
            borderColor: color,
            borderWidth: 2,
            pointBackgroundColor: color
        };
    });

    const radarOptions = {
        'MOMENTUM': 'Based on recent finishing positions vs qualifying performance.',
        'CONFIDENCE': 'AI confidence in the predicted finishing position.',
        'RELIABILITY': 'Predicted probability of finishing without a technical failure (DNF).',
        'PACE': 'Calculated speed rating based on historical and practice telemetry.',
        'EXPERIENCE': 'Driver career experience and track-specific history.'
    };

    _charts.radar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['MOMENTUM', 'CONFIDENCE', 'RELIABILITY', 'PACE', 'EXPERIENCE'],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#f0f2f5', font: { family: 'JetBrains Mono' } }
                },
                tooltip: {
                    callbacks: {
                        footer: (tooltipItems) => {
                            const label = tooltipItems[0].label;
                            return `\n${radarOptions[label] || ''}`;
                        }
                    }
                }
            },
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.1)' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    pointLabels: { color: '#5c647b', font: { family: 'JetBrains Mono', size: 12 } },
                    ticks: { display: false, max: 100, min: 0 }
                }
            }
        }
    });
}

// ── NEW PHASE 3 & 4 FUNCTIONS ─────────────────────────────────

function setupComparisonTool(preds) {
    const s1 = document.getElementById('comp-driver-1');
    const s2 = document.getElementById('comp-driver-2');
    if (!s1 || !s2 || !preds.race_order) return;

    const drivers = [...preds.race_order].sort((a,b) => a.driver.localeCompare(b.driver));
    
    const optionsHtml = drivers.map(d => `<option value="${d.driver}">${d.driver}</option>`).join('');
    s1.innerHTML = optionsHtml;
    s2.innerHTML = optionsHtml;

    // Default to top 2 predicted
    const sorted = [...preds.race_order].sort((a,b) => (a.predicted_rank || 99) - (b.predicted_rank || 99));
    if (sorted.length >= 2) {
        s1.value = sorted[0].driver;
        s2.value = sorted[1].driver;
    }

    updateComparison();
}

function updateComparison() {
    const name1 = document.getElementById('comp-driver-1').value;
    const name2 = document.getElementById('comp-driver-2').value;
    const ctx = document.getElementById('chart-comparison-radar');
    const statsEl = document.getElementById('comparison-stats');

    if (!lastResults || !ctx || !statsEl) return;
    const preds = lastResults.predictions.race_order;
    const d1 = preds.find(p => p.driver === name1);
    const d2 = preds.find(p => p.driver === name2);

    if (!d1 || !d2) return;

    if (_charts.comparison) _charts.comparison.destroy();

    const getStats = (d) => [
        (d.momentum || 0) * 50 + 50,
        d.confidence_pct || 50,
        100 - (d.dnf_prob_pct || 10),
        (21 - (d.predicted_rank || 20)) * 5,
        (d.is_rookie ? 40 : 90)
    ];

    const data1 = getStats(d1);
    const data2 = getStats(d2);

    _charts.comparison = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['MOMENTUM', 'CONFIDENCE', 'RELIABILITY', 'PACE', 'EXPERIENCE'],
            datasets: [
                {
                    label: name1,
                    data: data1,
                    backgroundColor: `${teamColors[d1.team] || '#FF003C'}33`,
                    borderColor: teamColors[d1.team] || '#FF003C',
                    borderWidth: 3
                },
                {
                    label: name2,
                    data: data2,
                    backgroundColor: `${teamColors[d2.team] || '#00FF41'}33`,
                    borderColor: teamColors[d2.team] || '#00FF41',
                    borderWidth: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    pointLabels: { color: '#f0f2f5', font: { family: 'JetBrains Mono', size: 12 } },
                    ticks: { display: false, max: 100, min: 0 }
                }
            }
        }
    });

    // Render Stats Grid
    const rows = [
        { label: 'Pred. Position', val1: `P${d1.predicted_pos.toFixed(1)}`, val2: `P${d2.predicted_pos.toFixed(1)}`, win: d1.predicted_pos < d2.predicted_pos },
        { label: 'DNF Risk', val1: `${d1.dnf_prob_pct.toFixed(0)}%`, val2: `${d2.dnf_prob_pct.toFixed(0)}%`, win: d1.dnf_prob_pct < d2.dnf_prob_pct },
        { label: 'Confidence', val1: `${d1.confidence_pct.toFixed(0)}%`, val2: `${d2.confidence_pct.toFixed(0)}%`, win: d1.confidence_pct > d2.confidence_pct },
        { label: 'Momentum', val1: d1.momentum > 0 ? 'Positive' : 'Neutral', val2: d2.momentum > 0 ? 'Positive' : 'Neutral', win: d1.momentum > d2.momentum }
    ];

    statsEl.innerHTML = rows.map(r => `
        <div class="comp-stat-row">
            <div class="comp-stat-val ${r.win ? 'winner' : ''}">${r.val1}</div>
            <div class="comp-stat-label">${r.label}</div>
            <div class="comp-stat-val ${!r.win ? 'winner' : ''}">${r.val2}</div>
        </div>
    `).join('');
}
let currentTz = 'UTC';
try {
    currentTz = localStorage.getItem('f1_timezone') || Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
} catch (e) {
    console.warn('Timezone access restricted, defaulting to UTC', e);
}

function setupTimezoneSelector(race) {
    const sel = document.getElementById('timezone-selector');
    if(!sel) return;
    if(sel.options.length === 0) {
        const tzs = Intl.supportedValuesOf('timeZone');
        if(!tzs.includes('Africa/Johannesburg')) tzs.push('Africa/Johannesburg');
        tzs.sort().forEach(tz => {
            const opt = document.createElement('option');
            opt.value = tz; opt.text = tz;
            if(tz === currentTz) opt.selected = true;
            sel.appendChild(opt);
        });
        sel.addEventListener('change', (e) => {
            currentTz = e.target.value;
            localStorage.setItem('f1_timezone', currentTz);
            renderSessionTimes(race);
        });
    }
}

function renderSessionTimes(race) {
    const container = document.getElementById('session-times-list');
    if(!container) return;
    container.innerHTML = '';
    const sessions = [
        {name: 'FP1', date: race.fp1_date, time: race.fp1_time},
        {name: 'FP2', date: race.fp2_date, time: race.fp2_time},
        {name: 'FP3', date: race.fp3_date, time: race.fp3_time},
        {name: 'Sprint', date: race.sprint_date, time: race.sprint_time},
        {name: 'Qualifying', date: race.quali_date, time: race.quali_time},
        {name: 'Race', date: race.date, time: race.time}
    ];
    sessions.forEach(s => {
        if(!s.date || !s.time) return;
        try {
            const isoString = s.date + 'T' + s.time;
            const d = new Date(isoString);
            const formatter = new Intl.DateTimeFormat('en-GB', {
                timeZone: currentTz, weekday: 'short', month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit'
            });
            const html = '<div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:4px;"><span style="font-weight:600;">' + s.name + '</span><span style="color:#aaa;">' + formatter.format(d) + '</span></div>';
            container.insertAdjacentHTML('beforeend', html);
        } catch(e) {}
    });
}

// --- Standings & Past Races ---
async function fetchStandings() {
    const dList = document.getElementById('standings-driver-list');
    const cList = document.getElementById('standings-ctor-list');
    
    dList.innerHTML = '<div class="dim-text" style="padding:16px;">Loading driver standings...</div>';
    cList.innerHTML = '<div class="dim-text" style="padding:16px;">Loading constructor standings...</div>';
    
    try {
        const [driversRes, ctorsRes] = await Promise.all([
            fetch('/api/standings/drivers'),
            fetch('/api/standings/constructors')
        ]);
        
        let drivers = await driversRes.json();
        let ctors = await ctorsRes.json();
        
        // Sort by position ascending
        if (Array.isArray(drivers)) drivers.sort((a, b) => (a.position || 0) - (b.position || 0));
        if (Array.isArray(ctors)) ctors.sort((a, b) => (a.position || 0) - (b.position || 0));
        
        if (!drivers || drivers.length === 0) {
            dList.innerHTML = '<div class="dim-text" style="padding:16px;">No driver standings data found.</div>';
        } else {
            dList.innerHTML = drivers.map((d, index) => {
                const pos = d.position || (index + 1);
                const posClass = pos === 1 ? 'pos-1' : pos === 2 ? 'pos-2' : pos === 3 ? 'pos-3' : 'pos-other';
                const team = driverTeamMap[d.name] || d.constructor || '';
                const teamColor = teamColors[team] || '#FF003C';
                const imgUrl = driverHeadshots[d.name];
                const code = driverCodes[d.name] || d.name.split(' ').pop().slice(0, 3).toUpperCase();
                const thumbHtml = imgUrl
                    ? `<img class="ro-thumb" src="${imgUrl}" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="ro-thumb" style="background:${teamColor}22;color:${teamColor};display:none;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;">${code}</div>`
                    : `<div class="ro-thumb" style="background:${teamColor}22;color:${teamColor};display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;">${code}</div>`;
                return `
                    <div class="standings-row" style="border-left: 3px solid ${teamColor};">
                        <div class="pos-badge ${posClass}">${pos}</div>
                        <div style="display:flex; align-items:center; gap:10px;">
                            ${thumbHtml}
                            <div>
                                <div style="font-weight:700; color:var(--text-main);">${d.name}</div>
                                <div style="font-size:0.75rem; color:${teamColor}; font-weight:600;">${team}</div>
                            </div>
                        </div>
                        <div style="text-align:right; font-weight:600; color:var(--text-main);">${d.points} pts</div>
                        <div style="text-align:right; font-size:0.8rem; color:var(--color-green); font-weight:600;">${d.wins || 0} Wins</div>
                    </div>
                `;
            }).join('');
        }
        
        if (!ctors || ctors.length === 0) {
            cList.innerHTML = '<div class="dim-text" style="padding:16px;">No constructor standings data found.</div>';
        } else {
            cList.innerHTML = ctors.map((c, index) => {
                const pos = c.position || (index + 1);
                const posClass = pos === 1 ? 'pos-1' : pos === 2 ? 'pos-2' : pos === 3 ? 'pos-3' : 'pos-other';
                const color = teamColors[c.constructor] || '#FF003C';
                const logoUrl = constructorLogos[c.constructor];
                const logoHtml = logoUrl
                    ? `<img class="ro-thumb" src="${logoUrl}" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';" style="object-fit:contain;"><div class="ro-thumb" style="background:${color}22;color:${color};display:none;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;">${c.constructor.slice(0,3).toUpperCase()}</div>`
                    : `<div class="ro-thumb" style="background:${color}22;color:${color};display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;">${c.constructor.slice(0,3).toUpperCase()}</div>`;
                return `
                    <div class="standings-row" style="grid-template-columns: 48px 1fr 100px; border-left: 3px solid ${color};">
                        <div class="pos-badge ${posClass}">${pos}</div>
                        <div style="display:flex; align-items:center; gap:10px;">
                            ${logoHtml}
                            <div style="font-weight:700; color:var(--text-main);">${c.constructor}</div>
                        </div>
                        <div style="text-align:right; font-weight:600; color:var(--text-main);">${c.points} pts</div>
                    </div>
                `;
            }).join('');
        }
    } catch (e) {
        dList.innerHTML = '<div class="dim-text" style="padding:16px; color:#ff4444;">Error loading standings.</div>';
        cList.innerHTML = '<div class="dim-text" style="padding:16px; color:#ff4444;">Error loading standings.</div>';
    }
}

async function initPastRaces() {
    const yearSelector = document.getElementById('past-year-selector');
    const raceSelector = document.getElementById('past-race-selector');
    if (!raceSelector) return;
    
    // Populate year selector if empty
    if (yearSelector && yearSelector.options.length === 0) {
        const currentYear = 2026;
        for (let y = currentYear; y >= 2018; y--) {
            const opt = document.createElement('option');
            opt.value = y;
            opt.text = y;
            yearSelector.appendChild(opt);
        }
        yearSelector.value = currentYear;
    }
    
    await loadRacesForYear();
}

async function onYearChange() {
    const emptyState = document.getElementById('past-race-empty-state');
    const content = document.getElementById('past-race-content');
    if (emptyState) emptyState.style.display = 'block';
    if (content) content.style.display = 'none';
    
    await loadRacesForYear();
}

async function loadRacesForYear() {
    const yearSelector = document.getElementById('past-year-selector');
    const raceSelector = document.getElementById('past-race-selector');
    if (!raceSelector) return;
    
    const year = yearSelector ? yearSelector.value : '2026';
    
    try {
        const res = await fetch(`/api/races?year=${year}`);
        const races = await res.json();
        
        raceSelector.innerHTML = '<option value="">Select a race...</option>';
        let latestCompletedRound = null;
        races.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r.round;
            opt.dataset.gpName = r.circuit_id || r.name.toLowerCase().replace(/ /g, '_');
            opt.dataset.name = r.name;
            const statusLabel = r.is_completed ? ' (Completed)' : '';
            opt.text = `Round ${r.round}: ${r.name}${statusLabel}`;
            raceSelector.appendChild(opt);
            if (r.is_completed) {
                latestCompletedRound = r.round;
            }
        });
        if (latestCompletedRound) {
            raceSelector.value = latestCompletedRound;
            loadPastRaceDetails();
        }
    } catch (e) {
        showToast('Error loading race list', 'error');
    }
}

async function loadPastRaceDetails() {
    const selector = document.getElementById('past-race-selector');
    const round = selector.value;
    const emptyState = document.getElementById('past-race-empty-state');
    const content = document.getElementById('past-race-content');
    
    if (!round) {
        emptyState.style.display = 'block';
        content.style.display = 'none';
        return;
    }
    
    const selectedOpt = selector.options[selector.selectedIndex];
    const gpName = selectedOpt.dataset.gpName;
    
    const yearSelector = document.getElementById('past-year-selector');
    const year = yearSelector ? yearSelector.value : '2026';
    
    emptyState.style.display = 'none';
    content.style.display = 'grid';
    
    const resultsContainer = document.getElementById('past-race-results');
    resultsContainer.innerHTML = '<div class="dim-text" style="padding:16px;">Loading results...</div>';
    
    if (_charts.telemetry) {
        _charts.telemetry.destroy();
        _charts.telemetry = null;
    }
    
    try {
        const resRes = await fetch(`/api/race/${round}/results?year=${year}`);
        if (!resRes.ok) throw new Error('Results not found');
        const results = await resRes.json();
        
        if (!results || results.length === 0) {
            resultsContainer.innerHTML = '<div class="dim-text" style="padding:16px;">No results available for this round.</div>';
        } else {
            resultsContainer.innerHTML = `
                <div class="race-order-header">
                    <span class="ro-col-pos">POS</span>
                    <span class="ro-col-driver">DRIVER</span>
                    <span class="ro-col-team">TEAM</span>
                    <span class="ro-col-pred">STATUS</span>
                    <span class="ro-col-conf">POINTS</span>
                </div>
            ` + results.map(r => {
                const posClass = r.position <= 3 ? 'ro-pos-top3' : r.position <= 10 ? 'ro-pos-points' : 'ro-pos-out';
                const teamColor = teamColors[r.constructor] || '#FF003C';
                const imgUrl = driverHeadshots[r.name];
                const code = driverCodes[r.name] || r.name.split(' ').pop().slice(0, 3).toUpperCase();
                const thumbHtml = imgUrl ? `<img class="ro-thumb" src="${imgUrl}" alt="" onerror="this.style.visibility='hidden'">` : `<div class="ro-thumb" style="background:${teamColor}22;color:${teamColor};display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;">${code}</div>`;
                return `
                    <div class="race-order-row">
                        <span class="ro-col-pos ${posClass}">${r.position}</span>
                        <span class="ro-col-driver">
                            ${thumbHtml}
                            <span class="ro-name">${r.name}</span>
                        </span>
                        <span class="ro-col-team" style="color:${teamColor}">${r.constructor}</span>
                        <span class="ro-col-pred">${r.status || 'Finished'}</span>
                        <span class="ro-col-conf" style="font-weight:700; color:var(--text-main);">${r.points || 0} pts</span>
                    </div>
                `;
            }).join('');
        }
    } catch (e) {
        resultsContainer.innerHTML = '<div class="dim-text" style="padding:16px; color:#ff4444;">Could not load race results.</div>';
    }
    
    try {
        const telRes = await fetch(`/api/telemetry/${gpName}?year=${year}&round_num=${round}`);
        if (!telRes.ok) throw new Error('Telemetry not found');
        const telemetry = await telRes.json();
        const stints = telemetry.stints || {};
        
        if (!stints || Object.keys(stints).length === 0) {
            throw new Error('Telemetry data empty');
        }
        renderTelemetryChart(stints);
    } catch (e) {
        const chartCanvas = document.getElementById('past-race-telemetry-chart');
        if (chartCanvas) {
            const ctx = chartCanvas.getContext('2d');
            ctx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
            ctx.fillStyle = '#5c647b';
            ctx.font = '14px Barlow Condensed';
            ctx.textAlign = 'center';
            ctx.fillText('Telemetry not available (cache empty or session not run)', chartCanvas.width / 2, chartCanvas.height / 2);
        }
    }
}

function renderTelemetryChart(telemetry) {
    const ctx = document.getElementById('past-race-telemetry-chart');
    if (!ctx) return;
    
    if (_charts.telemetry) _charts.telemetry.destroy();
    
    const drivers = Object.keys(telemetry).slice(0, 12);
    const datasets = [];
    
    const compoundColors = {
        'SOFT': '#ff5f5f',
        'MEDIUM': '#ffe853',
        'HARD': '#ffffff',
        'INTERMEDIATE': '#4b9d3f',
        'WET': '#2c6fff',
        'UNKNOWN': '#888888',
        'HYPERSOFT': '#ff8bb3',
        'ULTRASOFT': '#d172e8',
        'SUPERSOFT': '#ff3b3b',
        'C1': '#ffffff', 'C2': '#ffffff', 'C3': '#ffe853', 'C4': '#ffe853', 'C5': '#ff5f5f'
    };
    
    const compoundStints = {};
    
    drivers.forEach((driver, driverIdx) => {
        const stints = telemetry[driver] || [];
        stints.forEach(stint => {
            let comp = (stint.compound || 'UNKNOWN').toUpperCase();
            if (!compoundStints[comp]) compoundStints[comp] = [];
            
            const dataPoint = {
                x: [stint.start_lap, stint.end_lap],
                y: driver.split(' ').pop(),
                deg: stint.deg_per_lap,
                len: stint.stint_length
            };
            compoundStints[comp].push(dataPoint);
        });
    });
    
    Object.entries(compoundStints).forEach(([comp, dataPoints]) => {
        datasets.push({
            label: comp,
            data: dataPoints,
            backgroundColor: compoundColors[comp] || '#888888',
            borderColor: 'rgba(0,0,0,0.5)',
            borderWidth: 1,
            barPercentage: 0.6,
            categoryPercentage: 0.8
        });
    });
    
    _charts.telemetry = new Chart(ctx, {
        type: 'bar',
        data: {
            datasets: datasets
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#f0f2f5', font: { family: 'JetBrains Mono' } }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const raw = context.raw;
                            const deg = raw.deg !== null && raw.deg !== undefined ? `${raw.deg > 0 ? '+' : ''}${raw.deg.toFixed(3)}s/lap` : 'N/A';
                            return `${context.dataset.label}: Lap ${raw.x[0]}-${raw.x[1]} (${raw.len} laps). Deg: ${deg}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Laps', color: '#5c647b', font: { family: 'JetBrains Mono' } },
                    grid: { color: 'rgba(28, 32, 45, 0.5)' },
                    ticks: { color: '#5c647b', font: { family: 'JetBrains Mono' } },
                    min: 1
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f0f2f5', font: { family: 'JetBrains Mono', size: 12 } }
                }
            }
        }
    });
}

// ---- RESULTS SYMBOL LEGEND TOGGLE ----
function toggleResultsLegend() {
    const legend = document.getElementById('race-order-legend');
    const icon = document.querySelector('#legend-toggle-btn .toggle-icon');
    if (!legend) return;
    if (legend.classList.contains('collapsed')) {
        legend.classList.remove('collapsed');
        if (icon) icon.innerText = '▲';
    } else {
        legend.classList.add('collapsed');
        if (icon) icon.innerText = '▼';
    }
}

// ---- PREDICTION ANALYSIS & SELF-IMPROVEMENT ----
async function initPredictionAnalysis() {
    const selector = document.getElementById('post-race-round-selector');
    if (selector && selector.options.length <= 1) { // only "Select a round..." option exists
        try {
            const res = await fetch('/api/races');
            const races = await res.json();
            let latestCompletedRound = null;
            races.forEach(r => {
                const opt = document.createElement('option');
                opt.value = r.round;
                const statusLabel = r.is_completed ? ' (Completed)' : '';
                opt.text = `Round ${r.round}: ${r.name}${statusLabel}`;
                selector.appendChild(opt);
                if (r.is_completed) {
                    latestCompletedRound = r.round;
                }
            });
            if (latestCompletedRound) {
                selector.value = latestCompletedRound;
            }
        } catch (e) {
            console.error('Error populating post-race round selector:', e);
        }
    }

    try {
        const res = await fetch('/api/model/health');
        const health = await res.json();
        
        // Render active driver bias corrections table
        renderBiasTable(health.bias_corrections || {});
        
        // Render circuit-type weight adjustments
        renderWeightAdjustments(health.weight_adjustments || {});
        
        // Render season trend chart
        renderSeasonTrendChart(health.accuracy_log || []);
    } catch (e) {
        console.error('Error loading prediction analysis data:', e);
        showToast('Error loading prediction analysis details', 'error');
    }
}

async function triggerPostRaceCheck() {
    const selector = document.getElementById('post-race-round-selector');
    const round = selector.value;
    if (!round) {
        showToast('Please select a round number first', 'warning');
        return;
    }

    const runBtn = document.getElementById('btn-run-post-race');
    const progressPanel = document.getElementById('post-race-progress-panel');
    const progressStage = document.getElementById('post-race-progress-stage');
    const progressStatus = document.getElementById('post-race-progress-status');
    const progressBar = document.getElementById('post-race-progress-bar');
    const progressMsg = document.getElementById('post-race-progress-message');
    const resultsContainer = document.getElementById('post-race-results-container');

    runBtn.disabled = true;
    runBtn.classList.add('disabled');
    progressPanel.classList.remove('hidden');
    progressStage.innerText = "INITIALIZING";
    progressStatus.className = "status-badge loading";
    progressStatus.innerText = "RUNNING";
    progressBar.style.width = "10%";
    progressMsg.innerText = "Checking prediction cache and starting validation...";
    resultsContainer.style.display = 'none';

    try {
        const res = await fetch(`/api/post-race/${round}`, { method: 'POST' });
        const result = await res.json();

        if (result.status === 'error') {
            showToast(result.message, 'error');
            runBtn.disabled = false;
            runBtn.classList.remove('disabled');
            progressPanel.classList.add('hidden');
            return;
        }

        const runId = result.run_id;
        const evtSource = new EventSource(`/api/run/stream/${runId}`);

        evtSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.stage) {
                progressStage.innerText = data.stage.replace('_', ' ');
            }
            if (data.message) {
                progressMsg.innerText = data.message;
            }

            if (data.stage === 'FETCH_RESULTS') {
                progressBar.style.width = "40%";
            } else if (data.stage === 'COMPLETE') {
                progressBar.style.width = "100%";
                progressStatus.className = "status-badge success";
                progressStatus.innerText = "DONE";
                evtSource.close();
                
                // Show comparisons & metrics
                const metrics = data.data;
                if (metrics) {
                    renderComparisonTable(metrics.driver_comparisons || []);
                    renderAccuracyMetrics(metrics);
                    resultsContainer.style.display = 'block';
                    
                    // Reload overall health data
                    setTimeout(() => {
                        initPredictionAnalysis();
                    }, 500);
                }
                
                runBtn.disabled = false;
                runBtn.classList.remove('disabled');
                showToast('Post-race verification completed!', 'success');
            } else if (data.stage === 'ERROR') {
                progressStatus.className = "status-badge error";
                progressStatus.innerText = "ERROR";
                progressMsg.innerText = data.message || "An error occurred during verification.";
                progressBar.style.width = "0%";
                evtSource.close();
                runBtn.disabled = false;
                runBtn.classList.remove('disabled');
                showToast(data.message || 'Error executing post-race check', 'error');
            }
        };

        evtSource.onerror = (e) => {
            console.error('SSE Error in post-race check:', e);
            progressStatus.className = "status-badge error";
            progressStatus.innerText = "ERROR";
            progressMsg.innerText = "Connection lost during verification.";
            evtSource.close();
            runBtn.disabled = false;
            runBtn.classList.remove('disabled');
        };

    } catch (e) {
        showToast('Failed to trigger post-race verification', 'error');
        runBtn.disabled = false;
        runBtn.classList.remove('disabled');
        progressPanel.classList.add('hidden');
    }
}

async function resetModelCorrections() {
    if (!confirm('Are you sure you want to reset all driver bias corrections and circuit weight adjustments? This cannot be undone.')) {
        return;
    }

    try {
        const res = await fetch('/api/model/reset-corrections', { method: 'POST' });
        const result = await res.json();
        if (result.status === 'ok') {
            showToast('All corrections have been reset to zero.', 'success');
            // Refresh screen
            initPredictionAnalysis();
        } else {
            showToast(result.message || 'Failed to reset corrections', 'error');
        }
    } catch (e) {
        showToast('Error resetting corrections', 'error');
    }
}

function renderComparisonTable(comparison) {
    const wrapper = document.getElementById('comparison-table-wrapper');
    if (!wrapper) return;

    if (!comparison || comparison.length === 0) {
        wrapper.innerHTML = `<div style="text-align:center; padding:20px; color:var(--text-dim);">No comparison details available.</div>`;
        return;
    }

    let html = `
        <div class="table-responsive-container">
        <div class="race-order-header" style="grid-template-columns: 50px 1fr 140px 100px 100px 80px; font-weight: bold; border-bottom: 2px solid var(--border-dim); padding-bottom: 8px;">
            <div>Pos</div>
            <div>Driver</div>
            <div>Team</div>
            <div style="text-align:center;">Predicted</div>
            <div style="text-align:center;">Actual</div>
            <div style="text-align:center;">Delta</div>
        </div>
    `;

    // Sort by actual position
    const sortedComp = [...comparison].sort((a, b) => a.actual - b.actual);
    sortedComp.forEach((c, idx) => {
        const team = c.team || 'Unknown';
        const color = teamColors[team] || '#fff';
        const actStr = c.actual >= 99 ? 'DNF' : `P${c.actual}`;
        const delta = c.delta;
        
        let deltaHtml = '';
        if (c.actual >= 99) {
            deltaHtml = `<span style="color:var(--text-dim); font-size:0.85rem;">-</span>`;
        } else if (delta === 0) {
            deltaHtml = `<span class="delta-exact">✓</span>`;
        } else {
            const sign = delta > 0 ? '+' : '';
            const absDelta = Math.abs(delta);
            let clrClass = 'delta-green';
            if (absDelta > 5) {
                clrClass = 'delta-red';
            } else if (absDelta > 2) {
                clrClass = 'delta-yellow';
            }
            deltaHtml = `<span class="${clrClass}">${sign}${delta}</span>`;
        }

        html += `
            <div class="race-order-row" style="grid-template-columns: 50px 1fr 140px 100px 100px 80px;">
                <div class="ro-col-pos ro-pos-out" style="width:36px; font-size:0.9rem; padding: 1px 0;">${c.actual >= 99 ? 'DNF' : idx + 1}</div>
                <div class="ro-col-driver" title="${c.driver}">
                    <span style="border-left: 3px solid ${color}; padding-left: 8px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0;">${c.driver}</span>
                </div>
                <div class="ro-col-team" title="${team}" style="color: var(--text-dim); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${team}</div>
                <div style="text-align:center; font-family:var(--font-heading); font-weight:700;">P${c.predicted}</div>
                <div style="text-align:center; font-family:var(--font-heading); font-weight:700;">${actStr}</div>
                <div style="text-align:center; font-family:var(--font-heading); font-weight:700;">${deltaHtml}</div>
            </div>
        `;
    });
    html += `</div>`;

    wrapper.innerHTML = html;
}

function renderAccuracyMetrics(metrics) {
    const wrapper = document.getElementById('accuracy-metrics-pills');
    if (!wrapper) return;

    const mae = metrics.mae || 0.0;
    const rmse = metrics.rmse || 0.0;
    const rho = metrics.spearman_rho || 0.0;
    const winnerHit = metrics.winner_correct;
    const top3 = metrics.top3_hits;
    const top5 = metrics.top5_hits;

    const maeColor = mae <= 3.5 ? 'green' : mae <= 5.0 ? 'yellow' : 'red';
    const rhoColor = rho >= 0.65 ? 'green' : rho >= 0.45 ? 'yellow' : 'red';
    const winColor = winnerHit ? 'green' : 'red';

    wrapper.innerHTML = `
        <div class="metric-pill">
            <span class="metric-pill-label" title="Mean Absolute Error (MAE)">Mean Absolute Error (MAE)</span>
            <span class="metric-pill-val ${maeColor}">${mae.toFixed(2)} places</span>
        </div>
        <div class="metric-pill">
            <span class="metric-pill-label" title="RMSE">RMSE</span>
            <span class="metric-pill-val">${rmse.toFixed(2)}</span>
        </div>
        <div class="metric-pill">
            <span class="metric-pill-label" title="Spearman Correlation (ρ)">Spearman Correlation (ρ)</span>
            <span class="metric-pill-val ${rhoColor}">${rho.toFixed(3)}</span>
        </div>
        <div class="metric-pill">
            <span class="metric-pill-label" title="Winner Predicted Correctly">Winner Predicted Correctly</span>
            <span class="metric-pill-val ${winColor}">${winnerHit ? '✅ YES' : '❌ NO'}</span>
        </div>
        <div class="metric-pill">
            <span class="metric-pill-label" title="Top-3 Predicted Hit Rate">Top-3 Predicted Hit Rate</span>
            <span class="metric-pill-val">${top3}/3 (${(top3/3*100).toFixed(0)}%)</span>
        </div>
        <div class="metric-pill">
            <span class="metric-pill-label" title="Top-5 Predicted Hit Rate">Top-5 Predicted Hit Rate</span>
            <span class="metric-pill-val">${top5}/5 (${(top5/5*100).toFixed(0)}%)</span>
        </div>
    `;
}

function renderBiasTable(biasData) {
    const biases = biasData.driver_biases || {};
    const wrapper = document.getElementById('bias-table-wrapper');
    if (!wrapper) return;

    if (Object.keys(biases).length === 0) {
        wrapper.innerHTML = `<div style="text-align:center; padding:20px; color:var(--text-dim);">No bias correction data found. Run post-race verifications to build active biases.</div>`;
        return;
    }

    let html = `
        <div class="table-responsive-container">
        <div class="race-order-header" style="grid-template-columns: minmax(140px, 1.5fr) minmax(110px, 1fr) repeat(4, minmax(70px, 1fr)); font-weight: bold; border-bottom: 2px solid var(--border-dim); padding-bottom: 8px;">
            <div>Driver</div>
            <div>Team</div>
            <div style="text-align:center;">Permanent</div>
            <div style="text-align:center;">Street</div>
            <div style="text-align:center;">Hybrid</div>
            <div style="text-align:center;">Overall</div>
        </div>
    `;

    const drivers = Object.keys(biases).sort();
    drivers.forEach(drv => {
        const team = driverTeamMap[drv] || 'Unknown';
        const color = teamColors[team] || '#fff';
        const b = biases[drv] || {};
        
        function getChipHtml(val) {
            if (val === undefined || val === null || val === 0) return `<span style="color:var(--text-dim);">-</span>`;
            const sign = val > 0 ? '+' : '';
            const absVal = Math.abs(val);
            let chipClass = 'bias-chip-green';
            if (absVal > 1.5) {
                chipClass = 'bias-chip-red';
            } else if (absVal > 0.5) {
                chipClass = 'bias-chip-yellow';
            }
            return `<span class="bias-correction-chip ${chipClass}">${sign}${val.toFixed(2)}</span>`;
        }

        html += `
            <div class="race-order-row" style="grid-template-columns: minmax(140px, 1.5fr) minmax(110px, 1fr) repeat(4, minmax(70px, 1fr));">
                <div class="ro-col-driver" title="${drv}">
                    <span style="border-left: 3px solid ${color}; padding-left: 8px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0;">${drv}</span>
                </div>
                <div class="ro-col-team" title="${team}" style="color: var(--text-dim); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${team}</div>
                <div style="text-align:center;">${getChipHtml(b.permanent)}</div>
                <div style="text-align:center;">${getChipHtml(b.street)}</div>
                <div style="text-align:center;">${getChipHtml(b.hybrid)}</div>
                <div style="text-align:center;">${getChipHtml(b.overall)}</div>
            </div>
        `;
    });
    html += `</div>`;

    wrapper.innerHTML = html;
}

function renderWeightAdjustments(adjData) {
    const adjustments = adjData.adjustments || {};
    const wrapper = document.getElementById('weight-adjustments-wrapper');
    if (!wrapper) return;

    let hasAdjustments = false;
    let html = `<div style="display:flex; flex-direction:column; gap:12px;">`;

    const ctypes = ['permanent', 'street', 'hybrid'];
    ctypes.forEach(ctype => {
        const adj = adjustments[ctype] || {};
        const sc = adj.sc_prob || 0.0;
        const form = adj.form_score || 0.0;

        if (sc !== 0.0 || form !== 0.0) {
            hasAdjustments = true;
        }

        function formatNudge(val) {
            if (val === 0.0) return `<span style="color:var(--text-dim);">0.0%</span>`;
            const pct = (val * 100).toFixed(1);
            const sign = val > 0 ? '+' : '';
            const color = val > 0 ? 'var(--color-green)' : 'var(--f1-red)';
            return `<span style="color:${color}; font-weight:bold;">${sign}${pct}%</span>`;
        }

        html += `
            <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border-dim); padding:12px; border-radius:8px;">
                <div style="font-family:var(--font-heading); font-size:1.1rem; font-weight:600; color:var(--f1-red); text-transform:uppercase; margin-bottom:8px;">
                    ${ctype} circuits
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:4px;">
                    <span class="dim-text">Safety Car weight:</span>
                    <span>${formatNudge(sc)}</span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.9rem;">
                    <span class="dim-text">Driver Form weight:</span>
                    <span>${formatNudge(form)}</span>
                </div>
            </div>
        `;
    });

    html += `</div>`;

    if (!hasAdjustments) {
        wrapper.innerHTML = `<div style="text-align:center; padding:20px; color:var(--text-dim);">No active feature weight nudges. Weights are neutral.</div>`;
    } else {
        wrapper.innerHTML = html;
    }
}

function renderSeasonTrendChart(accuracyLog) {
    const canvas = document.getElementById('accuracy-trend-chart');
    if (!canvas) return;

    if (!accuracyLog || accuracyLog.length === 0) {
        if (_charts.accuracyTrend) {
            _charts.accuracyTrend.destroy();
            _charts.accuracyTrend = null;
        }
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#5c647b';
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('No accuracy trend data available yet. Run verifications first.', canvas.width / 2, canvas.height / 2);
        return;
    }

    const sortedLog = [...accuracyLog].sort((a, b) => {
        if (a.season !== b.season) return a.season - b.season;
        return a.round - b.round;
    });

    const labels = sortedLog.map(e => `R${e.round}: ${e.race.replace(' Grand Prix', '')}`);
    const maeData = sortedLog.map(e => e.mae);
    const spearmanData = sortedLog.map(e => e.spearman_rho);

    if (_charts.accuracyTrend) {
        _charts.accuracyTrend.destroy();
    }

    _charts.accuracyTrend = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Mean Absolute Error (MAE)',
                    data: maeData,
                    borderColor: '#FF003C',
                    backgroundColor: 'rgba(255, 0, 60, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointBackgroundColor: '#FF003C',
                    yAxisID: 'y',
                    tension: 0.15
                },
                {
                    label: 'Spearman Correlation (ρ)',
                    data: spearmanData,
                    borderColor: '#00F0FF',
                    backgroundColor: 'rgba(0, 240, 255, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointBackgroundColor: '#00F0FF',
                    yAxisID: 'y1',
                    tension: 0.15
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#5c647b', font: { family: 'Inter', size: 10 } }
                },
                y: {
                    position: 'left',
                    title: { display: true, text: 'MAE (lower is better)', color: '#FF003C', font: { family: 'Inter', weight: 'bold' } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#5c647b' },
                    min: 0,
                    max: Math.max(8, Math.ceil(Math.max(...maeData) + 1))
                },
                y1: {
                    position: 'right',
                    title: { display: true, text: 'Spearman Correlation (ρ)', color: '#00F0FF', font: { family: 'Inter', weight: 'bold' } },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#5c647b' },
                    min: -1.0,
                    max: 1.0
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f0f2f5', font: { family: 'Inter' } }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// ── NEW PHASE 5 FUNCTIONS ─────────────────────────────────────

async function initPastPredictions() {
    const selector = document.getElementById('past-pred-selector');
    if (!selector) return;
    
    try {
        const res = await fetch('/api/predictions');
        const data = await res.json();
        const predictions = data.predictions || [];
        
        if (predictions.length === 0) {
            selector.innerHTML = '<option value="">No past predictions found</option>';
            return;
        }
        
        selector.innerHTML = '<option value="">Select a prediction file...</option>';
        predictions.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.filename;
            
            let displayName = p.race ? p.race : p.filename.replace('.json', '').replace(/_/g, ' ').replace('race ', 'Race ');
            if (p.generated_at) {
                const dateStr = new Date(p.generated_at).toLocaleString();
                displayName += ` - ${dateStr}`;
            }
            opt.text = displayName;
            selector.appendChild(opt);
        });
    } catch (e) {
        showToast('Error loading predictions list', 'error');
    }
}

let _selectedPastPredictionData = null;

async function loadPastPrediction() {
    const filename = document.getElementById('past-pred-selector').value;
    const emptyState = document.getElementById('past-pred-empty-state');
    const content = document.getElementById('past-pred-content');
    const btnCompare = document.getElementById('btn-compare-pred');
    const container = document.getElementById('past-pred-details-container');
    const comparePanel = document.getElementById('past-pred-compare-panel');
    
    if (!filename) {
        emptyState.classList.remove('hidden');
        content.classList.add('hidden');
        btnCompare.classList.add('hidden');
        return;
    }
    
    emptyState.classList.add('hidden');
    content.classList.remove('hidden');
    btnCompare.classList.remove('hidden');
    comparePanel.classList.add('hidden'); // Hide until compared
    
    container.innerHTML = '<div class="dim-text" style="padding:16px;">Loading prediction details...</div>';
    
    try {
        const res = await fetch(`/api/predictions/${filename}`);
        if (!res.ok) throw new Error('Prediction not found');
        const pred = await res.json();
        _selectedPastPredictionData = pred;
        
        const timestamp = new Date(pred.generated_at || Date.now()).toLocaleString();
        
        let html = `
            <div style="margin-bottom: 20px; background: rgba(28, 32, 45, 0.5); padding: 15px; border-radius: 0;">
                <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 5px;">${pred.race?.name || 'Unknown Race'} (Round ${pred.race?.round || '?'})</div>
                <div class="dim-text" style="font-size: 0.9rem;">Predicted at: ${timestamp}</div>
            </div>
            <div class="race-order-header">
                <span class="ro-col-pos">PRED</span>
                <span class="ro-col-driver">DRIVER</span>
                <span class="ro-col-team">TEAM</span>
            </div>
        `;
        
        const order = pred.race_order || [];
        order.sort((a,b) => (a.predicted_rank||99) - (b.predicted_rank||99)).forEach(r => {
            const posClass = r.predicted_rank <= 3 ? 'ro-pos-top3' : r.predicted_rank <= 10 ? 'ro-pos-points' : 'ro-pos-out';
            const teamColor = teamColors[r.team] || '#FF003C';
            const imgUrl = driverHeadshots[r.driver];
            const code = driverCodes[r.driver] || r.driver.split(' ').pop().slice(0, 3).toUpperCase();
            const thumbHtml = imgUrl ? `<img class="ro-thumb" src="${imgUrl}" alt="" onerror="this.style.visibility='hidden'">` : `<div class="ro-thumb" style="background:${teamColor}22;color:${teamColor};display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;">${code}</div>`;
            
            html += `
                <div class="race-order-row">
                    <span class="ro-col-pos ${posClass}">P${r.predicted_rank}</span>
                    <span class="ro-col-driver">
                        ${thumbHtml}
                        <span class="ro-name">${r.driver}</span>
                    </span>
                    <span class="ro-col-team" style="color:${teamColor}">${r.team}</span>
                </div>
            `;
        });
        
        container.innerHTML = html;
        
    } catch (e) {
        container.innerHTML = '<div class="dim-text" style="padding:16px; color:#ff4444;">Could not load prediction details.</div>';
    }
}

async function compareSelectedPrediction() {
    const filename = document.getElementById('past-pred-selector').value;
    if (!filename) return;
    
    const btnCompare = document.getElementById('btn-compare-pred');
    btnCompare.disabled = true;
    btnCompare.innerText = '📊 Comparing...';
    
    try {
        const res = await fetch(`/api/post-race/compare/${filename}`, { method: 'POST' });
        const result = await res.json();
        
        if (result.status === 'error') {
            showToast(result.message || 'Error comparing prediction', 'error');
            return;
        }
        
        document.getElementById('past-pred-compare-panel').classList.remove('hidden');
        
        renderCustomComparisonTable(result.metrics.driver_comparisons || [], 'past-pred-compare-table');
        renderCustomAccuracyMetrics(result.metrics, 'past-pred-compare-metrics');
        
        showToast('Comparison successful!', 'success');
        
    } catch (e) {
        showToast('Failed to compare prediction', 'error');
    } finally {
        btnCompare.disabled = false;
        btnCompare.innerText = '📊 Compare to Actual Race Data';
    }
}

function renderCustomComparisonTable(comparison, elementId) {
    const wrapper = document.getElementById(elementId);
    if (!wrapper) return;

    if (!comparison || comparison.length === 0) {
        wrapper.innerHTML = '<div style="text-align:center; padding:20px; color:var(--text-dim);">No comparison details available.</div>';
        return;
    }

    let html = `
        <div class="table-responsive-container">
        <div class="race-order-header" style="grid-template-columns: 50px 1fr 140px 100px 100px 80px; font-weight: bold; border-bottom: 2px solid var(--border-dim); padding-bottom: 8px;">
            <div>Pos</div>
            <div>Driver</div>
            <div>Team</div>
            <div style="text-align:center;">Predicted</div>
            <div style="text-align:center;">Actual</div>
            <div style="text-align:center;">Delta</div>
        </div>
    `;

    const sortedComp = [...comparison].sort((a, b) => a.actual - b.actual);
    sortedComp.forEach((c, idx) => {
        const team = c.team || 'Unknown';
        const color = teamColors[team] || '#fff';
        const actStr = c.actual >= 99 ? 'DNF' : `P${c.actual}`;
        const delta = c.delta;
        
        let deltaHtml = '';
        if (c.actual >= 99) {
            deltaHtml = '<span style="color:var(--text-dim); font-size:0.85rem;">-</span>';
        } else if (delta === 0) {
            deltaHtml = '<span class="delta-exact">✓</span>';
        } else {
            const sign = delta > 0 ? '+' : '';
            const absDelta = Math.abs(delta);
            let clrClass = 'delta-green';
            if (absDelta > 5) clrClass = 'delta-red';
            else if (absDelta > 2) clrClass = 'delta-yellow';
            deltaHtml = `<span class="${clrClass}">${sign}${delta}</span>`;
        }

        html += `
            <div class="race-order-row" style="grid-template-columns: 50px 1fr 140px 100px 100px 80px;">
                <div class="ro-col-pos ro-pos-out" style="width:36px; font-size:0.9rem; padding: 1px 0;">${c.actual >= 99 ? 'DNF' : idx + 1}</div>
                <div class="ro-col-driver" title="${c.driver}">
                    <span style="border-left: 3px solid ${color}; padding-left: 8px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0;">${c.driver}</span>
                </div>
                <div class="ro-col-team" title="${team}" style="color: var(--text-dim); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${team}</div>
                <div style="text-align:center; font-family:var(--font-heading); font-weight:700;">P${c.predicted}</div>
                <div style="text-align:center; font-family:var(--font-heading); font-weight:700;">${actStr}</div>
                <div style="text-align:center; font-family:var(--font-heading); font-weight:700;">${deltaHtml}</div>
            </div>
        `;
    });
    html += `</div>`;
    wrapper.innerHTML = html;
}

function renderCustomAccuracyMetrics(metrics, elementId) {
    const wrapper = document.getElementById(elementId);
    if (!wrapper) return;
    
    const mae = metrics.mae || 0.0;
    const rmse = metrics.rmse || 0.0;
    const rho = metrics.spearman_rho || 0.0;
    const winnerHit = metrics.winner_correct;
    const top3 = metrics.top3_hits;
    const top5 = metrics.top5_hits;

    const maeColor = mae <= 3.5 ? 'green' : mae <= 5.0 ? 'yellow' : 'red';
    const rhoColor = rho >= 0.65 ? 'green' : rho >= 0.45 ? 'yellow' : 'red';
    const winColor = winnerHit ? 'green' : 'red';

    wrapper.innerHTML = `
        <div class="metric-pill">
            <span class="metric-pill-label" title="MAE">MAE</span>
            <span class="metric-pill-val ${maeColor}">${mae.toFixed(2)} places</span>
        </div>
        <div class="metric-pill">
            <span class="metric-pill-label" title="RMSE">RMSE</span>
            <span class="metric-pill-val">${rmse.toFixed(2)}</span>
        </div>
        <div class="metric-pill">
            <span class="metric-pill-label" title="Spearman (ρ)">Spearman (ρ)</span>
            <span class="metric-pill-val ${rhoColor}">${rho.toFixed(3)}</span>
        </div>
        <div class="metric-pill">
            <span class="metric-pill-label" title="Winner Hit">Winner Hit</span>
            <span class="metric-pill-val ${winColor}">${winnerHit ? '✅ YES' : '❌ NO'}</span>
        </div>
        <div class="metric-pill">
            <span class="metric-pill-label" title="Top-3 Hits">Top-3 Hits</span>
            <span class="metric-pill-val">${top3}/3 (${(top3/3*100).toFixed(0)}%)</span>
        </div>
        <div class="metric-pill">
            <span class="metric-pill-label" title="Top-5 Hits">Top-5 Hits</span>
            <span class="metric-pill-val">${top5}/5 (${(top5/5*100).toFixed(0)}%)</span>
        </div>
    `;
}

async function rebuildCache() {
    if (!confirm('Rebuild all caches? This will clear everything and trigger a fresh download.')) return;
    
    const progressDiv = document.getElementById('global-download-progress');
    const statusSpan = document.getElementById('global-download-status');
    
    progressDiv.classList.remove('hidden');
    statusSpan.innerText = 'Requesting cache rebuild...';
    
    try {
        const res = await fetch('/api/cache/rebuild', { method: 'POST' });
        const result = await res.json();
        
        if (result.status === 'error') {
            showToast(result.message || 'Error rebuilding cache', 'error');
            progressDiv.classList.add('hidden');
            return;
        }
        
        const runId = result.run_id;
        
        // Plan I1: live download progress bar (injected once, reused)
        let bar = document.getElementById('cache-progress-bar');
        let barText = document.getElementById('cache-progress-text');
        if (!bar) {
            const wrap = document.createElement('div');
            wrap.style.cssText = 'margin-top:8px;display:flex;align-items:center;gap:10px;';
            bar = document.createElement('progress');
            bar.id = 'cache-progress-bar';
            bar.max = 100; bar.value = 0;
            bar.style.cssText = 'flex:1;height:10px;';
            barText = document.createElement('span');
            barText.id = 'cache-progress-text';
            barText.style.cssText = 'font-size:12px;color:var(--text-secondary);white-space:nowrap;';
            wrap.appendChild(bar); wrap.appendChild(barText);
            statusSpan.parentElement.appendChild(wrap);
        }
        bar.value = 0;
        progressDiv.classList.remove('hidden');
        
        const evtSource = new EventSource(`/api/run/stream/${runId}`);
        
        evtSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Plan I1: dict progress payloads from the manifest warmer
            if (data.stage === 'DOWNLOADING' && data.data && data.data.total) {
                const d = data.data;
                const pct = Math.round(100 * d.done / Math.max(1, d.total));
                bar.value = pct;
                const failTxt = d.failed > 0 ? ` · ⚠ ${d.failed} failed` : '';
                const etaTxt = d.eta_min != null ? ` · ETA ${Math.round(d.eta_min)}m` : '';
                barText.innerText = `${d.done} / ${d.total}${failTxt}${etaTxt}`;
                if (d.phase === 'fastf1_sessions') {
                    barText.innerText += ' · telemetry sessions';
                }
            }
            
            if (data.stage === 'COMPLETE') {
                bar.value = 100;
                barText.innerText = 'Complete ✓';
                statusSpan.innerText = 'Rebuild complete!';
                evtSource.close();
                setTimeout(() => {
                    progressDiv.classList.add('hidden');
                    location.reload(); // Reload to pick up fresh data
                }, 2000);
            } else if (data.stage === 'ERROR') {
                statusSpan.innerText = `Error: ${data.message}`;
                evtSource.close();
                setTimeout(() => progressDiv.classList.add('hidden'), 5000);
            } else {
                statusSpan.innerText = data.message || data.stage;
            }
        };
        
        evtSource.onerror = () => {
            statusSpan.innerText = 'Connection to rebuild stream lost.';
            evtSource.close();
            setTimeout(() => progressDiv.classList.add('hidden'), 5000);
        };
        
    } catch (e) {
        showToast('Failed to trigger cache rebuild', 'error');
        progressDiv.classList.add('hidden');
    }
}
