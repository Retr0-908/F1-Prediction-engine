    const msgs = [
        "[0.001] INITIALIZING TELEMETRY ENGINE...",
        "[0.045] ESTABLISHING PIT WALL LINK...",
        "[0.120] LOADING RACING LINE DATA...",
        "[0.294] CALIBRATING PREDICTION ALGORITHMS...",
        "[0.471] SYNCHRONIZING SECURE TUNNEL...",
        "[0.890] ALL SYSTEMS GO."
    ];
    let step = 0;
    const el = document.getElementById('launch-status');
    const interval = setInterval(() => {
        if(step < msgs.length) {
            el.innerText = msgs[step];
            step++;
        } else {
            clearInterval(interval);
        }
    }, 120);
