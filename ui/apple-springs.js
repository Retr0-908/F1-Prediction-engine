/**
 * apple-springs.js
 * Apple-grade fluid interaction system for the F1 Predictor UI.
 *
 * Principles implemented:
 *   1. Response    — pointer-down feedback, zero latency
 *   2. Interruptibility — springs from current rendered value, never target
 *   3. Springs     — GSAP spring physics, critically-damped (bounce:0) by default
 *   4. Velocity handoff — (placeholder, panels/cards do not have gesture releases)
 *   5. Spatial consistency — directional screen transitions
 *   6. Gesture hinting — status-dot pulse, launch sequence stagger
 */

(function () {
  'use strict';

  /* ─── REDUCED MOTION GUARD ─────────────────────────────────────── */
  const pRM = window.matchMedia('(prefers-reduced-motion: reduce)');
  const reducedMotion = () => pRM.matches;

  /**
   * springTo — wrapper around gsap.to that respects reduced motion.
   * Always reads the element's CURRENT rendered value so animations are
   * fully interruptible (principle 3).
   */
  function springTo(el, props, opts = {}) {
    if (reducedMotion()) {
      gsap.set(el, props);
      return;
    }
    // Read current rendered transform so we start from where it is, not where it should be
    return gsap.to(el, {
      ...props,
      ease: opts.ease || 'power2.out',
      duration: opts.duration ?? 0.35,
      overwrite: 'auto',        // cancel conflicting tweens on the same element
      ...opts,
    });
  }

  /* ─── SPRING PRESETS (Apple WWDC values translated) ────────────── */
  const SPRING = {
    // Critically damped — no overshoot.  damping=1.0 / response=0.35s
    crisp: { ease: 'power2.out', duration: 0.35 },
    // Snappy for small elements (nav, badges)
    snap:  { ease: 'power3.out', duration: 0.25 },
    // Slightly under-damped for momentum interactions (card flick-reveal)
    bounce: { ease: 'back.out(1.4)', duration: 0.35 },
    // Screen / panel transitions
    screen: { ease: 'power2.inOut', duration: 0.38 },
  };

  /* ─── PHASE 1: POINTER-DOWN FEEDBACK ───────────────────────────── */
  /**
   * Attach immediate press-state scale to any list of selectors.
   * Responds on pointerdown (not click), respects capture/cancel.
   */
  function attachPressState(selector, scaleDown = 0.96) {
    document.querySelectorAll(selector).forEach(el => {
      el.addEventListener('pointerdown', () => {
        springTo(el, { scale: scaleDown }, { duration: 0.1, ease: 'power2.out' });
      });
      const release = () => springTo(el, { scale: 1 }, SPRING.bounce);
      el.addEventListener('pointerup',     release);
      el.addEventListener('pointercancel', release);
      el.addEventListener('pointerleave',  release);
    });
  }

  /* ─── PHASE 2: PANEL HOVER SPRINGS ─────────────────────────────── */
  /**
   * Replace the CSS hover translateY with GSAP springs so they are
   * fully interruptible mid-flight (e.g. mouse leaves while entering).
   */
  function attachPanelHover() {
    document.querySelectorAll('.panel').forEach(el => {
      // Remove the CSS transition that would fight GSAP
      el.style.transition = 'border-color 0.25s ease';

      el.addEventListener('pointerenter', () => {
        springTo(el, { y: -4, scale: 1.002 }, SPRING.crisp);
      });
      el.addEventListener('pointerleave', () => {
        springTo(el, { y: 0, scale: 1 }, SPRING.crisp);
      });
    });
  }

  /* ─── PHASE 2: DRIVER CARD HOVER SPRINGS ───────────────────────── */
  function attachCardHover() {
    // Use event delegation so dynamically-created cards also get this
    document.addEventListener('pointerenter', e => {
      const card = e.target.closest('.driver-compact-card, .dt-card, .card, .standings-row, .race-order-row');
      if (!card) return;
      const isRow = card.classList.contains('standings-row') || card.classList.contains('race-order-row');
      springTo(card, isRow ? { x: 5 } : { x: 4, scale: 1.01 }, SPRING.snap);
    }, true);

    document.addEventListener('pointerleave', e => {
      const card = e.target.closest('.driver-compact-card, .dt-card, .card, .standings-row, .race-order-row');
      if (!card) return;
      springTo(card, { x: 0, scale: 1 }, SPRING.crisp);
    }, true);
  }

  /* ─── PHASE 3: DIRECTIONAL SCREEN TRANSITIONS ──────────────────── */
  /**
   * Nav order defines directionality: moving to a later item → slide from right,
   * moving to an earlier item → slide from left.
   */
  const NAV_ORDER = [
    'dashboard-screen',
    'team-screen',
    'prices-screen',
    'standings-screen',
    'past-archive-screen',
    'pred-analysis-screen',
    'settings-screen',
    'analysis-screen',
    'results-screen'
  ];

  let currentScreenId = 'dashboard-screen';

  /**
   * Animate screen out, then screen in.
   * direction: 'right' (new screen enters from right), 'left', or 'up'/'down'.
   */
  function transitionScreens(fromId, toId) {
    if (fromId === toId) return;

    const fromEl = document.getElementById(fromId);
    const toEl   = document.getElementById(toId);
    if (!fromEl || !toEl) return;

    // Determine directionality
    const fromIdx = NAV_ORDER.indexOf(fromId);
    const toIdx   = NAV_ORDER.indexOf(toId);
    const goRight = toIdx > fromIdx;

    const enterX  = goRight ?  40 : -40;
    const exitX   = goRight ? -20 :  20;

    if (reducedMotion()) {
      fromEl.classList.remove('active');
      fromEl.style.opacity = '';
      fromEl.style.transform = '';
      toEl.classList.add('active');
      toEl.style.opacity = '';
      toEl.style.transform = '';
      return;
    }

    // Kill any running tweens on these screens
    gsap.killTweensOf(fromEl);
    gsap.killTweensOf(toEl);

    // Prepare incoming screen (hidden, off-screen)
    gsap.set(toEl, { opacity: 0, x: enterX, pointerEvents: 'none' });
    toEl.classList.add('active');

    const tl = gsap.timeline({
      onComplete: () => {
        fromEl.classList.remove('active');
        gsap.set(fromEl, { clearProps: 'all' });
        gsap.set(toEl, { clearProps: 'all' });
        toEl.style.pointerEvents = '';
      }
    });

    // Exit current screen
    tl.to(fromEl, { opacity: 0, x: exitX, duration: 0.22, ease: 'power2.in' });
    // Enter new screen (overlap slightly)
    tl.to(toEl,   { opacity: 1, x: 0,     duration: 0.32, ease: 'power2.out' }, '-=0.1');
  }

  /* ─── PHASE 4: TAB SWITCHING SPRING ────────────────────────────── */
  function attachTabSprings() {
    // Hook into the existing setupTabs function's event listeners via delegation
    document.querySelectorAll('.results-tabs, .tab-row').forEach(tabBar => {
      tabBar.addEventListener('click', e => {
        const btn = e.target.closest('.tab-btn');
        if (!btn) return;
        const targetId = btn.getAttribute('data-tab');
        if (!targetId) return;

        const screen = btn.closest('.screen');
        if (!screen) return;

        const prevContent = screen.querySelector('.tab-content.active');
        const nextContent = document.getElementById(targetId);

        if (!nextContent || prevContent === nextContent) return;

        // Determine direction from DOM order
        const allTabs = Array.from(screen.querySelectorAll('.tab-content'));
        const prevIdx = allTabs.indexOf(prevContent);
        const nextIdx = allTabs.indexOf(nextContent);
        const dir = nextIdx > prevIdx ? 1 : -1;

        if (reducedMotion()) return; // let existing JS handle it

        // Intercept: animate out old, animate in new
        if (prevContent) {
          gsap.to(prevContent, {
            opacity: 0, x: dir * -20, duration: 0.09, ease: 'power2.in',
            onComplete: () => {
              prevContent.classList.remove('active');
              prevContent.style.display = 'none';
              gsap.set(prevContent, { clearProps: 'all' });

              nextContent.style.display = 'block';
              nextContent.classList.add('active');
              gsap.fromTo(nextContent,
                { opacity: 0, x: dir * 20 },
                { opacity: 1, x: 0, duration: 0.14, ease: 'power2.out',
                  onComplete: () => gsap.set(nextContent, { clearProps: 'all' }) }
              );
            }
          });
        }
      }, true); // capture phase so we run before the existing click handler
    });
  }

  /* ─── PHASE 7: LIVE STATUS DOT PULSE ───────────────────────────── */
  /**
   * Adds a living pulse to green status indicators.
   * Uses CSS custom animation class rather than GSAP to keep it performant.
   */
  function initStatusPulse() {
    // Observe for dynamic status updates (Disabled for debugging)
    /*
    const observer = new MutationObserver(() => {
      document.querySelectorAll('.status-indicator.green').forEach(dot => {
        dot.classList.add('apple-pulse');
      });
    });
    observer.observe(document.body, { subtree: true, attributes: true, attributeFilter: ['class'] });
    */

    // Initial pass
    document.querySelectorAll('.status-indicator.green').forEach(dot => {
      dot.classList.add('apple-pulse');
    });
  }

  /* ─── PHASE 9: LAUNCH SCREEN STAGGER ───────────────────────────── */
  function animateLaunchEntrance() {
    const title  = document.querySelector('.brand-title');
    const status = document.getElementById('launch-status');
    const spinner = document.querySelector('.spinner');

    if (!title) return;

    if (reducedMotion()) return;

    // Start invisible
    gsap.set([title, spinner, status].filter(Boolean), { opacity: 0, y: 24 });

    gsap.timeline()
      .to(title,   { opacity: 1, y: 0, duration: 0.55, ease: 'power2.out' }, 0.1)
      .to(spinner, { opacity: 1, y: 0, duration: 0.4,  ease: 'power2.out' }, 0.3)
      .to(status,  { opacity: 1, y: 0, duration: 0.35, ease: 'power2.out' }, 0.45);
  }

  /* ─── PHASE 3: PATCH NAVIGATION ────────────────────────────────── */
  /**
   * Wrap the existing navigation click handler to call our spring transition.
   * Uses a capturing listener at the nav level so it fires before app.js's listener.
   */
  function patchNavigation() {
    const nav = document.getElementById('main-nav');
    if (!nav) return;

    nav.addEventListener('click', e => {
      const btn = e.target.closest('.nav-btn');
      if (!btn) return;

      const toId = btn.getAttribute('data-target');
      if (!toId || toId === currentScreenId) return;

      // Pointer-down scale (even if click fires slightly after)
      springTo(btn, { scale: 0.94 }, { duration: 0.08, ease: 'power2.out' });
      setTimeout(() => springTo(btn, { scale: 1 }, SPRING.bounce), 80);

      // Directional transition
      transitionScreens(currentScreenId, toId);
      currentScreenId = toId;
    }, true); // capture phase
  }

  /* ─── PHASE 8: NAV ACTIVE INDICATOR SPRING ─────────────────────── */
  /**
   * Animate the active underline indicator sliding between nav buttons.
   * Creates a floating indicator <span> that springs to the active button.
   */
  function initNavIndicator() {
    const navLinks = document.querySelector('.nav-links');
    if (!navLinks) return;

    const indicator = document.createElement('span');
    indicator.id = 'nav-spring-indicator';
    indicator.style.cssText = `
      position: absolute;
      bottom: 0;
      height: 2px;
      background: var(--neon-cyan);
      border-radius: 2px 2px 0 0;
      pointer-events: none;
      box-shadow: 0 0 8px rgba(0,240,255,0.5);
      transition: none;
    `;
    navLinks.style.position = 'relative';
    navLinks.appendChild(indicator);

    function moveIndicator(btn) {
      if (!btn) { gsap.set(indicator, { scaleX: 0, opacity: 0 }); return; }
      const navRect  = navLinks.getBoundingClientRect();
      const btnRect  = btn.getBoundingClientRect();
      const left = btnRect.left - navRect.left;
      const width = btnRect.width;

      if (reducedMotion()) {
        gsap.set(indicator, { x: left, width, opacity: 1, scaleX: 1 });
      } else {
        gsap.to(indicator, { x: left, width, opacity: 1, scaleX: 1,
          duration: 0.3, ease: 'power2.inOut', overwrite: 'auto' });
      }
    }

    // Position on first active button
    const initialActive = navLinks.querySelector('.nav-btn.active');
    if (initialActive) {
      gsap.set(indicator, { scaleX: 0, opacity: 0 });
      setTimeout(() => moveIndicator(initialActive), 200);
    }

    // Move on nav click
    navLinks.addEventListener('click', e => {
      const btn = e.target.closest('.nav-btn');
      if (btn) moveIndicator(btn);
    });
  }

  /* ─── RES-CARD ENTRANCE STAGGER ────────────────────────────────── */
  /**
   * When results are rendered, stagger-animate each card in.
   * Called by app.js after populateResults() — exposed on window.
   */
  window.appleStaggerCards = function (containerSelector, delay = 0) {
    if (reducedMotion()) return;
    const cards = document.querySelectorAll(containerSelector);
    gsap.fromTo(cards,
      { opacity: 0, y: 16 },
      {
        opacity: 1, y: 0,
        duration: 0.32,
        ease: 'power2.out',
        stagger: 0.04,
        delay,
        clearProps: 'all',
      }
    );
  };

  /* ─── PANEL ENTRANCE ON SCREEN SHOW ────────────────────────────── */
  /**
   * When a screen becomes active, stagger its panels in with springs.
   */
  function staggerPanels(screenEl) {
    if (reducedMotion()) return;
    const panels = screenEl.querySelectorAll('.panel');
    gsap.fromTo(panels,
      { opacity: 0, y: 20 },
      {
        opacity: 1, y: 0,
        duration: 0.38,
        ease: 'power2.out',
        stagger: 0.055,
        clearProps: 'all',
        delay: 0.1,
      }
    );
  }

  /* ─── SCROLL CONTAINER MOMENTUM HINT ───────────────────────────── */
  /**
   * On momentum-capable scrollable panels, add a subtle rubber-band
   * resistance visual when the user hits the top/bottom boundary.
   * This is Apple Principle 6: momentum projection — the boundary resists.
   */
  function initScrollMomentum() {
    document.querySelectorAll('.sprint-col, .race-col, .terminal-log').forEach(el => {
      let lastY = 0;
      el.addEventListener('scroll', () => {
        const atTop = el.scrollTop <= 0;
        const atBot = el.scrollTop + el.clientHeight >= el.scrollHeight - 2;
        if (atTop && el.scrollTop < lastY) {
          springTo(el.firstElementChild, { y: 6  }, { duration: 0.12, ease: 'power2.out' });
          setTimeout(() => springTo(el.firstElementChild, { y: 0 }, SPRING.crisp), 80);
        } else if (atBot) {
          springTo(el.lastElementChild,  { y: -6 }, { duration: 0.12, ease: 'power2.out' });
          setTimeout(() => springTo(el.lastElementChild, { y: 0 }, SPRING.crisp), 80);
        }
        lastY = el.scrollTop;
      }, { passive: true });
    });
  }

  /* ─── BUTTON RIPPLE (Apple-style press feedback for CTA buttons) ── */
  function initButtonRipple() {
    document.querySelectorAll('.btn-primary, .btn-fantasy').forEach(btn => {
      btn.addEventListener('pointerdown', e => {
        const rect = btn.getBoundingClientRect();
        const ripple = document.createElement('span');
        const size = Math.max(rect.width, rect.height) * 1.6;
        ripple.style.cssText = `
          position: absolute;
          border-radius: 50%;
          width: ${size}px;
          height: ${size}px;
          left: ${e.clientX - rect.left - size / 2}px;
          top:  ${e.clientY - rect.top  - size / 2}px;
          background: rgba(255,255,255,0.18);
          pointer-events: none;
          transform: scale(0);
        `;
        btn.style.overflow = 'hidden';
        btn.style.position = btn.style.position || 'relative';
        btn.appendChild(ripple);

        if (!reducedMotion()) {
          gsap.to(ripple, {
            scale: 1, opacity: 0, duration: 0.5,
            ease: 'power2.out',
            onComplete: () => ripple.remove(),
          });
        } else {
          ripple.remove();
        }
      });
    });
  }

  /* ─── INIT ──────────────────────────────────────────────────────── */
  function init() {
    // Wait for GSAP to be available
    if (typeof gsap === 'undefined') {
      console.warn('[apple-springs] GSAP not found — spring animations disabled.');
      return;
    }

    // Phase 9: Launch screen entrance
    animateLaunchEntrance();

    // Wait for DOM to be fully interactive (app.js runs after us)
    requestAnimationFrame(() => {
      // Phase 1: Pointer-down press states
      attachPressState('.btn, .btn-primary, .btn-fantasy, .btn-danger, .tab-btn, .legend-toggle', 0.96);
      attachPressState('.nav-btn', 0.93);
      attachPressState('.card, .driver-compact-card, .dt-card', 0.97);
      attachPressState('.settings-cache-row button', 0.95);

      // Phase 2: Panel hover springs
      attachPanelHover();
      attachCardHover();

      // Phase 3: Directional nav transitions
      patchNavigation();

      // Phase 4: Tab spring transitions
      attachTabSprings();

      // Phase 7: Status dot pulse
      initStatusPulse();

      // Phase 8: Sliding nav indicator
      initNavIndicator();

      // Scroll momentum
      initScrollMomentum();

      // Button ripple
      initButtonRipple();

      // Stagger initial dashboard panels
      const dashScreen = document.getElementById('dashboard-screen');
      if (dashScreen && dashScreen.classList.contains('active')) {
        staggerPanels(dashScreen);
      }
    });

    // Listen for screen changes to stagger-animate incoming panels (Disabled for debugging)
    /*
    const screenObserver = new MutationObserver(mutations => {
      mutations.forEach(m => {
        if (m.type === 'attributes' && m.attributeName === 'class') {
          const el = m.target;
          if (el.classList.contains('screen') && el.classList.contains('active')) {
            staggerPanels(el);
          }
        }
      });
    });
    document.querySelectorAll('.screen').forEach(s => {
      screenObserver.observe(s, { attributes: true });
    });
    */
  }

  // Run after DOM ready, and after GSAP is parsed
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
