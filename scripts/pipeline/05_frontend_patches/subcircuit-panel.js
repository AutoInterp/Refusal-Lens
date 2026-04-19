// Stage 07 subcircuit filter panel.
//
// Reads `d.subcircuits` (array of subcircuit names) from D3's __data__ binding
// on feature nodes. No extra fetch — memberships are baked into each graph JSON
// by utils_viz.annotate_subcircuits() at stage_frontend() time.
//
// UX:
//   - right-rail panel with all 11 subcircuits, feature counts from current graph
//   - hover a row → preview-highlight (only when nothing pinned)
//   - click/check a row → pin highlight (dim non-members)
//   - multiple pins = union of highlighted members
//   - "Clear" resets all pins
(function () {
    const PANEL_ID = 'subcircuit-panel';
    const BODY_FILTER = 'sc-filter-active';
    const BODY_HOVER = 'sc-hover-active';

    // Per-subcircuit color. Keep synchronized with README §9 narrative:
    //   universal=green, canonical=orange, sign-flip=red, dampening=blue,
    //   anti-amp=purple, late-wave=grey, class-exclusives get distinct hues.
    const COLORS = {
        universal_refusal_core: '#2e7d32',
        canonical_pro_refusal: '#e65100',
        sign_flip_convergent: '#c62828',
        dampening_specialists: '#1565c0',
        anti_refusal_amplifiers: '#6a1b9a',
        late_wave_layer24_32: '#616161',
        roleplay_exclusive: '#f57c00',
        fiction_exclusive: '#7b1fa2',
        cognitive_reframe_exclusive: '#006064',
        completion_exclusive: '#e91e63',
        analytical_exclusive: '#8b4513',
    };

    // Canonical display order — research-load-bearing first, class-exclusives last
    const ORDER = [
        'universal_refusal_core',
        'canonical_pro_refusal',
        'sign_flip_convergent',
        'dampening_specialists',
        'anti_refusal_amplifiers',
        'late_wave_layer24_32',
        'roleplay_exclusive',
        'fiction_exclusive',
        'cognitive_reframe_exclusive',
        'completion_exclusive',
        'analytical_exclusive',
    ];

    const pinned = new Set();

    function countFromGraph() {
        const counts = Object.fromEntries(ORDER.map(n => [n, 0]));
        if (typeof d3 === 'undefined') return counts;
        d3.selectAll('.link-graph text.node').each(function (d) {
            if (!d || !Array.isArray(d.subcircuits)) return;
            for (const sc of d.subcircuits) {
                if (counts[sc] !== undefined) counts[sc]++;
            }
        });
        return counts;
    }

    function paintPinned() {
        if (typeof d3 === 'undefined') return;
        const anyPinned = pinned.size > 0;
        d3.selectAll('.link-graph text.node, .link-graph circle').each(function (d) {
            if (!d || !Array.isArray(d.subcircuits) || d.subcircuits.length === 0) {
                this.removeAttribute('data-sc-match');
                return;
            }
            const hit = d.subcircuits.some(s => pinned.has(s));
            if (hit) this.setAttribute('data-sc-match', '');
            else this.removeAttribute('data-sc-match');
        });
        document.body.classList.toggle(BODY_FILTER, anyPinned);
    }

    function paintHover(scName) {
        if (typeof d3 === 'undefined') return;
        d3.selectAll('.link-graph text.node, .link-graph circle').each(function (d) {
            if (!d || !Array.isArray(d.subcircuits)) {
                this.removeAttribute('data-sc-hover');
                return;
            }
            if (d.subcircuits.includes(scName)) this.setAttribute('data-sc-hover', '');
            else this.removeAttribute('data-sc-hover');
        });
        document.body.classList.add(BODY_HOVER);
    }

    function clearHover() {
        document.body.classList.remove(BODY_HOVER);
        document.querySelectorAll('[data-sc-hover]').forEach(el => {
            el.removeAttribute('data-sc-hover');
        });
    }

    function refreshCounts() {
        const counts = countFromGraph();
        for (const name of ORDER) {
            const el = document.getElementById(`sc-count-${name}`);
            if (el) el.textContent = counts[name];
        }
        // Panel stays visible even when counts are all zero — on first ticks the
        // graph JSON may still be loading, so zeros are transient, not a signal
        // to hide the control. If a graph has no annotations at all, zeros
        // themselves tell the user that (rather than the control vanishing).
    }

    function buildPanel() {
        if (document.getElementById(PANEL_ID)) return;
        const panel = document.createElement('div');
        panel.id = PANEL_ID;
        panel.className = 'subcircuit-panel';

        const header = document.createElement('header');
        const title = document.createElement('h3');
        title.textContent = 'Subcircuits · Stage 07';
        const clearBtn = document.createElement('button');
        clearBtn.className = 'toggle-all';
        clearBtn.type = 'button';
        clearBtn.textContent = 'Clear';
        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            pinned.clear();
            document.querySelectorAll(`#${PANEL_ID} li`).forEach(li =>
                li.classList.remove('active')
            );
            document.querySelectorAll(`#${PANEL_ID} input[type=checkbox]`).forEach(cb => {
                cb.checked = false;
            });
            paintPinned();
        });
        header.appendChild(title);
        header.appendChild(clearBtn);
        panel.appendChild(header);

        const ul = document.createElement('ul');
        for (const name of ORDER) {
            const li = document.createElement('li');
            li.dataset.sc = name;

            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.addEventListener('change', (e) => {
                e.stopPropagation();
                if (cb.checked) pinned.add(name);
                else pinned.delete(name);
                li.classList.toggle('active', cb.checked);
                paintPinned();
            });

            const sw = document.createElement('span');
            sw.className = 'sw';
            sw.style.background = COLORS[name] || '#888';

            const nm = document.createElement('span');
            nm.className = 'name';
            nm.textContent = name;

            const ct = document.createElement('span');
            ct.className = 'count';
            ct.id = `sc-count-${name}`;
            ct.textContent = '0';

            li.appendChild(cb);
            li.appendChild(sw);
            li.appendChild(nm);
            li.appendChild(ct);

            li.addEventListener('mouseenter', () => {
                if (pinned.size === 0) paintHover(name);
            });
            li.addEventListener('mouseleave', clearHover);
            li.addEventListener('click', (e) => {
                if (e.target === cb) return;
                cb.checked = !cb.checked;
                cb.dispatchEvent(new Event('change'));
            });

            ul.appendChild(li);
        }
        panel.appendChild(ul);

        const hint = document.createElement('div');
        hint.className = 'hint';
        hint.textContent = 'Hover row → preview. Click/check → pin. Multiple pins = union. Overlap colors stay applied underneath.';
        panel.appendChild(hint);

        document.body.appendChild(panel);
    }

    function tick() {
        refreshCounts();
        if (pinned.size > 0) paintPinned();  // re-tag nodes after graph re-render
    }

    window.addEventListener('load', function () {
        buildPanel();
        setTimeout(tick, 600);
        setInterval(tick, 1500);
    });
})();
