// Paint graph nodes based on their `overlap_bucket` field.
//
// The vendored circuit-tracer frontend renders each node as a D3-bound
// <text class="node"> element plus an adjacent <circle> hit-target inside
// a <g class="link-graph">. D3 stores the bound JSON node on element.__data__.
//
// Strategy: read the bucket from __data__, set a data-overlap attribute so
// CSS in overlap-colors.css paints it. Re-run on every DOM mutation to catch
// re-renders when switching graphs.
//
// Apr 22 — extended for 3-way bare/ctrl/jb overlap buckets. Legend adapts
// to whichever buckets are actually present in the current graph.
(function () {
    const ALL_BUCKETS = [
        'shared_with_bare_and_ctrl',
        'shared_with_bare',
        'shared_with_ctrl',
        'jb_unique',
        'bare',
        'ctrl',
        'ctrl_unique',
    ];
    const BUCKET_META = {
        shared_with_bare_and_ctrl: {
            color: '#1b5e20',
            label: 'Shared: bare ∩ ctrl ∩ jb',
            note: 'most-stable refusal core',
        },
        shared_with_bare: {
            color: '#2e7d32',
            label: 'Shared with bare',
            note: null,
        },
        shared_with_ctrl: {
            color: '#f9a825',
            label: 'Shared with ctrl',
            note: 'PREFIX-induced (not JB-semantic)',
        },
        jb_unique: {
            color: '#e65100',
            label: 'JB unique',
            note: 'true JB-semantic signal',
        },
        bare: { color: '#37474f', label: 'Bare', note: null },
        ctrl: { color: '#546e7a', label: 'Ctrl', note: null },
        ctrl_unique: {
            color: '#8e24aa',
            label: 'Ctrl unique',
            note: 'benign-prefix-only',
        },
    };

    function annotate() {
        if (typeof d3 === 'undefined') return 0;
        let count = 0;
        d3.selectAll('.link-graph text.node, .link-graph circle').each(function (d) {
            if (!d || !d.overlap_bucket) return;
            const current = this.getAttribute('data-overlap');
            if (current !== d.overlap_bucket) {
                this.setAttribute('data-overlap', d.overlap_bucket);
                count++;
            }
        });
        return count;
    }

    function computeCounts() {
        const counts = {};
        for (const b of ALL_BUCKETS) counts[b] = 0;
        if (typeof d3 === 'undefined') return counts;
        d3.selectAll('.link-graph text.node').each(function (d) {
            if (d && d.overlap_bucket && counts.hasOwnProperty(d.overlap_bucket)) {
                counts[d.overlap_bucket]++;
            }
        });
        return counts;
    }

    function rebuildLegend() {
        const counts = computeCounts();
        const present = ALL_BUCKETS.filter(b => counts[b] > 0);
        let legend = document.querySelector('.overlap-legend');
        if (!legend) {
            legend = document.createElement('div');
            legend.className = 'overlap-legend';
            document.body.appendChild(legend);
        }
        if (present.length === 0) {
            legend.innerHTML = '<b>Overlap</b><div style="color:#888">(no annotated features)</div>';
            return;
        }
        const rows = present.map(b => {
            const m = BUCKET_META[b];
            const note = m.note ? `<span class="note">${m.note}</span>` : '';
            return `<div><span class="sw" style="background:${m.color}"></span>`
                + `${m.label} <span class="count">${counts[b]}</span>${note}</div>`;
        });
        legend.innerHTML = '<b>Overlap (feature counts)</b>' + rows.join('');
    }

    function tick() {
        const n = annotate();
        if (n > 0) {
            console.log('[overlap] annotated ' + n + ' new elements');
        }
        rebuildLegend();
    }

    // Wait for DOM + D3 + initial render, then annotate repeatedly.
    // setInterval is crude but robust to the vendored JS's arbitrary render order.
    window.addEventListener('load', function () {
        setTimeout(tick, 500);
        setInterval(tick, 1500);
    });
})();
