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
            short: 'b∩c∩j',
            note: 'most-stable refusal core',
        },
        shared_with_bare: {
            color: '#2e7d32',
            label: 'Shared with bare',
            short: 'b∩j',
            note: null,
        },
        shared_with_ctrl: {
            color: '#f9a825',
            label: 'Shared with ctrl',
            short: 'c∩j',
            note: 'PREFIX-induced (not JB-semantic)',
        },
        jb_unique: {
            color: '#e65100',
            label: 'JB unique',
            short: 'jb',
            note: 'true JB-semantic signal',
        },
        bare: { color: '#37474f', label: 'Bare', short: 'bare', note: null },
        ctrl: { color: '#546e7a', label: 'Ctrl', short: 'ctrl', note: null },
        ctrl_unique: {
            color: '#8e24aa',
            label: 'Ctrl unique',
            short: 'ctrl-only',
            note: 'benign-prefix-only',
        },
    };

    // Vendor nav layout (built by index.html ~L59-125):
    //   .nav  (flex, justify space-between)
    //     ├─ .controls-container   (graph select + .slider-container)
    //     └─ .save-button
    // We insert the legend between controls-container and save-button.
    function findInsertionAnchor() {
        const saveBtn = document.querySelector('.nav .save-button');
        if (saveBtn && saveBtn.parentNode) {
            return { parent: saveBtn.parentNode, before: saveBtn };
        }
        const nav = document.querySelector('.nav');
        if (nav) return { parent: nav, before: null };
        return null;
    }

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
        const anchor = findInsertionAnchor();
        let legend = document.querySelector('.overlap-legend');

        // Re-place the legend if the vendor nav re-rendered and orphaned it,
        // or if the legend was previously in body-fallback mode and the nav has
        // since appeared.
        if (legend && anchor && legend.parentNode !== anchor.parent) {
            legend.remove();
            legend = null;
        }

        if (!legend) {
            legend = document.createElement('div');
            legend.className = 'overlap-legend';
            if (anchor) {
                legend.classList.add('overlap-legend-inline');
                if (anchor.before) anchor.parent.insertBefore(legend, anchor.before);
                else anchor.parent.appendChild(legend);
            } else {
                // Fallback: nav not yet built. Park at top-right floating.
                document.body.appendChild(legend);
            }
        }

        const inline = legend.classList.contains('overlap-legend-inline');

        if (present.length === 0) {
            // Inline: keep silent (no clutter in the top bar). Floating fallback:
            // surface the "no annotations" hint so user knows the panel is alive.
            legend.innerHTML = inline
                ? ''
                : '<b>Overlap</b><div style="color:#888">(no annotated features)</div>';
            return;
        }

        if (inline) {
            const items = present.map(b => {
                const m = BUCKET_META[b];
                const tip = m.note ? `${m.label} — ${m.note}` : m.label;
                return `<span class="legend-item" title="${tip}">`
                    + `<span class="sw" style="background:${m.color}"></span>`
                    + `<span class="lab">${m.short}</span>`
                    + `<span class="cnt">${counts[b]}</span>`
                    + `</span>`;
            });
            legend.innerHTML = '<span class="legend-label">Overlap</span>' + items.join('');
        } else {
            const rows = present.map(b => {
                const m = BUCKET_META[b];
                const note = m.note ? `<span class="note">${m.note}</span>` : '';
                return `<div><span class="sw" style="background:${m.color}"></span>`
                    + `${m.label} <span class="count">${counts[b]}</span>${note}</div>`;
            });
            legend.innerHTML = '<b>Overlap (feature counts)</b>' + rows.join('');
        }
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
