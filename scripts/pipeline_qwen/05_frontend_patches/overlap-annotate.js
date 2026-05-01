// Paint graph nodes based on their `overlap_bucket` field.
//
// The vendored circuit-tracer frontend renders each node as a D3-bound
// <text class="node"> element plus an adjacent <circle> hit-target inside
// a <g class="link-graph">. D3 stores the bound JSON node on element.__data__.
//
// Strategy: read the bucket from __data__ (not from a lookup map by id),
// and set a data-overlap attribute so CSS in overlap-colors.css paints it.
// Re-run on every DOM mutation to catch re-renders when switching graphs.
(function () {
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

    function updateLegendCounts() {
        if (typeof d3 === 'undefined') return;
        const counts = { shared_with_bare: 0, jb_unique: 0, bare: 0 };
        d3.selectAll('.link-graph text.node').each(function (d) {
            if (d && d.overlap_bucket && counts.hasOwnProperty(d.overlap_bucket)) {
                counts[d.overlap_bucket]++;
            }
        });
        const elShared = document.getElementById('overlap-count-shared');
        const elJb = document.getElementById('overlap-count-jb');
        const elBare = document.getElementById('overlap-count-bare');
        if (elShared) elShared.textContent = counts.shared_with_bare;
        if (elJb) elJb.textContent = counts.jb_unique;
        if (elBare) elBare.textContent = counts.bare;
    }

    function tick() {
        const n = annotate();
        if (n > 0) {                                                                                                            
            console.log('[overlap] annotated ' + n + ' new elements');
        }                                                                                                                       
        updateLegendCounts();  // always refresh counts, even when nothing new
    }

    function buildLegend() {
        if (document.querySelector('.overlap-legend')) return;
        const legend = document.createElement('div');
        legend.className = 'overlap-legend';
        legend.innerHTML = `
            <b>Overlap (feature counts)</b>
            <div><span class="sw" style="background:#2e7d32"></span>Shared with bare<span class="count" id="overlap-count-shared">0</span></div>
            <div><span class="sw" style="background:#e65100"></span>JB unique (subcircuit)<span class="count" id="overlap-count-jb">0</span></div>
            <div><span class="sw" style="background:#37474f"></span>Bare<span class="count" id="overlap-count-bare">0</span></div>
        `;
        document.body.appendChild(legend);
    }

    // Wait for DOM + D3 + initial render, then annotate repeatedly.
    // setInterval is crude but robust to the vendored JS's arbitrary render order.
    window.addEventListener('load', function () {
        buildLegend();
        setTimeout(tick, 500);
        setInterval(tick, 1500);
    });
})();
