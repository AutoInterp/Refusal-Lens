/* Paint graph nodes by their baked `rl_trace_class` field (refusal_centric /
   suppression / amplification / neutral). Mirrors overlap-annotate.js: the
   vendored circuit-tracer frontend renders each node as a D3-bound
   <text class="node"> + adjacent <circle> inside <g class="link-graph">, with
   the bound JSON node on element.__data__. We set data-rl-trace so
   trace-highlight.css paints it, and re-run on DOM mutations to catch graph
   switches and gridsnap re-renders.

   v2: upstream-propagated nodes (not themselves a seed class) are colored by
   `rl_trace_upstream_class` instead, faded by hop distance via inline opacity
   (1/(1+hop)), and bordered by `rl_trace_mechanism` (active_inhibitor / mixed /
   passive_cascade). window.rlSetDepth(d) hides nodes with hop > d; trace.html's
   toolbar #depth-slider input calls rlSetDepth(this.value) on each iframe's
   window so the slider progressively reveals upstream hops in both panels. */
(function () {
  let maxHop = 0;
  window.rlSetDepth = function (d) { maxHop = +d; paint(); };
  function paint() {
    if (typeof d3 === "undefined") return 0;
    let count = 0;
    d3.selectAll(".link-graph text.node, .link-graph circle").each(function (d) {
      if (!d) return;
      const seed = d.rl_trace_class;                 // v1 seed class (may be neutral)
      const up = d.rl_trace_upstream_class;          // v2 upstream class
      const hop = (typeof d.rl_trace_hop === "number") ? d.rl_trace_hop : null;
      const mech = d.rl_trace_mechanism || null;
      if (seed && seed !== "neutral") {
        if (this.getAttribute("data-rl-trace") !== seed) this.setAttribute("data-rl-trace", seed);
      } else if (up) {
        if (this.getAttribute("data-rl-upstream") !== up) this.setAttribute("data-rl-upstream", up);
      } else if (seed) {
        if (this.getAttribute("data-rl-trace") !== seed) this.setAttribute("data-rl-trace", seed);
      }
      if (mech) this.setAttribute("data-rl-mech", mech);
      if (hop !== null) {
        this.setAttribute("data-rl-hop", String(hop));
        this.style.opacity = String(1 / (1 + hop));
        this.classList.toggle("rl-hop-hidden", hop > maxHop);
      }
      count++;
    });
    return count;
  }
  function start() {
    paint();
    const obs = new MutationObserver(() => paint());
    obs.observe(document.body, { childList: true, subtree: true });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
