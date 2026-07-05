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
   passive_cascade). window.rlSetDepth(d) reveals upstream color only up to hop d
   — nodes beyond the current depth stay v1 neutral-gray (never hidden), so depth 0
   reproduces the exact v1 view; trace.html's toolbar #depth-slider input calls
   rlSetDepth(this.value) on each iframe's window to color deeper hops in both panels. */
(function () {
  let maxHop = 0;
  window.rlSetDepth = function (d) { maxHop = +d; paint(); };
  function paint() {
    if (typeof d3 === "undefined") return 0;
    let count = 0;
    d3.selectAll(".link-graph text.node, .link-graph circle").each(function (d) {
      if (!d) return;
      const seed = d.rl_trace_class;
      const up = d.rl_trace_upstream_class;
      const hop = (typeof d.rl_trace_hop === "number") ? d.rl_trace_hop : null;
      const mech = d.rl_trace_mechanism || null;
      const seedColored = seed && seed !== "neutral";
      const showUp = up && hop !== null && hop <= maxHop;   // reveal upstream color only within depth
      if (seedColored) {
        if (this.getAttribute("data-rl-trace") !== seed) this.setAttribute("data-rl-trace", seed);
        this.removeAttribute("data-rl-upstream");
        this.style.opacity = "";
      } else if (showUp) {
        if (this.getAttribute("data-rl-upstream") !== up) this.setAttribute("data-rl-upstream", up);
        this.removeAttribute("data-rl-trace");
        this.style.opacity = String(1 / hop);               // hop-1 fully opaque; deeper hops fade
      } else {
        const nv = seed || "neutral";                        // beyond depth OR neutral -> v1 gray
        if (this.getAttribute("data-rl-trace") !== nv) this.setAttribute("data-rl-trace", nv);
        this.removeAttribute("data-rl-upstream");
        this.style.opacity = "";
      }
      if (mech && (seedColored || showUp)) this.setAttribute("data-rl-mech", mech);
      else this.removeAttribute("data-rl-mech");
      if (hop !== null) this.setAttribute("data-rl-hop", String(hop));
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
