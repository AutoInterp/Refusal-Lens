/* Paint graph nodes by their baked `rl_trace_class` field (refusal_centric /
   suppression / amplification / neutral). Mirrors overlap-annotate.js: the
   vendored circuit-tracer frontend renders each node as a D3-bound
   <text class="node"> + adjacent <circle> inside <g class="link-graph">, with
   the bound JSON node on element.__data__. We set data-rl-trace so
   trace-highlight.css paints it, and re-run on DOM mutations to catch graph
   switches and gridsnap re-renders. */
(function () {
  function paint() {
    if (typeof d3 === "undefined") return 0;
    let count = 0;
    d3.selectAll(".link-graph text.node, .link-graph circle").each(function (d) {
      if (!d || !d.rl_trace_class) return;
      if (this.getAttribute("data-rl-trace") !== d.rl_trace_class) {
        this.setAttribute("data-rl-trace", d.rl_trace_class);
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
