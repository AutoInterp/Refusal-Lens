/* Sets html.rl-compact synchronously when ?compact=1 is present, so the
   compact-mode.css rules hide the nav chrome and fill the column regardless of
   when the gridsnap viewer builds its DOM. No-op without the query param. */
(function () {
  try {
    if (new URLSearchParams(location.search).get("compact")) {
      document.documentElement.classList.add("rl-compact");
    }
  } catch (e) { /* older browsers: leave full UI */ }
})();
