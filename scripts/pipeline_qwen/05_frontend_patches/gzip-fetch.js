// Transparent gzip-decompression for graph data fetches.
//
// Activated only when `window.REFUSAL_LENS_USE_GZIP` is true (set by a
// small inline <script> injected by utils_viz.stage_frontend). When active,
// any fetch to /graph_data/*.json is first attempted as `.json.gz` and
// decompressed client-side via the browser's DecompressionStream API.
// Falls through to plain .json if the .gz variant 404s.
//
// This lets us keep graph_data on disk and on HuggingFace in compressed form
// (~12x smaller) while still serving via a dumb `python -m http.server`.
//
// DecompressionStream support: Chrome 80+, Firefox 113+, Safari 16.4+
(function () {
    if (typeof window === 'undefined' || !window.fetch) return;
    if (!window.REFUSAL_LENS_USE_GZIP) return;
    if (typeof DecompressionStream === 'undefined') {
        console.warn('[gzip-fetch] DecompressionStream unavailable; falling back to plain .json');
        return;
    }

    const origFetch = window.fetch.bind(window);

    function urlOf(input) {
        if (typeof input === 'string') return input;
        if (input && typeof input.url === 'string') return input.url;
        return '';
    }

    window.fetch = async function (input, options) {
        const urlStr = urlOf(input);

        // Only intercept per-graph JSON fetches. graph-metadata.json and any
        // other non-graph_data fetches go through unchanged.
        if (!urlStr.includes('/graph_data/') || !urlStr.endsWith('.json')) {
            return origFetch(input, options);
        }

        const gzUrl = urlStr + '.gz';
        let resp;
        try {
            resp = await origFetch(gzUrl, options);
        } catch (e) {
            console.warn('[gzip-fetch] fetch failed for', gzUrl, e);
            return origFetch(input, options);
        }

        if (!resp.ok) {
            // .gz not present — fall back to plain
            return origFetch(input, options);
        }

        const decompressed = resp.body.pipeThrough(new DecompressionStream('gzip'));
        return new Response(decompressed, {
            status: 200,
            statusText: 'OK',
            headers: new Headers({'Content-Type': 'application/json'}),
        });
    };

    console.log('[gzip-fetch] patched window.fetch for transparent .json.gz decompression');
})();
