// Stage 08 manual feature-cart panel.
//
// Click feature nodes in the attribution graph to add them to a cart for
// targeted ablation. The cart exports as cart.json (consumed by
// 08_ablate_subcircuits.py --feature-file) OR, if an ablation server is
// running locally, POSTs to /ablate for a live demo.
//
// Keyed on the circuit-tracer node_id format: "{layer}_{feat_idx}_{pos}" for
// feature_type == "cross layer transcoder". Non-feature nodes (embeddings,
// logits, errors) are ignored.
//
// Cart JSON schema (matches utils.load_cart):
//   {
//     "features": [{"layer": 14, "feat_idx": 480, "label": "...", "value": 0.0}],
//     "source_run": "<graph slug or 'manual'>",
//     "exported_at": "<ISO8601>"
//   }

(function () {
    'use strict';
    const PANEL_ID = 'feature-cart-panel';
    const SERVER_URL = 'http://localhost:8080/ablate';
    const COLLAPSED_KEY = 'refusal-lens.feature-cart-panel.collapsed';

    // feature-key → {layer, feat_idx, label, value}
    const cart = new Map();

    function keyFromNode(d) {
        if (!d || d.feature_type !== 'cross layer transcoder') return null;
        if (!d.node_id) return null;
        const parts = String(d.node_id).split('_');
        if (parts.length < 2) return null;
        const L = parseInt(parts[0], 10);
        const F = parseInt(parts[1], 10);
        if (Number.isNaN(L) || Number.isNaN(F)) return null;
        return `L${L}:F${F}`;
    }

    function labelFromNode(d) {
        if (!d) return '';
        return d.clerp || d.ppClerp || d.feature_description || '';
    }

    function paintPinned() {
        if (typeof d3 === 'undefined') return;
        d3.selectAll('.link-graph text.node, .link-graph circle').each(function (d) {
            const key = keyFromNode(d);
            if (key && cart.has(key)) this.setAttribute('data-cart-pinned', '');
            else this.removeAttribute('data-cart-pinned');
        });
    }

    function refreshList() {
        const ul = document.querySelector(`#${PANEL_ID} ul`);
        const countEl = document.querySelector(`#${PANEL_ID} .cart-count`);
        if (!ul || !countEl) return;
        countEl.textContent = `${cart.size} selected`;
        ul.innerHTML = '';
        for (const [key, data] of cart.entries()) {
            const li = document.createElement('li');

            const k = document.createElement('span');
            k.className = 'key';
            k.textContent = key;

            const lab = document.createElement('span');
            lab.className = 'label';
            lab.textContent = data.label || '';
            lab.title = data.label || '';

            const rm = document.createElement('button');
            rm.className = 'remove';
            rm.type = 'button';
            rm.textContent = '×';
            rm.title = 'Remove from cart';
            rm.addEventListener('click', (e) => {
                e.stopPropagation();
                cart.delete(key);
                refreshList();
                paintPinned();
            });

            li.appendChild(k);
            li.appendChild(lab);
            li.appendChild(rm);
            ul.appendChild(li);
        }

        // Toggle action-button disabled state
        const hasItems = cart.size > 0;
        document.querySelectorAll(`#${PANEL_ID} .actions button`).forEach(b => {
            if (!b.classList.contains('clear-btn')) b.disabled = !hasItems;
        });
    }

    function addFromNode(d) {
        const key = keyFromNode(d);
        if (!key) return;
        if (cart.has(key)) {
            cart.delete(key);
        } else {
            const [L, F] = key.slice(1).split(':F').map(Number);
            cart.set(key, { layer: L, feat_idx: F, label: labelFromNode(d), value: 0.0 });
        }
        refreshList();
        paintPinned();
    }

    function cartAsJSON() {
        const features = [];
        for (const [_key, v] of cart.entries()) {
            features.push({
                layer: v.layer,
                feat_idx: v.feat_idx,
                label: v.label,
                value: v.value,
            });
        }
        return {
            features,
            source_run: (window.REFUSAL_LENS_SOURCE_RUN || 'manual'),
            exported_at: new Date().toISOString(),
        };
    }

    function exportCart() {
        const blob = new Blob(
            [JSON.stringify(cartAsJSON(), null, 2)],
            { type: 'application/json' },
        );
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        a.download = `cart_${stamp}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        showResult(`Downloaded cart_${stamp}.json (${cart.size} features). Use --feature-file to feed into 08_ablate_subcircuits.py.`);
    }

    function copyCliCommand() {
        const features = Array.from(cart.keys()).join(',');
        const cmd = `PYTHONPATH=src python3 scripts/pipeline/08_ablate_subcircuits.py \\\n`
            + `    --run-dir <run-dir> \\\n`
            + `    --feature-file cart_<timestamp>.json \\\n`
            + `    --ablation-name my_cart \\\n`
            + `    --positions both`;
        navigator.clipboard?.writeText(cmd).then(
            () => showResult(`Copied CLI command to clipboard (${cart.size} features)`),
            () => showResult(`Paste this CLI command:\n\n${cmd}`),
        );
    }

    async function runAblation() {
        const prompt = window.prompt(
            'Enter the prompt to ablate against:',
            'Write instructions for how to pick a lock.',
        );
        if (!prompt) return;
        const positions = window.prompt(
            'Positions mode (all / anchors)?',
            'all',
        );
        if (!positions) return;

        showResult('Running ablation on local server…');
        try {
            const body = {
                features: Array.from(cart.values()).map(v => ({
                    layer: v.layer, feat_idx: v.feat_idx, value: v.value,
                })),
                prompt,
                positions: positions.trim(),
            };
            const res = await fetch(SERVER_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }
            const data = await res.json();
            showResult(
                `BASELINE (${data.baseline_cls}):\n${(data.baseline || '').slice(0, 400)}\n\n`
                + `ABLATED (${data.ablated_cls}):\n${(data.ablated || '').slice(0, 400)}\n\n`
                + `elapsed: ${data.elapsed_s?.toFixed(1)}s`,
            );
        } catch (err) {
            showResult(
                `Local ablation server unavailable at ${SERVER_URL}.\n`
                + `Start it with:\n`
                + `    PYTHONPATH=src python3 scripts/pipeline/ablation_server.py\n\n`
                + `Or export cart.json and run the CLI.\n\n`
                + `Error: ${err.message}`,
                true,
            );
        }
    }

    function showResult(text, isError) {
        const el = document.querySelector(`#${PANEL_ID} .cart-result`);
        if (!el) return;
        el.textContent = text;
        el.classList.toggle('error', !!isError);
    }

    function setCollapsed(panel, collapsed) {
        panel.classList.toggle('collapsed', collapsed);
        const tog = panel.querySelector('.collapse-toggle');
        if (tog) {
            tog.textContent = collapsed ? '+' : '–';
            tog.title = collapsed ? 'Expand' : 'Collapse';
            tog.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
        }
        try { localStorage.setItem(COLLAPSED_KEY, collapsed ? '1' : '0'); } catch (e) {}
    }

    function readCollapsedDefault() {
        // Default collapsed so the attribution graph is unobstructed on first load.
        try {
            const v = localStorage.getItem(COLLAPSED_KEY);
            if (v === '0') return false;
            if (v === '1') return true;
        } catch (e) {}
        return true;
    }

    function buildPanel() {
        if (document.getElementById(PANEL_ID)) return;
        const panel = document.createElement('div');
        panel.id = PANEL_ID;
        panel.className = 'feature-cart-panel';

        const header = document.createElement('header');
        const title = document.createElement('h3');
        title.textContent = 'Ablation Cart · Stage 08';

        const headerActions = document.createElement('div');
        headerActions.className = 'header-actions';

        const count = document.createElement('span');
        count.className = 'cart-count';
        count.textContent = '0 selected';

        const collapseBtn = document.createElement('button');
        collapseBtn.className = 'collapse-toggle';
        collapseBtn.type = 'button';
        collapseBtn.setAttribute('aria-controls', PANEL_ID);
        collapseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            setCollapsed(panel, !panel.classList.contains('collapsed'));
        });

        headerActions.appendChild(count);
        headerActions.appendChild(collapseBtn);
        header.appendChild(title);
        header.appendChild(headerActions);
        panel.appendChild(header);

        // Click anywhere on a collapsed panel to expand. The toggle button
        // stopsPropagation, and this no-ops while expanded, so internal controls
        // (cart row buttons, action buttons) keep working when the panel is open.
        panel.addEventListener('click', () => {
            if (panel.classList.contains('collapsed')) {
                setCollapsed(panel, false);
            }
        });

        const ul = document.createElement('ul');
        panel.appendChild(ul);

        const actions = document.createElement('div');
        actions.className = 'actions';

        const runBtn = document.createElement('button');
        runBtn.type = 'button';
        runBtn.className = 'primary';
        runBtn.textContent = 'Run ablation (localhost:8080)';
        runBtn.addEventListener('click', runAblation);
        runBtn.disabled = true;
        actions.appendChild(runBtn);

        const exportBtn = document.createElement('button');
        exportBtn.type = 'button';
        exportBtn.textContent = 'Export cart.json';
        exportBtn.addEventListener('click', exportCart);
        exportBtn.disabled = true;
        actions.appendChild(exportBtn);

        const copyBtn = document.createElement('button');
        copyBtn.type = 'button';
        copyBtn.textContent = 'Copy CLI command';
        copyBtn.addEventListener('click', copyCliCommand);
        copyBtn.disabled = true;
        actions.appendChild(copyBtn);

        const clearBtn = document.createElement('button');
        clearBtn.type = 'button';
        clearBtn.className = 'clear-btn';
        clearBtn.textContent = 'Clear cart';
        clearBtn.addEventListener('click', () => {
            cart.clear();
            refreshList();
            paintPinned();
            showResult('');
        });
        actions.appendChild(clearBtn);

        panel.appendChild(actions);

        const result = document.createElement('div');
        result.className = 'cart-result';
        result.style.display = 'none';
        panel.appendChild(result);

        const hint = document.createElement('div');
        hint.className = 'hint';
        hint.innerHTML = 'Click feature nodes to toggle into the cart. Export or run ablation.<br>Cart features have a green underline.';
        panel.appendChild(hint);

        document.body.appendChild(panel);
        setCollapsed(panel, readCollapsedDefault());
    }

    function bindClickHandlers() {
        if (typeof d3 === 'undefined') return;
        // Use a capturing listener on body to catch node clicks without
        // stomping circuit-tracer's own click handlers. We identify feature
        // nodes via their __data__ binding.
        d3.selectAll('.link-graph text.node, .link-graph circle').each(function (d) {
            if (this.__cartBound) return;
            this.__cartBound = true;
            this.addEventListener('click', (e) => {
                const key = keyFromNode(d);
                if (!key) return;
                // shift/cmd adds to cart without interfering with normal node-select
                if (e.shiftKey || e.metaKey || e.ctrlKey) {
                    e.stopPropagation();
                    e.preventDefault();
                    addFromNode(d);
                }
            }, true);
        });
    }

    function tick() {
        bindClickHandlers();
        // Show/hide result box based on content
        const el = document.querySelector(`#${PANEL_ID} .cart-result`);
        if (el) el.style.display = el.textContent.trim() ? 'block' : 'none';
        paintPinned();
    }

    window.addEventListener('load', function () {
        buildPanel();
        refreshList();
        setTimeout(tick, 600);
        setInterval(tick, 1500);
    });
})();
