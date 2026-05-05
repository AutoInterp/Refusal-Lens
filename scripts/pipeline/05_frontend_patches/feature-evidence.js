// Floating evidence tooltip on hover. Reads `preview_top_logits` and
// `preview_examples` baked into each CLT node by inject_feature_evidence(),
// looks them up via the d3-bound datum on the hovered DOM element, and
// renders a positioned floating panel. Lets the user verify what a feature
// actually fires on without trusting the (sometimes-misleading) label.
(function () {
  const TIP_SELECTORS = '.feature-row, .pp-clerp, .feature-title, .clerp-list .feature';

  const tip = document.createElement('div');
  tip.id = 'feature-evidence-tip';
  tip.style.cssText = [
    'position:fixed', 'pointer-events:none',
    'background:#fff', 'border:1px solid #c8c8c8', 'border-radius:4px',
    'padding:8px 10px', 'box-shadow:0 4px 14px rgba(0,0,0,0.18)',
    'font:11px/1.45 ui-monospace, "SF Mono", Menlo, Consolas, monospace',
    'color:#222', 'max-width:520px', 'z-index:99999', 'display:none',
    'white-space:normal',
  ].join(';');
  document.body.appendChild(tip);

  function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, ch => (
      {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]
    ));
  }

  function showWhitespace(s) {
    return String(s ?? '').replace(/\n/g, '⏎').replace(/\t/g, '→').replace(/\r/g, '↵');
  }

  // Walk up the DOM looking for a d3-bound datum with the evidence we want.
  function findDatum(el) {
    let cur = el;
    while (cur && cur !== document.body) {
      const d = cur.__data__;
      if (d && (d.preview_top_logits || d.preview_examples || d.feature_type)) return d;
      cur = cur.parentElement;
    }
    return null;
  }

  function renderTip(d) {
    if (!d) return null;
    if (d.feature_type !== 'cross layer transcoder') return null;
    const logits = d.preview_top_logits || [];
    const examples = d.preview_examples || [];
    if (!logits.length && !examples.length) return null;

    let html = '';
    const title = d.ppClerp || d.clerp;
    if (title) {
      html += `<div style="font-weight:600;margin-bottom:2px;color:#111">${escapeHtml(title)}</div>`;
    }
    html += `<div style="color:#888;font-size:10px;margin-bottom:6px">L${escapeHtml(d.layer)} · F#${escapeHtml(d.feature)}</div>`;

    if (logits.length) {
      const chips = logits.map(t => `<code style="background:#f3f3f3;padding:1px 4px;border-radius:2px;color:#333">${escapeHtml(t)}</code>`).join(' ');
      html += `<div style="margin-bottom:6px"><span style="color:#666">Top logits:</span> ${chips}</div>`;
    }

    if (examples.length) {
      html += `<div style="color:#666;margin-bottom:2px">Top activations:</div>`;
      for (const ex of examples) {
        const ctx = showWhitespace(ex.context || '');
        const trig = showWhitespace(ex.trigger_token || '');
        const act = (typeof ex.trigger_activation === 'number')
          ? ex.trigger_activation.toFixed(0) : '';
        html += `
          <div style="margin:2px 0 4px;padding:3px 6px;border-left:3px solid #f60;background:#fafafa">
            <div style="color:#888;font-size:10px;margin-bottom:1px">
              trigger «<b style="color:#f60">${escapeHtml(trig)}</b>» · act ${act}
            </div>
            <div style="color:#222">${escapeHtml(ctx).slice(0, 260)}</div>
          </div>`;
      }
    }
    return html;
  }

  let lastEl = null;

  function move(ev) {
    if (tip.style.display === 'none') return;
    const pad = 14;
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let x = ev.clientX + pad;
    let y = ev.clientY + pad;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - pad;
    tip.style.left = Math.max(4, x) + 'px';
    tip.style.top = Math.max(4, y) + 'px';
  }

  document.addEventListener('mouseover', (ev) => {
    const target = ev.target.closest(TIP_SELECTORS);
    if (!target) return;
    if (target === lastEl) return;
    lastEl = target;
    const d = findDatum(target);
    const html = renderTip(d);
    if (!html) {
      tip.style.display = 'none';
      return;
    }
    tip.innerHTML = html;
    tip.style.display = 'block';
    // Strip any title attr so the native browser tooltip doesn't double up.
    if (target.hasAttribute('title')) {
      target.dataset.origTitle = target.getAttribute('title');
      target.removeAttribute('title');
    }
    move(ev);
  });

  document.addEventListener('mousemove', move);

  document.addEventListener('mouseout', (ev) => {
    const next = ev.relatedTarget && ev.relatedTarget.closest && ev.relatedTarget.closest(TIP_SELECTORS);
    if (next) return;
    tip.style.display = 'none';
    if (lastEl && lastEl.dataset.origTitle != null) {
      lastEl.setAttribute('title', lastEl.dataset.origTitle);
      delete lastEl.dataset.origTitle;
    }
    lastEl = null;
  });
})();
