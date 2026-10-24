/* Shared helpers for the Batch / Compare / Lab / Line pages. */
window.DefectTools = (function () {
    function esc(value) {
        return String(value == null ? '' : value)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    /** POST one image to an endpoint; resolves {ok, status, data} and never throws on HTTP errors. */
    async function postImage(url, blob, filename, params) {
        const fd = new FormData();
        fd.append('image', blob, filename || 'image.png');
        Object.keys(params || {}).forEach((k) => fd.append(k, params[k]));
        try {
            const res = await fetch(url, { method: 'POST', body: fd });
            let data = null;
            try { data = await res.json(); } catch (e) { data = { error: 'Unexpected server response.' }; }
            return { ok: res.ok && data && data.success !== false, status: res.status, data: data };
        } catch (e) {
            return { ok: false, status: 0, data: { error: 'Network error — is the server running?' } };
        }
    }

    async function urlToFile(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error('Could not load ' + url);
        const blob = await res.blob();
        return new File([blob], url.substring(url.lastIndexOf('/') + 1), { type: blob.type || 'image/png' });
    }

    function loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error('Image failed to load'));
            img.src = src;
        });
    }

    function fileToImage(file) {
        const url = URL.createObjectURL(file);
        return loadImage(url).then((img) => { img._objectUrl = url; return img; });
    }

    /** Make a canvas crisp on high-DPI screens; returns a 2D context in CSS pixels. */
    function prepCanvas(canvas) {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth || 600, h = canvas.clientHeight || 260;
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        const ctx = canvas.getContext('2d');
        if (ctx && ctx.setTransform) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return { ctx: ctx, w: w, h: h };
    }

    function cssVar(name, fallback) {
        const v = getComputedStyle(document.body).getPropertyValue(name).trim();
        return v || fallback;
    }

    /**
     * Minimal line chart. series: [{label, color, values:[number|null], dashed?}]
     * opts: {xLabels, yMin, yMax, hLines:[{y,label,color}], pointColors?:[[color|null]...], yFormat}
     */
    function lineChart(canvas, series, opts) {
        const p = prepCanvas(canvas);
        const ctx = p.ctx;
        if (!ctx) return;
        const W = p.w, H = p.h, padL = 44, padR = 14, padT = 14, padB = 32;
        const text = cssVar('--text-secondary', '#9ca3af');
        const grid = cssVar('--border-color', 'rgba(255,255,255,0.1)');
        const yMin = opts.yMin, yMax = opts.yMax;
        const n = opts.xLabels.length;
        const x = (i) => padL + (n <= 1 ? 0 : (i * (W - padL - padR)) / (n - 1));
        const y = (v) => padT + (1 - (v - yMin) / (yMax - yMin)) * (H - padT - padB);
        const fmt = opts.yFormat || ((v) => String(Math.round(v * 100) / 100));

        ctx.clearRect(0, 0, W, H);
        ctx.font = '11px sans-serif';
        ctx.fillStyle = text;
        ctx.strokeStyle = grid;
        ctx.lineWidth = 1;
        for (let t = 0; t <= 4; t++) {
            const v = yMin + ((yMax - yMin) * t) / 4;
            ctx.beginPath(); ctx.moveTo(padL, y(v)); ctx.lineTo(W - padR, y(v)); ctx.stroke();
            ctx.textAlign = 'right'; ctx.fillText(fmt(v), padL - 6, y(v) + 4);
        }
        const step = Math.max(1, Math.ceil(n / 12));
        ctx.textAlign = 'center';
        for (let i = 0; i < n; i += step) ctx.fillText(opts.xLabels[i], x(i), H - 10);

        (opts.hLines || []).forEach((h) => {
            if (h.y == null) return;
            ctx.save();
            ctx.strokeStyle = h.color || '#f59e0b';
            ctx.setLineDash(h.solid ? [] : [6, 4]);
            ctx.beginPath(); ctx.moveTo(padL, y(h.y)); ctx.lineTo(W - padR, y(h.y)); ctx.stroke();
            ctx.fillStyle = h.color || '#f59e0b';
            ctx.textAlign = 'right';
            ctx.fillText(h.label || '', W - padR, y(h.y) - 4);
            ctx.restore();
        });

        series.forEach((s, si) => {
            ctx.save();
            ctx.strokeStyle = s.color; ctx.lineWidth = 2;
            if (s.dashed) ctx.setLineDash([5, 4]);
            ctx.beginPath();
            let pen = false;
            s.values.forEach((v, i) => {
                if (v == null) { pen = false; return; }
                if (!pen) { ctx.moveTo(x(i), y(v)); pen = true; } else ctx.lineTo(x(i), y(v));
            });
            ctx.stroke();
            ctx.setLineDash([]);
            s.values.forEach((v, i) => {
                if (v == null) return;
                const pc = opts.pointColors && opts.pointColors[si] && opts.pointColors[si][i];
                ctx.fillStyle = pc || s.color;
                ctx.beginPath(); ctx.arc(x(i), y(v), pc ? 4.5 : 3, 0, Math.PI * 2); ctx.fill();
            });
            ctx.restore();
        });

        // legend
        let lx = padL + 4;
        series.forEach((s) => {
            ctx.fillStyle = s.color; ctx.fillRect(lx, padT - 2, 12, 3);
            ctx.fillStyle = text; ctx.textAlign = 'left'; ctx.fillText(s.label, lx + 16, padT + 4);
            lx += 24 + ctx.measureText(s.label).width;
        });
    }

    function debounce(fn, ms) {
        let t = null;
        return function () { const a = arguments; clearTimeout(t); t = setTimeout(() => fn.apply(null, a), ms); };
    }

    return { esc, postImage, urlToFile, loadImage, fileToImage, prepCanvas, lineChart, debounce, cssVar };
})();
