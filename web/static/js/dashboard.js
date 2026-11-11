/* Dashboard pages: reveal animation for bars + interactive training-curve charts (no libraries). */
(function () {
    'use strict';

    // ---- Reveal bars when they scroll into view ----
    var reveals = document.querySelectorAll('[data-reveal]');
    if ('IntersectionObserver' in window) {
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                if (e.isIntersecting) { e.target.classList.add('in-view'); io.unobserve(e.target); }
            });
        }, { threshold: 0.25 });
        reveals.forEach(function (el) { io.observe(el); });
    } else {
        reveals.forEach(function (el) { el.classList.add('in-view'); });
    }

    // ---- Training curves ----
    var dataEl = document.getElementById('chartData');
    var holders = document.querySelectorAll('[data-curve-chart]');
    if (!dataEl || !holders.length) return;
    var data;
    try { data = JSON.parse(dataEl.textContent); } catch (e) { return; }
    var curves = data.curves || [];
    var NS = 'http://www.w3.org/2000/svg';

    function el(name, attrs, parent) {
        var n = document.createElementNS(NS, name);
        Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
        if (parent) parent.appendChild(n);
        return n;
    }
    var LABELS = { train_loss: 'Train loss', val_loss: 'Validation loss', train_acc: 'Train accuracy', val_acc: 'Validation accuracy' };

    function draw(holder) {
        var metrics = (holder.getAttribute('data-metrics') || holder.getAttribute('data-metric') || 'val_loss').split(',');
        var percent = holder.getAttribute('data-percent') === '1';
        var series = [];
        curves.forEach(function (c) {
            metrics.forEach(function (m) {
                var ys = c.history && c.history[m];
                if (ys && ys.length) series.push({ key: m, curve: c, ys: ys, dashed: m.indexOf('train') === 0 });
            });
        });
        if (!series.length) { holder.textContent = 'No training history recorded yet.'; return; }

        var W = Math.max(320, holder.clientWidth || 560), H = 300, L = 46, R = 16, T = 16, B = 36;
        var maxLen = Math.max.apply(null, series.map(function (s) { return s.ys.length; }));
        var all = [].concat.apply([], series.map(function (s) { return s.ys; }));
        var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
        if (percent) { lo = Math.min(lo, 0.5); hi = 1; }
        var pad = (hi - lo) * 0.08 || 0.05; if (!percent) lo = Math.max(0, lo - pad); hi += pad;
        function X(i) { return L + (maxLen === 1 ? 0 : i * (W - L - R) / (maxLen - 1)); }
        function Y(v) { return T + (hi - v) * (H - T - B) / (hi - lo); }
        function fmt(v) { return percent ? (v * 100).toFixed(0) + '%' : v.toFixed(2); }

        holder.innerHTML = '';
        var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'curve-svg', role: 'img' }, holder);
        for (var g = 0; g <= 4; g++) {
            var gv = lo + (hi - lo) * g / 4, gy = Y(gv);
            el('line', { x1: L, x2: W - R, y1: gy, y2: gy, class: 'curve-grid' }, svg);
            var t = el('text', { x: L - 6, y: gy + 4, class: 'curve-axis', 'text-anchor': 'end' }, svg); t.textContent = fmt(gv);
        }
        var step = Math.max(1, Math.ceil(maxLen / 10));
        for (var i = 0; i < maxLen; i += step) {
            var tx = el('text', { x: X(i), y: H - 12, class: 'curve-axis', 'text-anchor': 'middle' }, svg); tx.textContent = i + 1;
        }
        var ax = el('text', { x: (W + L) / 2, y: H - 1, class: 'curve-axis', 'text-anchor': 'middle' }, svg); ax.textContent = 'Epoch';

        series.forEach(function (s) {
            var d = s.ys.map(function (v, i) { return (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1); }).join(' ');
            var p = el('path', { d: d, class: 'curve-line' + (s.dashed ? ' dashed' : ''), stroke: s.curve.colour }, svg);
            var len = p.getTotalLength ? p.getTotalLength() : 0;
            if (len && !s.dashed) { p.style.strokeDasharray = len; p.style.strokeDashoffset = len; p.getBoundingClientRect(); p.style.transition = 'stroke-dashoffset 1.2s ease'; p.style.strokeDashoffset = 0; }
            if (!s.dashed && s.curve.best_epoch && s.ys[s.curve.best_epoch - 1] !== undefined && s.key.indexOf('val') === 0) {
                el('circle', { cx: X(s.curve.best_epoch - 1), cy: Y(s.ys[s.curve.best_epoch - 1]), r: 5.5, class: 'curve-best', stroke: s.curve.colour }, svg);
            }
        });

        // hover read-out
        var cross = el('line', { y1: T, y2: H - B, class: 'curve-cross', style: 'opacity:0' }, svg);
        var tip = document.createElement('div'); tip.className = 'curve-tip'; holder.appendChild(tip);
        var hit = el('rect', { x: L, y: T, width: W - L - R, height: H - T - B, fill: 'transparent' }, svg);
        hit.addEventListener('mousemove', function (ev) {
            var r = svg.getBoundingClientRect();
            var px = (ev.clientX - r.left) * W / r.width;
            var idx = Math.max(0, Math.min(maxLen - 1, Math.round((px - L) * (maxLen - 1) / (W - L - R))));
            cross.setAttribute('x1', X(idx)); cross.setAttribute('x2', X(idx)); cross.style.opacity = 1;
            var rows = series.filter(function (s) { return s.ys[idx] !== undefined; }).map(function (s) {
                var name = curves.length > 1 ? s.curve.label : LABELS[s.key];
                return '<span style="color:' + s.curve.colour + '">●</span> ' + name + ': <b>' + fmt(s.ys[idx]) + '</b>';
            });
            tip.innerHTML = '<div>Epoch ' + (idx + 1) + '</div>' + rows.join('<br>');
            tip.style.opacity = 1;
            tip.style.left = Math.min(r.width - 150, Math.max(0, (X(idx) / W) * r.width + 12)) + 'px';
        });
        hit.addEventListener('mouseleave', function () { cross.style.opacity = 0; tip.style.opacity = 0; });

        var legend = document.createElement('div'); legend.className = 'bar-legend';
        var seen = {};
        series.forEach(function (s) {
            var name = curves.length > 1 ? s.curve.label : LABELS[s.key];
            if (seen[name]) return; seen[name] = 1;
            var i = document.createElement('span'); i.className = 'legend-item'; i.style.setProperty('--m-colour', s.curve.colour);
            i.innerHTML = '<span class="dash-dot"' + (s.dashed ? ' style="opacity:.5"' : '') + '></span>' + name; legend.appendChild(i);
        });
        holder.appendChild(legend);
    }
    holders.forEach(draw);
})();
