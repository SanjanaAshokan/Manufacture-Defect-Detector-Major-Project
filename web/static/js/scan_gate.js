/* Live-camera "hold steady" gate.
 *
 * The camera page samples the picture every ~120 ms *locally* (cheap) and only asks the
 * server to analyse once the scene has been calm for a few samples - one request per
 * steady moment instead of one per frame. After the server says "that's not a
 * transistor" the gate stays shut until the scene changes, so a person sitting in front
 * of the camera is not re-sent over and over.
 */
(function (root) {
    function createScanGate(opts) {
        const o = Object.assign({ baseThreshold: 8, stillNeeded: 4, relaxEveryMs: 4000, maxRelax: 3 }, opts || {});
        let still = 0;
        let blocked = false;

        // The longer nothing steady has been found, the more movement is tolerated
        // (hand tremor), up to maxRelax x the base threshold.
        function threshold(waitedMs) {
            return o.baseThreshold * Math.min(o.maxRelax, 1 + 0.5 * Math.floor((waitedMs || 0) / o.relaxEveryMs));
        }

        return {
            threshold: threshold,
            /** Feed one motion measurement (mean per-channel pixel difference, 0-255).
             *  Returns true when it is time to send the frame for analysis. */
            sample: function (diff, waitedMs) {
                if (diff < threshold(waitedMs)) {
                    still++;
                } else {
                    still = 0;
                    blocked = false;       // the scene changed: allow a new attempt
                }
                return !blocked && still >= o.stillNeeded;
            },
            /** The server rejected this scene: stay quiet until something moves. */
            block: function () { blocked = true; still = 0; },
            /** Allow another attempt immediately (e.g. after a network hiccup). */
            reset: function () { blocked = false; still = 0; },
            get stillCount() { return still; },
            get isBlocked() { return blocked; }
        };
    }
    root.createScanGate = createScanGate;
    if (typeof module !== 'undefined' && module.exports) module.exports = { createScanGate: createScanGate };
})(typeof window !== 'undefined' ? window : globalThis);
