(function (root, factory) {
    const api = factory();
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    if (root) root.ActionReadiness = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    const LANDMARK = Object.freeze({
        L_SHOULDER: 11, R_SHOULDER: 12,
        L_HIP: 23, R_HIP: 24,
        L_KNEE: 25, R_KNEE: 26,
        L_ANKLE: 27, R_ANKLE: 28,
    });

    const DEFAULTS = Object.freeze({
        stableFrames: 5,
        minimumVisibility: 0.5,
        frameMargin: 0.03,
        seatedKneeMax: 155,
        seatedThighMin: 0.18,
        standingKneeMin: 150,
        maximumHipDrift: 0.035,
    });

    function angleDegrees(a, b, c) {
        const ab = { x: a.x - b.x, y: a.y - b.y };
        const cb = { x: c.x - b.x, y: c.y - b.y };
        const denominator = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
        if (denominator < 1e-6) return null;
        const cosine = Math.max(-1, Math.min(1, (ab.x * cb.x + ab.y * cb.y) / denominator));
        return Math.acos(cosine) * 180 / Math.PI;
    }

    function distance(a, b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    function pointInFrame(point, config) {
        return Boolean(point)
            && Number.isFinite(point.x)
            && Number.isFinite(point.y)
            && (point.visibility ?? 1) >= config.minimumVisibility
            && point.x > config.frameMargin
            && point.x < 1 - config.frameMargin
            && point.y > config.frameMargin
            && point.y < 1 - config.frameMargin;
    }

    function computeMetrics(landmarks, options = {}) {
        const config = { ...DEFAULTS, ...options };
        if (!landmarks || landmarks.length < 29) {
            return { valid: false, reason: '找不到完整骨架' };
        }
        const core = [
            LANDMARK.L_SHOULDER, LANDMARK.R_SHOULDER,
            LANDMARK.L_HIP, LANDMARK.R_HIP,
            LANDMARK.L_KNEE, LANDMARK.R_KNEE,
        ];
        if (!core.every(index => pointInFrame(landmarks[index], config))) {
            return { valid: false, reason: '請讓肩膀、髖部與膝蓋完整入鏡' };
        }

        const sides = [
            { shoulder: LANDMARK.L_SHOULDER, hip: LANDMARK.L_HIP, knee: LANDMARK.L_KNEE, ankle: LANDMARK.L_ANKLE },
            { shoulder: LANDMARK.R_SHOULDER, hip: LANDMARK.R_HIP, knee: LANDMARK.R_KNEE, ankle: LANDMARK.R_ANKLE },
        ];
        const measurements = [];
        for (const side of sides) {
            const shoulder = landmarks[side.shoulder];
            const hip = landmarks[side.hip];
            const knee = landmarks[side.knee];
            const ankle = landmarks[side.ankle];
            if (![shoulder, hip, knee, ankle].every(point => pointInFrame(point, config))) continue;
            const kneeAngle = angleDegrees(hip, knee, ankle);
            const torsoHeight = distance(shoulder, hip);
            if (kneeAngle === null || torsoHeight < 0.04) continue;
            measurements.push({
                kneeAngle,
                hipY: hip.y,
                thighHorizontalTorso: Math.abs(knee.x - hip.x) / torsoHeight,
                confidence: Math.min(
                    shoulder.visibility ?? 1,
                    hip.visibility ?? 1,
                    knee.visibility ?? 1,
                    ankle.visibility ?? 1
                ),
            });
        }
        if (!measurements.length) {
            return { valid: false, reason: '請至少讓一側腳踝完整入鏡' };
        }
        measurements.sort((left, right) => right.confidence - left.confidence);
        return { valid: true, ...measurements[0] };
    }

    function createGate(actionId, options = {}) {
        if (!['sit', 'walk', 'squat'].includes(actionId)) {
            throw new Error(`不支援的動作：${actionId}`);
        }
        const config = { ...DEFAULTS, ...options };
        const state = {
            actionId,
            stableFrames: 0,
            lastHipY: null,
            forced: false,
        };

        function snapshot(ready, reason, extra = {}) {
            return {
                ready,
                reason,
                progress: ready ? 1 : Math.min(1, state.stableFrames / config.stableFrames),
                stableFrames: state.stableFrames,
                requiredFrames: config.stableFrames,
                forced: state.forced,
                ...extra,
            };
        }

        function reset(reason) {
            state.stableFrames = 0;
            state.lastHipY = null;
            return snapshot(false, reason);
        }

        function update(landmarks) {
            if (state.forced) return snapshot(true, '已由工作人員確認就緒');
            const metrics = computeMetrics(landmarks, config);
            if (!metrics.valid) return reset(metrics.reason);

            const postureReady = actionId === 'sit'
                ? metrics.kneeAngle <= config.seatedKneeMax
                    || metrics.thighHorizontalTorso >= config.seatedThighMin
                : metrics.kneeAngle >= config.standingKneeMin;
            if (!postureReady) {
                return reset(actionId === 'sit'
                    ? '請先坐穩，雙腳踩地'
                    : '請站直並保持身體穩定');
            }

            if (state.lastHipY !== null
                && Math.abs(metrics.hipY - state.lastHipY) > config.maximumHipDrift) {
                state.stableFrames = 0;
            }
            state.lastHipY = metrics.hipY;
            state.stableFrames += 1;

            const ready = state.stableFrames >= config.stableFrames;
            const label = actionId === 'sit' ? '坐姿' : '站姿';
            return snapshot(
                ready,
                ready ? `${label}已穩定，可以開始` : `${label}確認中 ${state.stableFrames}/${config.stableFrames}`,
                { metrics }
            );
        }

        function forceReady() {
            state.forced = true;
            state.stableFrames = config.stableFrames;
            return snapshot(true, '已由工作人員確認就緒');
        }

        return { update, forceReady, state };
    }

    return { DEFAULTS, LANDMARK, angleDegrees, computeMetrics, createGate };
});
