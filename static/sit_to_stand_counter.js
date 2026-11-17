(function (root, factory) {
    const api = factory();
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    if (root) root.SitToStandCounter = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    const DEFAULTS = Object.freeze({
        calibrationFrames: 6,
        stableFrames: 2,
        calibrationSeatedKneeMax: 155,
        calibrationSeatedThighMin: 0.18,
        calibrationKneeRangeMax: 12,
        calibrationHipRangeTorsoMax: 0.10,
        seatedKneeMax: 145,
        seatedKneeBaselineMargin: 14,
        seatedThighMin: 0.15,
        standingKneeMin: 150,
        standingRiseTorso: 0.22,
        seatedReturnTorso: 0.20,
        smoothing: 0.50,
        resetSmoothingAfterInvalidFrames: 3,
    });

    function median(values) {
        const sorted = [...values].sort((a, b) => a - b);
        const middle = Math.floor(sorted.length / 2);
        return sorted.length % 2
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) / 2;
    }

    function distance(a, b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    function angleDegrees(a, b, c) {
        const ab = { x: a.x - b.x, y: a.y - b.y };
        const cb = { x: c.x - b.x, y: c.y - b.y };
        const denominator = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
        if (denominator < 1e-6) return null;
        const cosine = Math.max(-1, Math.min(1, (ab.x * cb.x + ab.y * cb.y) / denominator));
        return Math.acos(cosine) * 180 / Math.PI;
    }

    function visible(point, minimum = 0.5) {
        return Boolean(point)
            && Number.isFinite(point.x)
            && Number.isFinite(point.y)
            && (point.visibility ?? 1) >= minimum;
    }

    function computeMetrics(landmarks) {
        if (!landmarks || landmarks.length < 29) {
            return { valid: false, reason: '找不到完整骨架' };
        }
        const sides = [
            { shoulder: 11, hip: 23, knee: 25, ankle: 27 },
            { shoulder: 12, hip: 24, knee: 26, ankle: 28 },
        ];
        const measurements = [];
        for (const side of sides) {
            const shoulder = landmarks[side.shoulder];
            const hip = landmarks[side.hip];
            const knee = landmarks[side.knee];
            const ankle = landmarks[side.ankle];
            if (![shoulder, hip, knee, ankle].every(point => visible(point))) continue;
            const kneeAngle = angleDegrees(hip, knee, ankle);
            const torsoHeight = distance(shoulder, hip);
            if (kneeAngle === null || torsoHeight < 0.04) continue;
            const thighHorizontalTorso = Math.abs(knee.x - hip.x) / torsoHeight;
            const confidence = Math.min(
                shoulder.visibility ?? 1,
                hip.visibility ?? 1,
                knee.visibility ?? 1,
                ankle.visibility ?? 1
            );
            measurements.push({
                kneeAngle,
                hipY: hip.y,
                torsoHeight,
                thighHorizontalTorso,
                confidence,
            });
        }
        if (!measurements.length) {
            return { valid: false, reason: '請讓髖、膝與腳踝完整入鏡' };
        }
        measurements.sort((left, right) => right.confidence - left.confidence);
        const best = measurements[0];
        return {
            valid: true,
            kneeAngle: best.kneeAngle,
            hipY: best.hipY,
            torsoHeight: best.torsoHeight,
            thighHorizontalTorso: best.thighHorizontalTorso,
            sidesUsed: measurements.length,
            confidence: best.confidence,
        };
    }

    function createCounter(options = {}) {
        const config = { ...DEFAULTS, ...options };
        const state = {
            phase: 'unknown',
            candidate: null,
            candidateFrames: 0,
            repCount: 0,
            baselineHipY: null,
            baselineTorso: null,
            baselineKneeAngle: null,
            calibration: [],
            smoothedKneeAngle: null,
            smoothedHipY: null,
            smoothedThighHorizontalTorso: null,
            invalidFrames: 0,
        };

        function snapshot(extra = {}) {
            return {
                phase: state.phase,
                repCount: state.repCount,
                calibrated: state.baselineHipY !== null,
                counted: false,
                ...extra,
            };
        }

        function update(metrics) {
            if (!metrics || !metrics.valid) {
                state.candidate = null;
                state.candidateFrames = 0;
                state.invalidFrames += 1;
                if (state.invalidFrames >= config.resetSmoothingAfterInvalidFrames) {
                    state.smoothedKneeAngle = null;
                    state.smoothedHipY = null;
                    state.smoothedThighHorizontalTorso = null;
                }
                return snapshot({ reason: metrics?.reason || '請完整入鏡' });
            }

            state.invalidFrames = 0;

            const alpha = config.smoothing;
            state.smoothedKneeAngle = state.smoothedKneeAngle === null
                ? metrics.kneeAngle
                : state.smoothedKneeAngle * (1 - alpha) + metrics.kneeAngle * alpha;
            state.smoothedHipY = state.smoothedHipY === null
                ? metrics.hipY
                : state.smoothedHipY * (1 - alpha) + metrics.hipY * alpha;
            state.smoothedThighHorizontalTorso = state.smoothedThighHorizontalTorso === null
                ? metrics.thighHorizontalTorso
                : state.smoothedThighHorizontalTorso * (1 - alpha)
                    + metrics.thighHorizontalTorso * alpha;

            if (state.baselineHipY === null) {
                const looksSeated = state.smoothedKneeAngle <= config.calibrationSeatedKneeMax
                    || state.smoothedThighHorizontalTorso >= config.calibrationSeatedThighMin;
                if (!looksSeated) {
                    state.calibration = [];
                    return snapshot({ reason: '請先坐穩，系統正在校正坐姿' });
                }
                state.calibration.push({
                    hipY: state.smoothedHipY,
                    kneeAngle: state.smoothedKneeAngle,
                    torsoHeight: metrics.torsoHeight,
                });
                if (state.calibration.length > config.calibrationFrames) {
                    state.calibration.shift();
                }
                if (state.calibration.length < config.calibrationFrames) {
                    return snapshot({ reason: `坐姿校正 ${state.calibration.length}/${config.calibrationFrames}` });
                }
                const calibrationTorso = median(
                    state.calibration.map(item => item.torsoHeight)
                );
                const kneeAngles = state.calibration.map(item => item.kneeAngle);
                const hipPositions = state.calibration.map(item => item.hipY);
                const kneeRange = Math.max(...kneeAngles) - Math.min(...kneeAngles);
                const hipRangeTorso = (Math.max(...hipPositions) - Math.min(...hipPositions))
                    / Math.max(calibrationTorso, 0.04);
                if (kneeRange > config.calibrationKneeRangeMax
                    || hipRangeTorso > config.calibrationHipRangeTorsoMax) {
                    return snapshot({ reason: '請先坐穩，校正期間不要起身' });
                }
                state.baselineHipY = median(state.calibration.map(item => item.hipY));
                state.baselineTorso = calibrationTorso;
                state.baselineKneeAngle = median(kneeAngles);
                state.phase = 'down';
                return snapshot({ reason: '已辨識坐姿，請完整站直' });
            }

            const hipRise = (state.baselineHipY - state.smoothedHipY)
                / Math.max(state.baselineTorso, 0.04);
            const seatedKneeThreshold = Math.min(
                config.calibrationSeatedKneeMax,
                Math.max(
                    config.seatedKneeMax,
                    state.baselineKneeAngle + config.seatedKneeBaselineMargin
                )
            );
            let candidate = null;
            if (state.smoothedKneeAngle >= config.standingKneeMin
                && hipRise >= config.standingRiseTorso) {
                candidate = 'up';
            } else if ((state.smoothedKneeAngle <= seatedKneeThreshold
                    || state.smoothedThighHorizontalTorso >= config.seatedThighMin)
                && hipRise <= config.seatedReturnTorso) {
                candidate = 'down';
            }

            if (candidate === null) {
                state.candidate = null;
                state.candidateFrames = 0;
                const reason = state.phase === 'up'
                    ? '請完整坐回椅面並停穩'
                    : '請將髖部抬高並完全站直';
                return snapshot({ reason, hipRise, kneeAngle: state.smoothedKneeAngle });
            }

            if (candidate === state.candidate) state.candidateFrames += 1;
            else {
                state.candidate = candidate;
                state.candidateFrames = 1;
            }
            if (state.candidateFrames < config.stableFrames || state.phase === candidate) {
                return snapshot({
                    reason: candidate === 'up' ? '站直確認中' : '坐穩確認中',
                    hipRise,
                    kneeAngle: state.smoothedKneeAngle,
                });
            }

            let counted = false;
            if (candidate === 'up' && state.phase === 'down') {
                state.repCount += 1;
                counted = true;
            }
            state.phase = candidate;
            state.candidate = null;
            state.candidateFrames = 0;
            if (candidate === 'down') {
                state.baselineHipY = state.baselineHipY * 0.9 + state.smoothedHipY * 0.1;
                state.baselineTorso = state.baselineTorso * 0.9 + metrics.torsoHeight * 0.1;
                state.baselineKneeAngle = state.baselineKneeAngle * 0.9
                    + state.smoothedKneeAngle * 0.1;
            }
            return snapshot({
                counted,
                reason: candidate === 'up' ? '已辨識站直，請坐回椅面' : '已辨識坐姿，請再次站起',
                hipRise,
                kneeAngle: state.smoothedKneeAngle,
            });
        }

        return { update, state };
    }

    return { DEFAULTS, angleDegrees, computeMetrics, createCounter };
});
