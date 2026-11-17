'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
    DEFAULTS,
    computeMetrics,
    createCounter,
} = require('../static/sit_to_stand_counter.js');

function metric(kneeAngle, hipY, torsoHeight = 0.20, thighHorizontalTorso = 0.25) {
    return {
        valid: true,
        kneeAngle,
        hipY,
        torsoHeight,
        thighHorizontalTorso,
        sidesUsed: 2,
    };
}

function feed(counter, value, frames) {
    let result;
    for (let index = 0; index < frames; index += 1) result = counter.update(value);
    return result;
}

function calibratedCounter() {
    const counter = createCounter({
        calibrationFrames: 4,
        stableFrames: 2,
        smoothing: 1,
    });
    const result = feed(counter, metric(100, 0.65), 4);
    assert.equal(result.calibrated, true);
    assert.equal(result.phase, 'down');
    return counter;
}

function landmarkPose(kneeAngle, hipY, hiddenSide = null) {
    const landmarks = Array.from({ length: 33 }, () => ({
        x: 0.5,
        y: 0.5,
        visibility: 0,
    }));
    const radians = kneeAngle * Math.PI / 180;
    const sides = [
        { name: 'left', shoulder: 11, hip: 23, knee: 25, ankle: 27, x: 0.42, direction: 1 },
        { name: 'right', shoulder: 12, hip: 24, knee: 26, ankle: 28, x: 0.58, direction: -1 },
    ];
    for (const side of sides) {
        const visibility = side.name === hiddenSide ? 0.1 : 1;
        landmarks[side.shoulder] = { x: side.x, y: hipY - 0.20, visibility };
        landmarks[side.hip] = { x: side.x, y: hipY, visibility };
        landmarks[side.knee] = { x: side.x, y: hipY + 0.20, visibility };
        landmarks[side.ankle] = {
            x: side.x + side.direction * Math.sin(radians) * 0.20,
            y: hipY + 0.20 - Math.cos(radians) * 0.20,
            visibility,
        };
    }
    return landmarks;
}

test('pose metrics use knee angle and tolerate one occluded side', () => {
    const metrics = computeMetrics(landmarkPose(165, 0.55, 'right'));
    assert.equal(metrics.valid, true);
    assert.equal(metrics.sidesUsed, 1);
    assert.ok(Math.abs(metrics.kneeAngle - 165) < 0.001);
    assert.ok(Math.abs(metrics.hipY - 0.55) < 0.001);
    assert.ok(Math.abs(metrics.torsoHeight - 0.20) < 0.001);
});

test('pose metrics prefer the more visible side instead of averaging a guessed far side', () => {
    const pose = landmarkPose(165, 0.55);
    const farSide = landmarkPose(105, 0.60);
    for (const index of [12, 24, 26, 28]) {
        pose[index] = { ...farSide[index], visibility: 0.60 };
    }
    for (const index of [11, 23, 25, 27]) pose[index].visibility = 0.95;

    const metrics = computeMetrics(pose);
    assert.equal(metrics.sidesUsed, 2);
    assert.ok(Math.abs(metrics.kneeAngle - 165) < 0.001);
    assert.ok(Math.abs(metrics.hipY - 0.55) < 0.001);
    assert.equal(metrics.confidence, 0.95);
});

test('seated calibration waits for a stable seated window', () => {
    const counter = createCounter({ calibrationFrames: 4, smoothing: 1 });
    [0.65, 0.60, 0.65, 0.60].forEach(hipY => counter.update(metric(100, hipY)));
    assert.equal(counter.state.baselineHipY, null);

    const result = feed(counter, metric(100, 0.65), 4);
    assert.equal(result.calibrated, true);
    assert.equal(result.phase, 'down');
});

test('seated calibration accepts chair geometry when the projected knee angle is high', () => {
    const counter = createCounter({ calibrationFrames: 4, smoothing: 1 });
    const result = feed(counter, metric(165, 0.65, 0.20, 0.30), 4);

    assert.equal(result.calibrated, true);
    assert.equal(result.phase, 'down');
});

test('standing requires both knee extension and hip rise', () => {
    const counter = calibratedCounter();

    feed(counter, metric(DEFAULTS.standingKneeMin - 1, 0.50, 0.20, 0.05), 3);
    assert.equal(counter.state.repCount, 0, 'hip rise alone must not count');

    feed(counter, metric(170, 0.62, 0.20, 0.05), 3);
    assert.equal(counter.state.repCount, 0, 'knee extension alone must not count');
});

test('a longer occlusion clears stale smoothing before the next valid pose', () => {
    const counter = calibratedCounter();
    counter.update(metric(165, 0.55, 0.20, 0.05));
    feed(counter, { valid: false, reason: '遮擋' }, 3);
    counter.update(metric(100, 0.65));

    assert.equal(counter.state.smoothedKneeAngle, 100);
    assert.equal(counter.state.smoothedHipY, 0.65);
});

test('stable-frame hysteresis counts five complete stands including the final stand', () => {
    const counter = calibratedCounter();
    const standing = metric(165, 0.55, 0.20, 0.05);
    const seated = metric(105, 0.65);

    for (let repetition = 1; repetition <= 5; repetition += 1) {
        const firstStandingFrame = counter.update(standing);
        assert.equal(firstStandingFrame.counted, false);
        const stableStanding = counter.update(standing);
        assert.equal(stableStanding.counted, true);
        assert.equal(stableStanding.repCount, repetition);

        if (repetition < 5) {
            const returned = feed(counter, seated, 2);
            assert.equal(returned.phase, 'down');
        }
    }

    assert.equal(counter.state.repCount, 5);
    assert.equal(counter.state.phase, 'up');
});
