const test = require('node:test');
const assert = require('node:assert/strict');
const readiness = require('../static/action_readiness.js');

function point(x, y, visibility = 0.99) {
    return { x, y, z: 0, visibility };
}

function pose(kind = 'standing') {
    const landmarks = Array.from({ length: 33 }, () => point(0.5, 0.5, 0));
    const seated = kind === 'seated';
    landmarks[11] = point(0.43, seated ? 0.35 : 0.22);
    landmarks[12] = point(0.57, seated ? 0.35 : 0.22);
    landmarks[23] = point(0.44, seated ? 0.55 : 0.48);
    landmarks[24] = point(0.56, seated ? 0.55 : 0.48);
    landmarks[25] = point(seated ? 0.66 : 0.45, seated ? 0.60 : 0.68);
    landmarks[26] = point(seated ? 0.34 : 0.55, seated ? 0.60 : 0.68);
    landmarks[27] = point(seated ? 0.66 : 0.46, 0.88);
    landmarks[28] = point(seated ? 0.34 : 0.54, 0.88);
    return landmarks;
}

test('sit readiness requires a stable seated pose', () => {
    const gate = readiness.createGate('sit', { stableFrames: 3 });
    assert.equal(gate.update(pose('standing')).ready, false);
    assert.match(gate.update(pose('standing')).reason, /坐穩/);
    assert.equal(gate.update(pose('seated')).ready, false);
    assert.equal(gate.update(pose('seated')).ready, false);
    assert.equal(gate.update(pose('seated')).ready, true);
});

test('walk and squat readiness require stable standing', () => {
    for (const actionId of ['walk', 'squat']) {
        const gate = readiness.createGate(actionId, { stableFrames: 2 });
        assert.equal(gate.update(pose('seated')).ready, false);
        assert.equal(gate.update(pose('standing')).ready, false);
        assert.equal(gate.update(pose('standing')).ready, true);
    }
});

test('readiness resets when required landmarks leave the frame', () => {
    const gate = readiness.createGate('walk', { stableFrames: 2 });
    assert.equal(gate.update(pose('standing')).ready, false);
    const missingAnkles = pose('standing');
    missingAnkles[27].visibility = 0.1;
    missingAnkles[28].visibility = 0.1;
    assert.equal(gate.update(missingAnkles).ready, false);
    assert.match(gate.update(missingAnkles).reason, /腳踝/);
    assert.equal(gate.update(pose('standing')).ready, false);
});

test('operator override provides an explicit fallback', () => {
    const gate = readiness.createGate('sit');
    const result = gate.forceReady();
    assert.equal(result.ready, true);
    assert.equal(result.forced, true);
    assert.match(result.reason, /工作人員/);
});
