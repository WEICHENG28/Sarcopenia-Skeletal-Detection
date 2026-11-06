// Exercise the actual inline application script with deterministic browser/device fakes.
// These regressions do not replace a real camera / MediaPipe smoke test.
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function app() {
    let nextId = 0;
    const timers = new Map();
    const elements = new Map();
    const recorders = [];
    const requests = [];
    function element(id) {
        if (!elements.has(id)) elements.set(id, {
            style: {}, classList: { add() {}, remove() {}, toggle() {} },
            textContent: '', innerText: '', disabled: false,
            getContext: () => ({ clearRect() {} }),
        });
        return elements.get(id);
    }
    class MediaRecorder {
        static isTypeSupported() { return false; }
        constructor(stream, options) {
            this.state = 'inactive';
            this.mimeType = options.mimeType;
            recorders.push(this);
        }
        start() { this.state = 'recording'; }
        stop() { this.state = 'inactive'; }
        // Browser delivers dataavailable/onstop after stop() returns.
        finish(text) {
            this.ondataavailable({ data: new Blob([text]) });
            this.onstop();
        }
    }
    const context = vm.createContext({
        window: { location: { origin: 'http://localhost:8000' } },
        document: { getElementById: element },
        console: { warn() {} }, MediaRecorder, Blob, FormData,
        confirm: () => true, performance: { now: () => 0 },
        setInterval: () => ++nextId, clearInterval() {},
        setTimeout: fn => { const id = ++nextId; timers.set(id, fn); return id; },
        clearTimeout: id => timers.delete(id),
        requestAnimationFrame: () => ++nextId,
        fetch: async (url, options) => {
            requests.push({ url, ...options });
            return { ok: true, json: async () => ({ uploaded: true }) };
        },
    });
    const run = code => vm.runInContext(code, context);
    run(script);
    return {
        run, timers, recorders, requests, element,
        start: action => run(`toggleRecording('${action}', BTN_ID_MAP['${action}'], ACTION_LABEL['${action}'])`),
        flush() { for (const [id, fn] of [...timers]) { timers.delete(id); fn(); } },
        frame(time, y = 0.5, x = 0.5, visible = true) {
            run(`evaluatePose({landmarks: [Array.from({length: 33}, () => ({
                x: ${x}, y: ${y}, visibility: ${visible ? 1 : 0}
            }))]}, ${time})`);
        },
    };
}

test('walking preparation, pauses and completed movement never auto-stop', () => {
    const a = app();
    a.start('walk');
    a.frame(0);
    a.frame(0.1, 0.5, 0.51);
    for (let t = 0.2; t <= 10; t += 0.1) a.frame(t, 0.5, 0.51);
    for (let t = 10; t <= 15; t += 0.1) a.frame(t, 0.5, 0.51 + (t - 10) * 0.05);
    for (let t = 15; t <= 30; t += 0.1) a.frame(t, 0.5, 0.76);
    assert.equal(a.timers.size, 0);
    assert.equal(a.recorders[0].state, 'recording');
    assert.match(a.element('liveRepBadge').textContent, /手動停止/);
});

test('squat gives one completion after three seconds and stays recording', () => {
    const a = app();
    a.start('squat');
    for (let i = 0; i < 30; i++) a.frame(i / 30, 0.4);
    for (let i = 30; i < 150; i++) a.frame(i / 30, 0.65);
    assert.equal(a.run('live.holdCompleted'), true);
    assert.match(a.element('liveRepBadge').textContent, /維持完成/);
    for (let rep = 0; rep < 5; rep++) {
        for (let i = 0; i < 60; i++) a.frame(5 + rep * 2 + i / 30, i < 30 ? 0.4 : 0.65);
    }
    assert.equal(a.run('live.repCount'), 0);
    assert.equal(a.timers.size, 0);
    assert.equal(a.recorders[0].state, 'recording');
});

test('a squat hold under three seconds does not complete and resets on rising', () => {
    const a = app();
    a.start('squat');
    a.run("live.repState='down'; handleSquatHold(1); handleSquatHold(3.9)");
    assert.equal(a.run('live.holdCompleted'), false);
    a.run("live.repState='up'; handleSquatHold(4); live.repState='down'; handleSquatHold(5); handleSquatHold(7.9)");
    assert.equal(a.run('live.holdCompleted'), false);
    a.run('handleSquatHold(8)');
    assert.equal(a.run('live.holdCompleted'), true);
});

for (const loss of ['missing', 'low visibility']) {
    test(`a ${loss} skeleton interrupts the squat hold`, () => {
        const a = app();
        a.start('squat');
        a.run("live.repState='down'; handleSquatHold(1)");
        if (loss === 'missing') a.run('evaluatePose({landmarks: []}, 2)');
        else a.frame(2, 0.5, 0.5, false);
        a.run("live.repState='down'; handleSquatHold(5)");
        assert.equal(a.run('live.holdCompleted'), false);
        a.run('handleSquatHold(8)');
        assert.equal(a.run('live.holdCompleted'), true);
    });
}

test('five sit-to-stand repetitions auto-stop and save the recording', async () => {
    const a = app();
    a.start('sit');
    for (let rep = 0; rep < 5; rep++) {
        for (let i = 0; i < 60; i++) a.frame(rep * 2 + i / 30, i < 30 ? 0.65 : 0.35);
    }
    assert.equal(a.run('live.repCount'), 5);
    assert.equal(a.timers.size, 1);
    a.flush();
    assert.equal(a.recorders[0].state, 'inactive');
    a.recorders[0].finish('sit video');
    assert.equal(await a.run('videoData.sit.text()'), 'sit video');
    assert.equal(a.element('btnWalk').disabled, false);
});

for (const nextAction of ['walk', 'sit']) {
    test(`a cancelled sit timer cannot stop a new ${nextAction} recording`, () => {
        const a = app();
        a.start('sit');
        a.run("live.repCount=5; maybeAutoStopByReps('sit',10)");
        const staleCallback = [...a.timers.values()][0];
        a.start('sit'); // Manual stop.
        assert.equal(a.timers.size, 0);
        a.start(nextAction);
        staleCallback(); // Even a callback already queued by the browser is harmless.
        assert.equal(a.recorders[1].state, 'recording');
    });
}

test('stillness stop is also scoped to its original recorder', () => {
    const a = app();
    a.start('sit');
    a.run('live.repCount=1; live.hasMoved=true; live.stillSince=1; maybeAutoStopByStillness(5,true,4,5)');
    const staleCallback = [...a.timers.values()][0];
    a.start('sit');
    a.start('squat');
    staleCallback();
    assert.equal(a.recorders[1].state, 'recording');
});

test('delayed recording events keep each video separate and upload the saved blob', async () => {
    const a = app();
    a.start('sit');
    a.start('sit');
    a.start('walk');
    a.recorders[0].finish('first video');
    a.start('walk');
    a.recorders[1].finish('second video');
    assert.equal(await a.run('videoData.sit.text()'), 'first video');
    assert.equal(await a.run('videoData.walk.text()'), 'second video');
    await a.run("uploadVideo('sit', {user_id:'TEST', age:70, height_cm:160, weight_kg:60, gender:'female'})");
    assert.equal(a.requests[0].body.get('action'), 'Sit');
    assert.equal(await a.requests[0].body.get('video_file').text(), 'first video');
});

test('pose inference failure cancels auto-stop while manual recording still works', async () => {
    const a = app();
    a.start('sit');
    a.run("live.repCount=5; maybeAutoStopByReps('sit',10)");
    Object.assign(a.element('webcam'), { readyState: 2, videoWidth: 640, videoHeight: 480 });
    a.run("poseLandmarker={detectForVideo(){throw new Error('GPU lost')}}; poseDetectLoop(10000)");
    assert.equal(a.timers.size, 0);
    assert.equal(a.recorders[0].state, 'recording');
    assert.match(a.element('liveBanner').textContent, /手動停止/);
    a.start('sit');
    a.recorders[0].finish('fallback video');
    assert.equal(await a.run('videoData.sit.text()'), 'fallback video');
});

test('module loading failure leaves manual recording usable', async () => {
    const a = app();
    // This VM has no dynamic import loader, simulating a failed CDN/module load.
    await a.run('initPoseTracking()');
    a.start('walk');
    a.start('walk');
    a.recorders[0].finish('offline video');
    assert.equal(await a.run('videoData.walk.text()'), 'offline video');
});
