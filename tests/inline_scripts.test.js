// 檢查 index.html 內嵌腳本可以被解析：字串中的真實換行等語法錯誤會讓整個凱比頁面失效，
// 而以原始碼字串比對的測試抓不到這類問題。
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

for (const page of ['index.html', 'control.html']) {
    test(`${page} inline scripts parse`, () => {
        const html = fs.readFileSync(path.join(__dirname, '..', 'static', page), 'utf8');
        const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
        assert.ok(scripts.length > 0);
        for (const source of scripts) {
            assert.doesNotThrow(() => new vm.Script(source, { filename: page }));
        }
    });
}
