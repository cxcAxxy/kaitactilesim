'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const [url, out, requirePlots] = process.argv.slice(2); fs.mkdirSync(out, {recursive: true});
  const browser = await chromium.launch({executablePath: process.env.KAI_CHROMIUM_EXECUTABLE, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  try {
    const page = await browser.newPage({viewport: {width: 1480, height: 1100}}), errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(url); await page.waitForSelector('.wb-expert-video');
    await page.waitForFunction(() => !document.querySelector('.wb-expert-slider').disabled);
    assert.equal(await page.locator('video').count(), 2);
    assert.equal(await page.locator('.ke-slider').count(), 1);
    assert.equal(await page.locator('.wb-expert-slider').count(), 1);
    assert.equal(await page.locator('.wb-chart').count(), 4);
    if (requirePlots === '--require-diagnostic-plots') assert.equal(await page.locator('.wb-plots-panel').count(), 1, 'new diagnostic workbench must mount the gallery');
    assert.match(await page.locator('.ke-time-heading .ke-kicker').textContent(), /模型/);
    const config = await page.evaluate(() => JSON.parse(document.getElementById('kai-workbench-config').textContent));
    assert.equal(config.expert_video.clock, 'independent_video_seconds');
    assert.match(config.expert_video.comparison_note, /初始/);
    assert(Math.abs(Number(await page.locator('.wb-expert-slider').getAttribute('max')) - 66.9) < .02);
    assert.equal(await page.locator('.wb-expert-video').evaluate(v => v.paused), true, 'no autoplay');
    assert.equal(await page.locator('.ke-video').evaluate(v => v.paused), true);
    async function settleVideo(selector) {
      await page.waitForFunction(s => { const v = document.querySelector(s); return !v.seeking && v.readyState >= 2; }, selector);
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    }
    async function modelSeek(time) {
      await page.locator('.ke-slider').evaluate((s, t) => { s.value = String(t); s.dispatchEvent(new Event('input', {bubbles: true})); }, time);
      await settleVideo('.ke-video');
    }
    async function expertSeek(time) {
      await page.locator('.wb-expert-slider').evaluate((s, t) => { s.value = String(t); s.dispatchEvent(new Event('input', {bubbles: true})); }, time);
      await settleVideo('.wb-expert-video');
    }
    async function modelState() {
      return page.evaluate(() => ({slider: document.querySelector('.ke-slider').value, time: document.querySelector('.ke-video').currentTime,
        paused: document.querySelector('.ke-video').paused, rate: document.querySelector('.ke-video').playbackRate,
        readings: [...document.querySelectorAll('.wb-reading')].map(n => n.textContent)}));
    }
    await modelSeek(30); await expertSeek(35);
    await page.evaluate(() => {
      window.modelInputCount = 0; document.querySelector('.ke-slider').addEventListener('input', () => { window.modelInputCount++; });
    });
    const frozenModel = await modelState();
    await expertSeek(20);
    assert.deepEqual(await modelState(), frozenModel, 'expert seek must not affect model clock/video/load values');
    await page.locator('.wb-expert-slider').focus(); await page.keyboard.press('ArrowRight'); await settleVideo('.wb-expert-video');
    assert(Math.abs(await page.locator('.wb-expert-video').evaluate(v => v.currentTime) - 21) < .01);
    await page.locator('.wb-expert-speed').selectOption('2');
    assert.equal(await page.locator('.ke-video').evaluate(v => v.playbackRate), 1);
    // A rejected play request is recoverable and must not leave a stale warning.
    await page.locator('.wb-expert-video').evaluate(v => {
      const original = v.play;
      v.play = function () { this.play = original; return Promise.reject(new DOMException('Test refusal', 'NotAllowedError')); };
    });
    await page.locator('.wb-expert-play').click();
    assert.match(await page.locator('.wb-expert-status').textContent(), /暂时无法播放/);
    assert.deepEqual(await modelState(), frozenModel);
    await page.locator('.wb-expert-play').click();
    await page.waitForFunction(() => document.querySelector('.wb-expert-video').currentTime > 21.25);
    assert.doesNotMatch(await page.locator('.wb-expert-status').textContent(), /无法播放/);
    await page.locator('.wb-expert-play').click();
    assert.deepEqual(await modelState(), frozenModel, 'expert playback must not advance paused model');
    assert.equal(await page.evaluate(() => window.modelInputCount), 0);
    await page.locator('.wb-expert-heading').click();
    await page.locator('.wb-expert-heading').click();
    assert.deepEqual(await modelState(), frozenModel, 'expert fold/unfold must not seek model');
    assert.equal(await page.evaluate(() => window.modelInputCount), 0, 'expert UI must never dispatch model input, even with unchanged time');
    for (const id of ['mosaic', 'head']) {
      await page.locator(`.wb-expert-camera[data-camera="${id}"]`).click();
      assert.deepEqual(await modelState(), frozenModel);
    }
    const cropRatio = await page.locator('.wb-expert-video').evaluate(v => { const b = v.getBoundingClientRect(); return b.width / b.height; });
    assert(Math.abs(cropRatio - 1920 / 1080) < .002, 'expert crop preserves intrinsic proportions');
    if (await page.locator('.wb-plots-panel').count()) {
      const pausedExpert = await page.locator('.wb-expert-video').evaluate(v => ({time: v.currentTime, paused: v.paused, rate: v.playbackRate}));
      assert.equal(await page.locator('.wb-plots-tabs [role="tab"]').count(), 3);
      for (const id of ['tactile', 'hand-dof', 'wrist']) {
        const tab = page.locator('.wb-plots-tabs [role="tab"]').nth(['tactile', 'hand-dof', 'wrist'].indexOf(id));
        await tab.click();
        assert.equal(await page.locator('.wb-plots-panel').getAttribute('data-plot-id'), id);
        await page.waitForFunction(() => { const img = document.querySelector('.wb-plots-image'); return img.complete && img.naturalWidth > 0; });
        await page.locator('.wb-plots-actions button').click();
        assert.equal(await page.locator('.wb-plot-dialog').evaluate(n => n.open), true);
        await page.locator('.wb-plot-dialog-tools button').filter({hasText: '原尺寸'}).click();
        assert.equal(await page.locator('.wb-plot-dialog-scroll').evaluate(n => n.classList.contains('wb-plot-native')), true);
        await page.keyboard.press('Escape');
        assert.equal(await page.locator('.wb-plot-dialog').evaluate(n => n.open), false);
        assert.deepEqual(await modelState(), frozenModel, 'static gallery must not seek or play model');
        assert.deepEqual(await page.locator('.wb-expert-video').evaluate(v => ({time: v.currentTime, paused: v.paused, rate: v.playbackRate})), pausedExpert, 'static gallery must not seek or play expert');
      }
      assert.equal(await page.evaluate(() => window.modelInputCount), 0, 'static gallery never dispatches model input');
      await page.locator('.wb-plots-tabs [role="tab"]').first().click();
      await page.screenshot({path: path.join(out, 'diagnostic-gallery-desktop.png')});
    }
    const expertPausedTime = await page.locator('.wb-expert-video').evaluate(v => v.currentTime);
    await modelSeek(10);
    assert.equal(await page.locator('.wb-expert-video').evaluate(v => v.currentTime), expertPausedTime, 'model seek must not seek expert');
    await page.locator('.wb-speed select').selectOption('0.5');
    assert.equal(await page.locator('.wb-expert-video').evaluate(v => v.playbackRate), 2);
    await page.locator('.ke-controls .ke-primary').click();
    await page.waitForFunction(() => !document.querySelector('.ke-video').paused && document.querySelector('.ke-video').currentTime > 10.15);
    assert.equal(await page.locator('.wb-expert-video').evaluate(v => v.currentTime), expertPausedTime, 'model playback must not advance paused expert');
    // Expert play, seek and collapsing while model plays must not interrupt model.
    await page.locator('.wb-expert-play').click();
    await page.waitForFunction(() => !document.querySelector('.wb-expert-video').paused);
    await expertSeek(25);
    assert.equal(await page.locator('.ke-video').evaluate(v => v.paused), false);
    await page.locator('.wb-expert-play').click();
    await page.locator('.wb-expert-heading').click();
    await page.waitForFunction(() => document.querySelector('.wb-expert-video').paused);
    assert.equal(await page.locator('.ke-video').evaluate(v => v.paused), false);
    await page.locator('.ke-controls .ke-primary').click();
    await page.locator('.wb-expert-heading').click();
    await modelSeek(63.334); await expertSeek(35);
    await page.locator('.wb-expert-speed').selectOption('1');
    await page.locator('.wb-speed select').selectOption('1');
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({path: path.join(out, 'expert-desktop.png')});
    for (const viewport of [{width: 1440, height: 900}, {width: 1366, height: 768}, {width: 390, height: 844}]) {
      await page.setViewportSize(viewport); await page.waitForTimeout(100);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.locator('.wb-expert-panel').scrollIntoViewIfNeeded();
      await page.screenshot({path: path.join(out, `expert-${viewport.width}x${viewport.height}.png`)});
      if (viewport.width >= 1080) {
        assert(await page.locator('.wb-inspector').evaluate(n => n.classList.contains('wb-pinned')), 'expert panel must not add height to model sticky column');
      }
    }
    await page.locator('.wb-expert-slider').focus(); await page.keyboard.press('End'); await settleVideo('.wb-expert-video');
    assert(Math.abs(await page.locator('.wb-expert-video').evaluate(v => v.currentTime) - 66.9) < .02);
    await page.locator('.wb-expert-play').click();
    await page.waitForFunction(() => document.querySelector('.wb-expert-video').currentTime < 5 && !document.querySelector('.wb-expert-video').paused);
    // Teardown releases media; mounting again gives exactly one independent player.
    const cleanup = await page.evaluate(() => {
      const expert = document.querySelector('.wb-expert-video'); window.kaiWorkbench.destroy(); window.kaiWorkbench.destroy();
      const state = {paused: expert.paused, noSource: !expert.hasAttribute('src'), connected: expert.isConnected};
      const p = JSON.parse(document.getElementById('kai-evaluation-data').textContent), d = JSON.parse(document.getElementById('kai-tactile-load-data').textContent), c = JSON.parse(document.getElementById('kai-workbench-config').textContent);
      window.kaiWorkbench = KaiEvaluationWorkbench.mount(document.getElementById('kai-evaluation'), p, d, c); return state;
    });
    assert.deepEqual(cleanup, {paused: true, noSource: true, connected: false});
    assert.equal(await page.locator('video').count(), 2); assert.equal(await page.locator('.wb-expert-slider').count(), 1);
    await page.waitForFunction(() => !document.querySelector('.wb-expert-slider').disabled);
    // A media load error stays isolated and does not disable the model.
    await page.locator('.wb-expert-video').evaluate(v => v.dispatchEvent(new Event('error')));
    assert.equal(await page.locator('.wb-expert-slider').isDisabled(), true);
    assert.equal(await page.locator('.ke-slider').isDisabled(), false);
    assert.match(await page.locator('.wb-expert-status').textContent(), /模型回放与评估图不受影响/);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed: true, checks: ['verified independent video seconds', 'two independent players', 'expert seek/play/pause/rate never changes model', 'expert UI dispatches zero model inputs', 'model seek/play/rate never changes expert', 'fold/unfold isolated', 'source crop independent', 'keyboard and replay from end', 'desktop/short screen/mobile', 'idempotent cleanup/remount', 'expert errors isolated']}));
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
