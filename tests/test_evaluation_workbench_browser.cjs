'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const [url, out] = process.argv.slice(2); fs.mkdirSync(out, {recursive: true});
  const browser = await chromium.launch({executablePath: process.env.KAI_CHROMIUM_EXECUTABLE, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  try {
    const page = await browser.newPage({viewport: {width: 1480, height: 1100}}), errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(url); await page.waitForSelector('.wb-root');
    // Core model regression can run with or without the optional expert player.
    await page.locator('.wb-expert-panel').evaluateAll(nodes => nodes.forEach(n => { n.open = false; }));
    assert.equal(await page.locator('.wb-start-error').count(), 0);
    assert.equal(await page.locator('.ke-transport').count(), 1);
    assert.equal(await page.locator('.ke-load-panel').count(), 0, 'old add-on must not double mount');
    assert.equal(await page.locator('.wb-chart').count(), 4);
    assert.equal(await page.locator('.wb-task-grid').first().locator('.ke-task-chart-card').count(), 2);
    assert.equal(await page.locator('.wb-finger').count(), 5);
    assert.match(await page.locator('.wb-outcome').textContent(), /64\.4%/);
    const data = await page.evaluate(() => JSON.parse(document.getElementById('kai-tactile-load-data').textContent));
    const index = 1900, time = data.time_s[index];
    async function seek(t) {
      await page.locator('.ke-slider').evaluate((s, value) => { s.value = String(value); s.dispatchEvent(new Event('input', {bubbles: true})); }, t);
      await page.waitForFunction(value => Math.abs(Number(document.querySelector('[data-load-chart="normal-total"]').getAttribute('aria-valuenow')) - value) < 1e-8, t);
      await page.waitForFunction(() => { const v = document.querySelector('.ke-video'); return !v.seeking && v.readyState >= 2; });
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    }
    await seek(time);
    assert.match(await page.locator('[data-load-panel="normal-total"] .wb-reading').textContent(), new RegExp(data.normal_total_n[index].toFixed(3)));
    assert.match(await page.locator('[data-load-panel="tangent-total"] .wb-reading').textContent(), new RegExp(data.tangent_total_n[index].toFixed(3)));
    const bounds = await page.locator('.wb-load-card').evaluateAll(cards => cards.map(c => { const b = c.getBoundingClientRect(); return {x: b.x, y: b.y, w: b.width, h: b.height}; }));
    assert.equal(bounds[0].y, bounds[1].y); assert.equal(bounds[2].y, bounds[3].y); assert.equal(bounds[0].x, bounds[2].x); assert(bounds[2].y > bounds[0].y);
    const scales = await page.locator('.wb-load-card').evaluateAll(cards => cards.map(c => c.dataset.yMax));
    const totalBefore = await page.locator('[data-load-panel="normal-total"] .wb-reading').textContent();
    await page.locator('.wb-finger').nth(2).click();
    await page.waitForFunction(() => document.querySelector('[data-load-panel="normal-fingers"]').dataset.selectedFinger === '2');
    assert.equal(await page.locator('.ke-heat-card:visible').count(), 1);
    assert.equal(await page.locator('.ke-heat-card[aria-hidden="false"]').count(), 1);
    assert.match(await page.locator('[data-load-panel="tangent-fingers"] .wb-reading').textContent(), /右中指/);
    assert.equal(await page.locator('[data-load-panel="normal-total"] .wb-reading').textContent(), totalBefore);
    assert.deepEqual(await page.locator('.wb-load-card').evaluateAll(cards => cards.map(c => c.dataset.yMax)), scales);
    await page.locator('.ke-mode-button').nth(1).click();
    assert.equal(await page.locator('.ke-glyph-card:visible').count(), 1);
    const glyphSizes = await page.locator('.ke-force-glyph').evaluateAll(canvases => canvases.map(c => ({w: c.getBoundingClientRect().width, h: c.getBoundingClientRect().height, pixels: c.width})));
    assert(glyphSizes.every(b => b.w === glyphSizes[0].w && b.h === glyphSizes[0].h && b.w > 100), 'inactive glyphs must preserve geometry');
    assert.match(await page.locator('.ke-glyph-card[aria-hidden="false"] .ke-glyph-tangent-reading').textContent(), /ΔFt/);
    assert.match(await page.locator('.wb-current').textContent(), new RegExp(data.tangent_force_n[index][2].toFixed(3)));
    // Cameras are exact aspect-preserving crops of verified archived video pixels.
    for (const camera of ['head', 'wrist', 'global', 'mosaic']) {
      await page.locator(`.wb-camera[data-camera="${camera}"]`).click();
      const rect = await page.locator('.ke-video').evaluate(v => { const r = v.getBoundingClientRect(); return {width: r.width, height: r.height}; });
      assert(Math.abs(rect.width / rect.height - 1920 / 1080) < .002, `${camera} distorted the video`);
    }
    await page.locator('.wb-camera[data-camera="head"]').click();
    await page.screenshot({path: path.join(out, 'desktop.png')});
    await page.evaluate(() => scrollTo(0, 200));
    await page.waitForFunction(() => document.querySelector('.wb-inspector').classList.contains('wb-pinned'));
    const sticky = await page.locator('.wb-inspector').boundingBox(); const bar = await page.locator('.ke-transport').boundingBox();
    assert(sticky.y >= bar.y + bar.height - 1 && sticky.y < bar.y + bar.height + 14, 'left column should stick just below transport');
    await page.screenshot({path: path.join(out, 'desktop-sticky.png')});
    for (const [width, height] of [[1440, 900], [1366, 768]]) {
      await page.setViewportSize({width, height});
      await page.evaluate(() => scrollTo(0, 300));
      await page.waitForFunction(() => document.querySelector('.wb-inspector').classList.contains('wb-pinned'));
      const inspector = await page.locator('.wb-inspector').boundingBox(), transport = await page.locator('.ke-transport').boundingBox();
      assert(inspector.y >= transport.y + transport.height - 1 && inspector.y + inspector.height <= height, `${width}×${height} inspector must fit below shared transport`);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.screenshot({path: path.join(out, `desktop-${width}x${height}.png`)});
    }
    await page.setViewportSize({width: 1480, height: 1100});
    const chart = page.locator('[data-load-chart="tangent-fingers"]');
    const beforeHover = await page.locator('.ke-slider').inputValue();
    await chart.hover({position: {x: 150, y: 85}});
    assert.equal(await page.locator('.ke-slider').inputValue(), beforeHover, 'hover must not seek');
    assert.match(await page.locator('[data-load-panel="tangent-fingers"] .wb-hover').textContent(), /悬停.*原生采样/);
    await chart.click({position: {x: 150, y: 85}});
    await page.waitForFunction(old => document.querySelector('.ke-slider').value !== old, beforeHover);
    await page.waitForFunction(() => !document.querySelector('.ke-video').seeking);
    const mapping = await page.evaluate(() => {
      const viz = JSON.parse(document.getElementById('kai-evaluation-data').textContent).visualization;
      const t = Number(document.querySelector('.ke-slider').value); let expected = 0;
      viz.media.time_s.forEach((v, i) => { if (v <= t + 1e-8) expected = viz.media.video_time_s[i]; });
      return {actual: document.querySelector('.ke-video').currentTime, expected};
    }); assert(Math.abs(mapping.actual - mapping.expected) < .001);
    await seek(45.01); await chart.focus(); await page.keyboard.press('ArrowLeft');
    assert(Math.abs(Number(await page.locator('.ke-slider').inputValue()) - 45) < .001, 'left arrow from between samples goes to immediately previous sample');
    await page.locator('.ke-video').evaluate(v => v.play());
    await page.waitForFunction(() => !document.querySelector('.ke-video').paused);
    const playTime = await page.locator('.ke-slider').inputValue();
    await page.waitForFunction(t => document.querySelector('.ke-slider').value !== t, playTime);
    await page.locator('.wb-finger').nth(1).click();
    assert.equal(await page.locator('.ke-video').evaluate(v => v.paused), false, 'finger selection must not pause');
    await page.locator('.ke-video').evaluate(v => v.pause());
    // Expanding archived charts redraws correct canvas sizes while paused.
    await page.getByText('取放与支撑', {exact: true}).click();
    assert(await page.locator('[data-task-key="table_support_force_n"]').isVisible());
    assert.match(await page.locator('[data-task-key="maximum_lift_m"] h3').textContent(), /累计最大/);
    await page.getByText('全程统计与原始法向力图', {exact: true}).click();
    await page.waitForFunction(() => { const c = document.querySelector('.ke-force-chart'); return Math.abs(c.width - c.getBoundingClientRect().width * devicePixelRatio) < 2; });
    const total = await page.locator('[data-load-panel="normal-total"] .wb-reading').textContent();
    await page.locator('.ke-channel-toggle input').first().uncheck();
    assert.equal(await page.locator('[data-load-panel="normal-total"] .wb-reading').textContent(), total);
    for (const width of [1024, 390]) {
      await page.setViewportSize({width, height: 844});
      await page.waitForTimeout(150);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${width} horizontal overflow`);
      assert.equal(await page.locator('.wb-inspector').evaluate(n => getComputedStyle(n).position), 'static');
    }
    await page.evaluate(() => scrollTo(0, 0)); await page.screenshot({path: path.join(out, 'mobile-top.png')});
    await page.locator('.ke-spatial-panel').scrollIntoViewIfNeeded(); await page.screenshot({path: path.join(out, 'mobile-tactile.png')});
    await chart.scrollIntoViewIfNeeded(); await page.screenshot({path: path.join(out, 'mobile-loads.png')});
    assert.deepEqual(errors, []);
    // Separate instances / unknown signals / teardown / remount use existing archived payload only.
    await page.evaluate(() => {
      window.kaiWorkbench.destroy();
      window.kaiWorkbench.destroy(); // Idempotent teardown.
      const p = JSON.parse(document.getElementById('kai-evaluation-data').textContent), d = JSON.parse(document.getElementById('kai-tactile-load-data').textContent), c = JSON.parse(document.getElementById('kai-workbench-config').textContent);
      d.tangent_force_n = d.time_s.map(() => [null, null, null, null, null]); d.tangent_total_n = d.time_s.map(() => null);
      window.kaiWorkbench = KaiEvaluationWorkbench.mount(document.getElementById('kai-evaluation'), p, d, c);
    });
    await page.waitForFunction(() => document.querySelector('[data-load-panel="tangent-total"] .wb-reading').textContent.includes('未知'));
    assert.doesNotMatch(await page.locator('[data-load-panel="normal-total"] .wb-reading').textContent(), /未知/);
    assert.equal(await page.locator('.ke-transport').count(), 1); assert.equal(await page.locator('.wb-chart').count(), 4);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed: true, checks: ['2×2 force panels', 'single shared cursor', 'native current Fn/Ft', 'per-finger highlight with fixed scale', 'single inspector heatmap/glyph', 'inactive cards retain dimensions', 'verified camera crops', 'desktop sticky', 'video bidirectional', 'hover no seek', 'keyboard', 'no pause on finger selection', 'archived details redraw', 'mobile/tablet no overflow', 'unknown not zero', 'teardown/remount', 'no browser errors']}));
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
