'use strict';
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const [url, out, ...flags] = process.argv.slice(2); fs.mkdirSync(out, {recursive: true});
  const requireWrist = flags.includes('--require-wrist');
  const browser = await chromium.launch({executablePath: process.env.KAI_CHROMIUM_EXECUTABLE, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 900}}), errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(url); await page.waitForSelector('.ks-case');
    const batch = await page.evaluate(() => JSON.parse(document.getElementById('kai-suite-data').textContent));
    assert.equal(await page.locator('.ks-case').count(), batch.trial_count);
    assert.equal(batch.task_count, Object.keys(batch.tasks).length);
    assert(batch.task_count >= 9);
    assert.equal(await page.locator('#ks-mode').count(), 0, 'internal run classification is not a public filter');
    assert.equal(await page.locator('.ks-case .ks-mode').count(), 0, 'internal run classification is not a case badge');
    assert.equal(await page.locator('.ks-stat').count(), 2, 'show only task and trajectory summary counts');
    assert.equal(await page.locator('#ks-coverage').count(), 0, 'internal coverage and provenance are not a public section');
    await page.locator('#ks-search').fill('seed_005');
    assert.equal(await page.locator('.ks-case').count(), batch.trials.filter(t => t.seed === 5).length);
    await page.locator('#ks-search').fill('');
    const selected = [...new Map(batch.trials.map(t => [t.task, t])).values()];
    for (const trial of selected) {
      await page.goto(new URL(trial.report_path, url).href);
      await page.waitForSelector('.wb-header');
      assert.equal(await page.locator('.wb-start-error').count(), 0);
      assert.match(await page.locator('.wb-header h1').textContent(), new RegExp(trial.title));
      assert.equal(await page.locator('.wb-chart').count(), 4);
      assert.equal(await page.locator('.wb-finger').count(), 5);
      assert.equal(await page.locator('.wb-plots-panel').count(), 1);
      const data = await page.evaluate(() => JSON.parse(document.getElementById('kai-tactile-load-data').textContent));
      assert.equal(data.prediction_horizon, 50);
      const testTime = data.time_s[Math.floor(data.time_s.length / 2)];
      await page.locator('.ke-slider').evaluate((s, t) => { s.value = String(t); s.dispatchEvent(new Event('input', {bubbles: true})); }, testTime);
      await page.waitForTimeout(100);
      const sample = data.time_s.findIndex(t => t === testTime);
      const normal = data.normal_total_n[sample], tangent = data.tangent_total_n[sample];
      assert.match(await page.locator('[data-load-panel="normal-total"] .wb-reading').textContent(), new RegExp(normal === null ? '未知' : normal.toFixed(3)));
      assert.match(await page.locator('[data-load-panel="tangent-total"] .wb-reading').textContent(), new RegExp(tangent === null ? '未知' : tangent.toFixed(3)));
      const config = await page.evaluate(() => JSON.parse(document.getElementById('kai-workbench-config').textContent));
      const wrist = config.media_views?.views?.find(view => view.id === 'wrist');
      if (requireWrist) assert(wrist, `${trial.task} is missing its audited model wrist view`);
      if (wrist) {
        const model = page.locator('.ke-media-panel .ke-video');
        await page.waitForFunction(() => { const v = document.querySelector('.ke-media-panel .ke-video'); return !v.seeking && v.readyState >= 2; });
        const before = await model.evaluate(v => ({src: v.currentSrc, time: v.currentTime, paused: v.paused}));
        const cursorBefore = await page.locator('.ke-slider').inputValue();
        const expertBefore = await page.locator('.wb-expert-video').evaluate(v => v.currentTime);
        for (const camera of ['wrist', 'mosaic', 'head', 'wrist']) {
          await page.locator(`.wb-cameras .wb-camera[data-camera="${camera}"]`).click();
          assert.equal(await model.getAttribute('data-camera'), camera);
          assert.equal(await page.locator('.ke-slider').inputValue(), cursorBefore, 'camera switching must not seek the model');
          assert.equal(await page.locator('.wb-expert-video').evaluate(v => v.currentTime), expertBefore);
          const after = await model.evaluate(v => ({src: v.currentSrc, time: v.currentTime, paused: v.paused}));
          assert.deepEqual(after, before, 'views must crop the same video without reloading or changing playback');
        }
        const dimensions = await page.locator('.ke-video-frame').evaluate(n => ({width: n.getBoundingClientRect().width, height: n.getBoundingClientRect().height}));
        assert(Math.abs(dimensions.width / dimensions.height - wrist.width / wrist.height) < .003, 'wrist crop aspect ratio must match the audited source pane');
        const sourceDimensions = await model.evaluate(n => ({width: n.getBoundingClientRect().width, height: n.getBoundingClientRect().height}));
        assert(Math.abs(sourceDimensions.width / sourceDimensions.height - config.media_views.source_width / config.media_views.source_height) < .003, 'view cropping must not distort the source video');
        await page.locator('.ke-media-panel').screenshot({path: path.join(out, trial.task + '-model-wrist.png')});
        const playFrom = await model.evaluate(v => v.currentTime);
        await model.evaluate(v => v.play());
        await page.locator('.wb-cameras .wb-camera[data-camera="head"]').click();
        await page.locator('.wb-cameras .wb-camera[data-camera="wrist"]').click();
        assert.equal(await model.evaluate(v => v.paused), false, 'switching views must not interrupt model playback');
        await page.waitForFunction(t => document.querySelector('.ke-media-panel .ke-video').currentTime > t + .08, playFrom);
        await model.evaluate(v => v.pause());
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        assert.equal(await page.locator('.wb-expert-video').evaluate(v => v.currentTime), expertBefore, 'model wrist playback must not start the expert');
      }
      const modelTime = await page.locator('.ke-slider').inputValue();
      await page.waitForFunction(() => !document.querySelector('.wb-expert-slider').disabled);
      await page.locator('.wb-expert-slider').evaluate(s => { s.value = String(Number(s.max) / 2); s.dispatchEvent(new Event('input', {bubbles: true})); });
      assert.equal(await page.locator('.ke-slider').inputValue(), modelTime, 'expert clock must be independent for every task');
      if (trial.expert_relation === 'same_task_other_recording') assert.match(await page.locator('.wb-expert-comparison').textContent(), /不同|不是|不保证/);
      if (trial.plots === 'available') {
        assert.equal(await page.locator('.wb-plots-tabs [role="tab"]').count(), 3);
        if (trial.plot_kind === 'model_only') {
          assert.equal(await page.locator('.wb-plots-reference').count(), 0, 'model-only plots must not imply an expert reference');
          assert.match(await page.locator('.wb-plots-panel').textContent(), /仅模型|模型原生记录/);
          assert.doesNotMatch(await page.locator('.wb-expert-comparison').textContent(), /PNG 蓝线/);
          assert.match(await page.locator('.wb-expert-comparison').textContent(), /未绘制专家参考线/);
        }
        await page.locator('.wb-plots-tabs [role="tab"]').nth(1).click();
        await page.waitForFunction(() => { const n = document.querySelector('.wb-plots-image'); return n.complete && n.naturalWidth === 1920; });
        await page.locator('.wb-plots-actions button').click();
        assert.equal(await page.locator('.wb-plot-dialog').evaluate(n => n.open), true);
        await page.keyboard.press('Escape');
      } else {
        assert.equal(await page.locator('.wb-plots-tabs').count(), 0);
        assert.match(await page.locator('.wb-plots-panel').textContent(), /未生成/);
      }
      if (trial.task === 'pick-place') assert.equal(await page.locator('.ke-task-chart-card').count(), 0, 'do not invent pick/place progress');
      if (trial.task === 'usb-insert') assert.match(await page.locator('.wb-task-field-note').textContent(), /未对准|aligned_with_socket=false/);
      await page.setViewportSize({width: 390, height: 844});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      if (wrist) await page.locator('.ke-media-panel').screenshot({path: path.join(out, trial.task + '-model-wrist-mobile.png')});
      await page.setViewportSize({width: 1440, height: 900});
      await page.evaluate(() => scrollTo(0, 0)); await page.waitForTimeout(150);
      await page.screenshot({path: path.join(out, trial.task + '.png')});
    }
    await page.goto(url); await page.waitForSelector('.ks-case');
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: path.join(out, 'portal-mobile.png')});
    await page.setViewportSize({width: 1440, height: 900});
    await page.screenshot({path: path.join(out, 'portal-desktop.png')});
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed: true, trial_count: batch.trial_count, task_count: batch.task_count, checked_tasks: selected.map(t => t.task), require_wrist: requireWrist, checks: ['clean summary without internal run classifications or provenance section', 'seed filtering', 'native Fn/Ft readouts', 'same-video wrist crops without seeking or distortion', 'independent expert clocks', 'task-specific fields', 'plots and declared missing', 'desktop/mobile']})); 
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
