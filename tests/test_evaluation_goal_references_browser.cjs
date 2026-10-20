'use strict';
// Optional local-only browser QA using an already installed Playwright/browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const [portal, previous, output] = process.argv.slice(2);
  if (!portal || !previous || !output) throw new Error('Expected new portal URL, preserved portal URL, and new QA directory');
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({executablePath: process.env.KAI_CHROMIUM_EXECUTABLE,
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const errors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(portal);
    await page.waitForSelector('.ks-case');
    const batch = await page.evaluate(() => JSON.parse(document.getElementById('kai-suite-data').textContent));
    assert.equal(batch.trial_count, 360);
    assert.equal(await page.locator('.ks-case').count(), 360);
    assert.equal(batch.goal_reference_extension.status_counts.verified, 360);
    assert.match(await page.locator('h1').textContent(), /目标参考线/);
    await page.screenshot({path: path.join(output, 'portal.png')});

    const tasks = ['bulb-screw', 'bulb-unscrew', 'install-ram', 'poker-draw',
      'usb-insert', 'usb-unplug', 'whiteboard-wipe', 'whiteboard-wipe-horizontal',
      'whiteboard-wipe-horizontal-line'];
    for (const task of tasks) {
      const trial = (task === 'whiteboard-wipe-horizontal-line' ? batch.trials.find(row =>
        row.task === task && row.seed === 7 && row.report_path.includes('step20000')) : null)
        || batch.trials.find(row => row.task === task && row.seed === 7) || batch.trials.find(row => row.task === task);
      await page.goto(new URL(trial.report_path, portal).href);
      await page.waitForSelector('.wb-header');
      assert.equal(await page.locator('.wb-start-error').count(), 0);
      const goalCount = task === 'usb-unplug' ? 1 : 2;
      assert.equal(await page.locator('.ke-goal-reference').count(), goalCount);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${task}: no desktop horizontal overflow`);
      await page.screenshot({path: path.join(output, `${task}-first-screen-desktop.png`)});
      const payloadBefore = await page.locator('#kai-evaluation-data').textContent();
      const payload = JSON.parse(payloadBefore), data = payload.visualization;
      if (task === 'bulb-screw' || task === 'bulb-unscrew') {
        assert.match(await page.locator('.ke-goal-references').first().textContent(), task === 'bulb-screw' ? /100°/ : /131\.5°/);
      }
      const goals = data.task_series.filter(series => series.goal_references?.length);
      assert.equal(goals.length, goalCount);
      assert.equal(data.task_series.find(s => /torque/.test(s.key))?.goal_references, undefined);
      if (task === 'install-ram') {
        assert.deepEqual(goals.map(series => series.key).sort(), ['insertion_depth_m', 'orientation_error_rad']);
        const depth = goals.find(series => series.key === 'insertion_depth_m').goal_references[0];
        assert.equal(depth.target, .006);
        assert(Math.abs(depth.lower - .0058) < 1e-12); assert(Math.abs(depth.upper - .0062) < 1e-12);
        const orientation = goals.find(series => series.key === 'orientation_error_rad').goal_references[0];
        assert.equal(orientation.relation, 'at_most'); assert.equal(orientation.lower, 0);
        assert(Math.abs(orientation.target - Math.PI / 180) < 1e-12); assert.equal(orientation.upper, orientation.target);
      }
      if (task === 'poker-draw') {
        assert.deepEqual(goals.map(series => series.key).sort(), ['card_displacement_now_mm', 'card_displacement_peak_mm']);
        for (const series of goals) {
          const reference = series.goal_references[0];
          assert.equal(reference.target, 50); assert.equal(reference.unit, 'mm'); assert.equal(reference.relation, 'at_least');
          assert.equal(reference.lower, undefined); assert.equal(reference.upper, undefined);
        }
      }
      if (task === 'usb-insert') {
        assert.deepEqual(goals.map(series => series.key).sort(), ['insertion_depth_mm', 'orientation_error_deg']);
        const depth = goals.find(series => series.key === 'insertion_depth_mm').goal_references[0];
        assert.equal(depth.target, 12);
        assert(Math.abs(depth.lower - 11.9) < 1e-12); assert(Math.abs(depth.upper - 12.1) < 1e-12);
        const orientation = goals.find(series => series.key === 'orientation_error_deg').goal_references[0];
        assert.equal(orientation.relation, 'at_most'); assert.equal(orientation.target, 3);
        assert.equal(orientation.lower, 0); assert.equal(orientation.upper, 3);
        assert.match(await page.locator('[data-task-key="insertion_depth_mm"]').textContent(), /未对准.*0/);
      }
      if (task === 'usb-unplug') {
        assert.equal(goals[0].key, 'insertion_depth_m');
        const clearance = goals[0].goal_references[0];
        assert.equal(clearance.target, -.002); assert.equal(clearance.relation, 'at_most');
        assert.equal(clearance.lower, undefined); assert.equal(clearance.upper, undefined);
      }
      if (task.startsWith('whiteboard-wipe')) {
        assert.deepEqual(goals.map(series => series.key).sort(), ['cleaned_fraction', 'maximum_ink_remaining']);
        const clean = goals.find(series => series.key === 'cleaned_fraction').goal_references[0];
        assert.equal(clean.target, 1);
        const remaining = goals.find(series => series.key === 'maximum_ink_remaining').goal_references[0];
        assert.equal(remaining.target, 1e-9); assert.equal(remaining.relation, 'at_most');
        const areaKeys = ['effective_pad_contact_area_ratio', 'contact_time_weighted_mean_effective_area_ratio'];
        for (const key of areaKeys) {
          const area = data.task_series.find(series => series.key === key);
          assert(area, `${task}: ${key} must be present`);
          assert.equal(area.presentation_extension, 'whiteboard-contact-area-v1');
          assert.equal(area.goal_references, undefined, 'No invented 100% contact-area success line');
          assert.equal(area.values.length, area.time_s.length);
          assert(area.values.every(value => value === null || Number.isFinite(value) && value >= 0 && value <= 1 + 1e-12));
          const card = page.locator(`[data-task-key="${key}"]`);
          assert.equal(await card.locator('.ke-goal-reference').count(), 0);
          assert.match(await card.textContent(), /几何|刚体/);
          await card.scrollIntoViewIfNeeded();
          await card.screenshot({path: path.join(output, `${task}-${key}-desktop.png`)});
        }
        const currentArea = data.task_series.find(series => series.key === areaKeys[0]);
        assert(currentArea.values.some(value => value === null), 'Out-of-contact samples must not be fabricated zero-area observations');
        const meanArea = data.task_series.find(series => series.key === areaKeys[1]);
        assert.equal(meanArea.values[0], null, 'No pre-contact cumulative average exists');
        const configuration = JSON.parse(await page.locator('#kai-workbench-config').textContent());
        const meanSummary = configuration.task_summary.find(item => item.key === areaKeys[1]);
        assert(meanSummary, 'Contact-period mean must be in the terminal summary');
        const durationSummary = configuration.task_summary.find(item => item.key === 'pad_board_contact_duration_s');
        assert(durationSummary, 'The contact-duration denominator must be inspectable');
        if (task === 'whiteboard-wipe-horizontal-line') assert.equal(durationSummary.value, 0,
          'Explicit step20000 seed007 exercises the real no-contact branch');
        if (durationSummary.value === 0) {
          assert.equal(meanSummary.value, null, 'No-contact mean is undefined, never zero-quality evidence');
          assert.equal(meanSummary.status, 'no_contact');
          assert.equal(meanSummary.recorded_value, 0, 'Original source zero remains separately auditable');
          assert(currentArea.values.every(value => value === null));
          assert(meanArea.values.every(value => value === null));
          assert.match(await page.locator('.wb-task-finals').textContent(), /不适用|无板面.*接触|无.*接触/);
        }
        assert(configuration.task_presentation.primary_task_keys.includes('maximum_ink_remaining'));
        for (const key of areaKeys) assert(configuration.task_presentation.primary_task_keys.includes(key));
      }
      for (const series of goals) {
        const card = page.locator(`[data-task-key="${series.key}"]`);
        await card.scrollIntoViewIfNeeded();
        await page.waitForTimeout(150);
        assert.match(await card.textContent(), /不代表完整任务成功/);
        // Verify the purple reference stroke is really on the rendered canvas,
        // including failed trajectories whose observations never reach it.
        const purple = await card.locator('canvas').evaluate(canvas => {
          const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
          let count = 0;
          for (let i = 0; i < pixels.length; i += 4) {
            if (Math.abs(pixels[i] - 128) < 9 && Math.abs(pixels[i + 1] - 96) < 9 && Math.abs(pixels[i + 2] - 159) < 9) count++;
          }
          return count;
        });
        assert(purple > 20, `${task}/${series.key}: reference line must be visible even if outside native data extent`);
        if (task === 'bulb-screw') assert.match(await card.textContent(), /淡色带/);
        if (task === 'bulb-unscrew' && series.key === 'insertion_depth_m') assert.match(await card.textContent(), /≤.*-0\.003/);
        if (task === 'install-ram') {
          assert.match(await card.textContent(), /淡色带/);
          if (series.key === 'orientation_error_rad') assert.match(await card.textContent(), /≤.*0\.017453/);
        }
        if (task === 'poker-draw') {
          assert.match(await card.textContent(), /≥.*50 mm/);
          assert.equal(await card.locator('.ke-goal-reference-band').count(), 0, 'No fabricated finite success band for displacement');
          assert.doesNotMatch(await card.textContent(), /100\s*%/);
        }
        if (task === 'usb-insert') assert.match(await card.textContent(), /淡色带/);
        if (task === 'usb-unplug') assert.match(await card.textContent(), /≤.*-0\.002/);
        if (task.startsWith('whiteboard-wipe') && series.key === 'maximum_ink_remaining') {
          assert.match(await card.textContent(), /1(?:\.0+)?e-0?9|1e−9|10.?⁻⁹|0\.000000001/,
            'Tiny exact residual threshold must not be displayed as misleading zero');
        }
        await card.screenshot({path: path.join(output, `${task}-${series.key}-desktop.png`)});
      }

      const slider = page.locator('.ke-slider');
      const testTime = data.task_series[0].time_s[Math.floor(data.task_series[0].time_s.length / 2)];
      await slider.evaluate((element, time) => { element.value = String(time); element.dispatchEvent(new Event('input', {bubbles: true})); }, testTime);
      await page.waitForTimeout(150);
      assert.equal(await page.locator('.ke-goal-reference').count(), goalCount);
      assert.equal(await page.locator('#kai-evaluation-data').textContent(), payloadBefore);
      assert.equal(await page.locator('.wb-chart').count(), 4);
      assert.equal(await page.locator('.wb-finger').count(), 5);
      const beforeModel = await slider.inputValue();
      await page.waitForFunction(() => !document.querySelector('.wb-expert-slider').disabled);
      await page.locator('.wb-expert-slider').evaluate(element => {
        element.value = String(Number(element.max) / 3); element.dispatchEvent(new Event('input', {bubbles: true}));
      });
      assert.equal(await slider.inputValue(), beforeModel, 'Expert time stays independent of model progress');

      await page.setViewportSize({width: 390, height: 844});
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForTimeout(150);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'No mobile horizontal overflow');
      await page.screenshot({path: path.join(output, `${task}-first-screen-mobile.png`)});
      for (const series of goals) {
        await page.locator(`[data-task-key="${series.key}"]`).screenshot({path: path.join(output, `${task}-${series.key}-mobile.png`)});
      }
      if (task.startsWith('whiteboard-wipe')) for (const key of ['effective_pad_contact_area_ratio', 'contact_time_weighted_mean_effective_area_ratio']) {
        await page.locator(`[data-task-key="${key}"]`).screenshot({path: path.join(output, `${task}-${key}-mobile.png`)});
      }
      await page.setViewportSize({width: 1440, height: 1000});
    }

    for (const task of tasks) {
      const old = batch.trials.find(row => row.task === task);
      await page.goto(new URL(old.report_path, previous).href);
      await page.waitForSelector('.wb-header');
      const oldPayload = JSON.parse(await page.locator('#kai-evaluation-data').textContent());
      const count = oldPayload.visualization.task_series.reduce((sum, series) => sum + (series.goal_references?.length || 0), 0);
      assert.equal(await page.locator('.ke-goal-reference').count(), count, 'Preserved H50 still uses its original assets and data');
      assert.equal(count, ['bulb-screw', 'bulb-unscrew', 'install-ram', 'poker-draw'].includes(task) ? 2 : 0,
        'Preserved parent retains only its existing bulb/RAM/Poker goals');
      assert.equal(oldPayload.visualization.task_series.filter(series => series.presentation_extension === 'whiteboard-contact-area-v1').length, 0,
        'Preserved parent must not acquire the new geometric-area series');
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed: true, trial_count: 360, verified_goal_trials: 360,
      checks: ['visible canvas reference strokes on all nine tasks', 'tolerance bands', 'USB one-sided clearance and orientation limits', 'Poker >=50 mm without invented success bands',
        'whiteboard exact per-patch residual threshold', 'instant and contact-weighted geometric area without invented success goals', 'desktop/mobile',
        'native payload unchanged', 'cursor and Fn/Ft panels', 'independent expert time', 'preserved H50 unchanged']}));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
