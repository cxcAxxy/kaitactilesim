'use strict';
// Usage: node tests/test_evaluation_unified_browser.cjs <portal-url> [screenshot-dir]
// Reads an already exported site; never starts a simulator, exporter, or server.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const QA_ROOT = '/tmp/contact_web_qa.CJzQS7';
function playwright() {
  if (process.env.KAI_PLAYWRIGHT_MODULE) return require(process.env.KAI_PLAYWRIGHT_MODULE);
  try { return require('playwright'); }
  catch (error) {
    if (error.code !== 'MODULE_NOT_FOUND') throw error;
    return require(path.join(QA_ROOT, 'node_modules/playwright'));
  }
}
function category(trial) {
  if (trial.staged.complete_success === true) return 'complete';
  return trial.staged.stages.some(stage => stage.reached === true) ? 'partial' : 'none';
}
async function embedded(page, id) {
  return page.locator('#' + id).evaluate(node => JSON.parse(node.textContent));
}
async function settledVideo(page, selector) {
  await page.waitForFunction(s => {
    const video = document.querySelector(s);
    return video && video.readyState >= 2 && !video.seeking;
  }, selector, {timeout: 30000});
}
async function assertNoOverflow(page, context) {
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), context);
}

(async () => {
  const [input, requestedOut] = process.argv.slice(2);
  if (!input) throw new Error('Usage: node test_evaluation_unified_browser.cjs <portal-url> [screenshot-dir]');
  const url = new URL(input).href;
  assert(['http:', 'https:'].includes(new URL(url).protocol), 'Provide an HTTP(S) exported portal URL');
  const out = requestedOut || fs.mkdtempSync(path.join(os.tmpdir(), 'kai-unified-browser-'));
  fs.mkdirSync(out, {recursive: true});
  const fallbackChrome = path.join(QA_ROOT, 'browsers/chromium-1248/chrome-linux64/chrome');
  const executablePath = process.env.KAI_CHROMIUM_EXECUTABLE || (fs.existsSync(fallbackChrome) ? fallbackChrome : undefined);
  const libraryPath = path.join(QA_ROOT, 'runtime-libs/usr/lib/x86_64-linux-gnu');
  const env = {...process.env};
  if (fs.existsSync(libraryPath)) env.LD_LIBRARY_PATH = [libraryPath, env.LD_LIBRARY_PATH].filter(Boolean).join(':');
  const browser = await playwright().chromium.launch({executablePath, env, headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 900}});
    page.setDefaultTimeout(30000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url, {waitUntil: 'domcontentloaded'});
    await page.waitForSelector('.ks-case');
    const batch = await embedded(page, 'kai-suite-data');
    assert.equal(batch.evaluation_profile, 'recorded-staged');
    assert.equal(batch.task_count, 9, 'Current staged suite has nine explicitly recorded tasks');
    assert.equal(await page.locator('.ks-case').count(), batch.trial_count);
    assert.equal(await page.locator('.ks-case .ks-stages').count(), batch.trial_count);
    assert.equal(await page.locator('#ks-stage').count(), 1);
    assert([9, 18].includes(batch.stage_groups.length), 'Expected nine preflight or eighteen complete cohort groups');
    assert.equal(await page.locator('#ks-stage-groups > tr').count(), batch.stage_groups.length);
    assert.equal(batch.stage_groups.reduce((total, group) => total + group.count, 0), batch.trial_count);
    const groupKeys = batch.stage_groups.map(group => JSON.stringify([
      group.task, group.model, group.checkpoint_identity, group.mode, group.rule_sha256, group.cohort]));
    assert.equal(new Set(groupKeys).size, groupKeys.length, 'Never merge or duplicate distinct archived rule/cohort identities');
    assert.equal(new Set(await page.locator('.ks-case').evaluateAll(nodes => nodes.slice(0, 9).map(node =>
      node.querySelector('.ks-case-top strong').textContent))).size, 9, 'Initial portal must round-robin tasks');
    for (const [index, group] of batch.stage_groups.entries()) {
      assert.match(group.rule_sha256, /^[a-f0-9]{64}$/);
      const row = page.locator('#ks-stage-groups > tr').nth(index);
      assert.equal((await row.locator('td').nth(1).textContent()).trim(), String(group.count));
      if (group.task.startsWith('whiteboard-')) {
        assert.equal(group.complete_success_defined, false);
        assert.equal((await row.locator('td').nth(3).textContent()).trim(), '未定义');
      }
    }
    for (const filter of ['complete', 'partial', 'none']) {
      await page.locator('#ks-stage').selectOption(filter);
      assert.equal(await page.locator('.ks-case').count(), batch.trials.filter(trial => category(trial) === filter).length);
    }
    await page.locator('#ks-stage').selectOption('');
    for (const task of Object.keys(batch.tasks)) {
      await page.locator(`#ks-task-tabs button[data-task="${task}"]`).click();
      assert.equal(await page.locator('.ks-case').count(), batch.trials.filter(trial => trial.task === task).length);
      assert.equal(await page.locator('#ks-stage-groups > tr').count(), batch.stage_groups.filter(group => group.task === task).length);
      if (task.startsWith('whiteboard-')) assert.match(await page.locator('.ks-stage-result').first().textContent(), /未定义（仅拿起）/);
    }
    await page.locator('#ks-task-tabs button[data-task=""]').click();
    await page.screenshot({path: path.join(out, 'portal-desktop.png')});
    await page.setViewportSize({width: 390, height: 844});
    await assertNoOverflow(page, 'Mobile staged portal overflows');
    await page.screenshot({path: path.join(out, 'portal-mobile.png')});
    await page.setViewportSize({width: 1440, height: 900});

    const selected = Object.keys(batch.tasks).map(task => {
      const candidates = batch.trials.filter(trial => trial.task === task);
      return candidates.find(trial => trial.staged.stages.some(stage => stage.reached === true &&
        typeof stage.first_time_s === 'number' && Number.isFinite(stage.first_time_s))) || candidates[0];
    });
    let jumps = 0;
    for (const trial of selected) {
      await page.goto(new URL(trial.report_path, url).href, {waitUntil: 'domcontentloaded'});
      await page.waitForSelector('.wb-staged-panel');
      assert.equal(await page.locator('.wb-start-error').count(), 0);
      const config = await embedded(page, 'kai-workbench-config');
      const staged = config.staged_evaluation;
      assert(staged && staged.staged_result.version === 'staged_v1');
      assert.equal(await page.locator('.wb-stage-card').count(), staged.stages.length);
      assert.deepEqual(staged.native_result, trial.staged.native_result);
      assert.match(await page.locator('.wb-staged-note').first().textContent(), /曾经达到.*不是当前接触状态/);
      if (trial.task.startsWith('whiteboard-')) {
        assert.equal(staged.staged_result.complete_success, null);
        assert.equal(staged.stages.length, 1);
        assert.equal(staged.stages[0].name, 'picked_up');
        assert.match(await page.locator('.wb-staged-scope').textContent(), /未定义.*完整成功.*拾取阶段不能代替擦拭完成/);
      }
      const model = page.locator('.ke-media-panel .ke-video');
      const expert = page.locator('.wb-expert-video');
      assert.equal(await expert.count(), 1, `${trial.task}: independent expert must remain available`);
      await settledVideo(page, '.wb-expert-video');
      await expert.evaluate(video => video.pause());
      await model.evaluate(video => video.pause());
      await page.locator('.wb-expert-slider').evaluate(slider => {
        slider.value = String(Math.min(1.25, Number(slider.max) / 2));
        slider.dispatchEvent(new Event('input', {bubbles: true}));
      });
      await settledVideo(page, '.wb-expert-video');
      const expertBefore = await expert.evaluate(video => ({time: video.currentTime, src: video.currentSrc, paused: video.paused}));
      const slider = page.locator('.ke-slider');
      const maximumTime = Number(await slider.getAttribute('max'));
      const mapped = staged.timeline.model_clock_mapping === 'mapped';
      let jumped = false;
      for (const stage of staged.stages) {
        const card = page.locator(`.wb-stage-card[data-stage="${stage.key}"]`);
        assert.equal(await card.getAttribute('data-reached'), stage.reached === true ? 'true' : stage.reached === false ? 'false' : 'unknown');
        assert.equal(await card.locator('.wb-stage-name').textContent(), stage.label || stage.name);
        const button = card.locator('.wb-stage-first-time');
        const eligible = mapped && typeof stage.first_time_s === 'number' && Number.isFinite(stage.first_time_s) &&
          stage.first_time_s >= 0 && stage.first_time_s <= maximumTime;
        assert.equal(await button.isEnabled(), eligible);
        if (eligible && !jumped) {
          await button.click();
          const tolerance = Math.max(.001, Number(await slider.getAttribute('step')) || .001);
          assert(Math.abs(Number(await slider.inputValue()) - stage.first_time_s) <= tolerance + 1e-8,
            `${trial.task}: stage jump must use its recorded time, not normalize or align to expert`);
          await settledVideo(page, '.ke-media-panel .ke-video');
          assert.deepEqual(await expert.evaluate(video => ({time: video.currentTime, src: video.currentSrc, paused: video.paused})), expertBefore,
            'Stage jump must not seek or play the expert');
          jumped = true; jumps += 1;
        }
      }
      await page.locator('.wb-staged-samples > summary').click();
      assert.match(await page.locator('.wb-staged-clock').textContent(), mapped ? /未重采样|无有效阶段采样/ : /独立阶段时钟/);
      assert.doesNotMatch(await page.locator('.wb-staged-observations').textContent(), /NaN|Infinity|undefined/);
      const wrist = config.media_views?.views?.find(view => view.id === 'wrist');
      assert(wrist && config.media_views.status === 'verified', `${trial.task}: audited wrist view is required`);
      await settledVideo(page, '.ke-media-panel .ke-video');
      const modelBefore = await model.evaluate(video => ({time: video.currentTime, src: video.currentSrc, paused: video.paused}));
      const cursorBefore = await slider.inputValue();
      for (const camera of ['wrist', 'mosaic', 'head', 'wrist']) {
        await page.locator(`.wb-cameras .wb-camera[data-camera="${camera}"]`).click();
        assert.equal(await model.getAttribute('data-camera'), camera);
        assert.equal(await slider.inputValue(), cursorBefore);
        assert.deepEqual(await model.evaluate(video => ({time: video.currentTime, src: video.currentSrc, paused: video.paused})), modelBefore);
        assert.deepEqual(await expert.evaluate(video => ({time: video.currentTime, src: video.currentSrc, paused: video.paused})), expertBefore);
      }
      const aspect = await page.locator('.ke-video-frame').evaluate(node => node.getBoundingClientRect().width / node.getBoundingClientRect().height);
      assert(Math.abs(aspect - wrist.width / wrist.height) < .003, 'Wrist cropping must retain its audited aspect ratio');
      await page.locator('.wb-staged-panel').screenshot({path: path.join(out, `${trial.task}-stages.png`)});
      await page.setViewportSize({width: 390, height: 844});
      await assertNoOverflow(page, `${trial.task}: mobile staged report overflows`);
      await page.locator('.wb-staged-panel').screenshot({path: path.join(out, `${trial.task}-stages-mobile.png`)});
      await page.setViewportSize({width: 1440, height: 900});
    }
    assert(jumps > 0, 'At least one achieved-stage jump must be exercised, not silently skipped');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed: true, trial_count: batch.trial_count, group_count: batch.stage_groups.length,
      tasks: selected.map(trial => trial.task), stage_jumps: jumps, screenshots: out,
      checks: ['staged portal counts and filtering', 'archived rule/cohort groups', 'native/staged separation',
        'whiteboard undefined completion', 'independent expert during stage jump', 'SHA-audited wrist crops', 'mobile overflow']}));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
