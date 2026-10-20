'use strict';
// Optional local-only branding QA. Reuses the existing report server and cached browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');

async function embedded(page, id) { return JSON.parse(await page.locator('#' + id).textContent()); }
async function artifact(page, url) {
  const response = await page.request.get(url);
  assert(response.ok(), `Expected readable local snapshot artifact: ${url}`);
  return response.text();
}
async function noOverflow(page, label) {
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${label}: no horizontal document overflow`);
}
async function statisticsPosition(page) {
  assert(await page.locator('.ks-stage-groups').evaluate(node =>
    node.previousElementSibling?.classList.contains('ks-results-bar') && node.nextElementSibling?.id === 'ks-trials'),
  'Same-rule/checkpoint statistics remain immediately below displayed-count toolbar and above trajectories');
}
async function seek(page, selector, value) {
  await page.locator(selector).evaluate((input, time) => {
    input.value = String(time); input.dispatchEvent(new Event('input', {bubbles: true}));
  }, value);
}
async function ready(page) {
  await page.waitForSelector('.wb-header-stages .wb-stage-badge');
  await page.waitForFunction(() => document.querySelector('.wb-trajectory-select')?.options.length > 1);
  assert.equal(await page.locator('.wb-start-error').count(), 0);
}

(async () => {
  const [portal, parent, output] = process.argv.slice(2);
  if (!portal || !parent || !output) throw new Error('Expected branded portal URL, preserved parent URL, and screenshot directory');
  assert.notEqual(new URL(portal).href, new URL(parent).href);
  for (const url of [portal, parent]) assert.equal(new URL(url).hostname, '127.0.0.1', 'QA is loopback-only');
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({executablePath: process.env.KAI_CHROMIUM_EXECUTABLE,
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const errors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(portal); await page.waitForSelector('.ks-case');
    assert.equal(await page.title(), 'KaiTactileSim 官方评估平台 | H50');
    assert.equal(await page.locator('#kai-suite > header h1').textContent(), 'KaiTactileSim 官方评估平台');
    assert.match(await page.locator('#kai-suite > header').textContent(), /KaiTactileSim Evaluation Platform/);
    assert.match(await page.locator('#kai-suite > header').textContent(), /机器人操作任务评估\s*·\s*动作回放\s*·\s*触觉分析/);
    assert.match(await page.locator('.ks-brand-footer').textContent(), /KaiTactileSim/);
    assert.match(await page.locator('.ks-old').textContent(), /历史.*版本/);
    assert.equal(await page.locator('#ks-mode, .ks-mode, .ks-audit, #ks-coverage').count(), 0);
    const batch = await embedded(page, 'kai-suite-data');
    const prior = JSON.parse(await artifact(page, new URL('batch.json', parent).href));
    assert.equal(batch.trial_count, 360); assert.equal(batch.task_count, 9);
    assert.deepEqual(batch.trials, prior.trials, 'Branding cannot change recorded trial outcomes or stages');
    assert.deepEqual(batch.stage_groups, prior.stage_groups, 'Branding cannot change stage statistics');
    assert.deepEqual(batch.mode_counts, prior.mode_counts, 'Official ownership does not upgrade evaluation protocol eligibility');
    assert.equal(await page.locator('.ks-case').count(), 360);
    assert.match(await page.locator('#ks-visible').textContent(), /360\s*\/\s*360/);
    await statisticsPosition(page); await noOverflow(page, 'Branded overview desktop');
    await page.screenshot({path: path.join(output, 'official-overview-desktop.png')});
    await page.locator('.ks-brand-footer').screenshot({path: path.join(output, 'official-overview-footer.png')});
    await page.locator('.ks-stage-groups > summary').click();
    assert.equal(await page.locator('#ks-stage-groups > tr').count(), batch.stage_groups.length);
    await statisticsPosition(page);
    await page.locator('.ks-stage-groups > summary').click();
    await page.locator('#ks-view-list').click();
    assert.equal(await page.locator('#ks-trials').getAttribute('data-view'), 'list');
    await page.locator('#ks-task-tabs button[data-task="usb-insert"]').click();
    assert.equal(await page.locator('.ks-case').count(), 40);
    await page.locator('#ks-checkpoint').selectOption('24999');
    assert.equal(await page.locator('.ks-case').count(), 20);
    await page.locator('#ks-search').fill('seed_019');
    assert.equal(await page.locator('.ks-case').count(), 1);
    await statisticsPosition(page);
    await page.setViewportSize({width: 390, height: 844});
    await noOverflow(page, 'Branded filtered overview mobile');
    await page.screenshot({path: path.join(output, 'official-overview-filtered-mobile.png')});
    await page.locator('#ks-search').fill(''); await page.locator('#ks-checkpoint').selectOption('');
    await page.locator('#ks-task-tabs button[data-task=""]').click(); await page.locator('#ks-view-cards').click();
    await statisticsPosition(page); await noOverflow(page, 'Branded overview cards mobile');
    await page.evaluate(() => scrollTo(0, 0));
    const titleLines = await page.locator('#kai-suite > header h1').evaluate(node => {
      const rows = new Map(), walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const text = walker.currentNode;
        for (let i = 0; i < text.length; i++) {
          const range = document.createRange(); range.setStart(text, i); range.setEnd(text, i + 1);
          const rect = range.getBoundingClientRect(), y = Math.round(rect.top);
          rows.set(y, (rows.get(y) || '') + text.textContent[i]);
        }
      }
      return [...rows.values()].map(row => row.trim()).filter(Boolean);
    });
    assert(titleLines.every(row => row.length > 1), `Mobile official title must not strand a single character: ${JSON.stringify(titleLines)}`);
    await page.screenshot({path: path.join(output, 'official-overview-mobile.png')});
    await page.setViewportSize({width: 1440, height: 1000});

    const tasks = [...new Set(batch.trials.map(row => row.task))];
    for (const task of tasks) {
      const trial = batch.trials.find(row => row.task === task && row.staged.stages.some(stage => stage.reached === true))
        || batch.trials.find(row => row.task === task);
      const trialURL = new URL(trial.report_path, portal).href;
      for (const filename of ['report.html', 'workbench.json', 'report.json', 'visualization.json', 'staged_evaluation.json', 'goal_references.json']) {
        const relative = trial.report_path.replace(/report\.html$/, filename);
        assert.equal(await artifact(page, new URL(relative, portal).href), await artifact(page, new URL(relative, parent).href),
          `${task}: archived ${filename} remains byte-identical to parent`);
      }
      await page.goto(trialURL); await ready(page);
      const config = await embedded(page, 'kai-workbench-config'), payload = await embedded(page, 'kai-evaluation-data');
      const title = config.task_presentation?.title || batch.tasks[task].title;
      assert.equal(await page.title(), title + ' | KaiTactileSim');
      assert.equal(await page.locator('.wb-header h1').textContent(), title + ' · 轨迹分析');
      assert.match(await page.locator('.wb-eyebrow').textContent(), /KaiTactileSim Evaluation Platform/);
      assert.match(await page.locator('.wb-footer').textContent(), /KaiTactileSim/);
      assert.equal(new URL(await page.getByRole('link', {name: '评估总览', exact: true}).getAttribute('href'), page.url()).href, new URL(portal).href);
      if (config.previous_workbench_url) assert.equal(await page.getByRole('link', {name: '历史版本', exact: true}).count(), 1);
      assert.equal(config.task, task); assert.equal(config.trial_id, trial.trial_id);
      assert.deepEqual(config.staged_evaluation.stages, trial.staged.stages);
      const badges = await page.locator('.wb-header-stages .wb-stage-badge').evaluateAll(nodes => nodes.map(node => ({
        key: node.dataset.stage, reached: node.dataset.reached, text: node.textContent,
      })));
      assert.equal(badges.length, trial.staged.stages.length);
      trial.staged.stages.forEach((stage, i) => {
        assert.equal(badges[i].key, stage.key);
        assert.equal(badges[i].reached, stage.reached === true ? 'true' : stage.reached === false ? 'false' : 'unknown');
        assert(badges[i].text.includes(stage.label));
      });
      const native = trial.staged.native_result.task_success;
      assert.match(await page.locator('.wb-verdict').textContent(), native === true ? /原任务结果.*成功/ : native === false ? /原任务结果.*失败/ : /原任务结果.*未知/);
      if (!task.startsWith('whiteboard-wipe')) assert.match(await page.locator('.wb-staged-scope').textContent(), /诊断记录.*不覆盖.*原任务判定/);
      const before = await page.locator('.wb-header-stages').innerHTML();
      await seek(page, '.ke-slider', Number(await page.locator('.ke-slider').getAttribute('max')) * .5);
      assert.equal(await page.locator('.wb-header-stages').innerHTML(), before, 'Branding preserves cumulative stage semantics');
      await seek(page, '.ke-slider', 0);
      assert.equal(await page.locator('.wb-chart').count(), 4); assert.equal(await page.locator('.wb-finger').count(), 5);
      assert.equal(await page.locator('.ke-goal-reference').count(), payload.visualization.task_series.reduce((sum, item) => sum + (item.goal_references?.length || 0), 0));
      if (task.startsWith('whiteboard-wipe')) {
        assert.match(await page.locator('.wb-header-stages').textContent(), /不能视为擦拭完成/);
        for (const key of ['effective_pad_contact_area_ratio', 'contact_time_weighted_mean_effective_area_ratio']) assert.equal(await page.locator(`[data-task-key="${key}"]`).count(), 1);
      }
      const wrist = page.locator('.wb-cameras .wb-camera[data-camera="wrist"]');
      assert.equal(await wrist.count(), 1); await wrist.click();
      assert.equal(await page.locator('.ke-video').getAttribute('data-camera'), 'wrist');
      await page.locator('.wb-cameras .wb-camera[data-camera="mosaic"]').click();
      await page.evaluate(() => scrollTo(0, 0)); await noOverflow(page, `${task} branded desktop`);
      await page.screenshot({path: path.join(output, task + '-official-desktop.png')});
      await page.setViewportSize({width: 390, height: 844}); await page.evaluate(() => scrollTo(0, 0));
      await noOverflow(page, `${task} branded mobile`);
      await page.screenshot({path: path.join(output, task + '-official-mobile.png')});
      await page.setViewportSize({width: 1440, height: 1000});
      assert.equal(await page.locator('.wb-expert-heading strong').textContent(), '专家示范');
      assert.equal(await page.locator('.wb-expert-panel').evaluate(node => node.open), false);
      await page.locator('.wb-expert-panel > summary').click();
      await page.waitForFunction(() => !document.querySelector('.wb-expert-slider').disabled);
      const modelTime = await page.locator('.ke-slider').inputValue();
      const expertTime = Number(await page.locator('.wb-expert-slider').getAttribute('max')) / 3;
      await seek(page, '.wb-expert-slider', expertTime);
      await page.waitForFunction(time => {const video = document.querySelector('.wb-expert-video');
        return video.readyState >= 2 && !video.seeking && Math.abs(video.currentTime - time) < .05;}, expertTime);
      await page.locator('.wb-expert-play').click();
      await page.waitForFunction(time => {const video = document.querySelector('.wb-expert-video');
        return !video.paused && video.currentTime > time + .2;}, expertTime);
      await page.locator('.wb-expert-play').click();
      assert.equal(await page.locator('.wb-expert-video').evaluate(video => video.paused), true);
      assert.equal(await page.locator('.ke-slider').inputValue(), modelTime, 'Expert playback stays independent of model clock');
      if (task === 'usb-insert') {
        await page.locator('.wb-expert-panel').screenshot({path: path.join(output, 'official-expert-demonstration.png')});
        await page.locator('.wb-footer').screenshot({path: path.join(output, 'official-detail-footer.png')});
      }
    }
    for (const trial of batch.trials.filter(row => row.staged.native_result.task_success === true)) {
      await page.goto(new URL(trial.report_path, portal).href); await ready(page);
      assert.match(await page.locator('.wb-verdict').textContent(), /原任务结果.*成功/);
      assert(await page.locator('.wb-header-stages .wb-stage-badge[data-reached="false"]').count() > 0,
        'Official platform does not upgrade partial stage records into full staged success');
    }
    await page.goto(parent); await page.waitForSelector('.ks-case');
    assert.notEqual(await page.locator('#kai-suite > header h1').textContent(), 'KaiTactileSim 官方评估平台', 'Preserved parent keeps its original branding');
    await statisticsPosition(page);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed: true, tasks: tasks.length, trials: batch.trial_count,
      checks: ['official Chinese/English identity and footer', 'all task trajectory titles and formal navigation labels',
        'same-rule statistics remain below count toolbar', 'cards/list/filter/search', '1440 desktop and 390 mobile no overflow',
        'all task expert demonstration videos play independently', 'native/stage/protocol labels never upgraded',
        'wrist views, tactile charts, goal references and area unchanged', 'nine representative archived HTML/JSON byte-identical', 'parent preserved']}));
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
