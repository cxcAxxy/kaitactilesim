'use strict';

// Local-only verification using the already installed browser runtime.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

function eligible(trial) {
  return trial.valid === true && !String(trial.publication_status || '').startsWith('quarantined') &&
    !['invalid', 'missing'].includes(trial.publication_status);
}
function ratio(successes, known) {
  return known ? successes + ' / ' + known + ' (' + (100 * successes / known).toFixed(1) + '%)' : '未定义';
}
function expectedGroup(group) {
  const valid = group.trials.filter(eligible);
  const native = valid.filter(trial => typeof trial.native_success === 'boolean');
  const staged = valid.filter(trial => trial.staged_evidence_status === 'valid');
  const complete = staged.filter(trial => typeof trial.complete_stage_success === 'boolean');
  const stages = new Map();
  group.trials.forEach(trial => (trial.stages || []).forEach(stage => {
    const key = JSON.stringify([stage.key, stage.name]);
    if (!stages.has(key)) stages.set(key, {key, known: 0, successes: 0});
  }));
  staged.forEach(trial => (trial.stages || []).forEach(stage => {
    if (typeof stage.reached !== 'boolean') return;
    const value = stages.get(JSON.stringify([stage.key, stage.name]));
    value.known += 1; value.successes += Number(stage.reached);
  }));
  return {members: group.trials.length, eligible: valid.length,
    native: ratio(native.filter(trial => trial.native_success).length, native.length),
    complete: ratio(complete.filter(trial => trial.complete_stage_success).length, complete.length),
    stages: [...stages.values()].map(stage => ratio(stage.successes, stage.known))};
}

async function main() {
  const [target, output] = process.argv.slice(2);
  assert(target && output, 'usage: node check_unified_workbench_browser.cjs URL QA_DIRECTORY');
  const browser = await chromium.launch({headless: true,
    ...(process.env.KAI_EVAL_CHROME ? {executablePath: process.env.KAI_EVAL_CHROME} : {}),
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const errors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1500, height: 1050}});
    page.on('pageerror', error => errors.push(String(error)));
    assert.equal((await page.goto(target, {waitUntil: 'domcontentloaded'})).status(), 200);
    await page.waitForSelector('.ks-case');
    const data = await page.evaluate(() => JSON.parse(document.getElementById('kai-suite-data').textContent));
    assert.equal(data.schema, 'kaihand-unified-evaluation-workbench-v1');
    assert.equal(data.counts.members, 595);
    assert.equal(data.counts.eligible, 575);
    assert.equal(data.counts.quarantined, 20);
    assert.equal(data.groups.length, 47);
    assert.equal(data.tasks.length, 10);
    assert.equal(await page.locator('h1').innerText(), 'KaiTactileSim 官方评估平台');
    assert.equal(await page.locator('.ks-case').count(), 595);
    assert.equal(await page.locator('#ks-summary-mode').inputValue(), 'policy');
    assert.equal(await page.locator('#ks-comparison tr').count(), 47);
    assert.equal(await page.locator('#ks-historical-summary').count(), 0,
      'The old 360 records belong to the one 595-record summary, not a second table');
    assert.equal(data.historical_summary.rows.length, 18, 'Original counts remain in audit data');
    assert.equal(await page.locator('#ks-comparison-title').innerText(), '595 条成功率与阶段汇总');
    const total = await page.locator('#ks-summary-total').innerText();
    assert.match(total, /595.*575.*20/);
    assert.doesNotMatch(total, /%|16\s*\/\s*575|3\s*\/\s*320/,
      'The overview must show coverage, not a mixed-model success rate');
    const headers = await page.locator('.ks-comparison-panel thead th').allTextContents();
    assert.equal(headers.length, 8);
    assert.match(headers[1], /^模型$/);
    assert.match(headers[2], /^checkpoint$/i);
    const policyRows = await page.locator('#ks-comparison tr').evaluateAll(nodes => nodes.map(row => ({
      id: row.dataset.group,
      task: row.dataset.task, modelKey: row.dataset.model, checkpointKey: row.dataset.checkpoint,
      summaryMode: row.dataset.summaryMode, protocol: row.dataset.protocol, cohort: row.dataset.cohort,
      cells: [...row.children].map(cell => cell.innerText.trim()),
      model: row.querySelector('.ks-summary-model-cell')?.innerText.trim(),
      checkpoint: row.querySelector('.ks-summary-checkpoint')?.innerText.trim(),
      stages: [...row.querySelectorAll('.ks-summary-stages > p')].map(node => node.innerText.trim()),
    })));
    assert.equal(new Set(policyRows.map(row => row.id)).size, 47);
    assert.equal(new Set(policyRows.map(row => JSON.stringify([row.task, row.modelKey, row.checkpointKey]))).size, 47);
    const completedTasks = new Set(); let currentTask;
    policyRows.forEach(row => {
      if (row.task !== currentTask) {
        assert(!completedTasks.has(row.task), 'All policy rows for one task must be contiguous');
        if (currentTask !== undefined) completedTasks.add(currentTask);
        currentTask = row.task;
      }
    });
    for (const group of data.groups) {
      const row = policyRows.find(item => item.task === group.task && item.modelKey === group.model &&
        item.checkpointKey === String(group.checkpoint));
      assert(row, 'Every original task/model/checkpoint/protocol group needs its own default row: ' + group.id);
      assert.equal(row.id, group.id);
      assert.equal(row.summaryMode, 'policy');
      assert(row.protocol, 'Every policy row must retain a protocol identity');
      assert.equal(row.cohort, String(group.cohort_id));
      assert.equal(row.cells.length, 8);
      assert(row.model, 'The model must have its own table cell');
      const modelName = {pi05: 'π0.5', egotouch: 'EgoTouch', ftp1: 'FTP-1', 'pi05+cpm': 'CPM'}[group.model];
      assert(row.model.startsWith(modelName), group.id + ' model identity must not be merged with other policies');
      assert.equal(row.checkpoint, /^\d+$/.test(group.checkpoint) ? 'step ' + group.checkpoint : group.checkpoint,
        'Checkpoints must not be combined into the model identity cell');
      const expected = expectedGroup(group);
      assert(row.cells[4].startsWith(expected.eligible + ' / ' + expected.members), group.id + ' eligible denominator');
      assert(row.cells[5].startsWith(expected.native), group.id + ' native denominator must stay within this group');
      assert(row.cells[6].startsWith(expected.complete), group.id + ' complete-stage denominator must stay within this group');
      expected.stages.forEach(value => assert(row.stages.some(stage => stage.endsWith('：' + value)),
        group.id + ' stage denominator must stay within this group: ' + value));
    }
    const pokerRows = policyRows.filter(row => row.task === 'poker-draw');
    assert.equal(pokerRows.length, 9);
    assert(policyRows.slice(0, 9).every(row => row.task === 'poker-draw'),
      'The first nine default rows must contain all Poker models, including CPM and FTP-1');
    const completedModels = new Set(); let currentModel;
    pokerRows.forEach(row => {
      if (row.modelKey !== currentModel) {
        assert(!completedModels.has(row.modelKey), 'Poker rows for one model must be contiguous');
        if (currentModel !== undefined) completedModels.add(currentModel);
        currentModel = row.modelKey;
      }
    });
    for (const model of ['pi05', 'ftp1']) {
      const checkpoints = pokerRows.filter(row => row.modelKey === model).map(row => row.checkpointKey);
      assert.deepEqual(checkpoints, ['10000', '15000', '20000', '24999'],
        'Poker checkpoints must be ascending within each model');
    }
    const models = await page.locator('#ks-model option').evaluateAll(nodes => nodes.map(node => node.value).filter(Boolean).sort());
    assert.deepEqual(models, ['egotouch', 'ftp1', 'pi05', 'pi05+cpm']);
    const evidence = data.groups.flatMap(group => group.trials.map(trial => ({group, trial})));
    assert.equal(evidence.filter(({trial}) => trial.valid === true).length, 575);
    assert.equal(evidence.filter(({trial}) => trial.valid === true && trial.native_success === true).length, 16);
    assert.equal(evidence.filter(({trial}) => trial.publication_status === 'quarantined_replay_only').length, 20);
    fs.mkdirSync(output, {recursive: true});
    await page.screenshot({path: path.join(output, 'overview.png'), fullPage: false});

    await page.selectOption('#ks-summary-mode', 'task');
    assert.equal(await page.locator('#ks-comparison tr').count(), 10);
    assert.doesNotMatch(await page.locator('#ks-summary-total').innerText(), /%|16\s*\/\s*575|3\s*\/\s*320/);
    await page.screenshot({path: path.join(output, 'optional_task_summary.png'), fullPage: false});
    await page.selectOption('#ks-summary-mode', 'policy');
    assert.equal(await page.locator('#ks-comparison tr').count(), 47);
    await page.locator('.ks-task-tabs button[data-task="poker-draw"]').click();
    assert.equal(await page.locator('#ks-comparison tr').count(), 9);
    const pokerPolicies = await page.locator('#ks-comparison tr').evaluateAll(nodes => nodes.map(row => ({
      id: row.dataset.group, n: row.querySelector('.ks-summary-n').innerText,
      native: row.children[5].innerText,
    })));
    pokerPolicies.forEach(row => assert.match(row.n, /^(?:5 \/ 5|20 \/ 20)$/,
      'Poker policy rows must retain five- or twenty-trial denominators, never the pooled 75'));
    const pi05Poker20k = data.groups.find(group => group.task === 'poker-draw' && group.model === 'pi05' && group.checkpoint === '20000');
    assert.match(pokerPolicies.find(row => row.id === pi05Poker20k.id).native, /^1 \/ 20 \(5\.0%\)/);
    await page.locator('.ks-task-tabs button[data-task=""]').click();

    await page.locator('.ks-task-tabs button[data-task="install-ram"]').click();
    const ramModels = await page.locator('#ks-model option').evaluateAll(nodes => nodes.map(node => node.value).filter(Boolean).sort());
    assert.deepEqual(ramModels, models, 'CPM and FTP-1 must not disappear on other tasks');
    await page.selectOption('#ks-model', 'pi05+cpm');
    assert.equal(await page.locator('#ks-model').inputValue(), 'pi05+cpm');
    assert.equal(await page.locator('.ks-case').count(), 0);
    assert.match(await page.locator('#ks-comparison').innerText(), /该任务暂无该模型评估记录/);
    assert.match(await page.locator('#ks-summary-total').innerText(), /0.*0.*0/);
    assert.doesNotMatch(await page.locator('#ks-summary-total').innerText(), /%|0\s*\/\s*0/);
    await page.selectOption('#ks-model', 'ftp1');
    assert.equal(await page.locator('#ks-model').inputValue(), 'ftp1');
    assert.equal(await page.locator('.ks-case').count(), 0);
    await page.screenshot({path: path.join(output, 'no_task_model_data.png'), fullPage: false});
    for (const task of ['usb-insert', 'usb-unplug']) {
      await page.locator('.ks-task-tabs button[data-task="' + task + '"]').click();
      const taskModels = await page.locator('#ks-model option').evaluateAll(nodes => nodes.map(node => node.value).filter(Boolean).sort());
      assert.deepEqual(taskModels, models);
      for (const model of ['pi05+cpm', 'ftp1']) {
        await page.selectOption('#ks-model', model);
        assert.equal(await page.locator('#ks-model').inputValue(), model);
        assert.equal(await page.locator('.ks-case').count(), 0);
        assert.match(await page.locator('#ks-comparison').innerText(), /该任务暂无该模型评估记录/);
      }
    }

    await page.goto(target, {waitUntil: 'domcontentloaded'});
    await page.waitForSelector('.ks-case');
    assert.equal(await page.locator('#ks-summary-mode').inputValue(), 'policy');
    assert.equal(await page.locator('#ks-comparison tr').count(), 47);
    await page.selectOption('#ks-summary-mode', 'task');
    await page.locator('.ks-task-tabs button[data-task="poker-draw"]').click();
    assert.equal(await page.locator('.ks-case').count(), 75);
    assert.equal(await page.locator('#ks-comparison tr').count(), 1);
    const poker = await page.locator('#ks-comparison').innerText();
    assert.match(poker, /CPM/); assert.match(poker, /FTP-1/);
    assert.match(poker, /51 \/ 70/);
    await page.selectOption('#ks-summary-mode', 'policy');
    assert.equal(await page.locator('#ks-comparison tr').count(), 9);
    await page.selectOption('#ks-model', 'ftp1');
    assert.equal(await page.locator('.ks-case').count(), 20);
    assert.equal(await page.locator('#ks-comparison tr').count(), 4);
    await page.selectOption('#ks-checkpoint', '24999');
    assert.equal(await page.locator('.ks-case').count(), 5);
    assert.equal(await page.locator('#ks-comparison tr').count(), 1);
    const before = await page.locator('#ks-comparison').innerText();
    assert.match(before, /2 \/ 5 \(40\.0%\)/);
    await page.selectOption('#ks-outcome', 'pass');
    assert.equal(await page.locator('.ks-case').count(), 0);
    assert.equal(await page.locator('#ks-comparison').innerText(), before,
      'Outcome filtering must not change success-rate denominators');
    await page.selectOption('#ks-outcome', '');
    await page.selectOption('#ks-stage', 'none');
    assert.equal(await page.locator('#ks-comparison').innerText(), before,
      'Stage filtering must not change success-rate denominators');
    await page.selectOption('#ks-stage', '');
    await page.selectOption('#ks-model', 'pi05+cpm');
    assert.equal(await page.locator('.ks-case').count(), 5);
    const cpm = await page.locator('#ks-comparison').innerText();
    assert.match(cpm, /未定义/);
    assert(!/4 \/ 5/.test(cpm), 'CPM unverified recorded stages must not become a score');
    await page.screenshot({path: path.join(output, 'cpm_exclusion.png'), fullPage: false});

    await page.goto(target, {waitUntil: 'domcontentloaded'});
    await page.waitForSelector('.ks-case');
    await page.locator('.ks-task-tabs button[data-task="whiteboard-wipe-horizontal-line"]').click();
    await page.selectOption('#ks-checkpoint', '20000');
    assert.equal(await page.locator('.ks-case').count(), 20);
    const quarantined = await page.locator('#ks-comparison').innerText();
    assert.match(quarantined, /排除 20/);
    assert.match(quarantined, /未定义/);
    await page.locator('#ks-view-list').click();
    assert.equal(await page.locator('#ks-trials').getAttribute('data-view'), 'list');
    await page.fill('#ks-search', 'seed_010');
    assert.equal(await page.locator('.ks-case').count(), 1);
    assert.equal(await page.locator('#ks-comparison').innerText(), quarantined,
      'Search filtering must not change success-rate denominators');

    await page.goto(target, {waitUntil: 'domcontentloaded'});
    await page.waitForSelector('.ks-case');
    await page.locator('.ks-task-tabs button[data-task="usb-unplug"]').click();
    assert.equal(await page.locator('#ks-comparison tr').count(), 4);
    await page.selectOption('#ks-summary-mode', 'task');
    const usbOut = await page.locator('#ks-comparison').innerText();
    assert.match(usbOut, /3 \/ 10/);
    assert.match(usbOut, /0 \/ 40/);
    assert.match(usbOut, /完成判据不同，分别统计/);
    assert(!/3 \/ 50/.test(usbOut), 'Distinct completion definitions must not become one stage rate');
    await page.locator('.ks-task-tabs button[data-task="whiteboard-wipe"]').click();
    const whiteboard = await page.locator('#ks-comparison').innerText();
    assert.match(whiteboard, /8\.99%/); assert.match(whiteboard, /51\.57%/);

    await page.goto(target, {waitUntil: 'domcontentloaded'});
    await page.waitForSelector('.ks-case');
    await page.selectOption('#ks-model', 'egotouch');
    assert.equal(await page.locator('.ks-case').count(), 120);
    assert.equal(await page.locator('#ks-comparison tr').count(), 6);
    await page.selectOption('#ks-summary-mode', 'task');
    assert.equal(await page.locator('#ks-comparison tr').count(), 1);
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true);
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: false});

    let linkedVideos = 0;
    for (let i = 0; i < evidence.length; i += 8) {
      await Promise.all(evidence.slice(i, i + 8).map(async ({trial}) => {
        assert(trial.detail_url, 'Every retained trial needs a readable detail or quarantine notice');
        assert.equal((await page.request.head(new URL(trial.detail_url, target).href)).status(), 200);
        if (trial.video_url) {
          assert.equal((await page.request.head(new URL(trial.video_url, target).href)).status(), 200);
          linkedVideos += 1;
        }
      }));
    }
    const ftpVideo = evidence.find(({group, trial}) => group.model === 'ftp1' && trial.video_url).trial;
    const range = await page.request.get(new URL(ftpVideo.video_url, target).href, {headers: {Range: 'bytes=0-255'}});
    assert.equal(range.status(), 206);
    assert.equal((await range.body()).length, 256);
    const portal = await browser.newPage();
    await portal.goto(new URL('/', target).href, {waitUntil: 'domcontentloaded'});
    await portal.waitForSelector('#kai-suite .ks-case');
    assert.equal(await portal.locator('.ks-case').count(), 595);
    assert.equal(await portal.locator('a.card').count(), 0, 'Root must not be an experiment selector');
    assert.equal(await portal.locator('#ks-summary-mode').inputValue(), 'policy');
    assert.equal(await portal.locator('#ks-comparison tr').count(), 47);
    assert.doesNotMatch(await portal.locator('#ks-summary-total').innerText(), /%|16\s*\/\s*575|3\s*\/\s*320/);
    assert.deepEqual(errors, []);
    const result = {status: 'passed', groups: 47, members: 595, eligible: 575, quarantined: 20,
      tasks: 10, model_options: models, merged_summary_tables: 1,
      default_policy_comparison_rows: 47, optional_task_summary_rows: 10, historical_audit_rows: 18,
      global_model_options_retained_on_other_tasks: true, zero_data_is_not_zero_success: true,
      separate_model_checkpoint_columns: true, group_denominators_checked: 47,
      task_model_checkpoint_dataset_identity_checked: true, task_rows_contiguous: true,
      poker_policy_rows: 9, poker_checkpoint_order: 'ascending within model',
      coverage_only_overview: true, mixed_model_success_rate_displayed: false,
      distinct_usb_completion_definitions_retained: true, whiteboard_known_weighting: 'passed',
      details_checked: 595, videos_checked: linkedVideos, denominator_filter_independence: 'passed',
      cpm_invalid_stages_excluded: true, quarantine_excluded: true, mobile_overflow: false,
      media_range: '206 / 256 bytes', root_is_unified_workbench: true, browser_errors: errors};
    fs.writeFileSync(path.join(output, 'browser_verification.json'), JSON.stringify(result, null, 2) + '\n');
    process.stdout.write(JSON.stringify(result) + '\n');
  } finally { await browser.close(); }
}

main().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1; });
