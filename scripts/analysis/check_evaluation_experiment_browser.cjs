'use strict';

// Optional local-only QA with an already installed Chromium and Playwright.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

async function main() {
  const [target, output] = process.argv.slice(2);
  assert(target && output, 'usage: node check_evaluation_experiment_browser.cjs URL QA_DIRECTORY');
  const browser = await chromium.launch({headless: true,
    ...(process.env.KAI_EVAL_CHROME ? {executablePath: process.env.KAI_EVAL_CHROME} : {}),
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const errors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1600, height: 1100}});
    page.on('pageerror', error => errors.push(String(error)));
    assert.equal((await page.goto(target, {waitUntil: 'domcontentloaded'})).status(), 200);
    await page.waitForSelector('.kex-group');
    const catalog = await page.evaluate(() => JSON.parse(document.getElementById('evaluation-status-data').textContent));
    assert.equal(catalog.experiment.member_trials, 450);
    assert.equal(catalog.experiment.eligible_trials, 430);
    assert.equal(catalog.experiment.quarantined_trials, 20);
    assert.equal(await page.locator('#kex-member-count').innerText(), '450');
    assert.equal(await page.locator('#kex-eligible-count').innerText(), '430');
    assert.equal(await page.locator('#kex-quarantine-count').innerText(), '20');
    assert.equal(await page.locator('.kex-task').count(), 9);
    assert.equal(await page.locator('.kex-checkpoint').count(), 4);
    assert.equal(await page.locator('.kex-group').count(), 36);
    assert.equal(await page.locator('.kex-detail-link').count(), 450);
    assert.equal(await page.locator('.kex-trial[data-publication-status="quarantined_replay_only"]').count(), 20);
    for (const [checkpoint, count] of [['10000', 45], ['15000', 45], ['20000', 180], ['24999', 180]]) {
      assert.equal(catalog.groups.filter(g => g.checkpoint === checkpoint).reduce((n, g) => n + g.trials.length, 0), count);
    }
    fs.mkdirSync(output, {recursive: true});
    await page.screenshot({path: path.join(output, 'overview.png'), fullPage: false});
    await page.locator('.kex-task[data-task="usb-unplug"]').click();
    assert.equal(await page.locator('.kex-group').count(), 4);
    assert.deepEqual(await page.locator('.kex-group').evaluateAll(nodes => nodes.map(node => node.dataset.checkpoint)),
      ['10000', '15000', '20000', '24999']);
    const usbText = await page.locator('#kex-groups').innerText();
    assert(usbText.includes('extracted_and_held_2s') || usbText.includes('2 秒') || usbText.includes('2s'));
    assert(usbText.includes('released_stably_on_table') || usbText.includes('放稳') || usbText.includes('放回'));
    await page.screenshot({path: path.join(output, 'checkpoint_comparison.png'), fullPage: false});
    await page.locator('#kex-clear-filters').click();
    await page.locator('.kex-checkpoint[data-checkpoint="20000"]').click();
    assert.equal(await page.locator('.kex-group').count(), 9);
    assert((await page.locator('#kex-visible-count').innerText()).includes('180 条成员（160 条计分，20 条隔离）'));
    await page.locator('.kex-task[data-task="whiteboard-wipe-horizontal-line"]').click();
    assert.equal(await page.locator('.kex-group').count(), 1);
    assert((await page.locator('.kex-group [data-score="native"]').innerText()).includes('未定义'));
    const quarantineLink = await page.locator('.kex-detail-link').first().getAttribute('href');
    const quarantine = await browser.newPage();
    quarantine.on('pageerror', error => errors.push(String(error)));
    assert.equal((await quarantine.goto(new URL(quarantineLink, page.url()).href)).status(), 200);
    assert((await quarantine.locator('body').innerText()).includes('仅回看，不计成绩'));
    const preserved = await quarantine.locator('.historical-replay-link').getAttribute('href');
    assert.equal((await page.request.head(new URL(preserved, quarantine.url()).href)).status(), 200);
    await page.locator('#kex-clear-filters').click();
    const latest = catalog.groups.find(group => group.checkpoint === '10000');
    const details = await browser.newPage();
    assert.equal((await details.goto(new URL(latest.trials[0].detail_url, page.url()).href)).status(), 200);
    const media = await details.locator('video').getAttribute('src');
    const ranged = await page.request.get(new URL(media, details.url()).href, {headers: {Range: 'bytes=0-255'}});
    assert.equal(ranged.status(), 206);
    assert.equal((await ranged.body()).length, 256);
    const egoLink = page.locator('.kex-external-links a').filter({hasText: 'EgoTouch'});
    assert.equal(await egoLink.count(), 1);
    const ego = await browser.newPage();
    ego.on('pageerror', error => errors.push(String(error)));
    assert.equal((await ego.goto(await egoLink.getAttribute('href'))).status(), 200);
    await ego.waitForSelector('.kes-group');
    assert.equal(await ego.locator('.kes-group').count(), 6);
    assert.equal(await ego.locator('.kes-detail-link').count(), 120);
    const egoData = await ego.evaluate(() => JSON.parse(document.getElementById('evaluation-status-data').textContent));
    assert(egoData.groups.every(group => group.model === 'egotouch'));
    const egoDetail = await ego.locator('.kes-detail-link').first().getAttribute('href');
    assert.equal((await page.request.head(egoDetail)).status(), 200);
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true);
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: false});
    const root = await browser.newPage();
    assert.equal((await root.goto(new URL('/', target).href)).status(), 200);
    await root.waitForURL(target);
    assert.deepEqual(errors, []);
    const result = {status: 'passed', tasks: 9, groups: 36, members: 450, eligible: 430, quarantined: 20,
      checkpoints: {'10000': 45, '15000': 45, '20000': 180, '24999': 180},
      ego_groups: 6, ego_trials: 120, quarantine_notice_and_old_replay: 'passed',
      video_range: '206 / 256 bytes', mobile_overflow: false, browser_errors: errors};
    fs.writeFileSync(path.join(output, 'browser_verification.json'), JSON.stringify(result, null, 2) + '\n');
    process.stdout.write(JSON.stringify(result) + '\n');
  } finally { await browser.close(); }
}

main().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1; });
