'use strict';

// Optional live QA. Use an already installed Playwright/Chromium; no downloads.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

async function main() {
  const target = process.argv[2];
  const output = process.argv[3];
  assert(target && output, 'usage: node check_evaluation_status_browser.cjs URL QA_DIRECTORY');
  const browser = await chromium.launch({headless: true,
    ...(process.env.KAI_EVAL_CHROME ? {executablePath: process.env.KAI_EVAL_CHROME} : {}),
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const errors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    page.on('pageerror', error => errors.push(String(error)));
    assert.equal((await page.goto(target, {waitUntil: 'domcontentloaded'})).status(), 200);
    await page.waitForSelector('.kes-group');
    const catalog = await page.evaluate(() => JSON.parse(document.getElementById('evaluation-status-data').textContent));
    assert.equal(await page.locator('.kes-task[data-task]').count(), catalog.tasks.length);
    assert.equal(await page.locator('.kes-group').count(), catalog.groups.length);
    const source = page.locator('#es-source-kind, #kes-source-kind');
    await source.selectOption('job');
    const latest = catalog.groups.filter(group => group.source_kind === 'job');
    assert.equal(await page.locator('.kes-group').count(), latest.length);
    assert.equal(latest.reduce((count, group) => count + group.valid_trials, 0), 90);
    assert.equal(latest.reduce((count, group) => count + group.native_success_count, 0), 0);
    const latestLink = await page.locator('.kes-detail-link').first().getAttribute('href');
    assert(latestLink, 'latest trial detail is linked');
    const details = await browser.newPage();
    details.on('pageerror', error => errors.push(String(error)));
    assert.equal((await details.goto(new URL(latestLink, page.url()).href)).status(), 200);
    const video = details.locator('video');
    assert.equal(await video.count(), 1);
    const media = await video.getAttribute('src');
    const ranged = await page.request.get(new URL(media, details.url()).href, {headers: {Range: 'bytes=0-255'}});
    assert.equal(ranged.status(), 206);
    assert.equal((await ranged.body()).length, 256);
    await page.locator('.kes-task[data-task="usb-unplug"]').click();
    assert.equal(await page.locator('.kes-group').count(), 2);
    await page.locator('.kes-task[data-task="usb-unplug"]').click();
    await source.selectOption('archive');
    await page.locator('#es-model, #kes-model').selectOption('egotouch');
    assert.equal(await page.locator('.kes-group').count(), 6);
    await page.locator('#es-model, #kes-model').selectOption('');
    await source.selectOption('');
    assert.equal(await page.locator('.kes-detail-link').count(), catalog.groups.reduce((n, group) => n + group.trials.length, 0));
    const preserved = catalog.groups.flatMap(group => group.trials).find(trial => trial.detail_type === 'preserved_workbench');
    assert(preserved, 'preserved detail is referenced by summary SHA');
    assert.equal((await page.request.head(new URL(preserved.detail_url, page.url()).href)).status(), 200);
    await page.locator('.kes-task[data-task="bulb-screw-inplace"]').click();
    assert.equal(await page.locator('.kes-group').count(), 0);
    assert((await page.locator('body').innerText()).includes('暂无评估'));
    await page.locator('.kes-task[data-task="bulb-screw-inplace"]').click();
    fs.mkdirSync(output, {recursive: true});
    await page.screenshot({path: path.join(output, 'overview.png'), fullPage: false});
    await details.screenshot({path: path.join(output, 'latest_trial.png'), fullPage: false});
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), true);
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: false});
    assert.deepEqual(errors, []);
    const result = {status: 'passed', tasks: catalog.tasks.length, groups: catalog.groups.length,
      trials: catalog.groups.reduce((n, group) => n + group.trials.length, 0), latest_groups: latest.length,
      latest_valid_trials: 90, latest_native_successes: 0, media_range: '206 / 256 bytes', browser_errors: errors};
    fs.writeFileSync(path.join(output, 'browser_verification.json'), JSON.stringify(result, null, 2) + '\n');
    process.stdout.write(JSON.stringify(result) + '\n');
  } finally {
    await browser.close();
  }
}

main().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1; });
