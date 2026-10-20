'use strict';

// Local-only acceptance with an already installed Playwright/Chromium.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

async function main() {
  const [target, output] = process.argv.slice(2);
  assert(target && output, 'usage: node check_tactile_experiment_browser.cjs URL QA_DIRECTORY');
  const browser = await chromium.launch({headless: true,
    ...(process.env.KAI_EVAL_CHROME ? {executablePath: process.env.KAI_EVAL_CHROME} : {}),
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const errors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1500, height: 1050}});
    page.on('pageerror', error => errors.push(String(error)));
    assert.equal((await page.goto(target, {waitUntil: 'domcontentloaded'})).status(), 200);
    await page.waitForSelector('.ktp-group');
    const data = await page.evaluate(() => JSON.parse(document.getElementById('tactile-experiment-data').textContent));
    assert.equal(data.schema, 'kaihand-tactile-policy-experiment-v1');
    const counts = data.experiment.counts;
    assert.equal(counts.planned, 25);
    assert.equal(counts.observed, 25);
    assert.equal(counts.missing, 0);
    assert.equal(counts.valid + counts.invalid, 25);
    assert.equal(await page.locator('#ktp-planned-count').innerText(), '25');
    assert.equal(await page.locator('#ktp-valid-count').innerText(), String(counts.valid));
    assert.equal(await page.locator('.ktp-group').count(), 5);
    assert.equal(await page.locator('.ktp-detail-link').count(), 25);
    assert.deepEqual(data.groups.map(group => group.label),
      ['cpm_best', 'ftp1_step10000', 'ftp1_step15000', 'ftp1_step20000', 'ftp1_step24999']);
    assert.deepEqual(data.groups.map(group => group.identity.prediction_horizon), [30, 32, 32, 32, 32]);
    assert(data.groups.every(group => group.identity.execute_steps === 5 && group.counts.planned === 5));
    assert(data.groups.every(group => group.trials.length === 5));
    const expectedCpmDrift = process.env.KAI_EVAL_EXPECT_CPM_WEIGHT_DRIFT === '1';
    if (expectedCpmDrift) {
      // Explicit negative acceptance: do not turn a changed checkpoint into
      // verified stages or a misleading zero-percent staged score.
      const cpm = data.groups[0];
      assert.equal(cpm.native.known, 5);
      assert.equal(cpm.staged.known, 0);
      assert.equal(cpm.staged.success_rate, null);
      assert.equal(cpm.staged.evidence_counts.invalid, 5);
      assert(cpm.trials.every(trial => trial.staged_evidence_status === 'invalid'
        && trial.staged_complete_success === null
        && trial.reasons.some(reason => reason.startsWith('strict_staged_import_failed:Tactile inventory file SHA mismatch: ')
          && reason.endsWith('/best.pt'))));
      assert(data.groups.slice(1).every(group => group.staged.known === 5));
      assert.equal(await page.locator('.ktp-group[data-group="cpm_best"] [data-score="staged"] strong').innerText(), '未定义');
    } else {
      assert(data.groups.every(group => group.staged.known === group.native.known),
        'Every eligible benchmark trial should retain strict staged evidence');
    }
    fs.mkdirSync(output, {recursive: true});
    await page.screenshot({path: path.join(output, 'overview.png'), fullPage: false});
    await page.locator('.ktp-group-tab[data-group="ftp1_step20000"]').click();
    assert.equal(await page.locator('.ktp-group').count(), 1);
    assert.equal(await page.locator('.ktp-detail-link').count(), 5);
    await page.locator('.ktp-group-tab[data-group="ftp1_step20000"]').click();
    const details = await browser.newPage();
    details.on('pageerror', error => errors.push(String(error)));
    let videos = 0;
    for (const group of data.groups) {
      for (const trial of group.trials) {
        assert.equal((await page.request.head(new URL(trial.detail_url, target).href)).status(), 200);
        if (trial.video_url) {
          assert.equal((await page.request.head(new URL(trial.video_url, target).href)).status(), 200);
          videos += 1;
        }
      }
    }
    assert.equal(videos, 25, 'Every requested trial should retain its review video');
    const withVideo = data.groups.flatMap(group => group.trials).find(trial => trial.video_url);
    assert(withVideo, 'At least one trial has a complete review video');
    assert.equal((await details.goto(new URL(withVideo.detail_url, target).href)).status(), 200);
    assert.equal(await details.locator('video').count(), 1);
    const video = await details.locator('video').getAttribute('src');
    const response = await page.request.get(new URL(video, details.url()).href, {headers: {Range: 'bytes=0-255'}});
    assert.equal(response.status(), 206);
    assert.equal((await response.body()).length, 256);
    const back = await details.locator('a').first().getAttribute('href');
    assert.equal(new URL(back, details.url()).href, target);
    await details.screenshot({path: path.join(output, 'trial.png'), fullPage: false});
    const h50 = page.locator('.ktp-links a').filter({hasText: 'H50'});
    const ego = page.locator('.ktp-links a').filter({hasText: 'EgoTouch'});
    assert.equal(await h50.count(), 1); assert.equal(await ego.count(), 1);
    assert.equal((await page.request.head(new URL(await h50.getAttribute('href'), target).href)).status(), 200);
    assert.equal((await page.request.head(new URL(await ego.getAttribute('href'), target).href)).status(), 200);
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true);
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: false});
    const portal = await browser.newPage();
    assert.equal((await portal.goto(new URL('/', target).href)).status(), 200);
    const links = await portal.locator('a.card').evaluateAll(nodes => nodes.map(node => node.href));
    assert(links.includes(target));
    assert.equal(links.length, 3);
    assert.deepEqual(errors, []);
    const result = {status: 'passed', groups: 5, counts, linked_details: 25, linked_videos: videos,
      native_horizons: [30, 32, 32, 32, 32], media_range: '206 / 256 bytes', mobile_overflow: false,
      root_portal_links: 3, browser_errors: errors,
      strict_staged_verified: data.groups.reduce((sum, group) => sum + group.staged.known, 0),
      cpm_checkpoint_drift_excluded: expectedCpmDrift};
    fs.writeFileSync(path.join(output, 'browser_verification.json'), JSON.stringify(result, null, 2) + '\n');
    process.stdout.write(JSON.stringify(result) + '\n');
  } finally { await browser.close(); }
}

main().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1; });
