'use strict';
// Optional, local-only browser QA; reuse the existing report server and cached browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');

async function artifact(page, url) {
  const response = await page.request.get(url);
  assert(response.ok(), `Local artifact must exist: ${url}`);
  return response.text();
}
async function position(page) {
  assert(await page.locator('.ks-stage-groups').evaluate(node =>
    node.previousElementSibling?.classList.contains('ks-results-bar') && node.nextElementSibling?.id === 'ks-trials'),
  'Statistics stay immediately below displayed-count toolbar and above trajectory grid');
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'No document horizontal overflow');
}
async function screenshot(page, output, name) {
  await page.locator('.ks-results-bar').evaluate(node => scrollTo(0, Math.max(0, node.getBoundingClientRect().top + scrollY - 60)));
  await page.screenshot({path: path.join(output, name + '.png')});
}

(async () => {
  const [portal, parent, output] = process.argv.slice(2);
  if (!portal || !parent || !output) throw new Error('Expected new portal URL, parent portal URL, and QA output directory');
  assert.notEqual(new URL(portal).href, new URL(parent).href);
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({executablePath: process.env.KAI_CHROMIUM_EXECUTABLE,
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const errors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(portal);
    await page.waitForSelector('.ks-case');
    const batch = JSON.parse(await page.locator('#kai-suite-data').textContent());
    const oldBatch = JSON.parse(await artifact(page, new URL('batch.json', parent).href));
    assert.equal(batch.trial_count, 360);
    assert.equal(batch.task_count, 9);
    assert.deepEqual(batch.trials, oldBatch.trials);
    assert.deepEqual(batch.stage_groups, oldBatch.stage_groups);
    assert.equal(await page.locator('.ks-case').count(), 360);
    assert.match(await page.locator('#ks-visible').textContent(), /360\s*\/\s*360/);
    assert.equal(await page.locator('.ks-stage-groups').evaluate(node => node.open), false);
    await position(page);
    await screenshot(page, output, 'statistics-collapsed-desktop');
    await page.locator('.ks-stage-groups > summary').click();
    assert.equal(await page.locator('.ks-stage-groups').evaluate(node => node.open), true);
    assert.equal(await page.locator('#ks-stage-groups > tr').count(), batch.stage_groups.length);
    await position(page);
    await screenshot(page, output, 'statistics-expanded-desktop');
    await page.locator('#ks-task-tabs button[data-task="usb-insert"]').click();
    const usbGroups = batch.stage_groups.filter(row => row.task === 'usb-insert');
    assert.equal(await page.locator('#ks-stage-groups > tr').count(), usbGroups.length);
    assert.equal(await page.locator('.ks-case').count(), 40);
    await page.locator('#ks-checkpoint').selectOption('24999');
    const filteredGroups = usbGroups.filter(row => String(row.checkpoint) === '24999');
    assert.equal(await page.locator('#ks-stage-groups > tr').count(), filteredGroups.length);
    const texts = await page.locator('#ks-stage-groups > tr').allTextContents();
    filteredGroups.forEach((group, i) => {
      assert(texts[i].includes(group.title + ' · ' + group.checkpoint));
      group.stages.forEach(stage => assert(texts[i].includes(stage.label + ' ' + stage.reached + '/' + group.count)));
      assert(texts[i].includes(group.native_success_count + '/' + group.count));
    });
    assert.equal(await page.locator('.ks-case').count(), 20);
    assert.equal(await page.locator('.ks-stage-groups').evaluate(node => node.open), true, 'Filtering does not close the disclosure');
    await position(page);
    await page.locator('#ks-view-list').click();
    assert.equal(await page.locator('#ks-trials').getAttribute('data-view'), 'list');
    assert.equal(await page.locator('.ks-case').count(), 20);
    await position(page);
    await screenshot(page, output, 'statistics-filtered-list-desktop');
    await page.setViewportSize({width: 390, height: 844});
    await position(page);
    await screenshot(page, output, 'statistics-filtered-expanded-mobile');
    const scroll = page.locator('.ks-group-scroll');
    assert(await scroll.evaluate(node => node.scrollWidth > node.clientWidth), 'Wide statistics table scrolls within its own mobile container');
    await scroll.evaluate(node => { node.scrollLeft = node.scrollWidth; });
    await position(page);
    await page.locator('.ks-stage-groups > summary').click();
    assert.equal(await page.locator('.ks-stage-groups').evaluate(node => node.open), false);
    await position(page);
    await screenshot(page, output, 'statistics-filtered-collapsed-mobile');
    await page.locator('#ks-view-cards').click();
    assert.equal(await page.locator('#ks-trials').getAttribute('data-view'), 'cards');
    assert.equal(await page.locator('.ks-case').count(), 20);
    await position(page);
    await screenshot(page, output, 'statistics-filtered-cards-mobile');
    await page.reload(); await page.waitForSelector('.ks-case');
    assert.equal(await page.locator('.ks-stage-groups').evaluate(node => node.open), false, 'Disclosure is collapsed on a fresh page');
    assert.equal(await page.locator('.ks-case').count(), 360);
    await position(page);
    for (const task of Object.keys(batch.tasks)) {
      const trial = batch.trials.find(row => row.task === task);
      assert.equal(await artifact(page, new URL(trial.report_path, portal).href), await artifact(page, new URL(trial.report_path, parent).href), `${task}: detail HTML and embedded records unchanged`);
    }
    await page.goto(parent); await page.waitForSelector('.ks-case');
    assert(await page.locator('.ks-stage-groups').evaluate(node => node.previousElementSibling?.id === 'ks-trials'), 'Parent retains its original below-grid statistics position');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed: true, trial_count: 360, task_count: 9,
      checks: ['statistics below count toolbar before trajectories', 'collapsed default and expansion',
        'task/checkpoint group counts and values', 'cards/list without moving statistics', '1440 desktop and 390 mobile no overflow',
        'mobile table internal scrolling', 'native trials/stage groups/detail evidence unchanged', 'parent position unchanged']}));
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
