/* Real local website QA; use explicitly installed Playwright/Chromium. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const url = process.argv[2], output = process.argv[3]; fs.mkdirSync(output,{recursive:true});
  const browser = await chromium.launch({executablePath:process.env.KAI_CHROMIUM_EXECUTABLE,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
  try {
    const page = await browser.newPage({viewport:{width:1700,height:1100}}), errors = [];
    page.on('pageerror',error => errors.push(error.message));
    await page.goto(url); await page.waitForSelector('.hub-task-card');
    assert.equal(await page.locator('.hub-task-card').count(),12);
    const total = await page.locator('tbody tr').count(); assert(total >= 2);
    const horizonAudit = await page.evaluate(() => JSON.parse(document.getElementById('hub-data').textContent).horizon_selection);
    if (horizonAudit?.required_prediction_horizon === 50) {
      assert(await page.locator('.hub-horizon-notice').textContent().then(text => text.includes('H50-only')));
      assert.equal(horizonAudit.included_count,total);
      assert(await page.evaluate(() => JSON.parse(document.getElementById('hub-data').textContent).trials.every(trial => trial.prediction_horizon === 50)));
    }
    await page.screenshot({path:path.join(output,'hub-desktop.png')});
    await page.locator('.hub-filters select').nth(0).selectOption('pick-place');
    assert.equal(await page.locator('tbody tr').count(),2);
    await page.locator('.hub-filters select').nth(1).selectOption('pass');
    assert.equal(await page.locator('tbody tr').count(),1);
    await page.locator('.hub-filters select').nth(1).selectOption('');
    await page.locator('tbody input[type=checkbox]').nth(0).check();
    await page.locator('tbody input[type=checkbox]').nth(1).check();
    await page.getByRole('button',{name:'载入所选轨迹'}).click();
    await page.waitForFunction(() => document.querySelectorAll('.hub-pane .kai-eval').length === 2);
    const sliders = page.locator('.hub-pane input[type=range]');
    const secondBefore = await sliders.nth(1).inputValue();
    const max = Number(await sliders.nth(0).getAttribute('max'));
    await sliders.nth(0).fill((max / 2).toFixed(3)); await sliders.nth(0).dispatchEvent('input');
    assert.equal(await sliders.nth(1).inputValue(),secondBefore,'comparison clocks must remain independent');
    await page.locator('.hub-pane .ke-spatial-mode').nth(0).getByText('接触力圆图').click();
    await page.locator('.hub-pane .ke-spatial-mode').nth(1).getByText('接触力圆图').click();
    const scales = await page.locator('.hub-pane .ke-glyph-legend-row span').allTextContents(); assert.equal(scales[0],scales[1]);
    assert.equal(await page.locator('.ke-v2-evidence').count(),2);
    const audit = await page.locator('.hub-pane .ke-header-links a').first().getAttribute('href'); assert(/^[^/]+\/audit.html$/.test(audit));
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),'desktop overflow');
    await page.locator('.hub-panes').screenshot({path:path.join(output,'hub-comparison.png')});
    await page.getByRole('button',{name:'清空',exact:true}).click(); assert.equal(await page.locator('.kai-eval').count(),0);
    await page.locator('.hub-filters select').nth(0).selectOption('');
    await page.locator('.hub-filters input').fill('<script>'); assert.equal(await page.locator('tbody tr').count(),0);
    await page.locator('.hub-filters input').fill(''); assert.equal(await page.locator('tbody tr').count(),total);
    const fixture = await page.evaluate(() => {
      // Explicit synthetic rendering test only; no archive/report is changed.
      const host = document.createElement('div'); document.body.appendChild(host);
      const controller = KaiEvaluationDashboard.mount(host,{report:{suite_schema_version:'synthetic-browser-test',
        trial:{task:'synthetic UI regression',trial_id:'not-a-simulation'},
        task_review:{review:{status:'unknown'}}, contact_quality:{contact_pairs:{test:{body_a:'a',body_b:'b',
          contact:{status:'missing',observed_duration_s:0},normal_load:{overload:{status:'missing',observed_duration_s:0}}}}},
        reference_comparison:{status:'available',metrics:[{scope:'fixture',channel:'fixture',metric:'fixture',unit:'N',value:3,status:'available',reference_group_id:'test'}]},
        reference_distributions:{test:{statistics:{p05:1,p25:2,median:3,p75:4,p95:5}}}},visualization:{}});
      const readings = [...host.querySelectorAll('.ke-quality-values strong')].map(el => el.textContent);
      const result = {plots:host.querySelectorAll('.ke-reference-distribution').length,contact:readings[0],overload:readings[5]};
      controller.destroy(); host.remove(); return result;
    });
    assert.equal(fixture.plots,1); assert.equal(fixture.contact,'— s'); assert.equal(fixture.overload,'— s');
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),'mobile overflow');
    await page.screenshot({path:path.join(output,'hub-mobile.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({passed:true,trials:total,checks:['12-task coverage','task/outcome/search filters','two independent clocks','shared glyph range','V2 evidence cards','correct audit URLs','clear destroys views','synthetic reference distribution rendering','missing evidence not rendered as zero','desktop/mobile no overflow','no browser errors']}));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
