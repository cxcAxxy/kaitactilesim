/* Optional browser QA. Use an installed Playwright and browser, no downloads.
 * KAI_PLAYWRIGHT_MODULE=/path/to/playwright
 * KAI_CHROMIUM_EXECUTABLE=/path/to/chrome
 * node tests/test_evaluation_dashboard_browser.cjs /path/to/report.html /tmp/qa
 */
"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const {pathToFileURL} = require("node:url");
const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || "playwright");

(async () => {
  const reportURL = /^https?:\/\//.test(process.argv[2]) ? process.argv[2] : pathToFileURL(path.resolve(process.argv[2])).href;
  const output = path.resolve(process.argv[3]);
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({
    executablePath: process.env.KAI_CHROMIUM_EXECUTABLE || undefined,
    headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1080}});
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(reportURL);
    await page.waitForSelector(".kai-eval");
    const original = await page.evaluate(() => {
      const data = JSON.parse(document.getElementById("kai-evaluation-data").textContent);
      return {channels: data.visualization.channels.length,
              samples: data.visualization.tactile_series.time_s.length,
              glyph: !!data.visualization.contact_glyph,
              duration: data.visualization.tactile_series.time_s.at(-1)};
    });
    assert(original.channels >= 1);
    assert(await page.locator(".kai-eval canvas").count() >= 3, "Expected several real-data charts");
    const slider = page.locator('.kai-eval input[type="range"]').first();
    await slider.fill(String(Math.min(45, original.duration / 2)));
    await slider.dispatchEvent("input");
    await page.waitForTimeout(200);
    if (await page.locator(".kai-eval video").count()) {
      await page.waitForFunction(() => {
        const video = document.querySelector(".kai-eval video");
        return video.readyState >= 2 && !video.seeking;
      });
      const synchronization = await page.evaluate(() => {
        const data = JSON.parse(document.getElementById("kai-evaluation-data").textContent);
        const media = data.visualization.media;
        if (media.sync_status !== "mapped") return null;
        const t = Number(document.querySelector('.kai-eval input[type="range"]').value);
        let i = -1;
        media.time_s.forEach((time, index) => {if (time <= t + 1e-8) i = index;});
        return {expected: media.video_time_s[i], actual: document.querySelector(".kai-eval video").currentTime};
      });
      if (synchronization) assert(Math.abs(synchronization.expected - synchronization.actual) < .001, "Video seek must match explicit mapping: " + JSON.stringify(synchronization));
    }
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Desktop overflow");
    await page.screenshot({path: path.join(output, "desktop.png")});
    await page.screenshot({path: path.join(output, "desktop-full.png"), fullPage: true});
    if (original.glyph) {
      const glyphButton = page.locator('.ke-spatial-mode button').filter({hasText: '接触力圆图'});
      await glyphButton.click();
      assert.equal(await glyphButton.getAttribute('aria-pressed'), 'true');
      assert.equal(await page.locator('.ke-glyph-grid').isVisible(), true);
      assert.equal(await page.locator('.ke-heat-grid').isVisible(), false);
      assert.equal(await page.locator('.ke-force-glyph:visible').count(), original.channels);
      await page.locator('.ke-spatial-panel').screenshot({path: path.join(output, 'force-circle-desktop.png')});
      const reading = await page.locator('.ke-glyph-reading').first().textContent();
      assert(!/NaN|Infinity|undefined/.test(reading), 'Circle readings must be finite or explicitly unknown');
      const nativeReadings = await page.evaluate(() => {
        const data = JSON.parse(document.getElementById('kai-evaluation-data').textContent);
        const glyph = data.visualization.contact_glyph;
        const sampleText = document.querySelector('.ke-spatial-clock strong').textContent;
        const sampledTime = Number(sampleText.match(/[0-9]+(?:\.[0-9]+)?/)[0]);
        let index = 0;
        glyph.time_s.forEach((time, i) => {
          if (Math.abs(time - sampledTime) < Math.abs(glyph.time_s[index] - sampledTime)) index = i;
        });
        const expected = glyph.normal_force_n[index].map(value => typeof value === 'number'
          ? value.toLocaleString('zh-CN', {minimumFractionDigits: 3, maximumFractionDigits: 3})
          : '—');
        const actual = [...document.querySelectorAll('.ke-glyph-reading')]
          .map(node => node.textContent.match(/^Fn (.*?) N/)[1]);
        return {actual, expected};
      });
      assert.deepEqual(nativeReadings.actual, nativeReadings.expected, 'Circle Fn values match the displayed native sample');
      // Start of a trial has no 0.1 s history; never invent a zero change.
      await slider.fill('0');
      await slider.dispatchEvent('input');
      assert(!/NaN|Infinity|undefined/.test(await page.locator('.ke-glyph-grid').textContent()));
      assert.match(await page.locator('.ke-glyph-normal-change').first().textContent(), /—|未知|不可用/,
        'No history means unknown normal-force change, not zero');
      assert.match(await page.locator('.ke-glyph-tangent-reading').first().textContent(), /—|未知|不可用/,
        'No history means unknown tangent-force change, not zero');
      await slider.fill(String(Math.min(45, original.duration / 2)));
      await slider.dispatchEvent('input');
    }
    if (await page.locator('.kai-eval input[type="checkbox"]').count()) {
      const checkbox = page.locator('.kai-eval input[type="checkbox"]').first();
      await checkbox.uncheck();
      assert.equal(await checkbox.isChecked(), false);
      await checkbox.check();
    }
    await page.setViewportSize({width: 390, height: 844});
    await page.evaluate(() => scrollTo(0, 0));
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Mobile overflow");
    await page.screenshot({path: path.join(output, "mobile.png")});
    await page.screenshot({path: path.join(output, "mobile-full.png"), fullPage: true});
    if (original.glyph) {
      await page.locator('.ke-spatial-panel').screenshot({path: path.join(output, 'force-circle-mobile.png')});
      await page.locator('.ke-spatial-mode button').filter({hasText: '阵列热图'}).click();
      assert.equal(await page.locator('.ke-heat-grid').isVisible(), true);
      assert.equal(await page.locator('.ke-glyph-grid').isVisible(), false);
    }
    await page.setViewportSize({width: 1440, height: 1080});
    // Mount and destroy a second instance without altering the first instance.
    const multi = await page.evaluate(() => {
      const data = JSON.parse(document.getElementById("kai-evaluation-data").textContent);
      data.visualization.media = {status: "missing", sync_status: "unverified", reason: "QA no media"};
      const host = document.createElement("div");
      document.body.append(host);
      const instance = KaiEvaluationDashboard.mount(host, data);
      const count = document.querySelectorAll(".kai-eval").length;
      let isolated = true;
      if (data.visualization.contact_glyph) {
        const buttons = host.querySelectorAll('.ke-spatial-mode button');
        [...buttons].find(button => button.textContent.includes('接触力圆图')).click();
        const first = document.querySelector('.kai-eval .ke-spatial-mode button[aria-pressed="true"]');
        isolated = first?.textContent.includes('阵列热图');
      }
      instance.destroy();
      const remaining = document.querySelectorAll(".kai-eval").length;
      host.remove();
      return {count, remaining, isolated};
    });
    assert.equal(multi.count, 2);
    assert.equal(multi.remaining, 1);
    assert.equal(multi.isolated, true, 'View selection is local to each dashboard instance');
    if (original.glyph) {
      const legacy = await page.evaluate(() => {
        const data = JSON.parse(document.getElementById('kai-evaluation-data').textContent);
        delete data.visualization.contact_glyph;
        data.visualization.media = {status: 'missing'};
        const host = document.createElement('div');
        document.body.append(host);
        const instance = KaiEvaluationDashboard.mount(host, data);
        [...host.querySelectorAll('.ke-spatial-mode button')]
          .find(button => button.textContent.includes('接触力圆图')).click();
        const notice = host.querySelector('.ke-glyph-unavailable');
        const result = {visible: notice.getBoundingClientRect().height > 0,
          text: notice.textContent, circles: host.querySelectorAll('.ke-force-glyph').length};
        instance.destroy();
        host.remove();
        return result;
      });
      assert.equal(legacy.visible, true);
      assert.equal(legacy.circles, 0);
      assert.match(legacy.text, /未提供|未导出|不可用/);
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({status: "passed", ...original, screenshots: output,
      checks: ["real-data charts", "time scrub", "channel toggle", ...(original.glyph ? ["heatmap/force-circle toggle", "native force glyphs", "isolated view modes"] : []), "desktop", "mobile", "two instances", "destroy", "no script errors"]}));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exit(1);});
