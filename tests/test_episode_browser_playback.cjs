// Read-only browser regression against an already running episode browser.
// Usage: NODE_PATH=<directory containing playwright> node tests/test_episode_browser_playback.cjs
// Optional: EPISODE_BROWSER_URL=http://127.0.0.1:8000/
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({ headless: true, args: ['--no-proxy-server'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    // Reproduce clients where original AV1 videos cannot be used. No original
    // source should be requested: the page must use prepared H.264 copies.
    const originalRequests = [];
    await page.route(/\/media\/\d+\/[^/]+$/, route => {
      originalRequests.push(route.request().url());
      return route.abort();
    });
    page.setDefaultTimeout(120000);
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.EPISODE_BROWSER_URL || 'http://127.0.0.1:8000/');
    await page.locator('.episode-item').first().click();
    await page.waitForFunction(() => document.querySelectorAll('video').length > 0
      && [...document.querySelectorAll('video')].every(video => video.readyState >= 2));
    const layout = await page.evaluate(() => ({
      welcome: getComputedStyle(document.querySelector('#emptyState')).display,
      cameraTop: document.querySelector('#cameraGrid').getBoundingClientRect().top,
      sidebarHeight: document.querySelector('.sidebar').clientHeight,
      viewportHeight: innerHeight,
      listHeight: document.querySelector('.episode-list').clientHeight,
      listScrollHeight: document.querySelector('.episode-list').scrollHeight,
    }));
    assert.equal(layout.welcome, 'none');
    assert(layout.cameraTop < 150, 'Player should be visible immediately after selection');
    assert(layout.sidebarHeight <= layout.viewportHeight, 'Episode list must not stretch the page');
    assert(layout.listScrollHeight > layout.listHeight, 'Episode list should scroll independently');
    assert.equal(await page.locator('.finger-heatmap').count(), 20);
    assert(await page.evaluate(() => [...document.querySelectorAll('video')]
      .every(video => new URL(video.currentSrc).pathname.endsWith('/h264'))));
    await page.waitForFunction(() => document.querySelector('.finger-heatmap:not(.is-missing)'));
    const firstHeatmap = await page.locator('.finger-heatmap').first().getAttribute('aria-label');
    assert(firstHeatmap.includes('SDK raw'));

    // A real timeupdate Event must never be interpreted as the numeric time.
    await page.evaluate(() => {
      const video = document.querySelector('video');
      video.currentTime = 2;
      video.dispatchEvent(new Event('timeupdate'));
    });
    assert.equal(await page.locator('#currentTime').textContent(), '00:02.0');
    await page.locator('#playButton').click();
    await page.waitForFunction(() => [...document.querySelectorAll('video')]
      .every(video => !video.paused && video.currentTime > 3));
    await page.locator('#playButton').click();
    assert(await page.evaluate(() => [...document.querySelectorAll('video')]
      .every(video => video.paused && !video.error
        && video.getVideoPlaybackQuality().totalVideoFrames > 0)));

    await page.locator('#timeline').fill('10');
    await page.locator('#timeline').dispatchEvent('change');
    await page.waitForFunction(() => [...document.querySelectorAll('video')]
      .every(video => !video.seeking && Math.abs(video.currentTime - 10) < 0.1));
    await page.locator('#speedSelect').selectOption('2');
    await page.locator('#playButton').click();
    await page.waitForFunction(() => [...document.querySelectorAll('video')]
      .every(video => Math.abs(video.playbackRate - 2) <= 0.11 && video.currentTime > 11));

    const previousTitle = await page.locator('#activeTitle').textContent();
    await page.locator('#nextButton').click();
    await page.waitForFunction(title => document.querySelector('#activeTitle').textContent !== title
      && [...document.querySelectorAll('video')].every(video => video.readyState >= 2 && video.paused), previousTitle);
    await page.locator('#playButton').click();
    await page.waitForFunction(() => [...document.querySelectorAll('video')]
      .every(video => !video.paused && video.currentTime > 1));
    await page.locator('#playButton').click();
    assert.deepEqual(errors, []);
    assert.deepEqual(originalRequests, []);
    if (process.env.EPISODE_BROWSER_SCREENSHOT) {
      await page.screenshot({path: process.env.EPISODE_BROWSER_SCREENSHOT, fullPage: true});
    }
    console.log('PASS: H.264 playback without AV1, 20 heatmaps, player visibility, timeupdate, play/pause, seeking, speed, episode switching');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
