(function () {
  'use strict';
  const batch = JSON.parse(document.getElementById('kai-suite-data').textContent), trials = batch.trials;
  const state = {task: '', model: '', checkpoint: '', outcome: '', search: '', stage: ''};
  const stagedProfile = batch.evaluation_profile === 'recorded-staged';
  const outcomes = {pass: '记录为通过', fail: '记录为未通过', unknown: '记录未知'};
  const platformTitle = 'KaiTactileSim 官方评估平台';
  const viewStorageKey = 'kai-evaluation-suite-view-v1';
  let view = 'cards';
  try { if (typeof localStorage !== 'undefined' && localStorage.getItem(viewStorageKey) === 'list') view = 'list'; } catch (_) { /* Browser storage is optional, including file:// previews. */ }
  function el(tag, cls, parent, text) { const n = document.createElement(tag); n.className = cls; if (text !== undefined) n.textContent = text; parent.appendChild(n); return n; }
  const counts = document.getElementById('ks-counts');
  [[batch.task_count, '类任务'], [batch.trial_count, '条 H50 轨迹']].forEach(function (item) {
    const card = el('div', 'ks-stat', counts); el('strong', '', card, item[0]); el('span', '', card, item[1]);
  });
  // Upgrade the existing static shell without changing archived batch or trial data.
  const grid = document.getElementById('ks-trials'), visible = document.getElementById('ks-visible');
  const viewButtons = [];
  if (typeof document.querySelector === 'function') {
    const header = document.querySelector('#kai-suite > header');
    if (header) {
      const title = header.querySelector('h1'); if (title) title.textContent = platformTitle;
      const brand = header.querySelector('.ks-eyebrow'); if (brand) brand.textContent = 'KaiTactileSim Evaluation Platform';
      const subtitle = header.querySelector('p:not(.ks-eyebrow)'); if (subtitle) subtitle.textContent = '机器人操作任务评估 · 动作回放 · 触觉分析';
      const previous = header.querySelector('.ks-old'); if (previous) previous.textContent = '历史 H50 版本';
      header.appendChild(counts);
    }
    document.title = platformTitle + ' | H50';
    const main = document.getElementById('kai-suite');
    if (main && !document.querySelector('.ks-brand-footer')) {
      el('p', 'ks-brand-footer', main, platformTitle + ' · 评估规则与结果以各批次归档为准');
    }
    const groupSummary = document.querySelector('.ks-stage-groups');
    if (groupSummary && grid.parentNode) grid.parentNode.insertBefore(groupSummary, grid);
  }
  if (visible.parentNode) {
    const bar = document.createElement('div'); bar.className = 'ks-results-bar';
    visible.parentNode.insertBefore(bar, visible); bar.appendChild(visible);
    const controls = el('div', 'ks-view-controls', bar); controls.setAttribute('role', 'group'); controls.setAttribute('aria-label', '轨迹显示方式');
    [['cards', '卡片'], ['list', '紧凑列表']].forEach(item => {
      const button = el('button', 'ks-view-button', controls, item[1]); button.type = 'button'; button.id = 'ks-view-' + item[0]; button.dataset.view = item[0];
      button.setAttribute('aria-controls', 'ks-trials');
      button.addEventListener('click', () => {
        view = item[0];
        try { if (typeof localStorage !== 'undefined') localStorage.setItem(viewStorageKey, view); } catch (_) { /* A denied preference write must not affect navigation. */ }
        renderView();
      });
      viewButtons.push(button);
    });
  }
  function renderView() {
    grid.dataset.view = view;
    viewButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
  }
  const tabs = document.getElementById('ks-task-tabs'), buttons = [];
  function taskButton(task, title, count) {
    const button = el('button', '', tabs, title + ' · ' + count); button.type = 'button'; button.dataset.task = task;
    button.addEventListener('click', () => { state.task = task; updateOptions(); render(); }); buttons.push(button);
  }
  taskButton('', '全部任务', trials.length);
  Object.entries(batch.tasks).forEach(function (item) { taskButton(item[0], item[1].title, item[1].trial_count); });
  function updateOptions() {
    ['model', 'checkpoint'].forEach(key => {
      const select = document.getElementById('ks-' + key), values = [...new Set(trials.filter(t => !state.task || t.task === state.task).map(t => t[key]))].sort((a, b) => String(a).localeCompare(String(b), undefined, {numeric: true}));
      if (!values.includes(state[key])) state[key] = '';
      select.replaceChildren(); const all = el('option', '', select, '全部'); all.value = '';
      values.forEach(value => { const option = el('option', '', select, value); option.value = value; }); select.value = state[key];
    });
  }
  ['model', 'checkpoint', 'outcome'].forEach(key => document.getElementById('ks-' + key).addEventListener('change', event => { state[key] = event.target.value; render(); }));
  const stageSelect = document.getElementById('ks-stage');
  if (stageSelect) stageSelect.addEventListener('change', event => { state.stage = event.target.value; render(); });
  function stageCategory(t) {
    if (t.staged.complete_success === true) return 'complete';
    return t.staged.stages.some(s => s.reached === true) ? 'partial' : 'none';
  }
  function renderGroups() {
    const body = document.getElementById('ks-stage-groups'); if (!body) return;
    body.replaceChildren();
    (batch.stage_groups || []).filter(g => ['task', 'model', 'checkpoint'].every(key => !state[key] || String(g[key]) === state[key])).forEach(g => {
      const row = el('tr', '', body);
      el('td', '', row, g.title + ' · ' + g.checkpoint); el('td', '', row, g.count);
      el('td', '', row, g.stages.map(s => s.label + ' ' + s.reached + '/' + g.count + (s.unknown ? '（未知 ' + s.unknown + '）' : '')).join(' → '));
      el('td', '', row, g.complete_success_defined ? g.complete_success_count + '/' + g.count : '未定义');
      el('td', '', row, g.native_success_count + '/' + g.count);
      const sha = el('td', 'ks-rule-id', row, g.rule_sha256.slice(0, 12)); sha.title = g.rule_sha256 + '\n' + g.cohort + '\n权重: ' + g.checkpoint_identity;
    });
  }
  document.getElementById('ks-search').addEventListener('input', event => { state.search = event.target.value.trim().toLowerCase(); render(); });
  function render() {
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.task === state.task)));
    const filtered = trials.filter(t => ['task', 'model', 'checkpoint', 'outcome'].every(key => !state[key] || String(t[key]) === state[key]) &&
      (!state.stage || (t.staged && stageCategory(t) === state.stage)) &&
      (!state.search || (t.task + ' ' + t.model + ' ' + t.checkpoint + ' seed_' + String(t.seed).padStart(3, '0') + ' ' + t.trial_id).toLowerCase().includes(state.search)));
    visible.textContent = '显示 ' + filtered.length + ' / ' + trials.length + ' 条轨迹';
    const note = document.getElementById('ks-task-note'); note.replaceChildren();
    if (state.task) { const info = batch.tasks[state.task]; el('h2', '', note, info.title); el('p', '', note, info.task_note || '原记录结果与任务响应分开展示。'); }
    else el('p', '', note, '选择任务后查看对应的动作、任务响应和接触载荷；各任务采用自己的记录字段和单位。');
    grid.replaceChildren(); renderView();
    let display = filtered;
    if (!state.task) {
      const groups = new Map(); filtered.forEach(t => { if (!groups.has(t.task)) groups.set(t.task, []); groups.get(t.task).push(t); });
      display = [];
      while (display.length < filtered.length) groups.forEach(group => { if (group.length) display.push(group.shift()); });
    }
    display.forEach(t => {
      if (!/^[A-Za-z0-9_-]+\/report\.html$/.test(t.report_path)) return;
      const card = el('a', 'ks-case', grid); card.href = t.report_path;
      const main = el('div', 'ks-case-main', card), top = el('div', 'ks-case-top', main);
      el('strong', '', top, t.title);
      const native = t.staged && t.staged.native_result ? t.staged.native_result.task_success : null;
      const nativeClass = native === true ? 'pass' : native === false ? 'fail' : 'unknown';
      if (t.staged) el('span', 'ks-native-status ks-native-' + nativeClass, top, '原任务 · ' + (native === true ? '成功' : native === false ? '未成功' : '未知'));
      el('p', 'ks-identity', main, t.model + ' · step ' + t.checkpoint + ' · seed ' + (Number.isInteger(t.seed) ? String(t.seed).padStart(3, '0') : '未知') + ' · H50');
      el('p', 'ks-outcome ks-' + t.outcome, main, (stagedProfile ? '数据审核口径 · ' : '') + (outcomes[t.outcome] || outcomes.unknown));
      const progress = el('div', 'ks-case-progress', card);
      if (t.staged) {
        const stageCount = t.staged.stages.filter(s => s.reached === true).length;
        el('p', 'ks-stage-caption', progress, '累计阶段 ' + stageCount + ' / ' + t.staged.stages.length);
        const stages = el('div', 'ks-stages', progress);
        t.staged.stages.forEach(s => el('span', s.reached === true ? 'ks-reached' : '', stages, (s.reached === true ? '✓ ' : s.reached === false ? '— ' : '? ') + s.label));
        el('p', 'ks-stage-result', progress, '完整阶段：' + (t.staged.complete_success === null ? '未定义（仅拿起）' : t.staged.complete_success ? '通过' : '未通过') + ' · 原任务：' + (native === true ? '成功' : native === false ? '未成功' : '未知'));
        if (t.staged.complete_success === null) el('p', 'ks-stage-limit', progress, '仅记录拿起阶段，不代表擦拭完成');
        if (t.staged.source_selection && t.staged.source_selection.group_status === 'failed') el('p', 'ks-source-warning', progress, '该试验记录完整，但所在运行批次标为失败');
      }
      const footer = el('div', 'ks-case-footer', card), coverage = el('div', 'ks-case-coverage', footer);
      [['Fn / Ft', t.tactile === 'available' && t.tangent === 'available'], ['专家示范', t.expert === 'available'], [t.plot_kind === 'model_only' ? '模型全程图' : '参考对照图', t.plots === 'available']].forEach(item => el('span', item[1] ? '' : 'ks-missing', coverage, (item[1] ? '✓ ' : '— ') + item[0]));
      el('span', 'ks-open', footer, '轨迹分析 →');
    });
    if (!filtered.length) el('p', 'ks-empty', grid, '没有符合当前筛选条件的轨迹。');
    renderGroups();
  }
  updateOptions(); render();
})();
