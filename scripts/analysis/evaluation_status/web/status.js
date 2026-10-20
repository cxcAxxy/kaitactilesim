(function (global, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (global && global.document) {
    global.KaiEvaluationStatus = api;
    const start = () => api.mount(global.document);
    if (global.document.readyState === 'loading') {
      global.document.addEventListener('DOMContentLoaded', start, {once: true});
    } else start();
  }
})(typeof globalThis === 'undefined' ? this : globalThis, function () {
  'use strict';

  const LIVE_STATES = new Set(['queued', 'pending', 'starting', 'running', 'in_progress']);
  const MODE_LABELS = {diagnostic: '诊断记录', formal: '来源标记 formal', unverified: '口径未核验'};
  const SOURCE_LABELS = {job: '最新任务', archive: '历史归档'};
  const PROTOCOL_LABELS = {
    max_sim_seconds: '时长', guard_mode: '守卫', controller: '控制器',
    action_interpolation: '插值', success_hold_seconds: '成功保持',
  };

  function array(value) { return Array.isArray(value) ? value : []; }
  function integer(value) { return Number.isInteger(value) && value >= 0 ? value : null; }
  function text(value, fallback) {
    return typeof value === 'string' && value.trim() ? value.trim() : fallback;
  }
  function el(document, tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined && content !== null) node.textContent = String(content);
    return node;
  }
  function append(parent, ...children) {
    children.filter(Boolean).forEach(child => parent.appendChild(child));
    return parent;
  }
  function finite(value) { return typeof value === 'number' && Number.isFinite(value); }
  function countLabel(value, denominator) {
    const count = integer(value);
    const known = integer(denominator);
    return `${count === null ? '未知' : count} / ${known === null ? '未知' : known}`;
  }
  function safeUrl(value, base) {
    if (typeof value !== 'string' || !value.trim() || /[\\\0\r\n]/.test(value)) return null;
    if (/%2e/i.test(value)) return null;
    const parentSegments = value.split(/[?#]/, 1)[0].split('/').filter(segment => segment === '..');
    const leadingSibling = value.startsWith('../') && parentSegments.length === 1;
    if (parentSegments.length && !leadingSibling) return null;
    try {
      const url = new URL(value, base || 'http://localhost/');
      if (!['http:', 'https:'].includes(url.protocol)) return null;
      if (/%2e/i.test(url.pathname) || url.pathname.split('/').includes('..')) return null;
      if (leadingSibling && url.origin !== new URL(base || 'http://localhost/').origin) return null;
      return url.href;
    } catch (_) { return null; }
  }
  function protocolLabel(protocol) {
    if (!protocol || typeof protocol !== 'object' || Array.isArray(protocol)) return '协议未声明';
    const order = Object.keys(PROTOCOL_LABELS);
    const keys = [...order.filter(key => key in protocol),
      ...Object.keys(protocol).filter(key => !order.includes(key)).sort()];
    if (!keys.length) return '协议未声明';
    return keys.map(key => {
      let value = protocol[key];
      if (key === 'max_sim_seconds' && finite(value)) value = `${value}s`;
      else if (key === 'success_hold_seconds' && finite(value)) value = `${value}s`;
      else if (value === null || value === undefined || value === '') value = '未声明';
      else if (typeof value === 'object') value = JSON.stringify(value);
      return `${PROTOCOL_LABELS[key] || key}: ${value}`;
    }).join(' · ');
  }
  function nativeLabel(trial) {
    if (trial && trial.valid === false) return '无效记录 · 原生结果不计';
    if (!trial || trial.valid !== true) return '有效性未知 · 原生结果未知';
    if (trial.native_success === true) return '原生成功';
    if (trial.native_success === false) return '原生未完成';
    return '原生结果未知';
  }
  function groupSort(a, b) {
    const source = (a.source_kind === 'job' ? 0 : 1) - (b.source_kind === 'job' ? 0 : 1);
    if (source) return source;
    return [a.task, a.model, a.checkpoint, a.prediction_horizon, a.execute_steps, a.id]
      .map((value, index) => String(value ?? '').localeCompare(
        String([b.task, b.model, b.checkpoint, b.prediction_horizon, b.execute_steps, b.id][index] ?? ''),
        undefined, {numeric: true})).find(value => value) || 0;
  }
  function filterGroups(groups, state) {
    return array(groups).filter(group =>
      (!state.task || group.task === state.task) &&
      (!state.model || group.model === state.model) &&
      (!state.checkpoint || group.checkpoint === state.checkpoint) &&
      (!state.source_kind || group.source_kind === state.source_kind) &&
      (!state.mode || group.mode === state.mode)
    ).sort(groupSort);
  }
  function taskCoverage(tasks, groups) {
    return array(tasks).map(task => {
      const matched = array(groups).filter(group => group.task === task.task);
      return {...task, group_count: matched.length,
        trial_count: matched.reduce((total, group) => total + array(group.trials).length, 0)};
    });
  }
  function option(document, value, label) {
    const node = el(document, 'option', '', label);
    node.value = value;
    return node;
  }
  function badge(document, value, kind) {
    return el(document, 'span', `kes-badge kes-${kind || 'neutral'}`, value);
  }
  function metric(document, label, value, note) {
    const node = el(document, 'div', 'kes-metric');
    append(node, el(document, 'span', 'kes-metric-label', label),
      el(document, 'strong', '', value));
    if (note) node.appendChild(el(document, 'small', '', note));
    return node;
  }
  function terminal(status) { return !LIVE_STATES.has(String(status || '').toLowerCase()); }

  function renderTrial(document, trial, base) {
    const row = el(document, 'li', 'kes-trial');
    const seed = Number.isInteger(trial.seed) ? `seed ${String(trial.seed).padStart(3, '0')}` : 'seed 未知';
    const result = nativeLabel(trial);
    const status = text(trial.status, '状态未知');
    append(row, el(document, 'span', 'kes-trial-seed', seed),
      el(document, 'span', 'kes-trial-status', status),
      el(document, 'span', `kes-trial-result ${trial.valid === false ? 'kes-invalid' : ''}`, result));
    if (finite(trial.sim_seconds)) row.appendChild(el(document, 'span', 'kes-trial-time', `${trial.sim_seconds.toFixed(2)}s`));
    const href = terminal(trial.status) ? safeUrl(trial.detail_url, base) : null;
    if (href) {
      const link = el(document, 'a', 'kes-detail-link', '查看记录');
      link.href = href;
      row.appendChild(link);
    } else {
      row.appendChild(el(document, 'span', 'kes-no-detail',
        LIVE_STATES.has(String(trial.status || '').toLowerCase()) ? '运行中' : '无可核验详情'));
    }
    return row;
  }

  function renderGroup(document, group, base) {
    const article = el(document, 'article', 'kes-group');
    article.dataset.sourceKind = text(group.source_kind, 'unknown');
    article.dataset.mode = text(group.mode, 'unverified');
    const header = el(document, 'header', 'kes-group-header');
    const heading = el(document, 'div', 'kes-group-heading');
    append(heading, el(document, 'p', 'kes-kicker', `${text(group.title, group.task)} · ${text(group.model, '模型未知')}`),
      el(document, 'h3', '', `${text(group.checkpoint, 'checkpoint 未知')} · H${integer(group.prediction_horizon) ?? '?'} · K${integer(group.execute_steps) ?? '?'}`));
    const badges = el(document, 'div', 'kes-badges');
    append(badges,
      badge(document, SOURCE_LABELS[group.source_kind] || '来源未知', group.source_kind === 'job' ? 'latest' : 'archive'),
      badge(document, MODE_LABELS[group.mode] || MODE_LABELS.unverified,
        group.mode === 'formal' ? 'formal' : group.mode === 'diagnostic' ? 'diagnostic' : 'unverified'));
    append(header, heading, badges);
    article.appendChild(header);
    article.appendChild(el(document, 'p', 'kes-protocol', protocolLabel(group.protocol)));

    const metrics = el(document, 'div', 'kes-metrics');
    const planned = integer(group.planned_trials);
    append(metrics,
      metric(document, '有效记录', planned === null ? String(integer(group.valid_trials) ?? '未知') : `${integer(group.valid_trials) ?? 0} / ${planned}`,
        '无效记录不进入成功分母'),
      metric(document, '无效记录', String(integer(group.invalid_trials) ?? '未知')),
      metric(document, '原生成功', countLabel(group.native_success_count, group.native_known_count), '分母仅含原生结果已知记录'));
    if (integer(group.complete_stage_known_count) !== null) {
      metrics.appendChild(metric(document, '完整阶段', countLabel(group.complete_stage_success_count, group.complete_stage_known_count), '仅同一归档规则与协议'));
    }
    article.appendChild(metrics);

    const stages = array(group.stages);
    if (stages.length) {
      const stageBox = el(document, 'div', 'kes-stages');
      stageBox.appendChild(el(document, 'h4', '', '累计阶段（非当前状态）'));
      const list = el(document, 'ul', 'kes-stage-list');
      stages.forEach(stage => {
        const item = el(document, 'li', '');
        append(item, el(document, 'span', '', text(stage.name, stage.key)),
          el(document, 'strong', '', countLabel(stage.reached, stage.known)));
        list.appendChild(item);
      });
      append(stageBox, list);
      article.appendChild(stageBox);
    }
    const warnings = array(group.warnings).filter(value => typeof value === 'string' && value.trim());
    if (warnings.length) {
      const list = el(document, 'ul', 'kes-warnings');
      warnings.forEach(value => list.appendChild(el(document, 'li', '', value)));
      article.appendChild(list);
    }
    const details = el(document, 'details', 'kes-trials');
    details.appendChild(el(document, 'summary', '', `逐条记录 · ${array(group.trials).length}`));
    const trialList = el(document, 'ul', '');
    array(group.trials).forEach(trial => trialList.appendChild(renderTrial(document, trial, base)));
    details.appendChild(trialList);
    article.appendChild(details);
    return article;
  }

  function mount(document) {
    const root = document.getElementById('evaluation-status');
    const source = document.getElementById('evaluation-status-data');
    if (!root || !source) return false;
    let data;
    try { data = JSON.parse(source.textContent); }
    catch (_) {
      root.replaceChildren(el(document, 'p', 'kes-error', '评估状态数据无法读取。'));
      return false;
    }
    if (!data || data.schema !== 'kaihand-evaluation-status-v1') {
      root.replaceChildren(el(document, 'p', 'kes-error', '评估状态数据版本不受支持。'));
      return false;
    }
    const groups = array(data.groups);
    const tasks = taskCoverage(data.tasks, groups);
    const state = {task: '', model: '', checkpoint: '', source_kind: '', mode: ''};
    const shell = el(document, 'div', 'kes-shell');
    const hero = el(document, 'header', 'kes-hero');
    append(hero, el(document, 'p', 'kes-eyebrow', 'KaiTactileSim Evaluation Platform'),
      el(document, 'h1', '', 'KaiTactileSim 官方评估平台'),
      el(document, 'p', 'kes-lead', '最新任务状态与历史归档分开展示；所有计数只在同一模型、权重、H、K 与控制协议内成立。formal 仅复述来源协议标记，不代表本页新增认证。'));
    const historical = safeUrl(data.historical_url, document.baseURI);
    if (historical) {
      const link = el(document, 'a', 'kes-history-link', '打开历史 H50 官方归档');
      link.href = historical;
      hero.appendChild(link);
    }
    shell.appendChild(hero);

    const links = array(data.links).map(row => ({
      label: text(row && row.label, ''),
      href: safeUrl(row && row.url, document.baseURI),
    })).filter(row => row.label && row.href);
    if (links.length) {
      const nav = el(document, 'nav', 'kes-links');
      nav.setAttribute('aria-label', '相关评估资料');
      nav.appendChild(el(document, 'strong', '', '相关资料'));
      links.forEach(row => {
        const link = el(document, 'a', '', row.label);
        link.href = row.href;
        nav.appendChild(link);
      });
      shell.appendChild(nav);
    }

    const overview = el(document, 'section', 'kes-overview');
    overview.setAttribute('aria-labelledby', 'kes-overview-title');
    const overviewTitle = el(document, 'h2', '', '13 项任务覆盖'); overviewTitle.id = 'kes-overview-title';
    const taskGrid = el(document, 'div', 'kes-task-grid');
    const taskButtons = [];
    const renderTaskButtons = () => taskButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.task === state.task)));
    tasks.forEach(task => {
      const button = el(document, 'button', `kes-task ${task.group_count ? '' : 'kes-empty-task'}`);
      button.type = 'button'; button.dataset.task = task.task;
      append(button, el(document, 'strong', '', text(task.title, task.task)),
        el(document, 'span', '', task.group_count ? `${task.group_count} 组 · ${task.trial_count} 条记录` : '暂无评估'));
      button.addEventListener('click', () => { state.task = state.task === task.task ? '' : task.task; renderTaskButtons(); refreshOptions(); renderGroups(); });
      taskButtons.push(button); taskGrid.appendChild(button);
    });
    append(overview, overviewTitle, taskGrid); shell.appendChild(overview);

    const controls = el(document, 'section', 'kes-controls');
    controls.setAttribute('aria-label', '评估记录筛选');
    const selects = {};
    [['model', '模型'], ['checkpoint', 'Checkpoint'], ['source_kind', '来源'], ['mode', '口径']].forEach(([key, label]) => {
      const wrapper = el(document, 'label', ''); wrapper.appendChild(el(document, 'span', '', label));
      const select = el(document, 'select', ''); select.id = `es-${key.replace('_', '-')}`;
      select.addEventListener('change', () => { state[key] = select.value; renderGroups(); });
      selects[key] = select; wrapper.appendChild(select); controls.appendChild(wrapper);
    });
    shell.appendChild(controls);

    const status = el(document, 'p', 'kes-visible'); status.id = 'es-visible-count'; status.setAttribute('role', 'status');
    const cards = el(document, 'section', 'kes-group-grid'); cards.id = 'es-groups'; cards.setAttribute('aria-label', '协议分组结果');
    append(shell, status, cards);

    const jobsSection = el(document, 'section', 'kes-jobs');
    append(jobsSection, el(document, 'h2', '', '待训练、归一化与运行任务'));
    const jobs = el(document, 'div', 'kes-job-list');
    array(data.jobs).forEach(job => {
      const item = el(document, 'article', 'kes-job');
      append(item, el(document, 'strong', '', text(job.name, '未命名任务')),
        badge(document, text(job.state, '状态未知'), LIVE_STATES.has(job.state) ? 'latest' : 'neutral'),
        el(document, 'p', '', text(job.description, '暂无说明')));
      if (text(job.progress, '')) item.appendChild(el(document, 'p', 'kes-job-progress', job.progress));
      jobs.appendChild(item);
    });
    if (!array(data.jobs).length) jobs.appendChild(el(document, 'p', 'kes-muted', '当前没有待处理任务。'));
    append(jobsSection, jobs); shell.appendChild(jobsSection);

    if (array(data.excluded).length) {
      const excluded = el(document, 'details', 'kes-excluded');
      excluded.appendChild(el(document, 'summary', '', `未纳入记录 · ${data.excluded.length}`));
      const list = el(document, 'ul', '');
      data.excluded.forEach(row => list.appendChild(el(document, 'li', '', `${text(row.source, '来源未知')}：${text(row.reason, '原因未知')}`)));
      excluded.appendChild(list); shell.appendChild(excluded);
    }
    shell.appendChild(el(document, 'footer', 'kes-footer', `快照生成时间：${text(data.generated_utc, '未知')} · 诊断记录不得解释为正式排名。`));
    root.replaceChildren(shell);

    function refreshOptions() {
      const scoped = groups.filter(group => !state.task || group.task === state.task);
      const labels = {
        model: value => value,
        checkpoint: value => value,
        source_kind: value => SOURCE_LABELS[value] || value,
        mode: value => MODE_LABELS[value] || value,
      };
      Object.keys(selects).forEach(key => {
        const values = [...new Set(scoped.map(group => group[key]).filter(value => value !== null && value !== undefined && value !== ''))]
          .sort((a, b) => String(a).localeCompare(String(b), undefined, {numeric: true}));
        if (!values.includes(state[key])) state[key] = '';
        selects[key].replaceChildren(option(document, '', '全部'));
        values.forEach(value => selects[key].appendChild(option(document, value, labels[key](value))));
        selects[key].value = state[key];
      });
    }
    function renderGroups() {
      const visible = filterGroups(groups, state);
      status.textContent = `显示 ${visible.length} / ${groups.length} 个协议组；不跨组汇总成功率。`;
      cards.replaceChildren();
      visible.forEach(group => cards.appendChild(renderGroup(document, group, document.baseURI)));
      if (!visible.length) cards.appendChild(el(document, 'p', 'kes-empty', '没有符合筛选条件的协议组。'));
    }
    refreshOptions(); renderTaskButtons(); renderGroups();
    return true;
  }

  return {filterGroups, groupSort, mount, nativeLabel, protocolLabel, safeUrl, taskCoverage};
});
