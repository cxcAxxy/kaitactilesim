(function (global, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (global && global.document) {
    global.KaiEvaluationExperiment = api;
    const start = () => api.mount(global.document);
    if (global.document.readyState === 'loading') {
      global.document.addEventListener('DOMContentLoaded', start, {once: true});
    } else start();
  }
})(typeof globalThis === 'undefined' ? this : globalThis, function () {
  'use strict';

  const QUARANTINED = new Set(['quarantined', 'quarantined_replay_only']);
  const MODE_LABELS = {
    diagnostic: '诊断记录',
    formal: '来源标记 formal',
    quarantined: '隔离回看',
    unverified: '口径未核验',
  };
  const SOURCE_LABELS = {job: '新增归档', archive: '历史归档'};
  const STAGE_LABELS = {
    picked_up: '拾取并稳定持有',
    fully_extracted: '完全拔出',
    released_stably_on_table: '放回桌面并稳定释放',
    extracted_and_held_2s: '完全拔出并连续手持 ≥2 秒',
  };

  function array(value) { return Array.isArray(value) ? value : []; }
  function integer(value) { return Number.isInteger(value) && value >= 0 ? value : null; }
  function text(value, fallback) {
    return typeof value === 'string' && value.trim() ? value.trim() : fallback;
  }
  function finite(value) { return typeof value === 'number' && Number.isFinite(value); }
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

  function isQuarantined(group, trial) {
    return QUARANTINED.has(group && group.publication_status) ||
      QUARANTINED.has(group && group.mode) ||
      QUARANTINED.has(trial && trial.publication_status) ||
      QUARANTINED.has(trial && trial.status);
  }
  function isEligible(group, trial) {
    return !isQuarantined(group, trial) && trial && trial.valid === true;
  }
  function denominatorLabel(numerator, denominator) {
    const known = integer(denominator);
    if (known === null || known === 0) return '未定义';
    const reached = integer(numerator);
    return `${reached === null ? '未知' : reached} / ${known}`;
  }
  function experimentCounts(data) {
    const experiment = data && typeof data.experiment === 'object' && !Array.isArray(data.experiment)
      ? data.experiment : {};
    return {
      member: integer(experiment.member_trials) ?? integer(experiment.total_records),
      eligible: integer(experiment.eligible_trials) ?? integer(experiment.eligible_records),
      quarantined: integer(experiment.quarantined_trials) ?? integer(experiment.quarantined_records),
    };
  }
  function derivedCounts(groups) {
    const result = {member: 0, eligible: 0, quarantined: 0};
    array(groups).forEach(group => array(group.trials).forEach(trial => {
      result.member += 1;
      if (isQuarantined(group, trial)) result.quarantined += 1;
      else if (isEligible(group, trial)) result.eligible += 1;
    }));
    return result;
  }
  function checkpointLabels(experiment, groups) {
    const declared = array(experiment && experiment.checkpoint_labels)
      .filter(value => typeof value === 'string' && value.trim()).map(value => value.trim());
    const observed = array(groups).map(group => String(group.checkpoint ?? '')).filter(Boolean);
    return [...new Set([...declared, ...observed])];
  }
  function checkpointSummary(experiment, groups) {
    const rows = new Map(checkpointLabels(experiment, groups).map(checkpoint => [checkpoint,
      {checkpoint, member: 0, eligible: 0, quarantined: 0}]));
    array(groups).forEach(group => {
      const checkpoint = String(group.checkpoint ?? '');
      if (!checkpoint) return;
      if (!rows.has(checkpoint)) rows.set(checkpoint, {checkpoint, member: 0, eligible: 0, quarantined: 0});
      const row = rows.get(checkpoint);
      array(group.trials).forEach(trial => {
        row.member += 1;
        if (isQuarantined(group, trial)) row.quarantined += 1;
        else if (isEligible(group, trial)) row.eligible += 1;
      });
    });
    return [...rows.values()];
  }
  function taskCoverage(tasks, groups) {
    const active = new Set(array(groups).map(group => group.task));
    const declared = array(tasks).filter(task => task && active.has(task.task));
    const known = new Set(declared.map(task => task.task));
    array(groups).forEach(group => {
      if (!known.has(group.task)) {
        declared.push({task: group.task, title: text(group.title, group.task)});
        known.add(group.task);
      }
    });
    return declared.map(task => {
      const matched = array(groups).filter(group => group.task === task.task);
      const counts = derivedCounts(matched);
      return {...task, group_count: matched.length, member_trials: counts.member,
        eligible_trials: counts.eligible, quarantined_trials: counts.quarantined};
    });
  }
  function indexOrder(values) {
    return new Map(array(values).map((value, index) => [String(value), index]));
  }
  function ordered(value, order) {
    return order.has(String(value)) ? order.get(String(value)) : Number.MAX_SAFE_INTEGER;
  }
  function sortGroups(groups, taskOrder, checkpointOrder) {
    const tasks = indexOrder(taskOrder);
    const checkpoints = indexOrder(checkpointOrder);
    return [...array(groups)].sort((a, b) =>
      ordered(a.task, tasks) - ordered(b.task, tasks) ||
      String(a.task ?? '').localeCompare(String(b.task ?? '')) ||
      ordered(a.checkpoint, checkpoints) - ordered(b.checkpoint, checkpoints) ||
      String(a.checkpoint ?? '').localeCompare(String(b.checkpoint ?? ''), undefined, {numeric: true}) ||
      Number(isQuarantined(a)) - Number(isQuarantined(b)) ||
      String(a.protocol && a.protocol.rule_sha256 || '').localeCompare(String(b.protocol && b.protocol.rule_sha256 || '')) ||
      String(a.id ?? '').localeCompare(String(b.id ?? '')));
  }
  function filterGroups(groups, state, taskOrder, checkpointOrder) {
    return sortGroups(array(groups).filter(group =>
      (!state.task || group.task === state.task) &&
      (!state.checkpoint || String(group.checkpoint) === String(state.checkpoint))),
    taskOrder, checkpointOrder);
  }
  function ruleLabel(group) {
    const names = new Set(array(group && group.stages).map(stage => stage && stage.name));
    if (group && group.task === 'usb-unplug') {
      if (names.has('extracted_and_held_2s')) return '新版规则：完全拔出后连续手持 ≥2 秒';
      if (names.has('released_stably_on_table')) return '历史规则：完全拔出后放回桌面并稳定释放';
    }
    const sha = group && group.protocol && group.protocol.rule_sha256;
    return typeof sha === 'string' && sha ? `归档规则 ${sha.slice(0, 12)}` : '归档规则身份未记录';
  }
  function protocolLabel(protocol) {
    if (!protocol || typeof protocol !== 'object' || Array.isArray(protocol)) return '控制协议未声明';
    const parts = [];
    if (finite(protocol.max_sim_seconds)) parts.push(`时长 ${protocol.max_sim_seconds}s`);
    if (finite(protocol.control_hz)) parts.push(`${protocol.control_hz} Hz`);
    if (text(protocol.guard_mode, '')) parts.push(`守卫 ${protocol.guard_mode}`);
    if (text(protocol.controller, '')) parts.push(`控制器 ${protocol.controller}`);
    if (text(protocol.action_interpolation, '')) parts.push(`插值 ${protocol.action_interpolation}`);
    return parts.length ? parts.join(' · ') : '控制协议未声明';
  }
  function trialLabel(group, trial) {
    if (isQuarantined(group, trial)) return '隔离回看 · 不计成绩';
    if (!trial || trial.valid !== true) return '无效记录 · 不计成绩';
    if (trial.native_success === true) return '原生成功';
    if (trial.native_success === false) return '原生未完成';
    return '原生结果未知';
  }
  function terminal(status) {
    return !new Set(['queued', 'pending', 'starting', 'running', 'in_progress'])
      .has(String(status || '').toLowerCase());
  }
  function badge(document, value, kind) {
    return el(document, 'span', `kex-badge kex-${kind || 'neutral'}`, value);
  }
  function metric(document, key, label, value, note) {
    const node = el(document, 'div', 'kex-metric');
    node.dataset.score = key;
    append(node, el(document, 'span', 'kex-metric-label', label), el(document, 'strong', '', value));
    if (note) node.appendChild(el(document, 'small', '', note));
    return node;
  }
  function renderTrial(document, group, trial, base) {
    const quarantined = isQuarantined(group, trial);
    const row = el(document, 'li', `kex-trial ${quarantined ? 'kex-trial-quarantined' : ''}`);
    row.dataset.publicationStatus = quarantined ? 'quarantined_replay_only' : 'eligible';
    const seed = Number.isInteger(trial.seed) ? `seed ${String(trial.seed).padStart(3, '0')}` : 'seed 未知';
    append(row, el(document, 'span', 'kex-trial-seed', seed),
      el(document, 'span', 'kex-trial-status', text(trial.status, '状态未知')),
      el(document, 'span', 'kex-trial-result', trialLabel(group, trial)));
    if (finite(trial.sim_seconds)) row.appendChild(el(document, 'span', 'kex-trial-time', `${trial.sim_seconds.toFixed(2)}s`));
    const href = terminal(trial.status) ? safeUrl(trial.detail_url, base) : null;
    if (href) {
      const link = el(document, 'a', 'kex-detail-link', quarantined ? '查看隔离原记录' : '查看记录');
      link.href = href;
      row.appendChild(link);
    } else row.appendChild(el(document, 'span', 'kex-no-detail', '无可核验详情'));
    return row;
  }
  function renderGroup(document, group, base) {
    const quarantined = isQuarantined(group);
    const article = el(document, 'article', `kex-group ${quarantined ? 'kex-group-quarantined' : ''}`);
    article.dataset.task = text(group.task, 'unknown');
    article.dataset.checkpoint = String(group.checkpoint ?? 'unknown');
    article.dataset.publicationStatus = quarantined ? 'quarantined_replay_only' : 'eligible';
    article.dataset.ruleSha = text(group.protocol && group.protocol.rule_sha256, 'unknown');

    const header = el(document, 'header', 'kex-group-header');
    const heading = el(document, 'div', 'kex-group-heading');
    append(heading, el(document, 'p', 'kex-kicker', text(group.title, group.task)),
      el(document, 'h3', '', `checkpoint ${text(String(group.checkpoint ?? ''), '未知')} · H${integer(group.prediction_horizon) ?? '?'} · K${integer(group.execute_steps) ?? '?'}`));
    const badges = el(document, 'div', 'kex-badges');
    append(badges,
      badge(document, quarantined ? '隔离回看 · 不计成绩' : (MODE_LABELS[group.mode] || MODE_LABELS.unverified),
        quarantined ? 'quarantine' : group.mode === 'formal' ? 'formal' : 'diagnostic'));
    append(header, heading, badges); article.appendChild(header);
    article.appendChild(el(document, 'p', 'kex-rule', ruleLabel(group)));
    article.appendChild(el(document, 'p', 'kex-protocol', protocolLabel(group.protocol)));
    article.appendChild(el(document, 'p', 'kex-provenance',
      `溯源：${SOURCE_LABELS[group.source_kind] || text(group.source_kind, '来源未知')} · ${text(group.batch_id, '批次未记录')}`));

    const trials = array(group.trials);
    const metrics = el(document, 'div', 'kex-metrics');
    append(metrics,
      metric(document, 'members', '组内成员', String(trials.length), quarantined ? '仅供隔离回看' : '同一任务、权重与协议'),
      metric(document, 'eligible', '计分记录', quarantined ? '未定义' : String(integer(group.valid_trials) ?? trials.filter(trial => isEligible(group, trial)).length),
        quarantined ? '隔离记录不进入分母' : '无效记录不进入分母'),
      metric(document, 'native', '原生成功', quarantined ? '未定义' : denominatorLabel(group.native_success_count, group.native_known_count),
        '只读取原生任务结果'),
      metric(document, 'complete-stage', '完整阶段', quarantined ? '未定义' : denominatorLabel(group.complete_stage_success_count, group.complete_stage_known_count),
        '与原生成功分开；零分母不显示 0%'));
    article.appendChild(metrics);

    if (!quarantined && array(group.stages).length) {
      const stageBox = el(document, 'section', 'kex-stages');
      stageBox.appendChild(el(document, 'h4', '', '累计阶段（不替代完整成功）'));
      const list = el(document, 'ul', '');
      array(group.stages).forEach(stage => {
        const item = el(document, 'li', '');
        append(item, el(document, 'span', '', STAGE_LABELS[stage.name] || text(stage.name, stage.key)),
          el(document, 'strong', '', denominatorLabel(stage.reached, stage.known)));
        list.appendChild(item);
      });
      stageBox.appendChild(list);
      if (String(group.task).startsWith('whiteboard-wipe')) {
        stageBox.appendChild(el(document, 'p', 'kex-stage-note', '白板“拾取并稳定持有”不是擦净或完整任务成功。'));
      }
      article.appendChild(stageBox);
    }
    if (quarantined) {
      article.appendChild(el(document, 'p', 'kex-quarantine-note',
        '本组属于原有未发布记录：保留逐条回看入口，但不恢复有效性、原生成功或阶段分母。'));
    }
    const warnings = array(group.warnings).filter(value => typeof value === 'string' && value.trim());
    if (warnings.length) {
      const list = el(document, 'ul', 'kex-warnings');
      warnings.forEach(value => list.appendChild(el(document, 'li', '', value)));
      article.appendChild(list);
    }
    const details = el(document, 'details', 'kex-trials');
    details.appendChild(el(document, 'summary', '', `逐条记录 · ${trials.length}`));
    const list = el(document, 'ul', '');
    trials.forEach(trial => list.appendChild(renderTrial(document, group, trial, base)));
    details.appendChild(list); article.appendChild(details);
    return article;
  }

  function mount(document) {
    const root = document.getElementById('evaluation-status');
    const source = document.getElementById('evaluation-status-data');
    if (!root || !source) return false;
    let data;
    try { data = JSON.parse(source.textContent); }
    catch (_) {
      root.replaceChildren(el(document, 'p', 'kex-error', '实验数据无法读取。'));
      return false;
    }
    if (!data || data.schema !== 'kaihand-evaluation-status-v1' ||
        !data.experiment || typeof data.experiment !== 'object') {
      root.replaceChildren(el(document, 'p', 'kex-error', '同实验成员清单缺失或版本不受支持。'));
      return false;
    }
    const experiment = data.experiment;
    const groups = array(data.groups);
    const tasks = taskCoverage(data.tasks, groups);
    const checkpoints = checkpointSummary(experiment, groups);
    const taskOrder = tasks.map(task => task.task);
    const checkpointOrder = checkpoints.map(row => row.checkpoint);
    const declared = experimentCounts(data);
    const observed = derivedCounts(groups);
    const state = {task: '', checkpoint: ''};

    const shell = el(document, 'div', 'kex-shell');
    const hero = el(document, 'header', 'kex-hero');
    append(hero, el(document, 'p', 'kex-eyebrow', `${text(experiment.model, 'pi05')} · H${integer(experiment.prediction_horizon) ?? 50} · SAME EXPERIMENT`),
      el(document, 'h1', '', text(experiment.title, 'pi05 H50 同一实验汇总')),
      el(document, 'p', 'kex-lead', '旧批次与新增批次按同一 pi05 H50 实验成员统一展示；隔离记录可回看但不进入成绩，协议不同的组不合并成功率。'));
    const heroMetrics = el(document, 'div', 'kex-hero-metrics');
    [['member', '全部成员', 'kex-member-count'], ['eligible', '计分记录', 'kex-eligible-count'],
      ['quarantined', '隔离可回看', 'kex-quarantine-count']].forEach(([key, label, id]) => {
      const item = el(document, 'div', 'kex-hero-metric');
      const value = el(document, 'strong', '', declared[key] ?? '未知'); value.id = id;
      append(item, value, el(document, 'span', '', label)); heroMetrics.appendChild(item);
    });
    hero.appendChild(heroMetrics);
    if (['member', 'eligible', 'quarantined'].some(key => declared[key] !== observed[key])) {
      hero.appendChild(el(document, 'p', 'kex-contract-warning',
        `成员清单与分组不一致：声明 ${declared.member ?? '未知'}/${declared.eligible ?? '未知'}/${declared.quarantined ?? '未知'}，分组 ${observed.member}/${observed.eligible}/${observed.quarantined}。`));
    }
    shell.appendChild(hero);

    const links = array(data.links).map(row => ({label: text(row && row.label, ''),
      href: safeUrl(row && row.url, document.baseURI)})).filter(row => row.label && row.href);
    if (links.length) {
      const nav = el(document, 'nav', 'kex-external-links'); nav.setAttribute('aria-label', '独立实验与相关资料');
      nav.appendChild(el(document, 'strong', '', '独立实验与资料'));
      links.forEach(row => { const link = el(document, 'a', '', row.label); link.href = row.href; nav.appendChild(link); });
      shell.appendChild(nav);
    }

    const method = el(document, 'section', 'kex-method');
    append(method, el(document, 'h2', '', '统计边界'), el(document, 'p', '',
      '原生成功、完整阶段与中间阶段分别展示；白板拾取不等于擦净，USB 拔出沿用各条归档规则并按规则身份分组。'));
    shell.appendChild(method);

    const checkpointSection = el(document, 'section', 'kex-checkpoint-section');
    checkpointSection.appendChild(el(document, 'h2', '', 'Checkpoint 成员概览'));
    const checkpointGrid = el(document, 'div', 'kex-checkpoint-grid'); checkpointGrid.id = 'kex-checkpoints';
    const checkpointButtons = [];
    checkpoints.forEach(row => {
      const button = el(document, 'button', `kex-checkpoint ${row.quarantined ? 'kex-checkpoint-has-quarantine' : ''}`);
      button.type = 'button'; button.dataset.checkpoint = row.checkpoint;
      append(button, el(document, 'span', '', 'checkpoint'), el(document, 'strong', '', row.checkpoint),
        el(document, 'b', '', `${row.member} 条成员`),
        el(document, 'small', '', `${row.eligible} 条计分${row.quarantined ? ` · ${row.quarantined} 条隔离` : ''}`));
      button.addEventListener('click', () => { state.checkpoint = state.checkpoint === row.checkpoint ? '' : row.checkpoint; refresh(); });
      checkpointButtons.push(button); checkpointGrid.appendChild(button);
    });
    checkpointSection.appendChild(checkpointGrid); shell.appendChild(checkpointSection);

    const taskSection = el(document, 'section', 'kex-task-section');
    taskSection.appendChild(el(document, 'h2', '', `${tasks.length} 项任务`));
    const taskGrid = el(document, 'div', 'kex-task-grid'); taskGrid.id = 'kex-task-grid';
    const taskButtons = [];
    tasks.forEach(task => {
      const button = el(document, 'button', `kex-task ${task.quarantined_trials ? 'kex-task-has-quarantine' : ''}`);
      button.type = 'button'; button.dataset.task = task.task;
      append(button, el(document, 'strong', '', text(task.title, task.task)),
        el(document, 'span', '', `${task.member_trials} 条成员 · ${task.eligible_trials} 条计分`),
        task.quarantined_trials ? el(document, 'small', '', `${task.quarantined_trials} 条隔离可回看`) : null);
      button.addEventListener('click', () => { state.task = state.task === task.task ? '' : task.task; refresh(); });
      taskButtons.push(button); taskGrid.appendChild(button);
    });
    taskSection.appendChild(taskGrid); shell.appendChild(taskSection);

    const controls = el(document, 'div', 'kex-filter-bar');
    const clear = el(document, 'button', 'kex-clear', '清除筛选'); clear.id = 'kex-clear-filters'; clear.type = 'button';
    clear.addEventListener('click', () => { state.task = ''; state.checkpoint = ''; refresh(); });
    const visible = el(document, 'p', 'kex-visible'); visible.id = 'kex-visible-count'; visible.setAttribute('role', 'status');
    append(controls, visible, clear); shell.appendChild(controls);
    const cards = el(document, 'section', 'kex-group-grid'); cards.id = 'kex-groups'; cards.setAttribute('aria-label', '任务与 checkpoint 协议分组');
    shell.appendChild(cards);
    shell.appendChild(el(document, 'footer', 'kex-footer',
      `静态快照：${text(data.generated_utc, '时间未知')} · 隔离回看、诊断记录和不同完成协议不得合并为总成功率。`));
    root.replaceChildren(shell);

    function refresh() {
      taskButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.task === state.task)));
      checkpointButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.checkpoint === state.checkpoint)));
      clear.disabled = !state.task && !state.checkpoint;
      const selected = filterGroups(groups, state, taskOrder, checkpointOrder);
      const counts = derivedCounts(selected);
      visible.textContent = `显示 ${selected.length} / ${groups.length} 个协议组 · ${counts.member} 条成员（${counts.eligible} 条计分，${counts.quarantined} 条隔离）；不跨协议汇总成功率。`;
      cards.replaceChildren();
      selected.forEach(group => cards.appendChild(renderGroup(document, group, document.baseURI)));
      if (!selected.length) cards.appendChild(el(document, 'p', 'kex-empty', '没有符合当前任务与 checkpoint 的协议组。'));
    }
    refresh();
    return true;
  }

  return {checkpointSummary, denominatorLabel, derivedCounts, experimentCounts, filterGroups,
    isEligible, isQuarantined, mount, protocolLabel, ruleLabel, safeUrl, sortGroups,
    taskCoverage, trialLabel};
});
