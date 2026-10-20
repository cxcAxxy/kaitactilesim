(function (global, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (global && global.document) {
    global.KaiTactilePolicyExperiment = api;
    const start = () => api.mount(global.document);
    if (global.document.readyState === 'loading') {
      global.document.addEventListener('DOMContentLoaded', start, {once: true});
    } else start();
  }
})(typeof globalThis === 'undefined' ? this : globalThis, function () {
  'use strict';

  const ACTIVE = new Set(['pending', 'queued', 'starting', 'running', 'in_progress']);
  const STATUS_LABELS = {
    accepted: '已接受', complete: '已完成', completed: '已完成',
    running: '运行中', in_progress: '运行中', starting: '启动中',
    queued: '排队中', pending: '待运行', failed: '失败', partial: '部分完成',
  };
  const STAGE_LABELS = {
    dragged_on_table: '拖牌',
    stable_grasp: '稳定夹持',
    terminal_hold_over_3s: '展示姿态持牌超过 3 秒',
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
    const path = value.split(/[?#]/, 1)[0];
    const parents = path.split('/').filter(part => part === '..');
    if (parents.length > 1 || (parents.length === 1 && !value.startsWith('../'))) return null;
    try {
      const origin = new URL(base || 'http://localhost/');
      const url = new URL(value, origin);
      if (!['http:', 'https:'].includes(url.protocol) || url.origin !== origin.origin) return null;
      return url.href;
    } catch (_) { return null; }
  }
  function ratioLabel(stats) {
    if (!stats || typeof stats !== 'object') return '未定义';
    const known = integer(stats.known);
    const successes = integer(stats.successes);
    if (!known) return '未定义';
    return `${successes === null ? '未知' : successes} / ${known} (${(100 * (successes || 0) / known).toFixed(1)}%)`;
  }
  function countsFromTrials(trials) {
    const counts = {planned: 0, observed: 0, valid: 0, invalid: 0, missing: 0};
    array(trials).forEach(trial => {
      counts.planned += 1;
      if (trial && trial.observed === true) counts.observed += 1;
      const status = trial && trial.evidence_status;
      if (status === 'valid') counts.valid += 1;
      else if (status === 'invalid') counts.invalid += 1;
      else counts.missing += 1;
    });
    return counts;
  }
  function aggregate(groups) {
    return array(groups).reduce((total, group) => {
      const counts = countsFromTrials(group && group.trials);
      Object.keys(total).forEach(key => { total[key] += counts[key]; });
      return total;
    }, {planned: 0, observed: 0, valid: 0, invalid: 0, missing: 0});
  }
  function protocolLabel(protocol) {
    if (!protocol || typeof protocol !== 'object') return '协议未记录';
    const parts = [];
    if (finite(protocol.max_sim_seconds)) parts.push(`${protocol.max_sim_seconds}s`);
    if (text(protocol.guard_mode, '')) parts.push(`guard=${protocol.guard_mode}`);
    if (text(protocol.controller, '')) parts.push(protocol.controller);
    if (text(protocol.action_interpolation, '')) parts.push(`interpolation=${protocol.action_interpolation}`);
    if (finite(protocol.success_hold_seconds)) parts.push(`hold=${protocol.success_hold_seconds}s`);
    if (text(protocol.staged_evaluation_version, '')) parts.push(protocol.staged_evaluation_version);
    return parts.length ? parts.join(' · ') : '协议未记录';
  }
  function evidenceLabel(trial) {
    if (!trial || trial.evidence_status === 'missing') return '未完成 / 缺证据';
    if (trial.evidence_status === 'invalid') return '无效（不计分）';
    if (trial.native_success === true) return '原生成功';
    if (trial.native_success === false) return '原生未完成';
    return '原生结果未知';
  }
  function statusLabel(status) {
    const normalized = text(status, 'pending').toLowerCase();
    return STATUS_LABELS[normalized] || normalized;
  }
  function stageLabel(stage) {
    const name = text(stage && stage.name, text(stage && stage.key, '阶段未知'));
    return STAGE_LABELS[name] || name;
  }
  function renderTrial(document, trial, base) {
    const status = text(trial && trial.evidence_status, 'missing');
    const row = el(document, 'li', `ktp-trial ktp-trial-${status}`);
    row.dataset.evidenceStatus = status;
    const seed = Number.isInteger(trial && trial.seed) ? `seed ${String(trial.seed).padStart(3, '0')}` : 'seed ?';
    append(row, el(document, 'span', 'ktp-seed', seed),
      el(document, 'span', 'ktp-trial-state', text(trial && trial.status, status)),
      el(document, 'strong', 'ktp-trial-result', evidenceLabel(trial)));
    const staged = trial && trial.staged_complete_success;
    row.appendChild(el(document, 'span', 'ktp-stage-result',
      staged === true ? '完整阶段达成' : staged === false ? '完整阶段未达成' : '阶段未知'));
    const actions = el(document, 'span', 'ktp-trial-actions');
    const detail = safeUrl(trial && trial.detail_url, base);
    if (detail) { const link = el(document, 'a', 'ktp-detail-link', '详情'); link.href = detail; actions.appendChild(link); }
    const video = safeUrl(trial && trial.video_url, base);
    if (video) { const link = el(document, 'a', 'ktp-video-link', '视频'); link.href = video; actions.appendChild(link); }
    if (!actions.children.length) actions.appendChild(el(document, 'span', 'ktp-no-artifact', '无可回看产物'));
    row.appendChild(actions);
    return row;
  }
  function metric(document, key, label, value, note) {
    const node = el(document, 'div', 'ktp-metric'); node.dataset.score = key;
    append(node, el(document, 'span', '', label), el(document, 'strong', '', value));
    if (note) node.appendChild(el(document, 'small', '', note));
    return node;
  }
  function renderGroup(document, group, base) {
    const article = el(document, 'article', 'ktp-group');
    article.dataset.group = text(group && group.label, 'unknown');
    article.dataset.jobStatus = text(group && group.status, 'pending');
    const identity = group && group.identity || {};
    const heading = el(document, 'header', 'ktp-group-header');
    const name = el(document, 'div', '');
    append(name, el(document, 'p', 'ktp-kicker', text(identity.model_family, '模型未知')),
      el(document, 'h2', '', text(group && group.title, group && group.label || '未命名组')),
      el(document, 'p', 'ktp-identity', `H${integer(identity.prediction_horizon) ?? '?'} / start ${integer(identity.action_start_index) ?? '?'} / K${integer(identity.execute_steps) ?? '?'} · ${text(identity.checkpoint_label, '权重未知')} · ${text(identity.evaluation_mode, '诊断评估')}`));
    append(heading, name, el(document, 'span', `ktp-status ktp-status-${text(group && group.status, 'pending')}`, statusLabel(group && group.status)));
    article.appendChild(heading);
    article.appendChild(el(document, 'p', 'ktp-protocol', protocolLabel(identity.protocol)));
    if (identity.guidance && identity.guidance.enabled === true) {
      article.appendChild(el(document, 'p', 'ktp-guidance', `触觉引导：${identity.guidance.variant || 'enabled'} · ${identity.guidance.steps} steps · scale ${identity.guidance.scale}`));
    } else article.appendChild(el(document, 'p', 'ktp-guidance', '触觉直接输入模型；无 CPM 引导'));
    const counts = group.counts || countsFromTrials(group.trials);
    const stagedEvidence = group.staged && group.staged.evidence_counts || {};
    const stagedNote = `strict ${integer(stagedEvidence.verified) ?? 0} · invalid ${integer(stagedEvidence.invalid) ?? 0} · missing ${integer(stagedEvidence.missing) ?? 0}`;
    const grid = el(document, 'div', 'ktp-metrics');
    append(grid,
      metric(document, 'progress', '进度', `${integer(counts.observed) ?? 0} / ${integer(counts.planned) ?? array(group.trials).length}`, '已观测 / 计划'),
      metric(document, 'validity', '有效性', `${integer(counts.valid) ?? 0} valid · ${integer(counts.invalid) ?? 0} invalid`, '无效记录不进入成功率'),
      metric(document, 'native', '原生成功', ratioLabel(group.native), '仅 valid 原生结果'),
      metric(document, 'staged', '完整阶段', ratioLabel(group.staged), stagedNote));
    article.appendChild(grid);
    const stages = array(group.staged && group.staged.stages);
    if (stages.length) {
      const box = el(document, 'section', 'ktp-stages'); box.appendChild(el(document, 'h3', '', '累计阶段'));
      const list = el(document, 'ul', '');
      stages.forEach(stage => {
        const item = el(document, 'li', '');
        append(item, el(document, 'span', '', stageLabel(stage)),
          el(document, 'strong', '', integer(stage.known) ? `${integer(stage.reached) ?? 0} / ${stage.known}` : '未定义'));
        list.appendChild(item);
      });
      box.appendChild(list); article.appendChild(box);
    }
    const details = el(document, 'details', 'ktp-trials');
    details.appendChild(el(document, 'summary', '', `逐条记录 · ${array(group.trials).length}`));
    const list = el(document, 'ul', '');
    array(group.trials).forEach(trial => list.appendChild(renderTrial(document, trial, base)));
    details.appendChild(list); article.appendChild(details);
    return article;
  }

  function mount(document) {
    const root = document.getElementById('tactile-experiment');
    const source = document.getElementById('tactile-experiment-data');
    if (!root || !source) return false;
    let data;
    try { data = JSON.parse(source.textContent); }
    catch (_) { root.replaceChildren(el(document, 'p', 'ktp-error', '触觉实验数据无法读取。')); return false; }
    if (!data || data.schema !== 'kaihand-tactile-policy-experiment-v1' || !data.experiment) {
      root.replaceChildren(el(document, 'p', 'ktp-error', '触觉实验清单缺失或版本不受支持。')); return false;
    }
    const groups = array(data.groups);
    const derived = aggregate(groups);
    const declared = data.experiment.counts || {};
    const state = {group: ''};
    const shell = el(document, 'div', 'ktp-shell');
    const hero = el(document, 'header', 'ktp-hero');
    append(hero, el(document, 'p', 'ktp-eyebrow', 'POKER · TACTILE POLICY BENCHMARK'),
      el(document, 'h1', '', text(data.experiment.title, 'Poker 触觉模型对比')),
      el(document, 'p', 'ktp-lead', '五组模型按同一任务与 90 秒协议组织诊断评估；H30/H32、CPM/FTP-1、native/staged 始终分开，不升级为 formal。'));
    const totals = el(document, 'div', 'ktp-hero-metrics');
    [['planned', '计划', 'ktp-planned-count'], ['observed', '已观测', 'ktp-observed-count'],
      ['valid', '有效', 'ktp-valid-count'], ['invalid', '无效', 'ktp-invalid-count'], ['missing', '缺失', 'ktp-missing-count']]
      .forEach(([key, label, id]) => {
        const box = el(document, 'div', ''); const value = el(document, 'strong', '', integer(declared[key]) ?? derived[key]); value.id = id;
        append(box, value, el(document, 'span', '', label)); totals.appendChild(box);
      });
    hero.appendChild(totals);
    hero.appendChild(el(document, 'p', 'ktp-scope', `作业状态：${statusLabel(data.experiment.status)}。零分母显示“未定义”，不写成 0%。`));
    shell.appendChild(hero);
    const links = array(data.links).map(row => ({label: text(row && row.label, ''), href: safeUrl(row && row.url, document.baseURI)})).filter(row => row.label && row.href);
    if (links.length) {
      const nav = el(document, 'nav', 'ktp-links'); nav.setAttribute('aria-label', '其他评估入口');
      links.forEach(row => { const link = el(document, 'a', '', row.label); link.href = row.href; nav.appendChild(link); }); shell.appendChild(nav);
    }
    const method = el(document, 'section', 'ktp-method');
    append(method, el(document, 'h2', '', '口径'), el(document, 'p', '',
      '原生成功率只以 valid trial 为分母；staged 只以严格校验的累计阶段证据为分母。invalid、missing、running 都不会被当成失败或 0%。'));
    shell.appendChild(method);
    const tabs = el(document, 'div', 'ktp-tabs'); tabs.id = 'ktp-group-tabs';
    const buttons = [];
    groups.forEach(group => {
      const button = el(document, 'button', 'ktp-group-tab'); button.type = 'button'; button.dataset.group = group.label;
      append(button, el(document, 'strong', '', text(group.title, group.label)), el(document, 'small', '', `${statusLabel(group.status)} · ${group.counts.valid}/${group.counts.planned} valid`));
      button.addEventListener('click', () => { state.group = state.group === group.label ? '' : group.label; refresh(); });
      buttons.push(button); tabs.appendChild(button);
    });
    shell.appendChild(tabs);
    const visible = el(document, 'p', 'ktp-visible'); visible.id = 'ktp-visible-count'; shell.appendChild(visible);
    const cards = el(document, 'section', 'ktp-groups'); cards.id = 'ktp-groups'; shell.appendChild(cards);
    shell.appendChild(el(document, 'footer', 'ktp-footer', `静态快照：${text(data.generated_utc, '时间未知')} · 不与 pi05 H50 450 条实验混合。`));
    root.replaceChildren(shell);
    function refresh() {
      buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.group === state.group)));
      const selected = state.group ? groups.filter(group => group.label === state.group) : groups;
      const counts = aggregate(selected);
      visible.textContent = `显示 ${selected.length} / ${groups.length} 组 · ${counts.planned} 计划 · ${counts.valid} valid · ${counts.invalid} invalid · ${counts.missing} missing`;
      cards.replaceChildren(); selected.forEach(group => cards.appendChild(renderGroup(document, group, document.baseURI)));
    }
    refresh(); return true;
  }

  return {aggregate, countsFromTrials, evidenceLabel, mount, protocolLabel, ratioLabel, safeUrl, stageLabel, statusLabel};
});
