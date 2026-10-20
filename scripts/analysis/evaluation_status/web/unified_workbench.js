(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root && root.document) {
    const node = root.document.getElementById('kai-suite-data') || root.document.getElementById('kai-unified-data');
    if (node) api.mount(JSON.parse(node.textContent), root.document);
  }
})(typeof globalThis === 'undefined' ? this : globalThis, function () {
  'use strict';
  const schema = 'kaihand-unified-evaluation-workbench-v1';
  const defaultSummaryMode = 'policy';
  const structuralKeys = ['task', 'model', 'checkpoint', 'protocol', 'cohort'];
  function text(value) { return value === undefined || value === null ? '' : String(value); }
  function stable(value) {
    if (value === null || typeof value !== 'object') return JSON.stringify(value);
    if (Array.isArray(value)) return '[' + value.map(stable).join(',') + ']';
    return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + stable(value[key])).join(',') + '}';
  }
  function protocolKey(group) {
    return text(group.protocol_id) || stable({protocol: group.protocol || {},
      prediction_horizon: group.prediction_horizon, execute_steps: group.execute_steps,
      action_start_index: group.action_start_index, mode: group.mode});
  }
  function groupValue(group, key) {
    return key === 'protocol' ? protocolKey(group) : key === 'cohort' ? text(group.cohort_id) : text(group[key]);
  }
  function filterGroups(groups, state) {
    return groups.filter(group => structuralKeys.every(key => !state[key] || groupValue(group, key) === text(state[key])));
  }
  function eligible(trial) {
    const status = text(trial.publication_status).toLowerCase();
    return trial.valid === true && !status.startsWith('quarantined') && !['invalid', 'missing'].includes(status);
  }
  function stageEligible(trial) { return eligible(trial) && trial.staged_evidence_status === 'valid'; }
  function nativeOutcome(trial) {
    if (!eligible(trial)) return 'invalid';
    return trial.native_success === true ? 'pass' : trial.native_success === false ? 'fail' : 'unknown';
  }
  function stageCategory(trial) {
    if (!stageEligible(trial)) return 'unknown';
    if (trial.complete_stage_success === true) return 'complete';
    const stages = Array.isArray(trial.stages) ? trial.stages : [];
    if (stages.some(stage => stage.reached === true)) return 'partial';
    return stages.length && stages.every(stage => stage.reached === false) ? 'none' : 'unknown';
  }
  function trialMatches(trial, group, state) {
    if (state.outcome && nativeOutcome(trial) !== state.outcome) return false;
    if (state.stage && stageCategory(trial) !== state.stage) return false;
    if (!state.search) return true;
    const query = text(state.search).trim().toLowerCase();
    const source = [group.task, group.title, group.model, group.checkpoint, group.cohort_label,
      trial.id, trial.seed, 'seed_' + text(trial.seed).padStart(3, '0'),
      'seed_' + text(trial.seed).padStart(4, '0'), ...(Array.isArray(trial.notes) ? trial.notes : [trial.notes])].join(' ').toLowerCase();
    return source.includes(query);
  }
  function selectedTrials(groups, state) {
    return filterGroups(groups, state).flatMap(group => (group.trials || [])
      .filter(trial => trialMatches(trial, group, state)).map(trial => ({group, trial})));
  }
  function stageIdentity(stage, task) {
    return stable([text(task), text(stage.key), text(stage.name || stage.label || stage.key)]);
  }
  function summarizeGroup(group) {
    const trials = group.trials || [], valid = trials.filter(eligible);
    const native = valid.filter(trial => typeof trial.native_success === 'boolean');
    const staged = valid.filter(stageEligible);
    const complete = staged.filter(trial => typeof trial.complete_stage_success === 'boolean');
    const definitions = new Map();
    trials.forEach(trial => (trial.stages || []).forEach(stage => {
      const key = text(stage.key || stage.name || stage.label), identity = stageIdentity(stage, group.task);
      if (key && !definitions.has(identity)) definitions.set(identity, {key, name: stage.label || stage.name || key});
    }));
    return {
      members: trials.length, eligible: valid.length, excluded: trials.length - valid.length,
      native: {known: native.length, successes: native.filter(trial => trial.native_success).length},
      complete: {known: complete.length, successes: complete.filter(trial => trial.complete_stage_success).length},
      stageEvidence: staged.length,
      stages: [...definitions].map(([identity, definition]) => {
        const values = staged.map(trial => (trial.stages || []).find(stage =>
          stageIdentity(stage, group.task) === identity))
          .filter(stage => stage && typeof stage.reached === 'boolean');
        return {...definition, known: values.length, successes: values.filter(stage => stage.reached).length};
      }),
    };
  }
  function modelOptions(groups, task) {
    const models = new Map();
    groups.forEach(group => {
      const model = text(group.model);
      if (!models.has(model)) models.set(model, {value: model, label: modelLabel(model), total: 0, current: 0});
      const entry = models.get(model), count = (group.trials || []).length;
      entry.total += count; if (!task || group.task === task) entry.current += count;
    });
    return [...models.values()];
  }
  function aggregateMetrics(groups) {
    const metrics = new Map();
    groups.forEach(group => {
      if (!(group.trials || []).some(stageEligible)) return;
      (group.metrics || []).forEach(metric => {
        if (!Number.isFinite(metric.value) || !Number.isInteger(metric.known) || metric.known <= 0) return;
        const identity = stable([metric.key || metric.name || metric.label, metric.unit || '']);
        if (!metrics.has(identity)) metrics.set(identity, {...metric, weighted: 0, known: 0});
        const entry = metrics.get(identity);
        entry.weighted += metric.value * metric.known; entry.known += metric.known;
      });
    });
    return [...metrics.values()].map(metric => {
      const value = metric.weighted / metric.known;
      const {weighted, ...rest} = metric;
      return {...rest, value, display: value.toFixed(2), aggregation: 'known_episode_count_weighted_archive_mean'};
    });
  }
  function completionBreakdowns(groups) {
    const definitions = new Map();
    groups.forEach(group => (group.trials || []).filter(trial => stageEligible(trial) &&
      typeof trial.complete_stage_success === 'boolean').forEach(trial => {
      const stages = trial.stages || [];
      const signature = stable([group.task, stages.map(stage => stageIdentity(stage, group.task))]);
      if (!definitions.has(signature)) definitions.set(signature, {signature, task: group.task,
        terminal_stage: stages[stages.length - 1] || {name: '完整阶段'}, known: 0, successes: 0});
      const definition = definitions.get(signature);
      definition.known += 1; if (trial.complete_stage_success) definition.successes += 1;
    }));
    return [...definitions.values()];
  }
  function taskSummaryRows(groups) {
    const rows = new Map();
    groups.forEach(group => {
      if (!rows.has(group.task)) rows.set(group.task, []);
      rows.get(group.task).push(group);
    });
    return [...rows].map(([task, sourceGroups]) => {
      const trials = sourceGroups.flatMap(group => (group.trials || []).map(trial => ({...trial,
        staged_evidence_basis: trial.staged_evidence_basis || group.staged_evidence_basis})));
      const warnings = [...new Set(sourceGroups.flatMap(group => Array.isArray(group.warnings) ? group.warnings :
        group.warnings ? [group.warnings] : []).map(text))];
      const invalidStages = trials.filter(trial => eligible(trial) && trial.staged_evidence_status === 'invalid').length;
      if (invalidStages) warnings.push(invalidStages + ' 条阶段证据失效，不进入阶段分母；原生结果独立保留。');
      return {id: 'task:' + task, task, source_groups: sourceGroups, trials,
        metrics: aggregateMetrics(sourceGroups), complete_breakdowns: completionBreakdowns(sourceGroups), warnings, descriptive: true};
    });
  }
  function policySummaryRows(groups) {
    const taskOrder = new Map(), modelOrder = new Map();
    groups.forEach(group => {
      if (!taskOrder.has(group.task)) taskOrder.set(group.task, taskOrder.size);
      if (!modelOrder.has(group.task)) modelOrder.set(group.task, new Map());
      const models = modelOrder.get(group.task);
      if (!models.has(group.model)) models.set(group.model, models.size);
    });
    return groups.map((group, index) => ({group, index})).sort((a, b) =>
      taskOrder.get(a.group.task) - taskOrder.get(b.group.task) ||
      modelOrder.get(a.group.task).get(a.group.model) - modelOrder.get(b.group.task).get(b.group.model) ||
      text(a.group.checkpoint).localeCompare(text(b.group.checkpoint), undefined, {numeric: true}) || a.index - b.index)
      .map(item => item.group);
  }
  function summaryRows(groups, mode) { return mode === 'task' ? taskSummaryRows(groups) : policySummaryRows(groups); }
  function selectionSummary(groups) {
    return groups.reduce((total, group) => {
      const summary = summarizeGroup(group);
      ['members', 'eligible', 'excluded'].forEach(key => { total[key] += summary[key]; });
      ['native', 'complete'].forEach(key => {
        total[key].known += summary[key].known; total[key].successes += summary[key].successes;
      });
      return total;
    }, {members: 0, eligible: 0, excluded: 0, native: {known: 0, successes: 0}, complete: {known: 0, successes: 0}});
  }
  function ratioLabel(value) {
    return value.known > 0 ? value.successes + ' / ' + value.known + ' (' +
      (100 * value.successes / value.known).toFixed(1) + '%)' : '未定义';
  }
  function coverageLabel(groups) {
    const total = selectionSummary(groups);
    return '当前完整范围：' + total.members + ' 条归档 · 有效 ' + total.eligible + ' · 隔离 / 无效 ' + total.excluded +
      ' · ' + groups.length + ' 个原始评估分组。此处只展示覆盖数量，不显示跨模型、checkpoint 或任务混合的总体成功率。';
  }
  function stageLabel(stage, task) {
    if (stage.label) return text(stage.label);
    const name = text(stage.name || stage.key);
    if (name === 'picked_up' && text(task).startsWith('whiteboard-')) return '抓起板擦';
    if (name === 'fully_extracted' && task === 'bulb-unscrew') return '完全拧出';
    return ({dragged_on_table: '拖牌', stable_grasp: '稳定夹持',
      terminal_hold_over_3s: '展示姿态持牌超过 3 秒', picked_up: '抓起',
      held_near_socket: '移近孔口', threaded_and_seated: '旋入到位',
      stable_inplace_grasp: '原位稳定抓握', tightening_verified: '拧紧量已验证',
      released_stably: '稳定释放', fully_extracted: '完全拔出',
      released_stably_on_table: '放稳桌面', picked_up_and_held_near_slot: '抓起并移近插槽',
      inserted_and_seated: '插入到位', extracted_and_held_2s: '拔出并稳定持有 2 秒'})[name] || name;
  }
  function modelLabel(value) {
    return ({pi05: 'π0.5', 'pi0.5': 'π0.5', 'pi05+cpm': 'CPM（openpi-main + π0.5）',
      cpm: 'CPM（openpi-main + π0.5）', ftp1: 'FTP-1', 'ftp-1': 'FTP-1', egotouch: 'EgoTouch'})[text(value).toLowerCase()] || text(value);
  }
  function checkpointLabel(value) { return /^\d+$/.test(text(value)) ? 'step ' + value : text(value); }
  function protocolLabel(group) {
    const protocol = group.protocol || {};
    const pieces = [];
    if (group.prediction_horizon !== undefined && group.prediction_horizon !== null) pieces.push('H' + group.prediction_horizon);
    if (group.execute_steps !== undefined && group.execute_steps !== null) pieces.push('K' + group.execute_steps);
    const duration = protocol.max_sim_seconds === undefined ? protocol.max_seconds : protocol.max_sim_seconds;
    if (duration !== undefined && duration !== null) pieces.push(duration + 's');
    if (protocol.guard_mode) pieces.push('guard=' + protocol.guard_mode);
    if (protocol.action_interpolation) pieces.push(protocol.action_interpolation);
    if (protocol.controller) pieces.push(protocol.controller);
    if (protocol.success_hold_seconds !== undefined) pieces.push('hold=' + protocol.success_hold_seconds + 's');
    return pieces.join(' · ') || '按归档协议';
  }
  function evidenceBasisLabel(basis) {
    return ({recorded_archive: '归档阶段记录', tactile_manifest_inventory_v1: '触觉部署清单校验',
      strict_recorded: '已验证的记录阶段'})[basis] || text(basis) || '阶段溯源未说明';
  }
  function safeUrl(value, base) {
    if (typeof value !== 'string' || !value || /(?:^[a-z][a-z\d+.-]*:|^\/\/|\\|[\u0000-\u0020]|%2e|%2f|%5c)/i.test(value)) return null;
    try {
      const context = new URL(base), url = new URL(value, context);
      const reportsRoot = new URL('../', context).pathname;
      if (url.origin !== context.origin || !['http:', 'https:', 'file:'].includes(url.protocol) ||
          !url.pathname.startsWith(reportsRoot)) return null;
      return url.href;
    } catch (_) { return null; }
  }
  function mount(catalog, doc) {
    if (catalog.schema !== schema) throw new Error('Unsupported unified workbench schema');
    const groups = catalog.groups || [], tasks = catalog.tasks || [];
    const taskNames = new Map(tasks.map(task => [task.task, task.title || task.task]));
    const state = {task: '', model: '', checkpoint: '', protocol: '', cohort: '', outcome: '', stage: '', search: ''};
    let summaryMode = defaultSummaryMode;
    const byId = id => doc.getElementById(id);
    const grid = byId('ks-trials'), visible = byId('ks-visible'), tabs = byId('ks-task-tabs');
    function el(tag, cls, parent, value) {
      const node = doc.createElement(tag); if (cls) node.className = cls;
      if (value !== undefined) node.textContent = text(value);
      if (parent) parent.appendChild(node); return node;
    }
    function addText(parent, cls, value) { return el('p', cls, parent, value); }
    function noteText(value) { return Array.isArray(value) ? value.join('；') : text(value); }
    function shortIdentity(value) {
      if (value && typeof value === 'object') return Object.entries(value).map(([key, item]) => key + '=' + text(item)).join(' · ');
      return text(value);
    }
    function fillSelect(key, pairs) {
      const select = byId('ks-' + key); if (!select) return;
      const available = new Map(pairs);
      if (state[key] && !available.has(state[key])) state[key] = '';
      select.replaceChildren(); const all = el('option', '', select, '全部'); all.value = '';
      [...available].sort((a, b) => a[1].localeCompare(b[1], 'zh-CN', {numeric: true})).forEach(([value, label]) => {
        const option = el('option', '', select, label); option.value = value;
      });
      select.value = state[key];
    }
    function updateOptions() {
      fillSelect('model', modelOptions(groups, state.task).map(option => [option.value,
        option.label + (state.task ? ' · 当前任务 ' + option.current + ' / 总 ' + option.total : ' · ' + option.total + ' 条')]));
      const preceding = ['task', 'model'];
      ['checkpoint', 'protocol', 'cohort'].forEach(key => {
        const context = Object.fromEntries(preceding.map(field => [field, state[field]]));
        // Cascading options let users switch models without being trapped by an old checkpoint.
        const candidates = filterGroups(groups, context);
        fillSelect(key, candidates.map(group => [groupValue(group, key), key === 'checkpoint' ? checkpointLabel(group.checkpoint) : key === 'cohort' ?
            text(group.cohort_label || group.cohort_id) : protocolLabel(group)]).filter(pair => pair[0]));
        preceding.push(key);
      });
    }
    const counts = byId('ks-counts');
    if (counts) {
      counts.replaceChildren();
      const members = groups.reduce((count, group) => count + (group.trials || []).length, 0);
      const included = groups.reduce((count, group) => count + (group.trials || []).filter(eligible).length, 0);
      [[tasks.length, '类任务'], [members, '条归档轨迹'], [included, '条有效轨迹'],
        [members - included, '条隔离 / 无效']].forEach(([value, label]) => {
        const card = el('div', 'ks-stat', counts); el('strong', '', card, value); el('span', '', card, label);
      });
    }
    const taskButtons = [];
    function taskButton(task, label) {
      const count = groups.filter(group => !task || group.task === task).reduce((sum, group) => sum + (group.trials || []).length, 0);
      const button = el('button', '', tabs, label + ' · ' + count); button.type = 'button'; button.dataset.task = task;
      button.addEventListener('click', () => {
        state.task = task;
        updateOptions(); render();
      }); taskButtons.push(button);
    }
    taskButton('', '全部任务'); tasks.forEach(task => taskButton(task.task, task.title || task.task));
    ['model', 'checkpoint', 'protocol', 'cohort'].forEach(key => {
      const select = byId('ks-' + key); if (!select) return;
      select.addEventListener('change', event => {
        state[key] = event.target.value;
        if (key === 'model') { state.checkpoint = ''; state.protocol = ''; state.cohort = ''; }
        updateOptions(); render();
      });
    });
    fillSelect('outcome', [['pass', '原生成功'], ['fail', '原生未成功'], ['unknown', '原生未知'], ['invalid', '隔离 / 无效']]);
    fillSelect('stage', [['complete', '完整阶段通过'], ['partial', '部分阶段已达成'], ['none', '已知阶段均未达成'], ['unknown', '阶段未知 / 不计分']]);
    ['outcome', 'stage'].forEach(key => {
      const select = byId('ks-' + key); if (select) select.addEventListener('change', event => { state[key] = event.target.value; render(); });
    });
    const search = byId('ks-search');
    if (search) search.addEventListener('input', event => { state.search = event.target.value.trim().toLowerCase(); render(); });
    const summarySelect = byId('ks-summary-mode');
    if (summarySelect) {
      summarySelect.value = defaultSummaryMode;
      summarySelect.addEventListener('change', event => { summaryMode = event.target.value === 'task' ? 'task' : defaultSummaryMode; render(); });
    }
    let view = 'cards';
    let storage;
    try {
      storage = doc.defaultView && doc.defaultView.localStorage;
      if (storage && storage.getItem('kai-evaluation-suite-view-v1') === 'list') view = 'list';
    } catch (_) { /* Optional preference, including file:// previews. */ }
    const viewButtons = [];
    if (visible && visible.parentNode) {
      const bar = el('div', 'ks-results-bar'); visible.parentNode.insertBefore(bar, visible); bar.appendChild(visible);
      const controls = el('div', 'ks-view-controls', bar); controls.setAttribute('role', 'group'); controls.setAttribute('aria-label', '轨迹显示方式');
      [['cards', '卡片'], ['list', '紧凑列表']].forEach(([value, label]) => {
        const button = el('button', 'ks-view-button', controls, label); button.type = 'button'; button.id = 'ks-view-' + value;
        button.dataset.view = value; button.setAttribute('aria-controls', 'ks-trials');
        button.addEventListener('click', () => {
          view = value; try { if (storage) storage.setItem('kai-evaluation-suite-view-v1', view); } catch (_) { /* Optional preference. */ }
          renderView();
        }); viewButtons.push(button);
      });
    }
    function renderView() {
      grid.dataset.view = view;
      viewButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
    }
    function renderSummary(selected) {
      const body = byId('ks-comparison') || byId('ks-stage-groups'); if (!body) return;
      body.replaceChildren();
      summaryRows(selected, summaryMode).forEach(group => {
        const summary = summarizeGroup(group), row = el('tr', '', body); row.dataset.group = group.id;
        row.dataset.summaryMode = summaryMode;
        row.dataset.task = group.task;
        if (!group.descriptive) {
          row.dataset.model = group.model; row.dataset.checkpoint = text(group.checkpoint);
          row.dataset.protocol = protocolKey(group); row.dataset.cohort = text(group.cohort_id);
        }
        el('td', 'ks-summary-task', row, taskNames.get(group.task) || group.task);
        const identity = el('td', 'ks-summary-identity ks-summary-model-cell', row);
        const checkpoint = el('td', 'ks-summary-checkpoint', row);
        const protocol = el('td', 'ks-summary-protocol', row);
        if (group.descriptive) {
          const modelGroups = new Map();
          group.source_groups.forEach(source => {
            if (!modelGroups.has(source.model)) modelGroups.set(source.model, new Set());
            modelGroups.get(source.model).add(text(source.checkpoint));
          });
          modelGroups.forEach((checkpoints, model) => {
            addText(identity, 'ks-summary-model', modelLabel(model));
            addText(checkpoint, 'ks-summary-model', modelLabel(model) + '：' +
              [...checkpoints].sort((a, b) => a.localeCompare(b, undefined, {numeric: true})).join(' / '));
          });
          addText(identity, 'ks-small', group.source_groups.length + ' 个归档分组 · ' + modelGroups.size + ' 种模型');
          const protocols = new Set(group.source_groups.map(protocolKey));
          addText(protocol, '', protocols.size + ' 种评估协议');
          addText(protocol, 'ks-small', '按各记录原有判据的描述性汇总；不是同协议模型性能排名。');
          const sourceNote = addText(protocol, 'ks-small ks-rule-id', '原协议与规则溯源');
          sourceNote.title = group.source_groups.map(source => source.id + '\n' + protocolLabel(source) + '\n' +
            shortIdentity(source.rule_identity)).join('\n\n');
        } else {
          el('strong', '', identity, modelLabel(group.model));
          el('strong', '', checkpoint, checkpointLabel(group.checkpoint));
          addText(identity, 'ks-small', group.cohort_label || group.cohort_id || '归档批次');
          if (group.mode) addText(identity, 'ks-small', group.mode === 'diagnostic' || group.mode === 'diagnostic_only' ? '诊断评估 · 非正式指标' : group.mode);
          protocol.textContent = protocolLabel(group);
          const rule = shortIdentity(group.rule_identity);
          if (rule) { const node = addText(protocol, 'ks-small ks-rule-id', '规则 ' + rule.slice(0, 24)); node.title = rule; }
        }
        const basis = [...new Set((group.trials || []).filter(stageEligible).map(trial => trial.staged_evidence_basis || group.staged_evidence_basis))];
        if (basis.length) addText(protocol, 'ks-small', basis.map(evidenceBasisLabel).join(' / '));
        const n = el('td', 'ks-summary-n', row, summary.eligible + ' / ' + summary.members);
        if (summary.excluded) addText(n, 'ks-small ks-source-warning', '排除 ' + summary.excluded);
        const native = el('td', 'ks-rate', row, ratioLabel(summary.native));
        if (summary.native.known < summary.eligible) addText(native, 'ks-small', '原生未知 ' + (summary.eligible - summary.native.known));
        const complete = el('td', 'ks-rate', row);
        if (group.descriptive && group.complete_breakdowns.length > 1) {
          group.complete_breakdowns.forEach(definition => addText(complete, 'ks-complete-definition',
            stageLabel(definition.terminal_stage, group.task) + '：' + ratioLabel(definition)));
          addText(complete, 'ks-small', '完成判据不同，分别统计');
        } else complete.textContent = ratioLabel(summary.complete);
        if (summary.complete.known < summary.eligible) addText(complete, 'ks-small', '未定义 / 不计分 ' + (summary.eligible - summary.complete.known));
        const stageCell = el('td', 'ks-summary-stages', row);
        summary.stages.forEach(stage => addText(stageCell, '', stageLabel(stage, group.task) + '：' + ratioLabel(stage)));
        if (!summary.stages.length) addText(stageCell, 'ks-small', (group.trials || []).some(trial => trial.staged_evidence_status === 'invalid') ?
          '已记录阶段，但来源校验未通过（不计分）' : '未记录可验证的阶段数据');
        (group.metrics || []).forEach(metric => {
          const value = metric.display === undefined ? metric.value : metric.display;
          if (value !== null && value !== undefined) addText(stageCell, 'ks-summary-metric', text(metric.label || metric.name) + '：' +
            text(value) + text(metric.unit) + (Number.isInteger(metric.known) ? '（n=' + metric.known + '）' : ''));
        });
        const warnings = Array.isArray(group.warnings) ? group.warnings : group.warnings ? [group.warnings] : [];
        warnings.forEach(warning => addText(stageCell, 'ks-source-warning', noteText(warning)));
      });
      if (!selected.length) {
        const row = el('tr', '', body);
        const cell = el('td', 'ks-empty', row, state.task && state.model ?
          '该任务暂无该模型评估记录。模型选择已保留；未评估不代表成功率为 0。' :
          '没有符合模型、checkpoint 或协议筛选条件的统计分组。'); cell.colSpan = 8;
      }
      const summaryNote = byId('ks-summary-note');
      if (summaryNote) summaryNote.textContent = '成功率分母 = 各分组内有效且结果已知的轨迹；隔离、未知与失效阶段不补零。' +
        (summaryMode === 'task' ? '当前按任务汇总全部所选模型/checkpoint，按各记录原有判据作描述性汇总，不是同协议模型性能排名。' :
          '当前按任务 × 模型 × checkpoint 分别统计；即使三者相同，原协议、规则或批次不同的记录仍保持独立行。') +
        '原生结果、阶段和搜索筛选只筛下方轨迹，不改变本表分母。';
      const totalNode = byId('ks-summary-total');
      if (totalNode) totalNode.textContent = coverageLabel(selected);
    }
    function renderCard(group, trial) {
      const href = safeUrl(trial.detail_url, doc.baseURI);
      const card = el(href ? 'a' : 'article', 'ks-case', grid);
      if (href) card.href = href;
      card.dataset.trial = trial.id;
      const main = el('div', 'ks-case-main', card), top = el('div', 'ks-case-top', main);
      el('strong', '', top, taskNames.get(group.task) || group.title || group.task);
      const outcome = nativeOutcome(trial);
      el('span', 'ks-native-status ks-native-' + outcome, top, outcome === 'pass' ? '原任务 · 成功' :
        outcome === 'fail' ? '原任务 · 未成功' : outcome === 'invalid' ? '隔离 / 无效' : '原任务 · 未知');
      addText(main, 'ks-identity', modelLabel(group.model) + ' · ' + checkpointLabel(group.checkpoint) + ' · seed ' +
        (Number.isInteger(trial.seed) ? String(trial.seed).padStart(3, '0') : '未知'));
      addText(main, 'ks-outcome', protocolLabel(group));
      addText(main, 'ks-small', group.cohort_label || group.cohort_id || '');
      const progress = el('div', 'ks-case-progress', card), stages = trial.stages || [];
      const stageValid = stageEligible(trial), reached = stageValid ? stages.filter(stage => stage.reached === true).length : 0;
      addText(progress, 'ks-stage-caption', stageValid ? '累计阶段 ' + reached + ' / ' + stages.length :
        trial.staged_evidence_status === 'invalid' ? '已记录阶段，但来源校验未通过' : '阶段未知 / 不计分');
      const badges = el('div', 'ks-stages', progress);
      stages.forEach(stage => {
        const value = stageValid ? stage.reached : null;
        el('span', value === true ? 'ks-reached' : '', badges,
          (value === true ? '✓ ' : value === false ? '— ' : '? ') + stageLabel(stage, group.task));
      });
      const complete = stageValid ? trial.complete_stage_success : null;
      addText(progress, 'ks-stage-result', '完整阶段：' + (complete === true ? '通过' : complete === false ? '未通过' : '未定义 / 不计分'));
      if (stageValid) addText(progress, 'ks-small', evidenceBasisLabel(trial.staged_evidence_basis || group.staged_evidence_basis));
      if (!eligible(trial)) addText(progress, 'ks-source-warning', '该轨迹保留供审核，不进入成功率分母。');
      if (trial.staged_evidence_status === 'invalid') addText(progress, 'ks-source-warning', '阶段证据失效，原始记录不作为已达成阶段计分。');
      (Array.isArray(trial.notes) ? trial.notes : trial.notes ? [trial.notes] : []).forEach(note => addText(progress, 'ks-source-warning', noteText(note)));
      const footer = el('div', 'ks-case-footer', card), coverage = el('div', 'ks-case-coverage', footer);
      el('span', '', coverage, trial.publication_status || (eligible(trial) ? '有效归档' : '不计分归档'));
      if (trial.expert_used === true) el('span', '', coverage, 'CPM 专家引导');
      if (href) el('span', 'ks-open', footer, '轨迹分析 →');
    }
    function render() {
      taskButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.task === state.task)));
      const selected = filterGroups(groups, state), displayed = selectedTrials(groups, state);
      renderSummary(selected);
      const baseCount = selected.reduce((sum, group) => sum + (group.trials || []).length, 0);
      visible.textContent = '显示 ' + displayed.length + ' / ' + baseCount + ' 条筛选范围内轨迹';
      const note = byId('ks-task-note');
      if (note) {
        note.replaceChildren();
        const task = tasks.find(item => item.task === state.task);
        if (task) { el('h2', '', note, task.title || task.task); addText(note, '', task.note || '原任务成功、阶段进度与量化指标按各自归档口径展示。'); }
        else addText(note, '', '全部任务与策略在同一工作台对比；可按模型、checkpoint 和协议缩小范围，历史成功率与轨迹均保留。');
      }
      grid.replaceChildren(); renderView();
      // Interleave tasks in the all-task view, preserving each group's recorded seed ordering.
      if (!state.task) {
        const buckets = new Map(); displayed.forEach(item => {
          if (!buckets.has(item.group.task)) buckets.set(item.group.task, []);
          buckets.get(item.group.task).push(item);
        });
        let remaining = displayed.length;
        while (remaining) buckets.forEach(bucket => {
          if (bucket.length) { const item = bucket.shift(); renderCard(item.group, item.trial); remaining -= 1; }
        });
      } else displayed.forEach(item => renderCard(item.group, item.trial));
      if (!displayed.length) el('p', 'ks-empty', grid, !selected.length && state.task && state.model ?
        '该任务暂无该模型评估记录。已保留模型选择，可切换任务查看；未评估不代表成功率为 0。' :
        '没有符合当前筛选条件的轨迹；上方成功率仍使用完整分母。');
    }
    updateOptions(); render();
    return {state, render};
  }
  return {schema, defaultSummaryMode, stable, protocolKey, filterGroups, eligible, stageEligible, nativeOutcome, stageCategory,
    trialMatches, selectedTrials, stageIdentity, summarizeGroup, modelOptions, aggregateMetrics, completionBreakdowns,
    taskSummaryRows, policySummaryRows, summaryRows,
    selectionSummary, coverageLabel, ratioLabel, stageLabel, modelLabel, checkpointLabel, protocolLabel,
    evidenceBasisLabel, safeUrl, mount};
});
