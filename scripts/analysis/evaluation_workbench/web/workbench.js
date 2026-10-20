/* Additive layout adapter. Archived measurements and the original dashboard are read-only. */
(function (global) {
  'use strict';
  const COLORS = ['#087e94', '#df982a', '#6473b8', '#b76186', '#488763'];
  const PLATFORM_NAME = 'KaiTactileSim Evaluation Platform';
  const PLATFORM_FOOTER = 'KaiTactileSim 官方评估平台 · 数据与判定遵循各轨迹归档规则；平台归属不代表评估认证。';
  // Explicit task display names mirror task_fields.py, never infer task semantics
  // from a filename. Archived per-task presentation remains the first choice.
  const TASK_TITLES = Object.freeze({
    'bulb-screw': '灯泡旋入与释放', 'bulb-screw-inplace': '灯泡原位旋入',
    'bulb-unscrew': '灯泡旋出与放回', 'install-ram': '内存条安装',
    'poker-draw': '卡片抽取与举起', 'usb-insert': 'USB 插入', 'usb-unplug': 'USB 拔出',
    'whiteboard-wipe': '倾斜白板擦拭', 'whiteboard-wipe-horizontal': '水平白板擦拭',
    'whiteboard-wipe-horizontal-line': '水平白板线迹擦拭',
  });
  const LEGACY_BULB_ANGLE_SERIES = Object.freeze({
    clockwise_turns: Object.freeze({key: 'clockwise_angle_deg', label: '灯泡净旋入角度（顺时针）'}),
    turns_removed: Object.freeze({key: 'angle_removed_deg', label: '灯泡净旋出角度（逆时针）'}),
    rotation_from_initial_turns: Object.freeze({key: 'rotation_from_initial_deg', label: '相对实际初态的净旋入角度'}),
  });
  const scaledDegrees = value => finite(value) ? value * 360 : value;
  const angleNumber = value => {
    const rounded = Math.round(value * 1e10) / 1e10;
    return String(Object.is(rounded, -0) ? 0 : rounded);
  };
  function legacyRoundedMatch(raw, actual, expected) {
    if (!finite(actual) || !finite(expected)) return false;
    const decimals = raw.includes('.') ? raw.length - raw.indexOf('.') - 1 : 0;
    return Math.abs(actual - expected) <= .5 * 10 ** -decimals + 1e-12;
  }
  function canonicalLegacyAngleReferenceText(reference) {
    if (finite(reference.target)) reference.label = reference.key === 'angle_removed_deg'
      ? `名义螺纹脱扣角度：${angleNumber(reference.target)}°`
      : `名义机械到座旋入角度：${angleNumber(reference.target)}°`;
    if (typeof reference.note !== 'string') return reference;
    const targetTurns = finite(reference.target) ? reference.target / 360 : null;
    const toleranceCandidates = [reference.lower, reference.upper]
      .filter(finite).map(value => Math.abs(value - reference.target));
    const toleranceDegrees = toleranceCandidates.length ? Math.max(...toleranceCandidates) : null;
    reference.note = reference.note.replace(/(±|[+-])?\s*(\d+(?:\.\d+)?)\s*圈/g, (_match, sign = '', raw) => {
      const magnitude = Number(raw), turns = sign === '-' ? -magnitude : magnitude;
      let degrees = turns * 360;
      if (sign === '±' && finite(toleranceDegrees) && legacyRoundedMatch(raw, magnitude, toleranceDegrees / 360)) degrees = toleranceDegrees;
      else if (sign !== '±' && legacyRoundedMatch(raw, turns, targetTurns)) degrees = reference.target;
      return `${sign === '±' ? '±' : sign === '+' ? '+' : ''}${angleNumber(degrees)}°`;
    }).replace(/净顺时针旋入圈数/g, '净顺时针旋入角度')
      .replace(/净逆时针旋出圈数/g, '净逆时针旋出角度')
      .replace(/净旋入圈数/g, '净旋入角度').replace(/净旋出圈数/g, '净旋出角度')
      .replace(/净圈数/g, '净角度').replace(/圈数/g, '角度').replace(/圈/g, '角度');
    return reference;
  }
  function convertLegacyAngleReference(reference, legacyKey, convertedKey) {
    if (!reference || typeof reference !== 'object' || Array.isArray(reference) ||
        reference.key !== legacyKey || reference.unit !== 'turn') return reference;
    const converted = {...reference, key: convertedKey, unit: 'deg'};
    for (const field of ['target', 'lower', 'upper']) {
      if (Object.hasOwn(converted, field)) converted[field] = scaledDegrees(converted[field]);
    }
    return canonicalLegacyAngleReferenceText(converted);
  }
  function convertLegacyAngleSeries(series) {
    const specification = series && typeof series === 'object' && !Array.isArray(series)
      ? LEGACY_BULB_ANGLE_SERIES[series.key] : null;
    if (!specification || series.unit !== 'turn') return series;
    const converted = {...series, key: specification.key, unit: 'deg', label: specification.label};
    if (Array.isArray(series.values)) converted.values = series.values.map(scaledDegrees);
    if (Array.isArray(series.goal_references)) converted.goal_references = series.goal_references.map(
      reference => convertLegacyAngleReference(reference, series.key, specification.key));
    for (const field of ['target', 'lower', 'upper']) {
      if (Object.hasOwn(converted, field)) converted[field] = scaledDegrees(converted[field]);
    }
    return converted;
  }
  function remapLegacyAngleKeys(keys) {
    if (!Array.isArray(keys)) return keys;
    return keys.map(key => LEGACY_BULB_ANGLE_SERIES[key]?.key || key);
  }
  function normalizeBulbAngleDisplay(payload, config) {
    const taskSeries = payload?.visualization?.task_series;
    let normalizedPayload = payload, normalizedConfig = config, converted = false;
    if (Array.isArray(taskSeries)) {
      const normalizedSeries = taskSeries.map(series => {
        const value = convertLegacyAngleSeries(series);
        converted ||= value !== series;
        return value;
      });
      if (converted) normalizedPayload = {...payload, visualization: {...payload.visualization, task_series: normalizedSeries}};
    }
    const presentation = config?.task_presentation;
    if (presentation && typeof presentation === 'object' && !Array.isArray(presentation)) {
      const primary = remapLegacyAngleKeys(presentation.primary_task_keys);
      const secondary = remapLegacyAngleKeys(presentation.secondary_task_keys);
      const keysChanged = (Array.isArray(primary) && primary.some((key, index) => key !== presentation.primary_task_keys[index])) ||
        (Array.isArray(secondary) && secondary.some((key, index) => key !== presentation.secondary_task_keys[index]));
      if (keysChanged) {
        normalizedConfig = {...config, task_presentation: {...presentation,
          primary_task_keys: primary, secondary_task_keys: secondary}};
        converted = true;
      }
    }
    // Metadata and terminal summaries use the same units as the curves. Never
    // mutate the archived payload, and never scale a canonical degree twice.
    if (Array.isArray(config?.goal_references?.references)) {
      const references = config.goal_references.references.map(reference => {
        const specification = LEGACY_BULB_ANGLE_SERIES[reference?.key];
        return specification ? convertLegacyAngleReference(reference, reference.key, specification.key) : reference;
      });
      if (references.some((reference, index) => reference !== config.goal_references.references[index])) {
        normalizedConfig = {...normalizedConfig, goal_references: {...config.goal_references, references}};
        converted = true;
      }
    }
    if (Array.isArray(config?.task_summary)) {
      const summaries = config.task_summary.map(item => {
        const specification = LEGACY_BULB_ANGLE_SERIES[item?.key];
        return specification && item.unit === 'turn' ? {...item, key: specification.key,
          label: `终态${specification.label}`, unit: 'deg', value: scaledDegrees(item.value)} : item;
      });
      if (summaries.some((item, index) => item !== config.task_summary[index])) {
        normalizedConfig = {...normalizedConfig, task_summary: summaries};
        converted = true;
      }
    }
    return {payload: normalizedPayload, config: normalizedConfig, converted};
  }
  function degreeSymbolDashboardPayload(payload) {
    const taskSeries = payload?.visualization?.task_series;
    if (!Array.isArray(taskSeries)) return payload;
    let converted = false;
    const displaySeries = taskSeries.map(series => {
      if (!series || typeof series !== 'object' || Array.isArray(series) ||
          !Object.values(LEGACY_BULB_ANGLE_SERIES).some(specification => specification.key === series.key) ||
          series.unit !== 'deg') return series;
      converted = true;
      const display = {...series, unit: '°'};
      if (Array.isArray(series.goal_references)) display.goal_references = series.goal_references.map(reference =>
        reference && typeof reference === 'object' && !Array.isArray(reference) &&
        reference.key === series.key && reference.unit === 'deg' ? {...reference, unit: '°'} : reference);
      return display;
    });
    return converted ? {...payload, visualization: {...payload.visualization, task_series: displaySeries}} : payload;
  }
  function remountLegacyAngleDashboard(host, payload) {
    const dashboard = global.KaiEvaluationDashboard;
    if (!host || typeof dashboard?.mount !== 'function' ||
        typeof dashboard.normalizeBulbAngleSeries === 'function') return false;
    const displayPayload = degreeSymbolDashboardPayload(payload);
    if (displayPayload === payload) return false;
    global.kaiEvaluationView?.destroy?.();
    global.kaiEvaluationView = dashboard.mount(host, displayPayload);
    return true;
  }
  function taskDisplayTitle(config, task) {
    const title = config?.task_presentation?.title;
    return typeof title === 'string' && title.trim() ? title.trim() : Object.hasOwn(TASK_TITLES, task) ? TASK_TITLES[task] : typeof task === 'string' && task ? task : '轨迹分析';
  }
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = value => finite(value) ? value.toFixed(3) : '未知';
  const displayUnit = unit => unit === 'deg' ? '°' : unit;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  // Keep these acceptance rules aligned with dashboard.goalReferences: the
  // overview must not claim a target that the corresponding chart rejects.
  function displayedGoalReferences(series) {
    if (typeof series?.key !== 'string' || typeof series?.unit !== 'string') return [];
    const plottable = value => finite(value) && Math.abs(value) <= Number.MAX_VALUE / 8;
    return (Array.isArray(series.goal_references) ? series.goal_references : []).slice(0, 8).flatMap(reference => {
      if (!reference || reference.key !== series.key || reference.unit !== series.unit || !plottable(reference.target) ||
          !['target', 'at_most', 'at_least'].includes(reference.relation)) return [];
      const result = {...reference}; delete result.lower; delete result.upper;
      const directed = reference.relation === 'target' || reference.relation === 'at_most' && reference.upper === reference.target || reference.relation === 'at_least' && reference.lower === reference.target;
      if (directed && plottable(reference.lower) && plottable(reference.upper) && reference.lower <= reference.target && reference.target <= reference.upper && reference.lower < reference.upper) {
        result.lower = reference.lower; result.upper = reference.upper;
      }
      return [result];
    });
  }
  function goalReferenceStatus(visualization, config) {
    const metadata = config?.goal_references || {}, series = visualization?.task_series || [];
    const plotted = series.flatMap(item => displayedGoalReferences(item));
    const bands = plotted.filter(item => finite(item.lower) && finite(item.upper)).length;
    const reason = typeof metadata.reason === 'string' && metadata.reason.trim() ? metadata.reason.trim() : null;
    if (metadata.status === 'verified' && plotted.length) return {status: 'verified', count: plotted.length, bands,
      title: `参考条件已核验 · ${plotted.length} 条目标 / 阈值线${bands ? ` · ${bands} 项容差 / 允许范围` : ''}`,
      note: '具体参考值显示在对应曲线下方。参考线只说明单项条件，不等于完整任务成功，也不更改原归档判定。'};
    if (metadata.status === 'verified') return {status: 'unavailable', count: 0, bands: 0,
      title: '参考来源已核验，当前没有可配对的任务曲线', note: reason || '不将参考值套到单位或语义不同的曲线；详见数据来源。'};
    if (metadata.status === 'not_applicable') return {status: 'not_applicable', count: 0, bands: 0,
      title: '本条未配置任务参考条件', note: reason || '此类归档没有可用的任务参考定义；仍保留原始观测，不借用其他任务的阈值。'};
    return {status: 'unavailable', count: 0, bands: 0, title: '参考条件暂不可用',
      note: reason || '本条归档的参考规则或来源尚未核验，暂不绘制目标线或容差带；未知不表示零。'};
  }
  function mountGoalReferenceStatus(parent, visualization, config) {
    const doc = parent.ownerDocument, state = goalReferenceStatus(visualization, config);
    const panel = doc.createElement('div'); panel.className = 'wb-goal-status'; panel.dataset.status = state.status;
    panel.dataset.referenceCount = String(state.count); panel.dataset.bandCount = String(state.bands);
    const heading = doc.createElement('strong'); heading.textContent = state.title; panel.appendChild(heading);
    const note = doc.createElement('p'); note.textContent = state.note; panel.appendChild(note); parent.appendChild(panel);
    return {node: panel, destroy() { panel.remove(); }};
  }
  function bulbAngleState(visualization, config, time) {
    const task = config?.unified_identity?.task || config?.task || config?.task_presentation?.task;
    const key = {'bulb-screw': 'clockwise_angle_deg', 'bulb-unscrew': 'angle_removed_deg', 'bulb-screw-inplace': 'rotation_from_initial_deg'}[task];
    if (!key) return null;
    const matching = (visualization?.task_series || []).filter(item => item?.key === key && item.unit === 'deg');
    const series = matching.length === 1 ? matching[0] : null;
    const times = Array.isArray(series?.time_s) ? series.time_s : [];
    const deltas = times.slice(1).map((value, index) => value - times[index]).filter(value => finite(value) && value > 0).sort((a, b) => a - b);
    const nominal = finite(series?.nominal_interval_s) ? series.nominal_interval_s : deltas.length ? deltas[Math.floor(deltas.length / 2)] : null;
    const index = series && visualization.clock_aligned === true ? observedIndex({...series, time_s: times, nominal_interval_s: nominal}, time) : -1;
    const current = index >= 0 && finite(series.values?.[index]) ? series.values[index] : null;
    const summaries = (config.task_summary || []).filter(item => item?.key === key && item.unit === 'deg');
    const summary = summaries.length === 1 ? summaries[0] : null;
    // A missing terminal measurement is not the last valid time-series sample.
    const terminal = summary?.status === 'recorded' && finite(summary.value) ? summary.value : null;
    const references = config.goal_references?.status === 'verified' ? displayedGoalReferences(series).filter(reference => reference.relation === 'target') : [];
    const reference = references.length === 1 ? references[0] : null;
    return {key, label: series?.label || Object.values(LEGACY_BULB_ANGLE_SERIES).find(item => item.key === key)?.label,
      current, terminal, target: reference?.target ?? null, lower: reference?.lower ?? null, upper: reference?.upper ?? null,
      sampleTime: current === null ? null : times[index], time, relative: task === 'bulb-screw-inplace',
      targetNote: reference?.note || (task === 'bulb-screw-inplace' ? '相对初态的目标需绑定本条实际起始角度，不直接套用绝对到座角。' : '未核验本条角度目标，不以其他任务或轨迹的角度替代。')};
  }
  function mountBulbAngleSummary(parent, visualization, config) {
    let state = bulbAngleState(visualization, config, 0);
    if (!state) return null;
    const doc = parent.ownerDocument;
    function el(tag, cls, container, text) { const node = doc.createElement(tag); node.className = cls; if (text !== undefined) node.textContent = text; container.appendChild(node); return node; }
    const panel = el('section', 'wb-bulb-angle-summary', parent); panel.dataset.metricKey = state.key;
    el('h3', '', panel, `${state.label} · 角度读数`);
    const grid = el('div', 'wb-bulb-angle-grid', panel);
    function card(kind, title) { const node = el('div', `wb-bulb-angle-${kind}`, grid); el('span', '', node, title); return {value: el('strong', '', node), note: el('small', '', node)}; }
    const current = card('current', '当前角度 · 跟随游标'), terminal = card('terminal', '终态角度 · 固定归档值'), target = card('target', '目标参考角 · 不代表成功');
    terminal.value.textContent = state.terminal === null ? '未记录' : `${fmt(state.terminal)}°`;
    terminal.note.textContent = state.terminal === null ? '不使用最后一个有效采样替代' : '来自本条轨迹的终态记录';
    target.value.textContent = state.target === null ? '未设可核验目标' : `${angleNumber(state.target)}°`;
    target.note.textContent = finite(state.lower) && finite(state.upper) ? `参考范围 ${angleNumber(state.lower)}°–${angleNumber(state.upper)}°` : state.target === null && state.relative ? '需按本条实际初态换算' : '没有另行声明的角度容差带';
    target.value.title = state.targetNote;
    el('p', 'wb-bulb-angle-note', panel, '净角度保留方向：反向回拧会抵消，不是来回旋转的累计总角度。达到参考角仍须结合接合 / 脱扣、稳定与释放等归档条件。');
    function update(time) { state = bulbAngleState(visualization, config, time); current.value.textContent = state.current === null ? '该时刻未记录' : `${fmt(state.current)}°`; current.note.textContent = state.sampleTime === null ? '不外推、不补零' : `原始采样 ${fmt(state.sampleTime)} s`; }
    update(0);
    return {node: panel, update, destroy() { panel.remove(); }};
  }
  function leftIndex(times, time) {
    let lo = 0, hi = times.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (times[mid] <= time + 1e-8) lo = mid + 1; else hi = mid; }
    return lo - 1;
  }
  function observedIndex(data, time) {
    const times = data.time_s || [], i = leftIndex(times, time);
    if (!finite(time) || i < 0 || time > times[times.length - 1] + 1e-8) return -1;
    if (Math.abs(times[i] - time) < 1e-8) return i;
    if (data.gap_after?.[i] === true) return -1;
    const declared = Array.isArray(data.gap_after) && data.gap_after.length === times.length;
    if (!declared && finite(data.nominal_interval_s) && time - times[i] > data.nominal_interval_s * 1.5 + 1e-8) return -1;
    return i;
  }
  function stepTime(times, time, direction) {
    if (!times.length) return time;
    if (direction > 0) return times[Math.min(times.length - 1, leftIndex(times, time) + 1)];
    let i = leftIndex(times, time); if (Math.abs(times[i] - time) < 1e-8) i--;
    return times[Math.max(0, i)];
  }
  function yMaximum(series) {
    let maximum = 0;
    series.forEach(values => values.forEach(v => { if (finite(v)) maximum = Math.max(maximum, v); }));
    return maximum > 0 ? maximum * 1.08 : 1;
  }
  function knownRuns(data, values) {
    const times = data.time_s || [], declared = Array.isArray(data.gap_after) && data.gap_after.length === times.length;
    const runs = []; let run = [];
    times.forEach((t, i) => {
      const gap = i > 0 && (data.gap_after?.[i - 1] === true || (!declared && finite(data.nominal_interval_s) && t - times[i - 1] > data.nominal_interval_s * 1.5 + 1e-8));
      if (!finite(values[i]) || gap) { if (run.length) runs.push(run); run = []; }
      if (finite(values[i])) run.push([t, values[i]]);
    });
    if (run.length) runs.push(run); return runs;
  }
  function cropStyle(config, id) {
    const view = config?.status === 'verified' && config.views?.find(item => item.id === id);
    if (!view || ![config.source_width, config.source_height, view.width, view.height].every(v => finite(v) && v > 0) ||
        ![view.x, view.y].every(v => finite(v) && v >= 0) || view.x + view.width > config.source_width || view.y + view.height > config.source_height) return null;
    return {aspect: `${view.width} / ${view.height}`, width: `${config.source_width / view.width * 100}%`, height: `${config.source_height / view.height * 100}%`, left: `${-view.x / view.width * 100}%`, top: `${-view.y / view.height * 100}%`};
  }
  function modelCropConfig(views, media) {
    // Spatial camera layout and video/data clock mapping are independent.
    // A verified crop must nevertheless belong to the exact displayed media.
    return views?.status === 'verified' && typeof views.sha256 === 'string' &&
      /^[0-9a-f]{64}$/.test(views.sha256) && views.sha256 === media?.sha256 ? views : null;
  }
  function expertVideoURL(config) {
    return config?.status === 'available' && config.clock === 'independent_video_seconds' &&
      typeof config.url === 'string' && /^media\/[A-Za-z0-9_.-]+\.mp4$/.test(config.url) ? config.url : null;
  }
  function diagnosticPlotURL(plot) {
    return typeof plot?.url === 'string' && /^plots\/(?:[A-Za-z0-9_-]+\/)?[A-Za-z0-9_.-]+\.png$/.test(plot.url) ? plot.url : null;
  }
  function expertDisplayConfig(expert, plots) {
    if (!expert || plots?.status !== 'available' || plots.comparison_kind !== 'model_only') return expert;
    const note = expert.comparison_note || '';
    // Display-only wording: preserve source identity and unit warnings without
    // implying that a model-only gallery contains an expert reference line.
    return {...expert, comparison_note: note
      .replace('与 PNG 蓝线对应同一源录制', '与原记录声明的参考源属于同一录制；本页未绘制专家参考线')
      .replace('同任务的另一条录制，不是 PNG 蓝线对应 episode', '同任务的独立录制，与原记录声明的参考源不同；本页未绘制专家参考线')};
  }
  function nativeOutcomeStatus(report, staged, published) {
    if (published) {
      if (published.valid !== true || String(published.publication_status || '').startsWith('quarantined')) return 'unknown';
      return published.native_success === true ? 'pass' : published.native_success === false ? 'fail' : 'unknown';
    }
    // The data-audit verdict and native task success are different evidence.
    // A present but unknown native value must never fall back to an audit pass.
    if (staged) return staged.native_result?.task_success === true ? 'pass' : staged.native_result?.task_success === false ? 'fail' : 'unknown';
    return ['pass', 'fail', 'unknown'].includes(report?.outcome?.status) ? report.outcome.status : 'unknown';
  }
  function publishedStageStatus(published) {
    if (!published) return null;
    if (String(published.publication_status || '').startsWith('quarantined')) return 'quarantined';
    if (published.valid !== true) return 'invalid_trial';
    return ['valid', 'invalid', 'quarantined'].includes(published.staged_evidence_status)
      ? published.staged_evidence_status : 'unknown';
  }
  function stageEvidenceMessage(published) {
    const status = publishedStageStatus(published);
    if (status === 'quarantined') return {title: '隔离轨迹 · 阶段不计分',
      note: '仅保留历史录制供回看，不进入当前成功率或阶段分母；原始记录可在数据来源中查看。'};
    if (status === 'invalid_trial') return {title: '无效轨迹 · 阶段不计分', note: '该轨迹不进入成功率或阶段分母；已有原始记录仅供审核。'};
    if (status === 'invalid') return {title: '已记录阶段，但来源校验未通过',
      note: '阶段结果未知、暂不计分；原生任务结果与视频独立保留。具体来源差异见“数据来源”。'};
    if (status === 'unknown') return published?.stage_recorded === true ?
      {title: '阶段已记录，来源尚未确认', note: '阶段结果未知、不计分；不把未知记录视为未达到。'} :
      {title: '未记录可验证的阶段数据', note: '阶段结果未知，不补零，也不由原任务结果推断阶段是否达到。'};
    return null;
  }
  function stageBadgeDefinitions(staged, published) {
    const masked = published && publishedStageStatus(published) !== 'valid';
    return (Array.isArray(staged?.stages) ? staged.stages : []).map(stage => ({
      key: typeof stage?.key === 'string' ? stage.key : '',
      label: [stage?.label, stage?.name, stage?.key].find(value => typeof value === 'string' && value.trim()) || '未命名阶段',
      reached: !masked && stage?.reached === true ? true : !masked && stage?.reached === false ? false : null,
      first_time_s: !masked && finite(stage?.first_time_s) && stage.first_time_s >= 0 ? stage.first_time_s : null,
    }));
  }
  function modelDisplayName(model) {
    return {'pi05': 'π0.5', 'pi0.5': 'π0.5', 'pi05+cpm': 'CPM（openpi-main + π0.5）',
      'ftp1': 'FTP-1', 'ftp-1': 'FTP-1', 'egotouch': 'EgoTouch'}[String(model || '').toLowerCase()] || model || '模型未记录';
  }
  function canonicalModel(model) {
    const normalized = String(model || '').toLowerCase();
    return {'pi0.5': 'pi05', 'ftp-1': 'ftp1'}[normalized] || normalized;
  }
  function detailIdentity(payload, data, config) {
    const trial = payload.report?.trial || {}, published = config.unified_identity;
    return {task: published?.task || data.task, trial_id: published?.trial_id || data.trial_id,
      group_id: published?.group_id, model: published?.model || trial.cohort?.model_family || config.model_family,
      checkpoint: published?.checkpoint ?? config.checkpoint ?? String(trial.source || '').match(/step(\d+)/)?.[1] ?? null,
      seed: published ? published.seed : trial.seed ?? null,
      prediction_horizon: published?.prediction_horizon ?? data.prediction_horizon ?? config.prediction_horizon ?? trial.prediction_horizon,
      execute_steps: published?.execute_steps ?? config.execute_steps ?? trial.cohort?.execute_steps ?? null};
  }
  function validateWorkbenchIdentity(payload, data, config) {
    const trial = payload.report?.trial || {}, horizon = data.prediction_horizon;
    if (!Number.isSafeInteger(horizon) || horizon <= 0 || config.prediction_horizon !== horizon ||
        trial.prediction_horizon !== horizon || data.trial_id !== trial.trial_id || config.trial_id !== data.trial_id ||
        (trial.task !== undefined && trial.task !== data.task) || (config.task !== undefined && config.task !== data.task)) {
      throw new Error('Workbench horizon/trial identity mismatch.');
    }
    if (config.unified_identity) {
      const identity = config.unified_identity;
      if (identity.task !== data.task || identity.prediction_horizon !== horizon ||
          typeof identity.group_id !== 'string' || !identity.group_id || typeof identity.trial_id !== 'string' || !identity.trial_id ||
          typeof identity.model !== 'string' || !identity.model || typeof identity.valid !== 'boolean' ||
          !['string', 'number'].includes(typeof identity.checkpoint) || !String(identity.checkpoint) ||
          (typeof identity.checkpoint === 'number' && !finite(identity.checkpoint)) ||
          (identity.seed !== null && (!Number.isSafeInteger(identity.seed) || identity.seed < 0)) ||
          (trial.seed !== undefined && trial.seed !== identity.seed) ||
          (trial.cohort?.model_family && canonicalModel(trial.cohort.model_family) !== canonicalModel(identity.model)) ||
          (config.checkpoint !== undefined && config.checkpoint !== null && String(config.checkpoint) !== String(identity.checkpoint)) ||
          !Number.isSafeInteger(identity.execute_steps) || identity.execute_steps <= 0 || identity.execute_steps > horizon ||
          (config.execute_steps !== undefined && config.execute_steps !== identity.execute_steps) ||
          (data.execute_steps !== undefined && data.execute_steps !== identity.execute_steps)) {
        throw new Error('Unified workbench publication identity mismatch.');
      }
    }
    return true;
  }
  function mountFallbackAudit(root, published) {
    if (!published) return null;
    const doc = root.ownerDocument, moved = [];
    const panel = doc.createElement('details'); panel.className = 'wb-detail wb-fallback-audit'; panel.open = false;
    const heading = doc.createElement('summary'); heading.textContent = '原始仪表盘审核记录（非当前评估成绩）'; panel.appendChild(heading);
    const body = doc.createElement('div'); body.className = 'wb-detail-body'; panel.appendChild(body);
    const note = doc.createElement('p'); note.className = 'wb-section-note';
    note.textContent = published.valid !== true ? '本轨迹已隔离或无效，不计分。以下保留的原始通过/未通过标签只供审核，不代表当前任务或阶段成绩。' :
      '以下为原始仪表盘与审核标签；当前任务结果以页首归档身份为准，不据原审核通过标签重新评分。';
    body.appendChild(note);
    ['.ke-header', '.ke-summary', '.ke-meta-line', '.ke-v2-evidence', '.ke-warnings', '.ke-footer'].forEach(selector => {
      const node = root.querySelector(selector);
      if (node) { moved.push({node, parent: node.parentNode, next: node.nextSibling}); body.appendChild(node); }
    });
    root.appendChild(panel);
    return {node: panel, destroy() {
      moved.slice().reverse().forEach(({node, parent, next}) => {
        if (next && next.parentNode === parent) parent.insertBefore(node, next); else parent.appendChild(node);
      });
      panel.remove();
    }};
  }
  function whiteboardAreaMessage(area) {
    if (area?.status === 'verified') return '已补充归档的板擦贴合指标：接触期间均值按原物理步时长加权，不是视频帧平均。该面积是带载刚体几何估计，不是真实形变接触面积，也没有预设成功百分比。';
    if (area?.status === 'recorded') return '已记录的板擦贴合指标：保留归档曲线及原物理步加权均值，不重新积分。面积是带载刚体几何估计（未重新认证），不是真实形变接触面积，也不是任务成功率。';
    return '板擦贴合指标来源未能核实：不显示推测曲线，也不将缺测写成 0。';
  }
  // Page sections are display navigation only: never pass a clock or seek function.
  function mountSectionNavigation(parent, sections) {
    const doc = parent.ownerDocument, removers = [];
    const nav = doc.createElement('nav'); nav.className = 'wb-section-nav'; nav.setAttribute('aria-label', '轨迹分析分区');
    parent.appendChild(nav);
    sections.filter(section => section?.node && /^[a-z][a-z0-9-]+$/.test(section.id)).forEach(section => {
      section.node.id = section.id;
      const link = doc.createElement('a'); link.className = 'wb-section-link' + (section.id === 'wb-expert-reference' ? ' wb-reference-link' : '');
      link.href = '#' + section.id; link.textContent = section.label; nav.appendChild(link);
      const click = event => {
        event.preventDefault();
        if (section.node.tagName.toLowerCase() === 'details') section.node.open = true;
        section.node.scrollIntoView?.({behavior: 'auto', block: 'start'});
      };
      link.addEventListener('click', click); removers.push(() => link.removeEventListener('click', click));
    });
    return {node: nav, destroy() { removers.forEach(remove => remove()); nav.remove(); }};
  }
  function navigationSeedLabel(seed) { return seed === null ? '未知' : String(seed).padStart(3, '0'); }
  function unifiedTaskNavigationRows(index, identity) {
    if (index?.schema !== 'kaihand-unified-evaluation-workbench-v1' || !Array.isArray(index.groups) ||
        !index.groups.length || index.groups.length > 2048 || typeof identity?.task !== 'string' ||
        typeof identity?.trial_id !== 'string' || typeof identity?.group_id !== 'string') return null;
    const rows = [], ids = new Set(), paths = new Set(), groups = new Set();
    for (const group of index.groups) {
      if (!group || typeof group.id !== 'string' || !group.id || groups.has(group.id) ||
          typeof group.task !== 'string' || !group.task || typeof group.model !== 'string' || !group.model ||
          !['string', 'number'].includes(typeof group.checkpoint) || !String(group.checkpoint) ||
          (typeof group.checkpoint === 'number' && !finite(group.checkpoint)) ||
          !Number.isSafeInteger(group.prediction_horizon) || group.prediction_horizon <= 0 ||
          !Number.isSafeInteger(group.execute_steps) || group.execute_steps <= 0 || group.execute_steps > group.prediction_horizon ||
          !Array.isArray(group.trials)) return null;
      groups.add(group.id);
      for (const trial of group.trials) {
        if (!trial || typeof trial.id !== 'string' || !trial.id || ids.has(trial.id) ||
            (trial.seed !== null && (!Number.isSafeInteger(trial.seed) || trial.seed < 0)) ||
            typeof trial.valid !== 'boolean' || (trial.valid && typeof trial.native_success !== 'boolean') ||
            typeof trial.detail_url !== 'string' ||
            !/^(?:[A-Za-z0-9_-]+\/)+(?:report|index)\.html$/.test(trial.detail_url) || paths.has(trial.detail_url)) return null;
        ids.add(trial.id); paths.add(trial.detail_url);
        rows.push({task: group.task, group_id: group.id, trial_id: trial.id, model: group.model,
          checkpoint: String(group.checkpoint), seed: trial.seed, prediction_horizon: group.prediction_horizon,
          execute_steps: group.execute_steps, report_path: trial.detail_url, valid: trial.valid,
          publication_status: trial.publication_status, native_success: trial.valid &&
            !String(trial.publication_status || '').startsWith('quarantined') ? trial.native_success : null,
          staged_evidence_status: trial.staged_evidence_status, protocol: group.protocol || {}, cohort_id: group.cohort_id});
        if (rows.length > 20000) return null;
      }
    }
    const trials = rows.filter(trial => trial.task === identity.task).sort((a, b) =>
      a.model.localeCompare(b.model, undefined, {numeric: true}) ||
      a.checkpoint.localeCompare(b.checkpoint, undefined, {numeric: true}) || a.group_id.localeCompare(b.group_id) ||
      (a.seed === null ? b.seed === null ? 0 : 1 : b.seed === null ? -1 : a.seed - b.seed) || a.trial_id.localeCompare(b.trial_id));
    const currentIndex = trials.findIndex(trial => trial.trial_id === identity.trial_id && trial.group_id === identity.group_id);
    if (currentIndex < 0) return null;
    const current = trials[currentIndex];
    for (const key of ['task', 'model', 'prediction_horizon', 'execute_steps', 'seed', 'report_path']) {
      if (identity[key] !== undefined && current[key] !== identity[key]) return null;
    }
    if (identity.checkpoint !== undefined && current.checkpoint !== String(identity.checkpoint)) return null;
    return {trials, currentIndex, unified: true};
  }
  function taskNavigationRows(index, identity) {
    if (index?.schema === 'kaihand-unified-evaluation-workbench-v1') return unifiedTaskNavigationRows(index, identity);
    if (index?.schema_version !== 'evaluation-task-navigation-v1' || index.prediction_horizon !== 50 ||
        !Array.isArray(index.trials) || !index.trials.length || index.trials.length > 10000 ||
        typeof identity?.task !== 'string' || typeof identity?.trial_id !== 'string') return null;
    const ids = new Set(), paths = new Set();
    for (const trial of index.trials) {
      if (!trial || trial.prediction_horizon !== 50 || typeof trial.task !== 'string' || !trial.task ||
          typeof trial.trial_id !== 'string' || !trial.trial_id || typeof trial.model !== 'string' || !trial.model ||
          !['string', 'number'].includes(typeof trial.checkpoint) || !String(trial.checkpoint) ||
          (typeof trial.checkpoint === 'number' && !finite(trial.checkpoint)) ||
          (trial.seed !== null && (!Number.isSafeInteger(trial.seed) || trial.seed < 0)) || typeof trial.report_path !== 'string' ||
          !/^[A-Za-z0-9_-]+\/report\.html$/.test(trial.report_path) || ids.has(trial.trial_id) || paths.has(trial.report_path)) return null;
      ids.add(trial.trial_id); paths.add(trial.report_path);
    }
    const trials = index.trials.filter(trial => trial.task === identity.task).slice().sort((a, b) =>
      a.model.localeCompare(b.model, undefined, {numeric: true}) ||
      String(a.checkpoint).localeCompare(String(b.checkpoint), undefined, {numeric: true}) ||
      (a.seed === null ? b.seed === null ? 0 : 1 : b.seed === null ? -1 : a.seed - b.seed) || a.trial_id.localeCompare(b.trial_id));
    const currentIndex = trials.findIndex(trial => trial.trial_id === identity.trial_id);
    if (currentIndex < 0 || (identity.report_path !== undefined && trials[currentIndex].report_path !== identity.report_path)) return null;
    return {trials, currentIndex};
  }
  function mountTrajectoryNavigation(parent, identity, environment = global, options = {}) {
    options = options || {};
    const doc = parent.ownerDocument, removers = [];
    let destroyed = false, controller = null, validated = null;
    function el(tag, cls, container, text) { const n = doc.createElement(tag); n.className = cls; if (text !== undefined) n.textContent = text; container.appendChild(n); return n; }
    const nav = el('nav', 'wb-trajectory-nav', parent); nav.setAttribute('aria-label', '同任务评估轨迹切换'); nav.dataset.task = identity.task;
    const title = el('div', 'wb-trajectory-title', nav); el('strong', '', title, '同任务切换');
    const count = el('span', 'wb-trajectory-count', title, '正在读取…');
    const previous = el('a', 'wb-trajectory-prev', nav, '← 上一条');
    const select = el('select', 'wb-trajectory-select', nav); select.setAttribute('aria-label', '选择同任务评估轨迹'); select.disabled = true;
    const placeholder = el('option', '', select, '正在读取同任务轨迹…'); placeholder.value = '';
    const next = el('a', 'wb-trajectory-next', nav, '下一条 →');
    const status = el('p', 'wb-trajectory-status', nav); status.setAttribute('role', 'status');
    const unified = typeof options.catalog_url === 'string' && !!options.catalog_url;
    let catalogURL = null, currentURL = null;
    if (unified) {
      try {
        const base = environment.location?.href || 'http://localhost' + (environment.location?.pathname || '/');
        if (/^(?:[a-z]+:|\/\/)|[\\\u0000-\u0020]|%2e|%2f|%5c/i.test(options.catalog_url)) throw new Error('Unsafe catalog URL');
        currentURL = new URL(base); catalogURL = new URL(options.catalog_url, currentURL);
        if (catalogURL.origin !== currentURL.origin || !/^https?:$/.test(catalogURL.protocol) ||
            !catalogURL.pathname.endsWith('/catalog.json') || catalogURL.search || catalogURL.hash) throw new Error('Unsafe catalog URL');
      } catch (_) { catalogURL = null; }
    }
    function destination(trial) { return unified ? new URL(trial.report_path, catalogURL).href : '../' + trial.report_path; }
    function disableLink(link) { link.removeAttribute('href'); link.setAttribute('aria-disabled', 'true'); link.tabIndex = -1; }
    function updateLink(link, trial) {
      if (!trial) { disableLink(link); return; }
      link.href = destination(trial); link.setAttribute('aria-disabled', 'false'); link.tabIndex = 0;
      link.title = `${unified ? modelDisplayName(trial.model) : trial.model} · ${/^\d+$/.test(String(trial.checkpoint)) ? 'step ' : ''}${trial.checkpoint} · seed ${navigationSeedLabel(trial.seed)}`;
    }
    function unavailable(message) {
      if (destroyed) return;
      nav.dataset.status = 'unavailable'; count.textContent = '切换暂不可用';
      placeholder.textContent = '可通过“评估总览”选择轨迹';
      status.textContent = message + '；当前回放和指标不受影响，可使用页首“评估总览”。';
    }
    disableLink(previous); disableLink(next); nav.dataset.status = 'loading';
    function change() {
      if (destroyed || !validated) return;
      const trial = validated.trials.find(item => item.report_path === select.value);
      if (trial && trial.trial_id !== identity.trial_id) environment.location.assign(destination(trial));
    }
    select.addEventListener('change', change); removers.push(() => select.removeEventListener('change', change));
    let ready;
    if (unified && !catalogURL) {
      unavailable('统一导航清单地址未通过来源校验'); ready = Promise.resolve();
    } else if (!/^https?:$/.test(environment.location?.protocol || '') || typeof environment.fetch !== 'function') {
      unavailable('本地文件模式无法读取同任务导航索引'); ready = Promise.resolve();
    } else {
      if (typeof environment.AbortController === 'function') controller = new environment.AbortController();
      ready = Promise.resolve().then(() => {
        if (destroyed) return null;
        return environment.fetch(unified ? options.catalog_url : '../trajectory_index.json', {credentials: 'same-origin', mode: 'same-origin', ...(controller ? {signal: controller.signal} : {})});
      }).then(response => {
        if (destroyed) return null;
        if (!response?.ok) throw new Error('navigation index unavailable');
        return response.json();
      }).then(index => {
        if (destroyed) return;
        const reportPath = unified ? currentURL.pathname.startsWith(new URL('./', catalogURL).pathname) ?
          currentURL.pathname.slice(new URL('./', catalogURL).pathname.length) : null :
          environment.location.pathname?.match(/\/([A-Za-z0-9_-]+\/report\.html)$/)?.[1];
        validated = reportPath ? taskNavigationRows(index, {...identity, report_path: reportPath}) : null;
        if (!validated) throw new Error('navigation index identity mismatch');
        select.replaceChildren(); let group = null, groupKey = null;
        validated.trials.forEach(trial => {
          const key = JSON.stringify([trial.model, String(trial.checkpoint), unified ? trial.group_id : null]);
          if (key !== groupKey) {
            group = el('optgroup', '', select);
            group.label = unified ? `${modelDisplayName(trial.model)} · ${/^\d+$/.test(trial.checkpoint) ? 'step ' : ''}${trial.checkpoint} · H${trial.prediction_horizon}/K${trial.execute_steps}` : `${trial.model} · step ${trial.checkpoint}`;
            if (unified) group.title = JSON.stringify({group_id: trial.group_id, protocol: trial.protocol, cohort_id: trial.cohort_id});
            groupKey = key;
          }
          const outcome = unified && String(trial.publication_status || '').startsWith('quarantined') ? '隔离历史 · 不计分' :
            unified && !trial.valid ? '无效 · 不计分' : trial.native_success === true ? '原任务成功' : trial.native_success === false ? '原任务未成功' : '原任务结果未知';
          const option = el('option', '', group, `seed ${navigationSeedLabel(trial.seed)} · ${outcome}`);
          option.value = trial.report_path; option.selected = trial.trial_id === identity.trial_id;
        });
        select.value = validated.trials[validated.currentIndex].report_path; select.disabled = validated.trials.length < 2;
        count.textContent = `当前 ${validated.currentIndex + 1} / ${validated.trials.length} 条`;
        updateLink(previous, validated.trials[validated.currentIndex - 1]); updateLink(next, validated.trials[validated.currentIndex + 1]);
        status.textContent = (unified ? '来自统一工作台的全部同任务轨迹，包含不同模型及 checkpoint；' : '') + '切换后从该轨迹自身时间起点载入，不做时间对齐。';
        nav.dataset.status = 'ready';
      }).catch(() => { unavailable('同任务导航索引未能读取或身份校验未通过'); });
    }
    return {node: nav, ready, destroy() {
      if (destroyed) return; destroyed = true; controller?.abort(); removers.forEach(remove => remove()); nav.remove();
    }};
  }
  function mountDetailHeader(parent, payload, data, config, environment = global) {
    const doc = parent.ownerDocument, published = config.unified_identity, recorded = detailIdentity(payload, data, config);
    const originalTitle = doc.title, taskTitle = taskDisplayTitle(config, data.task), pageTitle = `${taskTitle} | KaiTactileSim`;
    let destroyed = false;
    doc.title = pageTitle;
    function el(tag, cls, container, text) { const n = doc.createElement(tag); n.className = cls; if (text !== undefined) n.textContent = text; container.appendChild(n); return n; }
    function link(container, text, href, cls = '') {
      if (typeof href !== 'string' || !href || /^(?:[a-z]+:|\/\/)|[\\\u0000-\u0020]/i.test(href)) return;
      const a = el('a', 'wb-link' + (cls ? ' ' + cls : ''), container, text); a.href = href;
    }
    const header = doc.createElement('header'); header.className = 'wb-header'; parent.prepend(header);
    Object.entries(recorded).forEach(([key, value]) => { if (value !== undefined && value !== null) header.dataset[key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase())] = String(value); });
    const identity = el('div', 'wb-identity', header), brand = el('p', 'wb-eyebrow', identity);
    const horizon = Number.isSafeInteger(recorded.prediction_horizon) ? 'H' + recorded.prediction_horizon : 'H 未记录';
    const execution = Number.isSafeInteger(recorded.execute_steps) ? ' / K' + recorded.execute_steps : '';
    el('span', 'wb-platform-name', brand, PLATFORM_NAME); el('span', 'wb-horizon', brand, horizon + execution);
    el('h1', '', identity, `${taskTitle} · 轨迹分析`);
    const checkpoint = recorded.checkpoint === null ? 'checkpoint 未记录' : (/^\d+$/.test(String(recorded.checkpoint)) ? 'step ' : '') + recorded.checkpoint;
    el('p', 'wb-trial', identity, `${modelDisplayName(recorded.model)} · ${checkpoint} · seed ${navigationSeedLabel(recorded.seed)} · ${horizon}${execution}`);
    const summary = el('div', 'wb-outcome', header), status = nativeOutcomeStatus(payload.report, config.staged_evaluation, published);
    const quarantined = publishedStageStatus(published) === 'quarantined';
    const outcomeLabel = quarantined ? '隔离（不计分）' : published && published.valid !== true ? '无效（不计分）' : {pass: '成功', fail: '失败', unknown: '未知'}[status];
    const outcome = el('span', `wb-verdict wb-${status}`, summary, `原任务结果 · ${outcomeLabel}`);
    outcome.title = config.staged_evaluation ? '来自归档的原生 task_success；阶段记录独立展示，不根据触觉曲线重新评分' : '保留原任务判定；不根据触觉曲线重新评分';
    const finalClean = payload.report.task_specific?.cleaned_fraction ?? payload.report.partial_state?.cleaned_fraction;
    if (finite(finalClean) && (!config.generic || data.task.startsWith('whiteboard-wipe'))) el('strong', '', summary, `最终清洁进度 ${(finalClean * 100).toFixed(1)}%`);
    el('span', 'wb-final-note', summary, '终态摘要，不是当前游标值');
    const links = el('nav', 'wb-links', header);
    if (config.unified_navigation?.overview_url) link(links, '返回统一工作台', config.unified_navigation.overview_url, 'wb-overview-link');
    else link(links, '评估总览', config.suite_url, 'wb-overview-link');
    link(links, '返回 r2 原版', config.baseline_url); link(links, 'H50 稳定版', config.stable_url); link(links, '历史版本', config.previous_workbench_url);
    const stages = stageBadgeDefinitions(config.staged_evaluation, published), evidence = stageEvidenceMessage(published);
    if (config.staged_evaluation || published) {
      const stageSummary = el('section', 'wb-header-stages', header); stageSummary.setAttribute('aria-label', '整条轨迹的任务阶段完成情况');
      if (published) stageSummary.dataset.evidenceStatus = publishedStageStatus(published);
      const stageHeading = el('div', 'wb-header-stage-heading', stageSummary);
      el('strong', '', stageHeading, '任务阶段');
      el('span', 'wb-header-stage-count', stageHeading, evidence ? evidence.title : stages.length ? `${stages.filter(stage => stage.reached === true).length} / ${stages.length} 曾达到` : '未记录可验证的阶段数据');
      const badges = el('div', 'wb-stage-badges', stageSummary);
      stages.forEach(stage => {
        const badge = el('span', 'wb-stage-badge', badges, `${stage.reached === true ? '✓' : stage.reached === false ? '—' : '?'} ${stage.label}`);
        badge.dataset.stage = stage.key; badge.dataset.reached = stage.reached === true ? 'true' : stage.reached === false ? 'false' : 'unknown';
        badge.title = `${stage.reached === true ? '整条轨迹曾达到' : stage.reached === false ? '整条轨迹未达到' : '未记录，不能当作未达到'}${stage.first_time_s === null ? '' : `；首次 ${fmt(stage.first_time_s)} s`}；不随当前游标改变`;
      });
      const whiteboardPickupOnly = data.task.startsWith('whiteboard-wipe') && config.staged_evaluation?.staged_result?.complete_success === null;
      el('p', 'wb-header-stage-note', stageSummary, evidence ? evidence.note : whiteboardPickupOnly ? '全程累计“曾达到”，不随游标变化；白板这里只记录拿起，不能视为擦拭完成。' : '全程累计“曾达到”，不随游标变化；阶段记录与原任务结果分别保留。');
    }
    if (published) {
      const audit = el('details', 'wb-evidence-audit', header); el('summary', '', audit, '数据来源与原始记录');
      (Array.isArray(config.audit_links) ? config.audit_links : []).forEach(item => link(audit, item.label || '查看原始记录', item.url, 'wb-original-record-link'));
      el('pre', 'ke-evidence-json', audit, JSON.stringify({publication: published,
        rule_identity: config.staged_evaluation?.rule_identity, warnings: config.staged_evaluation?.warnings}, null, 2));
    }
    const navigation = mountTrajectoryNavigation(header, recorded, environment, config.unified_navigation);
    return {node: header, navigation, destroy() {
      if (destroyed) return; destroyed = true;
      navigation.destroy(); header.remove();
      if (doc.title === pageTitle) doc.title = originalTitle;
    }};
  }
  let plotGalleryCount = 0;
  // Static source figures have their own controls and never seek either video.
  function mountDiagnosticPlots(parent, config) {
    const doc = parent.ownerDocument, removers = [], plots = (config?.plots || []).filter(plot => diagnosticPlotURL(plot));
    let destroyed = false, selected = 0;
    function el(tag, cls, container, text) { const n = doc.createElement(tag); n.className = cls; if (text !== undefined) n.textContent = text; container.appendChild(n); return n; }
    function on(node, type, fn) { node.addEventListener(type, fn); removers.push(() => node.removeEventListener(type, fn)); }
    const panel = el('section', 'wb-plots-panel', parent);
    const modelOnly = config.comparison_kind === 'model_only'; panel.classList.toggle('wb-model-only', modelOnly);
    const heading = el('div', 'wb-section-heading', panel); el('span', 'wb-step', heading, '03'); el('h2', '', heading, modelOnly ? '模型全程诊断图' : '全程诊断图');
    el('p', 'wb-plots-source', panel, config.source_label || config.task || '当前轨迹');
    if (!plots.length || config.status !== 'available') {
      el('p', 'wb-section-note', panel, '该轨迹的原评估未生成完整的三类参考对照图。模型视频、任务响应和触觉载荷仍可查看。');
      return {node: panel, destroy() { panel.remove(); }};
    }
    el('p', 'wb-plots-note', panel, config.comparison_note || '专家与模型按各自起点时间叠画；同一横坐标不表示处于相同任务阶段。原图供观察，未进行阶段对齐。');
    const legend = el('div', 'wb-plots-legend', panel);
    if (!modelOnly) el('span', 'wb-plots-reference', legend, config.reference_label || '蓝色实线 · 专家参考');
    el('span', 'wb-plots-rollout', legend, config.rollout_label || '红色虚线 · 模型推理');
    const tablist = el('div', 'wb-plots-tabs', panel); tablist.setAttribute('role', 'tablist'); tablist.setAttribute('aria-label', '选择全程诊断图');
    const content = el('div', 'wb-plots-content', panel), galleryId = `wb-plot-${++plotGalleryCount}`;
    content.id = galleryId; content.setAttribute('role', 'tabpanel');
    const toolbar = el('div', 'wb-plots-toolbar', content), description = el('p', 'wb-plots-description', toolbar);
    const actions = el('div', 'wb-plots-actions', toolbar), enlarge = el('button', '', actions, '放大查看'); enlarge.type = 'button';
    const download = el('a', '', actions, '下载原图'); download.setAttribute('download', '');
    const previewFrame = el('div', 'wb-plots-preview-frame', content);
    previewFrame.tabIndex = 0; previewFrame.setAttribute('role', 'region'); previewFrame.setAttribute('aria-label', '诊断图预览，可上下滚动浏览，点击图片放大');
    const preview = el('button', 'wb-plots-preview', previewFrame); preview.type = 'button';
    const image = el('img', 'wb-plots-image', preview); image.loading = 'lazy'; image.decoding = 'async';
    const status = el('p', 'wb-plots-status', content); status.setAttribute('role', 'status');
    const note = el('p', 'wb-plots-note wb-plots-tactile-note', content); note.hidden = true;
    const dialog = el('dialog', 'wb-plot-dialog', panel), dialogHeader = el('div', 'wb-plot-dialog-header', dialog);
    const dialogTitle = el('strong', '', dialogHeader), tools = el('div', 'wb-plot-dialog-tools', dialogHeader);
    const fit = el('button', '', tools, '适应宽度'), nativeSize = el('button', '', tools, '原尺寸'), close = el('button', '', tools, '关闭');
    [fit, nativeSize, close].forEach(button => { button.type = 'button'; });
    close.setAttribute('aria-label', '关闭放大图'); dialog.setAttribute('aria-label', '全程诊断图放大查看');
    const scroll = el('div', 'wb-plot-dialog-scroll', dialog), largeImage = el('img', '', scroll);
    scroll.tabIndex = 0; scroll.setAttribute('role', 'region'); scroll.setAttribute('aria-label', '放大图片，可上下左右滚动');
    el('p', 'wb-plot-dialog-note', dialog, '可滚动查看整张图 · Esc 关闭 · 此处操作不改变视频时间');
    function setSize(native) {
      scroll.classList.toggle('wb-plot-native', native);
      fit.setAttribute('aria-pressed', String(!native)); nativeSize.setAttribute('aria-pressed', String(native));
    }
    function openImage() {
      if (destroyed) return;
      largeImage.src = diagnosticPlotURL(plots[selected]); largeImage.alt = image.alt;
      dialogTitle.textContent = `${config.source_label || config.task} · ${plots[selected].label}`;
      setSize(false); dialog.showModal(); scroll.scrollTo(0, 0);
    }
    function select(index, focus = false) {
      selected = index; const plot = plots[index], url = diagnosticPlotURL(plot);
      tabs.forEach((button, i) => { button.setAttribute('aria-selected', String(index === i)); button.tabIndex = index === i ? 0 : -1; });
      content.setAttribute('aria-labelledby', tabs[index].id);
      description.textContent = plot.description || plot.label;
      image.alt = `${config.source_label || config.task} · ${plot.label}，${modelOnly ? '仅模型原生记录' : '原始专家与模型对照图'}`;
      image.src = url; image.width = plot.width || 1560; image.height = plot.height || 1200;
      previewFrame.scrollTo(0, 0);
      preview.setAttribute('aria-label', `放大查看${plot.label}`); download.href = url;
      status.textContent = '预览框内可滚动浏览整张图；点击图片放大，或选“原尺寸”读刻度。图中时间轴不联动视频。';
      note.hidden = plot.id !== 'tactile' || !config.tactile_note; note.textContent = config.tactile_note || '';
      panel.dataset.plotId = plot.id; if (focus) tabs[index].focus();
    }
    const tabs = plots.map((plot, index) => {
      const button = el('button', '', tablist, plot.label); button.type = 'button'; button.setAttribute('role', 'tab');
      button.id = `${galleryId}-tab-${index}`; button.setAttribute('aria-controls', galleryId);
      on(button, 'click', () => select(index));
      on(button, 'keydown', event => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault(); select(event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length, true);
      }); return button;
    });
    on(preview, 'click', openImage); on(enlarge, 'click', openImage);
    on(fit, 'click', () => setSize(false)); on(nativeSize, 'click', () => setSize(true)); on(close, 'click', () => dialog.close());
    on(dialog, 'click', event => { if (event.target === dialog) { const b = dialog.getBoundingClientRect(); if (event.clientX < b.left || event.clientX > b.right || event.clientY < b.top || event.clientY > b.bottom) dialog.close(); } });
    on(image, 'error', () => { status.textContent = '原图未能加载，可通过下载入口检查文件。'; });
    if (config.diagnostic_example_url && /^\.\.\/diagnostics\/[A-Za-z0-9_.-]+\.html$/.test(config.diagnostic_example_url)) {
      const example = el('a', 'wb-link wb-plots-example', panel, '查看灯泡旋出示例的三类原图'); example.href = config.diagnostic_example_url;
    }
    select(0);
    return {node: panel, destroy() { if (destroyed) return; destroyed = true; if (dialog.open) dialog.close(); removers.forEach(remove => remove()); panel.remove(); }};
  }
  // Deliberately receives no model slider, model clock, charts or seek callback.
  function mountExpertPlayer(parent, config) {
    const doc = parent.ownerDocument, removers = []; let destroyed = false, frame = null, failed = false;
    function el(tag, cls, container, text) { const n = doc.createElement(tag); n.className = cls; if (text !== undefined) n.textContent = text; container.appendChild(n); return n; }
    function on(node, type, fn) { node.addEventListener(type, fn); removers.push(() => node.removeEventListener(type, fn)); }
    const panel = el('details', 'wb-expert-panel', parent); panel.open = false;
    const heading = el('summary', 'wb-expert-heading', panel);
    const expertTitle = el('strong', '', heading, '专家示范');
    if (typeof config.title === 'string') expertTitle.title = config.title;
    el('span', 'wb-expert-badge', heading, '独立时间线');
    el('span', 'wb-expert-invitation', heading, '展开查看示范');
    const body = el('div', 'wb-expert-body', panel);
    el('p', 'wb-expert-source', body, config.source_label || '已声明的专家示范视频');
    el('p', 'wb-expert-note', body, '专家播放与模型播放互不联动；本页任务曲线和力图始终属于模型轨迹。');
    if (config.comparison_note) el('p', 'wb-expert-note wb-expert-comparison', body, config.comparison_note);
    const cameras = el('div', 'wb-expert-cameras', body); cameras.setAttribute('role', 'group'); cameras.setAttribute('aria-label', '专家视频视角');
    const stage = el('div', 'wb-expert-stage', body), video = el('video', 'wb-expert-video', stage);
    video.muted = true; video.playsInline = true; video.preload = 'metadata'; video.controls = false;
    video.setAttribute('aria-label', '专家示范视频，使用独立专家时间线控制');
    const cameraConfig = config.media_views?.sha256 === config.sha256 ? config.media_views : null;
    const views = (Array.isArray(cameraConfig?.views) ? cameraConfig.views : []).filter(view => cropStyle(cameraConfig, view.id));
    const cameraButtons = [];
    function setCamera(id) {
      const crop = cropStyle(cameraConfig, id); stage.classList.toggle('wb-expert-cropped', !!crop); stage.style.aspectRatio = crop ? crop.aspect : '';
      ['width', 'height', 'left', 'top'].forEach(key => { video.style[key] = crop ? crop[key] : ''; }); video.dataset.camera = id;
      cameraButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.camera === id)));
    }
    [...views, {id: 'mosaic', label: '完整画面'}].forEach(view => {
      const button = el('button', 'wb-expert-camera', cameras, view.label); button.type = 'button'; button.dataset.camera = view.id;
      cameraButtons.push(button); on(button, 'click', () => setCamera(view.id));
    });
    if (!views.length) cameras.hidden = true;
    setCamera(views[0]?.id || 'mosaic');
    const transport = el('div', 'wb-expert-transport', body); transport.setAttribute('role', 'group'); transport.setAttribute('aria-label', '专家独立视频时间线');
    const row = el('div', 'wb-expert-clock-row', transport), timeBox = el('div', 'wb-expert-clock', row);
    el('span', 'wb-expert-clock-label', timeBox, '专家视频时间');
    const time = el('strong', 'wb-expert-time', timeBox, '0.000'), end = el('span', 'wb-expert-end', timeBox, ' / — s');
    const controls = el('div', 'wb-expert-controls', row);
    const play = el('button', 'wb-expert-play', controls, '▶ 播放专家'); play.type = 'button';
    const rateLabel = el('label', 'wb-expert-speed-label', controls, '倍速 '), rate = el('select', 'wb-expert-speed', rateLabel);
    rate.setAttribute('aria-label', '专家视频播放速度');
    [.25, .5, 1, 2].forEach(value => { const option = el('option', '', rate, `${value}×`); option.value = value; option.selected = value === 1; });
    const fullscreen = el('button', 'wb-expert-fullscreen', controls, '全屏'); fullscreen.type = 'button'; fullscreen.disabled = !video.requestFullscreen;
    fullscreen.title = '全屏查看专家原始完整视频';
    const slider = el('input', 'wb-expert-slider', transport); slider.type = 'range'; slider.min = '0'; slider.max = '1'; slider.step = 'any'; slider.value = '0'; slider.disabled = true;
    slider.setAttribute('aria-label', '专家视频时间，秒；独立于模型轨迹');
    const status = el('p', 'wb-expert-status', transport, '正在读取专家视频…'); status.setAttribute('role', 'status');
    const provenance = el('details', 'wb-expert-provenance', body); el('summary', '', provenance, '示范来源与对照范围');
    el('p', '', provenance, config.description || '独立示范视频，仅用于回看。');
    el('p', '', provenance, config.comparison_note || '未建立与模型轨迹的时间或任务阶段对应关系，不参与模型指标计算。');
    el('p', '', provenance, '时间轴使用此视频自身的编码秒数，不把视频秒数当作模型仿真时间。');
    el('pre', '', provenance, JSON.stringify({task: config.task, recorded_title: config.title, video_sha256: config.sha256, clock: config.clock, provenance: config.provenance}, null, 2));
    function duration() { return !failed && finite(video.duration) && video.duration > 0 ? video.duration : null; }
    function update() {
      if (destroyed) return;
      const total = duration(), current = finite(video.currentTime) ? video.currentTime : 0;
      time.textContent = fmt(current); end.textContent = total === null ? ' / — s' : ` / ${fmt(total)} s`;
      slider.disabled = play.disabled = rate.disabled = total === null;
      if (total !== null) { slider.max = String(total); slider.value = String(clamp(current, 0, total)); }
      slider.setAttribute('aria-valuetext', `专家视频 ${fmt(current)} 秒${total === null ? '' : '，共 ' + fmt(total) + ' 秒'}；不联动模型`);
      slider.style.setProperty('--expert-progress', `${total ? current / total * 100 : 0}%`);
      play.textContent = video.paused || video.ended ? '▶ 播放专家' : 'Ⅱ 暂停专家'; play.setAttribute('aria-pressed', String(!video.paused && !video.ended));
    }
    function tick() { frame = null; update(); if (!destroyed && !video.paused && !video.ended) frame = global.requestAnimationFrame(tick); }
    function stopTick() { if (frame !== null) global.cancelAnimationFrame(frame); frame = null; }
    function seek(value) { const total = duration(); if (total === null || !finite(value)) return; video.pause(); video.currentTime = clamp(value, 0, total); update(); }
    on(play, 'click', () => {
      if (duration() === null) return;
      if (!video.paused) { video.pause(); return; }
      if (video.ended || video.currentTime >= video.duration) video.currentTime = 0;
      const promise = video.play(); if (promise?.catch) promise.catch(error => { if (!destroyed && error.name !== 'AbortError') { status.textContent = '专家视频暂时无法播放，请重试或检查媒体文件。'; update(); } });
    });
    on(slider, 'input', () => seek(Number(slider.value)));
    on(slider, 'keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key) || duration() === null) return;
      event.preventDefault(); seek(event.key === 'Home' ? 0 : event.key === 'End' ? video.duration : video.currentTime + (event.key === 'ArrowRight' ? 1 : -1));
    });
    on(rate, 'change', () => { video.playbackRate = Number(rate.value); });
    on(video, 'loadedmetadata', () => {
      status.textContent = duration() === null ? '专家视频时长不可用；独立时间线已禁用。' : '仅控制专家视频 · 拖动定位 · ← → 前后 1 秒'; update();
    });
    ['durationchange', 'timeupdate', 'seeking', 'seeked', 'ratechange'].forEach(event => on(video, event, update));
    on(video, 'play', () => {
      if (!failed) status.textContent = '仅控制专家视频 · 拖动定位 · ← → 前后 1 秒';
      stopTick(); tick();
    });
    ['pause', 'ended'].forEach(event => on(video, event, () => { stopTick(); update(); }));
    on(video, 'error', () => { failed = true; stopTick(); status.textContent = '专家视频未能加载；模型回放与评估图不受影响。'; update(); });
    on(panel, 'toggle', () => { if (!panel.open) video.pause(); });
    on(fullscreen, 'click', () => {
      const promise = video.requestFullscreen?.(); if (promise?.catch) promise.catch(() => { if (!destroyed) status.textContent = '此环境不支持全屏，可继续在页内播放专家视频。'; });
    });
    on(doc, 'fullscreenchange', () => { video.controls = doc.fullscreenElement === video; });
    const url = expertVideoURL(config);
    if (url) video.src = url;
    else { failed = true; status.textContent = '没有经过来源核验的专家视频，不使用其他轨迹替代。'; }
    update();
    return {node: panel, destroy() {
      if (destroyed) return; destroyed = true; stopTick(); removers.forEach(remove => remove());
      video.pause(); video.removeAttribute('src'); video.load(); panel.remove();
    }};
  }
  function mountStagedPanel(parent, config, seek, maximumTime, published) {
    config = config || {};
    const unavailable = stageEvidenceMessage(published);
    if (unavailable) {
      const doc = parent.ownerDocument;
      function el(tag, cls, container, text) { const node = doc.createElement(tag); node.className = cls; if (text !== undefined) node.textContent = text; container.appendChild(node); return node; }
      const panel = el('section', 'wb-staged-panel wb-staged-unverified', parent);
      panel.dataset.evidenceStatus = publishedStageStatus(published);
      const heading = el('div', 'wb-staged-heading', panel); el('h2', '', heading, '任务阶段 · 来源说明');
      el('p', 'wb-staged-scope', panel, unavailable.title); el('p', 'wb-staged-note', panel, unavailable.note);
      const cards = el('div', 'wb-staged-cards', panel);
      stageBadgeDefinitions(config, published).forEach(stage => {
        const card = el('article', 'wb-stage-card', cards); card.dataset.stage = stage.key; card.dataset.reached = 'unknown';
        el('span', 'wb-stage-name', card, stage.label); el('strong', '', card, '未知 / 不计分');
      });
      const audit = el('details', 'wb-staged-rules', panel); el('summary', '', audit, '展开阶段来源记录');
      const download = el('a', 'wb-link', audit, '下载本页配置（含阶段记录与来源）'); download.href = 'workbench.json';
      el('pre', 'ke-evidence-json', audit, JSON.stringify({publication: published,
        rule_identity: config.rule_identity, warnings: config.warnings}, null, 2));
      return {update() {}, destroy() { panel.remove(); }};
    }
    const doc = parent.ownerDocument, removers = [], timeline = config.timeline || {};
    const mapped = timeline.model_clock_mapping === 'mapped';
    function el(tag, cls, container, text) { const n = doc.createElement(tag); n.className = cls; if (text !== undefined) n.textContent = text; container.appendChild(n); return n; }
    const panel = el('section', 'wb-staged-panel', parent);
    const heading = el('div', 'wb-staged-heading', panel);
    el('h2', '', heading, '任务阶段 · 归档记录');
    el('span', 'wb-staged-version', heading, config.staged_result?.version || '版本未记录');
    const complete = config.staged_result?.complete_success;
    el('p', 'wb-staged-scope', panel, complete === null ? '本任务未定义 staged 完整成功；白板拾取阶段不能代替擦拭完成。' : `staged 全程结果：${complete === true ? '达到末阶段' : complete === false ? '未达到末阶段' : '未记录'}。这是诊断记录，不覆盖页首的原任务判定。`);
    el('p', 'wb-staged-note', panel, '阶段为全程累计“曾经达到”，不是当前接触状态。首次达到时间来自物理观测记录；点击只定位本条模型轨迹，不联动专家。');
    const cards = el('div', 'wb-staged-cards', panel);
    (config.stages || []).forEach(stage => {
      const card = el('article', 'wb-stage-card', cards); card.dataset.stage = stage.key;
      card.dataset.reached = stage.reached === true ? 'true' : stage.reached === false ? 'false' : 'unknown';
      el('span', 'wb-stage-name', card, stage.label || stage.name);
      el('strong', '', card, stage.reached === true ? '已达到' : stage.reached === false ? '未达到' : '未记录');
      const button = el('button', 'wb-stage-first-time', card, finite(stage.first_time_s) ? `首次 ${fmt(stage.first_time_s)} s · 定位` : '没有首次达到时间');
      button.type = 'button';
      button.disabled = !mapped || !seek || !finite(stage.first_time_s) || stage.first_time_s < 0 || stage.first_time_s > maximumTime;
      if (!mapped) button.title = '阶段时钟未与模型游标核验同源，不进行隐式对齐。';
      const handler = () => seek(stage.first_time_s); button.addEventListener('click', handler); removers.push(() => button.removeEventListener('click', handler));
    });
    if (config.source_selection?.group_status) el('p', 'wb-staged-source-warning', panel, `来源批处理状态：${config.source_selection.group_status}；本条轨迹已完成，保留独立运行警告，不据批处理状态覆盖其记录。`);
    const samples = el('details', 'wb-staged-samples', panel); el('summary', '', samples, '游标处的阶段观测 · 原生控制端点');
    const clock = el('p', 'wb-staged-clock', samples);
    const observations = el('div', 'wb-staged-observations', samples);
    const readings = (timeline.signals || []).map(signal => { const n = el('div', '', observations); el('span', '', n, signal.label); return {value: el('strong', '', n, '未记录'), unit: signal.unit}; });
    const conditionReadings = (timeline.conditions || []).map(condition => { const n = el('div', '', observations); el('span', '', n, condition.label); return el('strong', '', n, '未记录'); });
    el('p', 'wb-staged-note', samples, '手—物体、物体—台面和插口载荷来自接触求解器，不等同于五指触觉标量总量；缺测不补零，也不据此自动作因果判断。');
    const audit = el('details', 'wb-staged-rules', panel); el('summary', '', audit, '归档评估规则与数据来源');
    el('p', 'wb-staged-note', audit, '使用本次归档定义、阈值和源码身份；当前整合工程的规则不重算旧结果。不同规则身份不合并统计。');
    const download = el('a', 'wb-link', audit, published ? '下载本页配置（含阶段记录与来源）' : '下载阶段记录与来源');
    download.href = published ? 'workbench.json' : 'staged_evaluation.json';
    el('pre', 'ke-evidence-json', audit, JSON.stringify({rule_identity: config.rule_identity, thresholds: config.staged_result?.thresholds, stage_definitions: config.staged_result?.stage_definitions, data_evidence: config.data_evidence, warnings: config.warnings}, null, 2));
    function update(time) {
      const i = mapped ? observedIndex(timeline, time) : -1;
      clock.textContent = !mapped ? '独立阶段时钟未与模型游标对齐；仅展示全程阶段结果。' : i < 0 ? '当前游标无有效阶段采样；不外推、不补零。' : `阶段原始采样 ${fmt(timeline.time_s[i])} s · 未重采样`;
      readings.forEach((reading, j) => { const v = timeline.values?.[i]?.[j]; reading.value.textContent = finite(v) ? `${fmt(v)} ${reading.unit}` : '未记录'; });
      conditionReadings.forEach((reading, j) => { const v = timeline.condition_values?.[i]?.[j]; reading.textContent = v === true ? '是' : v === false ? '否' : '未记录'; });
    }
    update(0);
    return {update, destroy() { removers.forEach(remove => remove()); panel.remove(); }};
  }
  function mount(host, payload, data, config) {
    ({payload, config} = normalizeBulbAngleDisplay(payload, config));
    const root = host.querySelector('.kai-eval'), doc = host.ownerDocument;
    if (!root || root.classList.contains('wb-root')) throw new Error('Workbench requires one unmodified base dashboard.');
    validateWorkbenchIdentity(payload, data, config);
    const viz = payload.visualization || {}, channels = data.channels || [];
    if (config.generic && (!data.time_s?.length || channels.length !== 5)) {
      root.classList.add('wb-root', 'wb-studio');
      const notice = doc.createElement('p'); notice.className = 'wb-section-note'; notice.textContent = '右手五指原生载荷不可用；保留原任务记录、已有通道、回放与原图。'; root.prepend(notice);
      const detailHeader = mountDetailHeader(root, payload, data, config);
      const fallbackAudit = mountFallbackAudit(root, config.unified_identity);
      const goalStatus = mountGoalReferenceStatus(root, viz, config);
      const stages = config.staged_evaluation || config.unified_identity ? mountStagedPanel(root, config.staged_evaluation, null, 0, config.unified_identity) : null;
      const expert = config.expert_video ? mountExpertPlayer(root, expertDisplayConfig(config.expert_video, config.diagnostic_plots)) : null;
      const plots = config.diagnostic_plots ? mountDiagnosticPlots(root, config.diagnostic_plots) : null;
      const footer = doc.createElement('p'); footer.className = 'wb-footer'; footer.textContent = PLATFORM_FOOTER; root.appendChild(footer);
      let disposed = false;
      return {destroy() { if (disposed) return; disposed = true; detailHeader.destroy(); expert?.destroy(); plots?.destroy(); stages?.destroy(); fallbackAudit?.destroy(); goalStatus.destroy(); notice.remove(); footer.remove(); root.classList.remove('wb-root', 'wb-studio'); }};
    }
    if (channels.length !== 5 || channels.some((ch, i) => ch.name !== viz.channels?.[i]?.name)) throw new Error('Workbench channel mapping mismatch.');
    if (!Array.isArray(data.time_s) || !data.time_s.length || data.time_s.some((t, i) => !finite(t) || (i > 0 && t <= data.time_s[i - 1]))) throw new Error('Invalid native time axis.');
    for (const key of ['normal_force_n', 'tangent_force_n']) if (!Array.isArray(data[key]) || data[key].length !== data.time_s.length || data[key].some(row => !Array.isArray(row) || row.length !== 5)) throw new Error(`Invalid ${key}`);
    for (const key of ['normal_total_n', 'tangent_total_n']) if (!Array.isArray(data[key]) || data[key].length !== data.time_s.length) throw new Error(`Invalid ${key}`);
    const slider = root.querySelector('.ke-slider'), clock = root.querySelector('.ke-time-value'), transport = root.querySelector('.ke-transport');
    const taskCards = [...root.querySelectorAll('.ke-task-chart-card')];
    if (taskCards.length !== (viz.task_series || []).length) throw new Error('Workbench task card mapping mismatch.');
    const cleanups = [], charts = []; let selected = 0, destroyed = false, raf = null;
    const original = [...root.childNodes];
    const savedTexts = [];
    function replaceText(node, text) { savedTexts.push([node, node.textContent]); node.textContent = text; }
    // Preserve original node locations so destroy() also restores the original layout.
    const moved = [];
    function move(node, parent) { if (node) { moved.push([node, node.parentNode, node.nextSibling]); parent.appendChild(node); } return node; }
    function el(tag, cls, parent, text) { const n = doc.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; if (parent) parent.appendChild(n); return n; }
    function on(node, type, fn) { node.addEventListener(type, fn); cleanups.push(() => node.removeEventListener(type, fn)); }
    function link(parent, text, href) { if (!href || /^(?:[a-z]+:|\/\/)/i.test(href)) return; const a = el('a', 'wb-link', parent, text); a.href = href; }
    function detail(parent, title, subtitle) { const d = el('details', 'wb-detail', parent); const s = el('summary', '', d); el('strong', '', s, title); el('span', '', s, subtitle); const content = el('div', 'wb-detail-body', d); on(d, 'toggle', () => { if (d.open) refreshBase(); }); return content; }
    function section(parent, number, title, note) { const p = el('section', 'wb-section', parent); const h = el('div', 'wb-section-heading', p); el('span', 'wb-step', h, number); el('h2', '', h, title); if (note) el('p', 'wb-section-note', p, note); return p; }
    const video = root.querySelector('.ke-media-panel .ke-video'), originalVideoStyle = video?.getAttribute('style'), originalControls = video?.controls, originalPlaybackRate = video?.playbackRate;
    const oldFrame = root.querySelector('.ke-video-frame'), originalFrameStyle = oldFrame?.getAttribute('style');
    function refreshSpatial() { root.querySelector('.ke-mode-button[aria-pressed="true"]')?.click(); }
    function refreshBase() {
      if (destroyed) return;
      const playing = root.querySelector('.ke-controls .ke-primary')?.getAttribute('aria-pressed') === 'true';
      if (!playing) slider.dispatchEvent(new global.Event('input', {bubbles: true}));
      else refreshSpatial();
      schedule();
    }
    function seek(time) { slider.value = String(clamp(time, 0, Number(slider.max))); slider.dispatchEvent(new global.Event('input', {bubbles: true})); }
    root.classList.add('wb-root', 'wb-studio');
    const detailHeader = mountDetailHeader(root, payload, data, config), header = detailHeader.node;
    const workspace = el('div', 'wb-workspace', null), left = el('aside', 'wb-inspector', workspace), right = el('main', 'wb-analysis', workspace);
    workspace.setAttribute('aria-label', '模型回放与任务进度');
    const tactile = el('section', 'wb-tactile-workspace', null); tactile.setAttribute('aria-label', '模型触觉分析');
    const bottom = el('section', 'wb-secondary', null);
    move(transport, root); root.append(workspace, tactile, bottom);
    // Reuse the model transport's sticky position so a long metric page can
    // switch trials without returning to the top or the task list.
    transport.prepend(detailHeader.navigation.node);
    const speedLabel = el('label', 'wb-speed', transport.querySelector('.ke-controls'), '速度 '), speed = el('select', '', speedLabel);
    speed.setAttribute('aria-label', '视频播放速度'); [.25, .5, 1, 2].forEach(rate => { const option = el('option', '', speed, `${rate}×`); option.value = rate; if (rate === 1) option.selected = true; });
    speed.disabled = !video || viz.media?.sync_status !== 'mapped'; on(speed, 'change', () => { if (video) video.playbackRate = Number(speed.value); });
    const media = move(root.querySelector('.ke-media-panel'), left);
    replaceText(media.querySelector('h2'), config.expert_video ? '模型动作回放' : '动作回放');
    if (config.expert_video) replaceText(transport.querySelector('.ke-kicker'), '模型时间游标');
    const cameras = el('div', 'wb-cameras', media); cameras.setAttribute('role', 'group'); cameras.setAttribute('aria-label', '模型视频视角');
    const cameraButtons = [];
    const cameraConfig = modelCropConfig(config.media_views, viz.media);
    const views = (Array.isArray(cameraConfig?.views) ? cameraConfig.views : []).filter(v => cropStyle(cameraConfig, v.id));
    if (oldFrame) media.insertBefore(cameras, oldFrame);
    function setCamera(id) {
      if (!video || !oldFrame) return;
      const crop = cropStyle(cameraConfig, id);
      oldFrame.classList.toggle('wb-cropped', !!crop); oldFrame.style.aspectRatio = crop ? crop.aspect : '16 / 9';
      ['width', 'height', 'left', 'top'].forEach(key => { video.style[key] = crop ? crop[key] : ''; });
      video.controls = !crop; video.dataset.camera = id;
      cameraButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.camera === id)));
      schedule();
    }
    [...views, {id: 'mosaic', label: '完整画面'}].forEach(view => { const button = el('button', 'wb-camera', cameras, view.label); button.type = 'button'; button.dataset.camera = view.id; button.title = view.note || (view.id === 'wrist' ? '同一模型视频的右腕画面，裁去拼图文字覆盖带；切换不改变播放时间，完整画面保留原拼图。' : '切换同一段模型视频的显示区域，不改变播放时间。'); on(button, 'click', () => setCamera(view.id)); cameraButtons.push(button); });
    if (!views.some(view => view.id === 'wrist')) el('p', 'wb-model-view-note', media, config.media_views?.reason || '本条视频的腕部画面位置尚未核验，保留完整画面，不借用其他轨迹的裁剪框。');
    if (config.generic && video) {
      const full = el('button', 'wb-camera wb-model-fullscreen', cameras, '全屏'); full.type = 'button'; full.disabled = !video.requestFullscreen;
      on(full, 'click', () => { const promise = video.requestFullscreen?.(); if (promise?.catch) promise.catch(() => {}); });
      on(doc, 'fullscreenchange', () => { video.controls = doc.fullscreenElement === video || video.dataset.camera === 'mosaic'; });
      if (viz.media?.mapping_method === 'recorded_camera_pose_time_s') el('p', 'wb-model-clock-note', media, '视频按记录的相机位姿时钟定位；相机与帧仿真时间最多相差 2 ms，Fn 保留原记录时钟。');
    }
    const spatial = move(root.querySelector('.ke-spatial-panel'), tactile);
    replaceText(spatial.querySelector('h2'), '当前触觉 · 单指检查');
    const fingerStrip = el('div', 'wb-fingers', null); fingerStrip.setAttribute('role', 'group'); fingerStrip.setAttribute('aria-label', '选择检查手指，同时高亮曲线');
    spatial.querySelector('.ke-section-title').after(fingerStrip);
    const fingerButtons = channels.map((ch, i) => {
      const button = el('button', 'wb-finger', fingerStrip); button.type = 'button'; button.style.setProperty('--finger-color', COLORS[i]); button.dataset.finger = i;
      el('span', '', button, ch.label || ch.name); const number = el('small', '', button, 'Fn —');
      on(button, 'click', () => selectFinger(i)); return {button, number};
    });
    const current = el('div', 'wb-current', null); fingerStrip.after(current);
    const currentNormal = el('strong', '', current), currentTangent = el('strong', '', current), currentTime = el('span', '', current);
    const heatCards = [...spatial.querySelectorAll('.ke-heat-card')], glyphCards = [...spatial.querySelectorAll('.ke-glyph-card')];
    [heatCards, glyphCards].forEach(cards => cards.forEach((card, i) => replaceText(card.querySelector('h3'), channels[i].label || channels[i].name)));
    function selectFinger(index) {
      selected = index; root.dataset.selectedFinger = String(index);
      fingerButtons.forEach(({button}, i) => button.setAttribute('aria-pressed', String(i === index)));
      [heatCards, glyphCards].forEach(cards => cards.forEach((card, i) => { card.classList.toggle('wb-inactive', i !== index); card.setAttribute('aria-hidden', String(i !== index)); card.inert = i !== index; }));
      refreshSpatial(); schedule();
    }
    // Keep explanations nearby but avoid a tall repeated block above the inspector.
    const spatialHelp = detail(spatial, '读图说明', '快照、力圆和 ΔFt 的含义');
    move(spatial.querySelector(':scope > .ke-note:not(.ke-spatial-selection):not(.ke-glyph-footnote)'), spatialHelp);
    ['.ke-glyph-legend', '.ke-glyph-footnote', '.ke-spatial-selection'].forEach(selector => move(spatial.querySelector(selector), spatialHelp));
    const expertPlayer = config.expert_video ? mountExpertPlayer(root, expertDisplayConfig(config.expert_video, config.diagnostic_plots)) : null;
    if (expertPlayer) root.insertBefore(expertPlayer.node, bottom);
    const stageBody = config.staged_evaluation || config.unified_identity ? detail(bottom, '阶段记录与评估规则', '全程累计 · 首次达到时间 · 当前观测') : null;
    const stagedPanel = stageBody ? mountStagedPanel(stageBody, config.staged_evaluation, seek, Number(slider.max), config.unified_identity) : null;
    const presentation = config.task_presentation || {}, whiteboard = data.task.startsWith('whiteboard-wipe');
    const taskSection = section(right, '01', '任务关键指标', '曲线与读数随模型游标变化；虚线 / 色带标出已核验的参考条件，不据此改写原任务结果。');
    const goalStatus = mountGoalReferenceStatus(taskSection, viz, config);
    const bulbAngles = mountBulbAngleSummary(taskSection, viz, {...config, task: data.task});
    const primaryTask = el('div', 'wb-task-grid', taskSection);
    const taskNotes = detail(taskSection, '终态记录与指标说明', '整条轨迹摘要 · 不随当前游标变化');
    el('p', 'wb-section-note', taskNotes, presentation.task_note || '任务交互载荷与手指载荷来自不同接触位置，应分别读图。');
    if (whiteboard && config.whiteboard_contact_area) {
      const area = config.whiteboard_contact_area;
      const areaNote = el('p', 'wb-section-note wb-contact-area-status', taskNotes, whiteboardAreaMessage(area));
      areaNote.dataset.whiteboardContactAreaStatus = area.status || 'unavailable';
      if (area.status !== 'verified') (area.warnings || []).slice(0, 3).forEach(message => el('p', 'wb-section-note', taskNotes, String(message)));
    }
    if (config.task_summary?.length) {
      el('p', 'wb-section-note', taskNotes, '终态记录 · 以下字段不随当前游标变化');
      const finals = el('div', 'wb-task-finals', taskNotes);
      config.task_summary.forEach(item => {
        const card = el('div', 'wb-task-final', finals); card.dataset.metricKey = item.key; el('span', '', card, item.label);
        el('strong', '', card, item.value === null || item.value === undefined ? item.status === 'no_contact' ? '无接触，均值未定义' : '未记录' : typeof item.value === 'boolean' ? item.value ? '是' : '否' : item.unit === 'fraction' ? item.value !== 0 && Math.abs(item.value) < 1e-6 ? item.value.toExponential(4) + ' fraction' : (item.value * 100).toFixed(1) + '%' : finite(item.value) ? fmt(item.value) + (item.unit ? ' ' + displayUnit(item.unit) : '') : String(item.value));
        card.title = [item.source || '没有终态字段记录', item.semantics].filter(Boolean).join('；');
      });
    }
    const diagnosticGallery = config.diagnostic_plots ? mountDiagnosticPlots(bottom, {...config.diagnostic_plots, diagnostic_example_url: config.diagnostic_example_url}) : null;
    const pickupBody = detail(bottom, whiteboard ? '取放与支撑' : '补充任务响应', whiteboard ? '板擦—桌面载荷 / 累计最大抬升高度' : '本任务的其他原始记录字段');
    const secondaryTask = el('div', 'wb-task-grid', pickupBody);
    taskCards.forEach((card, i) => {
      const key = viz.task_series[i].key; card.dataset.taskKey = key;
      move(card, (presentation.primary_task_keys || ['cleaned_fraction', 'board_normal_force_n']).includes(key) ? primaryTask : secondaryTask);
      if (viz.task_series[i].note) el('p', 'wb-task-field-note', card, viz.task_series[i].note);
      if (key === 'table_support_force_n') replaceText(card.querySelector('h3'), '板擦—桌面法向接触载荷');
      if (key === 'maximum_lift_m') replaceText(card.querySelector('h3'), '累计最大抬升高度');
    });
    if (!primaryTask.childElementCount) el('p', 'wb-section-note', primaryTask, '此轨迹没有可同步的任务响应字段；终态保留在下方完整记录。');
    if (whiteboard) el('p', 'wb-section-note', pickupBody, '桌面载荷不是白板载荷；累计最大抬升高度不是当前高度。');
    if (config.generic && !secondaryTask.childElementCount) pickupBody.parentElement.hidden = true;
    const loadSection = section(tactile, '02', '五指接触载荷', '上排看总量，下排看各指贡献；与左侧单指检查共享模型游标。Fn 与 |Ft| 是载荷，不是载荷变化量。');
    const legends = el('div', 'wb-chart-legend', loadSection);
    const legendButtons = channels.map((ch, i) => { const button = el('button', '', legends, ch.label || ch.name); button.type = 'button'; button.style.setProperty('--finger-color', COLORS[i]); button.dataset.finger = i; on(button, 'click', () => selectFinger(i)); return button; });
    el('p', 'wb-semantics', loadSection, 'ΣFn 与 Σ|Ft| 是五指标量加总，不是空间矢量合力。各图纵轴独立、范围固定；缺测不补零。');
    const loadGrid = el('div', 'wb-load-grid', loadSection), span = Number(slider.max) || 1;
    const times = data.time_s || [];
    const definitions = [
      ['normal-total', '五指法向载荷 · ΣFn', [data.normal_total_n || []], '#087e94', false],
      ['tangent-total', '五指切向载荷 · Σ|Ft|', [data.tangent_total_n || []], '#b76e39', false],
      ['normal-fingers', '各指法向载荷 · Fn', channels.map((_, i) => (data.normal_force_n || []).map(row => row?.[i])), null, true],
      ['tangent-fingers', '各指切向载荷 · |Ft|', channels.map((_, i) => (data.tangent_force_n || []).map(row => row?.[i])), null, true],
    ];
    definitions.forEach(([key, title, series, color, fingers]) => {
      const panel = el('article', 'wb-load-card', loadGrid); panel.dataset.loadPanel = key;
      el('h3', '', panel, title); const reading = el('p', 'wb-reading', panel);
      const canvas = el('canvas', 'wb-chart', panel); canvas.dataset.loadChart = key; canvas.tabIndex = 0; canvas.setAttribute('role', 'slider'); canvas.setAttribute('aria-label', `${title}，点击或按方向键选择时刻`); canvas.setAttribute('aria-valuemin', '0'); canvas.setAttribute('aria-valuemax', String(span));
      const hover = el('p', 'wb-hover', panel, '悬停读数 · 点击定位 · ← → 逐样本');
      const chart = {panel, canvas, reading, hover, series, runs: series.map(values => knownRuns(data, values)), color, fingers, title, maximum: yMaximum(series), hoverTime: null}; charts.push(chart);
      function eventTime(event) { const bounds = canvas.getBoundingClientRect(); return clamp((event.clientX - bounds.left - 44) / Math.max(1, bounds.width - 58) * span, 0, span); }
      on(canvas, 'click', event => seek(eventTime(event)));
      on(canvas, 'pointermove', event => { chart.hoverTime = eventTime(event); renderChart(chart); });
      on(canvas, 'pointerleave', () => { chart.hoverTime = null; renderChart(chart); });
      on(canvas, 'keydown', event => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        seek(event.key === 'Home' ? 0 : event.key === 'End' ? span : stepTime(times, Number(slider.value), event.key === 'ArrowRight' ? 1 : -1));
      });
    });
    const warningText = [...(data.warnings || [])];
    if (!viz.clock_aligned) warningText.push('任务曲线时钟未核实，请勿据此判断先后关系。');
    if (viz.media?.sync_status !== 'mapped') warningText.push('视频与数据未可靠同步，视频仅独立回放。');
    if (data.normal_status !== 'available' || data.tangent_status !== 'available') warningText.push('部分载荷数据不可用；未知不表示零。');
    el('p', 'wb-caution', tactile, warningText.length ? warningText.join(' / ') : '这里展示同一轨迹内的时间关联，不单凭力的变化判定滑移、动作好坏或因果关系。');
    const sectionNavigation = mountSectionNavigation(transport, [
      {id: 'wb-task-progress', label: '任务进度', node: workspace},
      {id: 'wb-tactile-analysis', label: '触觉分析', node: tactile},
      {id: 'wb-expert-reference', label: '专家示范 ↗', node: expertPlayer?.node},
      {id: 'wb-full-diagnostics', label: '全程图', node: diagnosticGallery?.node},
    ]);
    const contactBody = detail(bottom, '接触片段与事件', '检出的接触持续时间与原始事件');
    move(root.querySelector('.ke-contact-bands')?.closest('.ke-panel'), contactBody); move(root.querySelector('.ke-events')?.closest('.ke-panel'), contactBody);
    const statsBody = detail(bottom, '全程统计与原始法向力图', '均值 / P95 / 峰值，以及原版通道筛选');
    move(root.querySelector('.ke-metric-legend')?.closest('.ke-panel'), statsBody); move(root.querySelector('.ke-force-chart')?.closest('.ke-panel'), statsBody);
    const auditBody = detail(bottom, '评估规则与数据审计', '保留原结果、规则复核、数据来源、缺失字段和下载入口');
    ['.ke-header', '.ke-summary', '.ke-meta-line', '.ke-v2-evidence', '.ke-warnings', '.ke-footer'].forEach(selector => move(root.querySelector(selector), auditBody));
    const loadAudit = el('details', 'ke-more', auditBody); el('summary', '', loadAudit, '五指载荷定义与来源');
    link(loadAudit, '下载载荷 JSON', 'tactile_loads.json');
    el('pre', 'ke-evidence-json', loadAudit, JSON.stringify({semantics: data.semantics, warnings: data.warnings, provenance: data.provenance, trial_id: data.trial_id, prediction_horizon: data.prediction_horizon}, null, 2));
    const viewAudit = el('details', 'ke-more', auditBody); el('summary', '', viewAudit, '模型视频视角与裁剪来源');
    link(viewAudit, '下载视角配置', 'workbench.json');
    el('p', 'wb-section-note', viewAudit, '视角切换只裁剪同一段存档视频，不重渲染、拉伸相机画面或改变模型与专家时间。腕部视窗避开拼图文字覆盖带，完整画面仍保留原拼图。');
    el('pre', 'ke-evidence-json', viewAudit, JSON.stringify(config.media_views || {status: 'full_frame_only'}, null, 2));
    // Empty original shell remains available to restore without touching archived code.
    const oldLayout = root.querySelector('.ke-layout'); oldLayout.hidden = true;
    el('p', 'wb-footer', bottom, PLATFORM_FOOTER);
    function readDescription(chart, index) {
      const values = chart.fingers ? `${channels[selected].label || channels[selected].name} ${fmt(chart.series[selected]?.[index])} N` : `${fmt(chart.series[0]?.[index])} N`;
      return `${values}${index < 0 ? ' · 无有效观测' : ''}`;
    }
    function renderChart(chart) {
      if (destroyed) return;
      const {canvas, series, maximum} = chart, box = canvas.getBoundingClientRect(); if (box.width <= 0) return;
      const width = box.width, height = box.height, dpr = global.devicePixelRatio || 1;
      const pixelWidth = Math.round(width * dpr), pixelHeight = Math.round(height * dpr);
      if (canvas.width !== pixelWidth) canvas.width = pixelWidth; if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
      const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
      const left = 44, rightEdge = width - 14, top = 18, bottomEdge = height - 29;
      const x = t => left + t / span * (rightEdge - left), y = v => bottomEdge - v / maximum * (bottomEdge - top);
      ctx.font = '12px system-ui'; ctx.lineWidth = 1;
      for (let tick = 0; tick <= 3; tick++) {
        const value = maximum * tick / 3, yy = y(value); ctx.strokeStyle = '#e5edf1'; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(rightEdge, yy); ctx.stroke();
        ctx.fillStyle = '#748696'; ctx.textAlign = 'right'; ctx.fillText(value.toFixed(1), left - 7, yy + 3);
      }
      ctx.fillStyle = '#748696'; ctx.textAlign = 'left'; ctx.fillText('N', 9, 11);
      for (let tick = 0; tick <= 3; tick++) { const t = tick * span / 3; ctx.textAlign = tick === 0 ? 'left' : tick === 3 ? 'right' : 'center'; ctx.fillText(`${t.toFixed(0)} s`, x(t), height - 7); }
      ctx.save(); ctx.beginPath(); ctx.rect(left, top - 2, rightEdge - left, bottomEdge - top + 4); ctx.clip();
      const order = series.map((_, i) => i).sort((a, b) => Number(a === selected) - Number(b === selected));
      order.forEach(k => {
        ctx.strokeStyle = chart.color || COLORS[k]; ctx.fillStyle = chart.color || COLORS[k]; ctx.globalAlpha = chart.fingers && k !== selected ? .36 : 1; ctx.lineWidth = chart.fingers && k !== selected ? 1.1 : 1.8;
        chart.runs[k].forEach(run => {
          ctx.beginPath(); run.forEach(([t, value], i) => { if (i === 0) ctx.moveTo(x(t), y(value)); else ctx.lineTo(x(t), y(value)); }); ctx.stroke();
          if (run.length === 1) { ctx.beginPath(); ctx.arc(x(run[0][0]), y(run[0][1]), 2, 0, Math.PI * 2); ctx.fill(); }
        });
      });
      ctx.globalAlpha = 1; const time = Number(slider.value), index = observedIndex(data, time);
      ctx.strokeStyle = '#263d50'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x(time), top); ctx.lineTo(x(time), bottomEdge); ctx.stroke();
      series.forEach((values, k) => { const value = values[index]; if (!finite(value)) return; ctx.fillStyle = chart.color || COLORS[k]; ctx.globalAlpha = chart.fingers && k !== selected ? .4 : 1; ctx.beginPath(); ctx.arc(x(times[index]), y(value), k === selected || !chart.fingers ? 3 : 2, 0, Math.PI * 2); ctx.fill(); }); ctx.globalAlpha = 1;
      if (chart.hoverTime !== null) { ctx.setLineDash([3, 4]); ctx.strokeStyle = '#8896a1'; ctx.beginPath(); ctx.moveTo(x(chart.hoverTime), top); ctx.lineTo(x(chart.hoverTime), bottomEdge); ctx.stroke(); ctx.setLineDash([]); }
      ctx.restore(); chart.reading.textContent = readDescription(chart, index);
      chart.panel.dataset.yMax = String(maximum); chart.panel.dataset.selectedFinger = String(selected);
      canvas.setAttribute('aria-valuenow', String(time)); canvas.setAttribute('aria-valuetext', `${fmt(time)} 秒；${chart.reading.textContent}`);
      if (chart.hoverTime === null) chart.hover.textContent = index < 0 ? '当前时刻无有效采样 · 不补零' : `原生采样 ${fmt(times[index])} s · 点击图中定位`;
      else { const i = observedIndex(data, chart.hoverTime); chart.hover.textContent = `悬停 ${fmt(chart.hoverTime)} s · ${i < 0 ? '无有效观测' : '原生采样 ' + fmt(times[i]) + ' s'} · ${readDescription(chart, i)}`; }
    }
    function render() {
      if (destroyed) return; const time = Number(slider.value), index = observedIndex(data, time);
      stagedPanel?.update(time);
      bulbAngles?.update(time);
      fingerButtons.forEach(({number}, i) => { number.textContent = `Fn ${fmt(data.normal_force_n?.[index]?.[i])}`; });
      currentNormal.textContent = `Fn ${fmt(data.normal_force_n?.[index]?.[selected])} N`;
      currentTangent.textContent = `|Ft| ${fmt(data.tangent_force_n?.[index]?.[selected])} N`;
      currentTime.textContent = index < 0 ? '当前无有效观测' : `原生采样 ${fmt(times[index])} s`;
      legendButtons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === selected)));
      charts.forEach(renderChart);
      const barHeight = transport.getBoundingClientRect().height; root.style.setProperty('--wb-sticky-top', `${barHeight + 10}px`);
      left.classList.toggle('wb-pinned', global.innerWidth >= 1080 && left.getBoundingClientRect().height + barHeight + 22 <= global.innerHeight);
    }
    function schedule() { if (destroyed || raf !== null) return; raf = global.requestAnimationFrame(() => { raf = null; render(); }); }
    const clockObserver = new global.MutationObserver(schedule); clockObserver.observe(clock, {childList: true, characterData: true, subtree: true});
    const resizeObserver = new global.ResizeObserver(schedule); [transport, left, ...charts.map(c => c.canvas)].forEach(n => resizeObserver.observe(n));
    on(global, 'resize', () => { refreshBase(); });
    on(spatial.querySelector('.ke-spatial-mode'), 'click', schedule);
    setCamera(views[0]?.id || 'mosaic'); selectFinger(0); refreshBase(); schedule();
    return {destroy() {
      if (destroyed) return;
      detailHeader.destroy();
      sectionNavigation.destroy();
      expertPlayer?.destroy();
      stagedPanel?.destroy();
      goalStatus.destroy();
      bulbAngles?.destroy();
      diagnosticGallery?.destroy();
      destroyed = true; clockObserver.disconnect(); resizeObserver.disconnect(); cleanups.forEach(fn => fn()); if (raf !== null) global.cancelAnimationFrame(raf);
      moved.reverse().forEach(([node, parent, next]) => parent.insertBefore(node, next?.parentNode === parent ? next : null));
      savedTexts.forEach(([node, text]) => { node.textContent = text; }); taskCards.forEach(card => { delete card.dataset.taskKey; });
      original.forEach(node => root.appendChild(node)); [header, workspace, tactile, bottom, speedLabel, fingerStrip, current, cameras].forEach(node => node.remove());
      root.classList.remove('wb-root', 'wb-studio'); oldLayout.hidden = false; root.style.removeProperty('--wb-sticky-top'); delete root.dataset.selectedFinger;
      [heatCards, glyphCards].forEach(cards => cards.forEach(card => { card.classList.remove('wb-inactive'); card.removeAttribute('aria-hidden'); card.inert = false; }));
      spatialHelp.parentNode.remove();
      if (video) { video.controls = originalControls; video.playbackRate = originalPlaybackRate; delete video.dataset.camera; if (originalVideoStyle === null) video.removeAttribute('style'); else video.setAttribute('style', originalVideoStyle); }
      if (oldFrame) { oldFrame.classList.remove('wb-cropped'); if (originalFrameStyle === null) oldFrame.removeAttribute('style'); else oldFrame.setAttribute('style', originalFrameStyle); }
      if (root.querySelector('.ke-controls .ke-primary')?.getAttribute('aria-pressed') !== 'true') slider.dispatchEvent(new global.Event('input', {bubbles: true}));
      else refreshSpatial();
    }};
  }
  global.KaiEvaluationWorkbench = Object.freeze({mount, mountDiagnosticPlots, normalizeBulbAngleDisplay});
  if (typeof module !== 'undefined' && module.exports) module.exports = {leftIndex, observedIndex, stepTime, knownRuns, yMaximum, cropStyle, modelCropConfig, expertVideoURL, diagnosticPlotURL, expertDisplayConfig, taskDisplayTitle, nativeOutcomeStatus, publishedStageStatus, stageEvidenceMessage, stageBadgeDefinitions, modelDisplayName, detailIdentity, validateWorkbenchIdentity, mountFallbackAudit, whiteboardAreaMessage, unifiedTaskNavigationRows, taskNavigationRows, mountTrajectoryNavigation, mountDetailHeader, mountStagedPanel, mountSectionNavigation, normalizeBulbAngleDisplay, degreeSymbolDashboardPayload, remountLegacyAngleDashboard, displayUnit, displayedGoalReferences, goalReferenceStatus, mountGoalReferenceStatus, bulbAngleState, mountBulbAngleSummary};
  if (typeof document !== 'undefined') {
    try {
      const diagnosticHost = document.getElementById('kai-diagnostic-plots');
      if (diagnosticHost) {
        const config = JSON.parse(document.getElementById('kai-diagnostic-config').textContent);
        const originalTitle = document.title, taskTitle = taskDisplayTitle(config, config.task), pageTitle = `${taskTitle} · 全程诊断图 | KaiTactileSim`;
        document.title = pageTitle;
        const brand = document.createElement('p'); brand.className = 'wb-eyebrow'; brand.textContent = PLATFORM_NAME; diagnosticHost.appendChild(brand);
        const h1 = document.createElement('h1'); h1.textContent = `${taskTitle} · 全程诊断图`; diagnosticHost.appendChild(h1);
        let back = null;
        if (typeof config.workbench_url === 'string' && /^\.\.\/[A-Za-z0-9_-]+\/report\.html$/.test(config.workbench_url)) {
          back = document.createElement('a'); back.className = 'wb-link'; back.href = config.workbench_url; back.textContent = '返回轨迹分析'; diagnosticHost.appendChild(back);
        }
        const gallery = mountDiagnosticPlots(diagnosticHost, config);
        const footer = document.createElement('p'); footer.className = 'wb-footer'; footer.textContent = PLATFORM_FOOTER; diagnosticHost.appendChild(footer);
        let disposed = false;
        global.kaiDiagnosticPlots = {node: gallery.node, destroy() {
          if (disposed) return; disposed = true;
          gallery.destroy(); brand.remove(); h1.remove(); back?.remove(); footer.remove();
          if (document.title === pageTitle) document.title = originalTitle;
        }}; return;
      }
      const archivedPayload = JSON.parse(document.getElementById('kai-evaluation-data').textContent), data = JSON.parse(document.getElementById('kai-tactile-load-data').textContent), archivedConfig = JSON.parse(document.getElementById('kai-workbench-config').textContent);
      const normalized = normalizeBulbAngleDisplay(archivedPayload, archivedConfig);
      const host = document.getElementById('kai-evaluation');
      remountLegacyAngleDashboard(host, normalized.payload);
      global.kaiWorkbench = mount(host, normalized.payload, data, normalized.config);
    } catch (error) {
      const warning = document.createElement('p'); warning.className = 'wb-start-error'; warning.textContent = `页面未能加载：${error.message}`;
      (document.getElementById('kai-evaluation') || document.getElementById('kai-diagnostic-plots'))?.prepend(warning); console.error(error);
    }
  }
})(typeof window !== 'undefined' ? window : globalThis);
