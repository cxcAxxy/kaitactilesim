/* Offline evaluation dashboard. No dependencies, no executable report content. */
(function (global) {
  'use strict';
  const COLORS = ['#087e94', '#df982a', '#6473b8', '#b76186', '#488763', '#c8714c', '#668994', '#927645', '#66598c', '#528fa5'];
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const array = value => Array.isArray(value) ? value : [];
  const object = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  const fmt = (value, digits = 2) => finite(value) ? value.toLocaleString('zh-CN', {maximumFractionDigits: digits, minimumFractionDigits: digits}) : '—';
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const LEGACY_BULB_ANGLE_SERIES = Object.freeze({
    clockwise_turns: Object.freeze({key: 'clockwise_angle_deg', label: '灯泡净旋入角度（顺时针）'}),
    turns_removed: Object.freeze({key: 'angle_removed_deg', label: '灯泡净旋出角度（逆时针）'}),
    rotation_from_initial_turns: Object.freeze({key: 'rotation_from_initial_deg', label: '相对实际初态的净旋入角度'}),
  });
  const scaledDegrees = value => finite(value) ? value * 360 : value;
  const displayUnit = unit => unit === 'deg' ? '°' : unit;
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
  function normalizeBulbAngleSeries(series) {
    const specification = object(series) && LEGACY_BULB_ANGLE_SERIES[series.key];
    if (!specification || series.unit !== 'turn') return series;
    const converted = {...series, key: specification.key, unit: 'deg', label: specification.label};
    if (Array.isArray(series.values)) converted.values = series.values.map(scaledDegrees);
    if (Array.isArray(series.goal_references)) converted.goal_references = series.goal_references.map(reference => {
      if (!reference || typeof reference !== 'object' || Array.isArray(reference) ||
          reference.key !== series.key || reference.unit !== 'turn') return reference;
      const result = {...reference, key: specification.key, unit: 'deg'};
      for (const field of ['target', 'lower', 'upper']) {
        if (Object.hasOwn(result, field)) result[field] = scaledDegrees(result[field]);
      }
      return canonicalLegacyAngleReferenceText(result);
    });
    for (const field of ['target', 'lower', 'upper']) {
      if (Object.hasOwn(converted, field)) converted[field] = scaledDegrees(converted[field]);
    }
    return converted;
  }
  const median = values => { const a = values.filter(finite).sort((x, y) => x - y); return a.length ? a[Math.floor(a.length / 2)] : null; };
  function cadence(times) { return median(times.slice(1).map((t, i) => t - times[i]).filter(v => v > 0)); }
  function leftIndex(times, time) {
    if (!times.length || !finite(time) || time < times[0] - 1e-8) return -1;
    let lo = 0, hi = times.length;
    while (lo < hi) { const middle = (lo + hi) >>> 1; if (times[middle] <= time + 1e-8) lo = middle + 1; else hi = middle; }
    return lo - 1;
  }
  function observedIndex(times, time, nominal, gaps) {
    if (!times.length || time > times[times.length - 1] + 1e-8) return -1;
    const i = leftIndex(times, time);
    if (i < 0) return -1;
    const age = time - times[i];
    const authoritativeGaps = Array.isArray(gaps) && gaps.length === times.length;
    if (age > 1e-8 && (gaps?.[i] === true || (!authoritativeGaps && finite(nominal) && age > 1.5 * nominal + 1e-8))) return -1;
    return i;
  }
  function relativeURL(value) {
    if (typeof value !== 'string' || !value || /[\u0000-\u0020\\]/.test(value)) return null;
    let decoded;
    try { decoded = decodeURIComponent(value); } catch (_) { return null; }
    if (/^[a-z][a-z\d+.-]*:/i.test(decoded) || decoded.startsWith('//') || /[\u0000-\u0020\\]/.test(decoded) || decoded.includes('%') || decoded.split('/').includes('..')) return null;
    return value;
  }
  function knownSegments(times, values, gaps, nominal) {
    const runs = []; let run = [];
    const authoritativeGaps = Array.isArray(gaps) && gaps.length === times.length;
    for (let i = 0; i < Math.min(times.length, values.length); i++) {
      const connected = i > 0 && gaps?.[i - 1] !== true && (authoritativeGaps || !finite(nominal) || times[i] - times[i - 1] <= 1.5 * nominal + 1e-8);
      if (!finite(times[i]) || !finite(values[i])) { if (run.length) runs.push(run); run = []; continue; }
      if (run.length && !connected) { runs.push(run); run = []; }
      run.push([times[i], values[i]]);
    }
    if (run.length) runs.push(run);
    return runs;
  }
  // Reference metadata is additive presentation evidence, never a success test.
  // Reject mismatched units/keys and unplotable extremes instead of coercing them.
  const plottable = value => finite(value) && Math.abs(value) <= Number.MAX_VALUE / 8;
  function goalReferences(series) {
    if (typeof series?.key !== 'string' || typeof series?.unit !== 'string') return [];
    return array(series.goal_references).slice(0, 8).flatMap(raw => {
      const ref = object(raw);
      if (ref.key !== series.key || ref.unit !== series.unit || !plottable(ref.target) || !['target', 'at_most', 'at_least'].includes(ref.relation)) return [];
      const result = {key: ref.key, unit: ref.unit, target: ref.target, relation: ref.relation,
        label: typeof ref.label === 'string' ? ref.label.trim() : '', note: typeof ref.note === 'string' ? ref.note.trim() : ''};
      if (ref.reference_kind === 'ideal') result.reference_kind = 'ideal';
      // A one-sided finite allowed range must end at its declared threshold.
      // Never extend it past the limit or invent an unbounded success region.
      const directedBand = ref.relation === 'target' || (ref.relation === 'at_most' && ref.upper === ref.target) || (ref.relation === 'at_least' && ref.lower === ref.target);
      if (directedBand && plottable(ref.lower) && plottable(ref.upper) && ref.lower <= ref.target && ref.target <= ref.upper && ref.lower < ref.upper) {
        result.lower = ref.lower; result.upper = ref.upper;
      }
      return [result];
    });
  }
  function chartRange(series, references = []) {
    let low = 0, high = 0;
    const include = value => { if (finite(value)) { low = Math.min(low, value); high = Math.max(high, value); } };
    series.forEach(s => s.values.forEach(include));
    references.forEach(ref => { include(ref.target); include(ref.lower); include(ref.upper); });
    const extent = high - low;
    if (high === low) high = low + 1; else high += extent * .08;
    if (references.length && extent > 0) low -= extent * .04;
    return {low, high};
  }
  function referenceNumber(value) {
    if (!finite(value)) return '—';
    return value !== 0 && (Math.abs(value) < 1e-6 || Math.abs(value) >= 1e9)
      ? value.toExponential(4) : value.toLocaleString('zh-CN', {maximumFractionDigits: 6});
  }
  function referenceText(ref) {
    const kind = ref.relation === 'at_most' ? '参考阈值 ≤' : ref.relation === 'at_least' ? '参考阈值 ≥' : ref.reference_kind === 'ideal' ? '理想参考' : '机械目标';
    return `${kind} ${referenceNumber(ref.target)}${ref.unit ? ' ' + displayUnit(ref.unit) : ''}`;
  }
  const chartPrecision = extent => clamp(Math.ceil(-Math.log10(extent / 4)) + 1, 0, 6);
  const vector2 = value => Array.isArray(value) && value.length === 2 && value.every(finite);
  function glyphSample(glyph, time, channel) {
    glyph = object(glyph);
    const clock = array(glyph.time_s), step = finite(glyph.nominal_interval_s) ? glyph.nominal_interval_s : cadence(clock);
    const index = observedIndex(clock, time, step, glyph.gap_after);
    const normal = index >= 0 ? glyph.normal_force_n?.[index]?.[channel] : null;
    const center = index >= 0 ? glyph.center_xy?.[index]?.[channel] : null;
    const normalChange = index >= 0 ? glyph.normal_change_n?.[index]?.[channel] : null;
    const tangentChange = index >= 0 ? glyph.tangent_change_n?.[index]?.[channel] : null;
    return {index, sampleTime: index >= 0 ? clock[index] : null, lag: index >= 0 ? Math.max(0, time - clock[index]) : null,
      normal: finite(normal) && normal >= 0 ? normal : null,
      center: vector2(center) ? center : null, normalChange: finite(normalChange) ? normalChange : null,
      tangentChange: vector2(tangentChange) ? tangentChange : null};
  }
  function glyphRadius(force, maximum, radius) {
    if (!finite(force) || force < 0 || !finite(maximum) || maximum < 0 || !finite(radius) || radius <= 0) return null;
    if (force === 0) return 0;
    return maximum > 0 ? radius * Math.sqrt(force / maximum) : null;
  }
  function glyphBounds(channels) {
    const result = {};
    array(channels).forEach(channel => {
      const unit = channel.coordinate_status === 'physical' && channel.unit === 'mm' ? 'mm' : 'index';
      const points = array(channel.taxel_xy).filter(vector2);
      if (!points.length) return;
      const bounds = result[unit] || {minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity};
      points.forEach(point => { bounds.minX = Math.min(bounds.minX, point[0]); bounds.maxX = Math.max(bounds.maxX, point[0]); bounds.minY = Math.min(bounds.minY, point[1]); bounds.maxY = Math.max(bounds.maxY, point[1]); });
      result[unit] = bounds;
    });
    Object.values(result).forEach(bounds => { bounds.centerX = (bounds.minX + bounds.maxX) / 2; bounds.centerY = (bounds.minY + bounds.maxY) / 2; bounds.range = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) || 1; });
    return result;
  }
  function glyphPoint(point, bounds, side, centerX, centerY, yDown = false) {
    if (!vector2(point) || !bounds || !(bounds.range > 0) || !finite(side)) return null;
    const scale = side / bounds.range;
    return [centerX + (point[0] - bounds.centerX) * scale, centerY + (yDown ? 1 : -1) * (point[1] - bounds.centerY) * scale];
  }
  function glyphVector(vector, maximum, radius) {
    if (!vector2(vector) || !finite(maximum) || maximum < 0 || !finite(radius) || radius <= 0) return null;
    const magnitude = Math.hypot(vector[0], vector[1]);
    if (magnitude === 0) return {magnitude: 0, dx: 0, dy: 0};
    if (maximum === 0) return null;
    return {magnitude, dx: vector[0] / maximum * radius, dy: -vector[1] / maximum * radius};
  }
  function mount(element, options) {
    if (!element || typeof element.appendChild !== 'function') throw new TypeError('mount needs a DOM element');
    const report = object(options?.report), viz = object(options?.visualization);
    const doc = element.ownerDocument || global.document;
    const trial = object(report.trial), source = object(viz.tactile_series), spatial = object(viz.spatial_frames), glyph = object(viz.contact_glyph);
    // Presentation-only scales: never shrink native ranges or mutate evidence.
    const displayScale = (native, requested) => finite(requested) && requested >= 0 && (!finite(native) || requested >= native) ? requested : native;
    const normalScale = displayScale(glyph.force_scale_max_n, options?.scales?.normal_force_max_n);
    const taxelScale = displayScale(spatial.scale_max_n, options?.scales?.taxel_force_max_n);
    const tangentScale = displayScale(glyph.tangent_change_scale_max_n, options?.scales?.tangent_change_max_n);
    const channels = array(viz.channels), times = array(source.time_s), forces = array(source.normal_force_n);
    const nominal = finite(source.nominal_interval_s) ? source.nominal_interval_s : cadence(times);
    const taskSeries = array(viz.task_series).map(normalizeBulbAngleSeries), spatialTimes = array(spatial.time_s), media = object(viz.media);
    const allTimes = [times, spatialTimes, ...taskSeries.map(s => array(s.time_s))];
    const endpoints = allTimes.map(t => t.length ? t[t.length - 1] : null).filter(finite);
    if (finite(report.efficiency?.simulation_duration_s)) endpoints.push(report.efficiency.simulation_duration_s - (finite(viz.time_origin_s) ? viz.time_origin_s : 0));
    if (array(media.time_s).length) endpoints.push(media.time_s[media.time_s.length - 1]);
    const end = Math.max(0, ...endpoints), span = end || 1;
    let destroyed = false, playing = false, animation = null, previousTick = null, currentTime = 0;
    let video = null, mappedMedia = false, programmaticSeek = false, pendingVideoSeek = null;
    const cleanups = [], charts = [], contactRows = [], gridViews = [], glyphViews = [], readingNodes = [], objectURLs = new Set();
    let spatialMode = 'heatmap';
    const coordinateBounds = glyphBounds(glyph.channels);
    const selected = new Set(channels.map((_, i) => i));
    const root = doc.createElement('section'); root.className = 'kai-eval';
    root.setAttribute('aria-label', '轨迹评估可视化'); element.appendChild(root);
    const node = (tag, cls, parent, text) => {
      const el = doc.createElement(tag); if (cls) el.className = cls;
      if (text !== undefined && text !== null) el.textContent = String(text);
      if (parent) parent.appendChild(el); return el;
    };
    const on = (target, name, handler, options) => { target.addEventListener(name, handler, options); cleanups.push(() => target.removeEventListener(name, handler, options)); };
    const button = (parent, text, cls = 'ke-button') => { const el = node('button', cls, parent, text); el.type = 'button'; return el; };
    const pill = (parent, text, state) => node('span', `ke-badge ${['pass', 'fail', 'unknown'].includes(state) ? 'ke-' + state : ''}`, parent, text);
    const titleRow = (parent, title, note) => { const row = node('div', 'ke-section-title', parent); node('h2', '', row, title); if (note) node('span', 'ke-subtle', row, note); return row; };
    const statusText = status => ({pass: '通过', fail: '未通过', unknown: '无法判定'})[status] || '无法判定';
    const channelName = i => channels[i]?.name || `通道 ${i + 1}`;
    const nativeValue = (i, time) => { const index = observedIndex(times, time, nominal, source.gap_after); return index >= 0 && finite(forces[index]?.[i]) ? forces[index][i] : null; };
    const canvas = (parent, cls, label) => { const el = node('canvas', cls, parent); el.setAttribute('role', 'img'); el.setAttribute('aria-label', label); return el; };
    const fit = view => {
      const rect = view.getBoundingClientRect(), width = Math.max(1, rect.width || 600), height = Math.max(1, rect.height || 260);
      const dpr = Math.min(global.devicePixelRatio || 1, 2);
      if (view.width !== Math.round(width * dpr) || view.height !== Math.round(height * dpr)) { view.width = Math.round(width * dpr); view.height = Math.round(height * dpr); }
      const ctx = view.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
      return {ctx, width, height, dpr};
    };
    const text = (ctx, value, x, y, align = 'left', color = '#758294', size = 11) => { ctx.fillStyle = color; ctx.font = `${size}px system-ui,sans-serif`; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(value, x, y); };
    function hatch(ctx, x, y, w, h) {
      ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); ctx.fillStyle = '#eef1f4'; ctx.fillRect(x, y, w, h); ctx.strokeStyle = '#cad3dd'; ctx.lineWidth = .7;
      for (let i = -h; i < w + h; i += 8) { ctx.beginPath(); ctx.moveTo(x + i, y + h); ctx.lineTo(x + i + h, y); ctx.stroke(); } ctx.restore();
    }

    // Header and outcome: recorded task monitor labels are not re-certified.
    const header = node('header', 'ke-header', root), identity = node('div', '', header);
    node('p', 'ke-eyebrow', identity, '轨迹评估 / OFFLINE REVIEW');
    node('h1', '', identity, trial.task || '任务未标注');
    node('p', 'ke-run-id', identity, trial.trial_id || '试验编号未提供');
    const links = node('div', 'ke-header-links', header);
    const audit = node('a', 'ke-button', links, '审计详情 ↗'); audit.href = relativeURL(options?.auditURL) || 'audit.html';
    const download = button(links, '下载 JSON ↓');
    on(download, 'click', () => {
      if (!global.Blob || !global.URL?.createObjectURL) return;
      const url = global.URL.createObjectURL(new global.Blob([JSON.stringify(report, null, 2)], {type: 'application/json;charset=utf-8'})); objectURLs.add(url);
      const a = node('a', '', root); a.href = url; a.download = 'evaluation-report.json'; a.hidden = true; a.click(); a.remove();
    });
    const summary = node('div', 'ke-summary', root), outcome = object(report.outcome), safety = object(report.safety);
    const verdict = node('article', 'ke-summary-card ke-verdict', summary);
    node('p', 'ke-kicker', verdict, '任务结果');
    pill(verdict, statusText(outcome.status), outcome.status);
    node('p', 'ke-summary-note', verdict, '原任务监视器记录 · 非独立重新认证');
    const duration = node('article', 'ke-summary-card', summary);
    node('p', 'ke-kicker', duration, '仿真经过时间');
    node('p', 'ke-big-number', duration, `${fmt(report.efficiency?.simulation_duration_s, 2)} s`);
    node('p', 'ke-summary-note', duration, `实际耗时 ${fmt(report.efficiency?.wall_duration_s, 1)} s · 不混作仿真时间`);
    const safe = node('article', 'ke-summary-card', summary); node('p', 'ke-kicker', safe, '已记录安全检查');
    node('p', 'ke-big-number', safe, statusText(safety.status));
    node('p', 'ke-summary-note', safe, `${safety.enforcement === 'enforce' ? '执行保护' : safety.enforcement === 'observe' ? '只观测、不强制停止' : '保护模式未知'} · 仅限有记录的检查`);
    const lastCard = node('article', 'ke-summary-card', summary);
    const clean = report.task_specific?.cleaned_fraction ?? report.task_specific?.partial_state?.cleaned_fraction;
    node('p', 'ke-kicker', lastCard, finite(clean) ? '记录的终态清洁进度' : '触觉观测');
    node('p', 'ke-big-number', lastCard, finite(clean) ? `${fmt(clean * 100, 1)}%` : `${channels.length} 个通道`);
    node('p', 'ke-summary-note', lastCard, finite(clean) ? '任务状态量，不等同图像面积或成功条件' : `${times.length.toLocaleString('zh-CN')} 个原始样本 · ${finite(nominal) && nominal > 0 ? fmt(1 / nominal, 1) + ' Hz' : '采样率未知'}`);
    const metaLine = node('div', 'ke-meta-line', root);
    node('span', '', metaLine, `数据来源：${trial.source || '未提供'}`);
    if (Number.isInteger(trial.prediction_horizon)) pill(metaLine, `预测 H${trial.prediction_horizon} · 非控制频率`, 'unknown');
    if (outcome.formal_metrics_valid === false || outcome.diagnostic_only === true) pill(metaLine, '诊断运行：不作为正式性能结果', 'unknown');

    if (report.suite_schema_version) {
      const panel = node('section', 'ke-panel ke-v2-evidence', root);
      titleRow(panel, '规则复核与接触质量', '证据充分才下结论 · 不替换原任务结果');
      const review = object(report.task_review), quality = object(report.contact_quality), reference = object(report.reference_comparison);
      const cards = node('div', 'ke-summary', panel);
      const reviewCard = node('article', 'ke-summary-card', cards); node('p','ke-kicker',reviewCard,'独立字段规则复核'); pill(reviewCard,statusText(review.review?.status),review.review?.status);
      node('p','ke-note',reviewCard,review.review?.status === 'unknown' ? '未满足复核所需的契约、证据或有效性条件；展开下方详情查看具体原因。' : '按明确选择的规则与独立声明字段重算；不等同重新运行物理仿真。');
      const evidenceCard = node('article','ke-summary-card',cards); node('p','ke-kicker',evidenceCard,'版本化证据');
      node('p','ke-big-number',evidenceCard,report.evidence_v2?.source === 'sidecar' ? '新证据记录' : report.evidence_v2?.source === 'legacy' ? '旧记录 · 部分' : '缺失');
      node('p','ke-note',evidenceCard,`${report.evidence_v2?.sample_count || 0} 个样本 · ${report.evidence_v2?.sampled_complete ? '采样覆盖完整（输入声明）' : '不具备完整复核证据'}`);
      const refCard = node('article','ke-summary-card',cards); node('p','ke-kicker',refCard,'专家条件分布');
      node('p','ke-big-number',refCard,reference.status === 'available' ? '可比较' : reference.status === 'not_configured' ? '未配置' : '见逐项状态');
      node('p','ke-note',refCard,reference.status === 'not_configured' ? '未提供显式声明的专家参考集。' : reference.status === 'unavailable' ? '参考成员、传感器、采样、任务条件或样本量尚不满足可比性要求；详情列出原因。' : '逐 episode 比较，未按专家时间轴对齐；分位数不是质量评分。');
      Object.entries(object(quality.contact_pairs)).forEach(([name,pair]) => {
        const article = node('article','ke-quality-pair',panel); node('h3','',article,`${name} · ${pair.body_a} ↔ ${pair.body_b}`);
        node('p','ke-note',article,`${pair.target ? '目标接触对' : '非目标接触对'} · 允许相对运动：${pair.allowed_relative_motion || '未知'}`);
        const row = node('div','ke-quality-values',article);
        [['已观测接触',pair.contact?.status === 'missing' ? null : pair.contact?.observed_duration_s,'s'],['接触观测覆盖',finite(pair.contact?.coverage_fraction) ? pair.contact.coverage_fraction * 100 : null,'%'],['采样法向峰值',pair.normal_load?.sampled_peak,'N'],['力时长加权均值',pair.normal_load?.duration_weighted_mean,'N'],['预期接触中断',pair.expected_contact_interruptions?.observed_interruption_duration_s,'s'],['显式阈值过载',pair.normal_load?.overload?.status === 'missing' ? null : pair.normal_load?.overload?.observed_duration_s,'s']].forEach(([label,value,unit]) => {
          const item = node('div','',row); node('span','ke-kicker',item,label); node('strong','',item,`${fmt(value,3)} ${unit}`);
        });
        node('p','ke-note',article,`滑移证据：${pair.independently_labeled_slip?.status || 'missing'}；相对运动判据：${pair.relative_motion_supported_slip?.status || 'missing'}。力变化不直接作为滑移标签；— 表示无法计算，不表示零。`);
      });
      if (!Object.keys(object(quality.contact_pairs)).length) node('p','ke-empty',panel,'接触对证据缺失：仍可检查下方触觉强度，但无法据此判断碰到了哪个对象、接触是否正确或是否滑移。');
      array(reference.metrics).forEach(metric => {
        node('p','ke-note',panel,`${metric.scope} / ${metric.channel} / ${metric.metric}：${fmt(metric.value,3)} ${metric.unit || ''} · ${metric.status} · 参考 ${metric.reference_episode_count ?? '—'} 条 · 百分位区间 ${JSON.stringify(metric.percentile_rank_interval ?? null)}`);
        const stats = report.reference_distributions?.[metric.reference_group_id]?.statistics;
        if (metric.status === 'available' && stats && [stats.p05,stats.p25,stats.median,stats.p75,stats.p95,metric.value].every(finite)) {
          const low = Math.min(stats.p05,metric.value), high = Math.max(stats.p95,metric.value), width = high - low || 1;
          const x = value => 30 + (value - low) / width * 440;
          const svg = doc.createElementNS('http://www.w3.org/2000/svg','svg'); svg.setAttribute('viewBox','0 0 500 60'); svg.setAttribute('class','ke-reference-distribution'); svg.setAttribute('role','img');
          svg.setAttribute('aria-label',`参考分布 P05 ${stats.p05}，P25 ${stats.p25}，中位数 ${stats.median}，P75 ${stats.p75}，P95 ${stats.p95}；本次 ${metric.value} ${metric.unit}`); panel.appendChild(svg);
          const shape = (tag,attributes) => { const el = doc.createElementNS('http://www.w3.org/2000/svg',tag); Object.entries(attributes).forEach(([key,value]) => el.setAttribute(key,String(value))); svg.appendChild(el); return el; };
          shape('line',{x1:x(stats.p05),x2:x(stats.p95),y1:22,y2:22,stroke:'#75949e','stroke-width':2});
          shape('rect',{x:x(stats.p25),y:13,width:Math.max(1,x(stats.p75)-x(stats.p25)),height:18,fill:'#b9d9df',stroke:'#6495a0'});
          shape('line',{x1:x(stats.median),x2:x(stats.median),y1:12,y2:32,stroke:'#3e717d','stroke-width':2});
          shape('circle',{cx:x(metric.value),cy:22,r:5,fill:'#d18845',stroke:'#fff','stroke-width':1.5});
          const label = shape('text',{x:30,y:52,fill:'#728997','font-size':10}); label.textContent = `参考 P05–P95 / 方框 P25–P75 / 竖线中位数 · 橙点：本次 ${fmt(metric.value,3)} ${metric.unit}`;
        }
      });
      const details = node('details','ke-more',panel); node('summary','',details,'查看条件、事件、缺失字段与完整质量指标');
      node('pre','ke-evidence-json',details,JSON.stringify({task_review:review,contact_quality:quality,reference_comparison:reference,reference_distributions:report.reference_distributions},null,2));
    }

    const transport = node('section', 'ke-transport', root); transport.setAttribute('aria-label', '共享时间游标');
    const timeHeading = node('div', 'ke-time-heading', transport), clockGroup = node('div', '', timeHeading);
    node('span', 'ke-kicker', clockGroup, '共享时间游标');
    const timeText = node('strong', 'ke-time-value', clockGroup, '0.000'), timeEnd = node('span', 'ke-time-total', clockGroup, ` / ${fmt(end, 3)} s`);
    const controls = node('div', 'ke-controls', timeHeading), play = button(controls, '▶ 播放', 'ke-button ke-primary');
    const previous = button(controls, '‹', 'ke-button ke-square'); previous.setAttribute('aria-label', '前一个原生观测');
    const next = button(controls, '›', 'ke-button ke-square'); next.setAttribute('aria-label', '后一个原生观测');
    const slider = node('input', 'ke-slider', transport); slider.type = 'range'; slider.min = '0'; slider.max = String(end || 1); slider.step = 'any'; slider.value = '0'; slider.disabled = end <= 0; slider.setAttribute('aria-label', '共享时间，秒');
    const ruler = node('div', 'ke-ruler', transport); node('span', '', ruler, '0 s'); node('span', '', ruler, '点击曲线或拖动游标，联动已有观测'); node('span', '', ruler, `${fmt(end, 2)} s`);
    const grid = node('div', 'ke-layout', root), main = node('div', 'ke-main', grid), side = node('aside', 'ke-side', grid);

    // Scene media remains independent unless frame timestamps prove the mapping.
    const mediaPanel = node('section', 'ke-panel ke-media-panel', main);
    titleRow(mediaPanel, '场景回放', '原始评估视频');
    const mediaURL = relativeURL(media.url), mediaTimes = array(media.time_s), videoTimes = array(media.video_time_s);
    const mediaNote = node('p', 'ke-note', mediaPanel);
    const validClock = a => a.length > 0 && a.every((v, i) => finite(v) && (i === 0 || v > a[i - 1]));
    mappedMedia = media.sync_status === 'mapped' && mediaTimes.length === videoTimes.length && validClock(mediaTimes) && validClock(videoTimes);
    if (media.status === 'available' && mediaURL) {
      const frame = node('div', 'ke-video-frame', mediaPanel); video = node('video', 'ke-video', frame);
      video.controls = true; video.muted = true; video.preload = 'metadata'; video.playsInline = true; video.src = mediaURL;
      video.setAttribute('aria-label', '原始轨迹场景视频');
      mediaNote.textContent = mappedMedia ? '逐帧时间映射已验证；游标选择最近非未来视频帧。视频编码秒数不作为仿真秒数。' : `独立回放，未建立可靠时间对应：${media.reason || '缺少逐帧仿真时间证据'}。`;
      on(video, 'error', () => { mediaNote.textContent = '视频未能加载。曲线与触觉数据仍可独立查看；请检查离线媒体文件是否完整。'; });
      if (mappedMedia) {
        const followVideo = () => {
          if (programmaticSeek || destroyed) return;
          const i = leftIndex(videoTimes, video.currentTime); if (i >= 0) seek(mediaTimes[i], false);
        };
        on(video, 'timeupdate', followVideo);
        on(video, 'seeked', () => { if (programmaticSeek) { programmaticSeek = false; pendingVideoSeek = null; } else followVideo(); });
        on(video, 'loadedmetadata', () => { if (pendingVideoSeek !== null) seekVideo(pendingVideoSeek); });
        on(video, 'play', () => { playing = true; updatePlay(); });
        on(video, 'pause', () => { playing = false; updatePlay(); });
        on(video, 'ended', () => { playing = false; updatePlay(); followVideo(); });
      }
    } else {
      node('div', 'ke-empty ke-video-empty', mediaPanel, '未提供可用场景视频');
      mediaNote.textContent = media.reason || '没有关联媒体；此处不使用其他轨迹视频代替。';
    }

    const forcePanel = node('section', 'ke-panel', main);
    titleRow(forcePanel, '法向力时间线', '原始样本 · N');
    const legend = node('div', 'ke-channel-controls', forcePanel), legendActions = node('div', 'ke-channel-actions', forcePanel);
    const allButton = button(legendActions, '全部通道', 'ke-text-button'), clearButton = button(legendActions, '清空选择', 'ke-text-button');
    channels.forEach((channel, i) => {
      const label = node('label', 'ke-channel-toggle', legend), check = node('input', '', label); check.type = 'checkbox'; check.checked = true;
      const dot = node('i', 'ke-dot', label); dot.style.backgroundColor = COLORS[i % COLORS.length];
      const labelText = node('span', '', label, channelName(i)); labelText.title = channelName(i);
      const reading = node('strong', '', label, '—'); readingNodes.push({check, reading});
      on(check, 'change', () => { if (check.checked) selected.add(i); else selected.delete(i); rebuildCharts(); render(); });
    });
    function selectAll(enabled) { channels.forEach((_, i) => { if (enabled) selected.add(i); else selected.delete(i); readingNodes[i].check.checked = enabled; }); rebuildCharts(); render(); }
    on(allButton, 'click', () => selectAll(true)); on(clearButton, 'click', () => selectAll(false));
    const forceCanvas = canvas(forcePanel, 'ke-force-chart', '多通道法向力原生时间曲线，点击选时刻');
    node('p', 'ke-note', forcePanel, '曲线保留全部原始样本与短峰；缺测和采样间断处断线。游标读数取最近非未来观测；单位为归档声明的 N。');
    const forceSeries = channels.map((channel, i) => ({time: times, values: forces.map(row => finite(row?.[i]) ? row[i] : null), gaps: source.gap_after, nominal, color: COLORS[i % COLORS.length], channelIndex: i, label: channelName(i)}));
    addChart(forceCanvas, forceSeries, {unit: 'N', filter: series => selected.has(series.channelIndex)});

    const contactPanel = node('section', 'ke-panel', main); titleRow(contactPanel, '按阈值检出的接触片段', '原生时长 · 迟滞与最短片段规则');
    const contacts = node('div', 'ke-contact-bands', contactPanel), overall = object(report.tactile?.scopes?.overall);
    channels.forEach((channel, i) => {
      const row = node('div', 'ke-contact-row', contacts); node('span', 'ke-band-label', row, channelName(i)).title = channelName(i);
      const track = node('div', 'ke-band-track', row); track.setAttribute('aria-label', `${channelName(i)}确认接触片段`);
      const unknownBand = (start, stop) => {
        if (!finite(start) || !finite(stop) || stop <= start) return;
        const mark = node('span', 'ke-band-unknown', track); mark.style.left = `${clamp(start / span * 100, 0, 100)}%`;
        mark.style.width = `${Math.max(0, Math.min(stop, span) - Math.max(0, start)) / span * 100}%`;
        mark.title = `无观测或采样缺口 · ${fmt(start, 3)}–${fmt(stop, 3)} s`;
      };
      if (!times.length) unknownBand(0, end);
      else {
        unknownBand(0, times[0]); unknownBand(times[times.length - 1], end);
        const authoritativeGaps = Array.isArray(source.gap_after) && source.gap_after.length === times.length;
        for (let j = 0; j < times.length - 1; j++) if (!finite(forces[j]?.[i]) || source.gap_after?.[j] === true || (!authoritativeGaps && finite(nominal) && times[j + 1] - times[j] > nominal * 1.5 + 1e-8)) unknownBand(times[j], times[j + 1]);
      }
      const continuity = object(overall.channels?.[channelName(i)]?.continuity);
      array(continuity.episodes).forEach(episode => {
        if (!finite(episode.start_s) || !finite(episode.end_s) || episode.end_s <= episode.start_s) return;
        const segment = button(track, '', 'ke-band-segment'); segment.style.left = `${clamp(episode.start_s / span * 100, 0, 100)}%`; segment.style.width = `${Math.max(0, Math.min(episode.end_s, span) - Math.max(0, episode.start_s)) / span * 100}%`; segment.style.backgroundColor = COLORS[i % COLORS.length];
        segment.title = `${channelName(i)} · ${fmt(episode.start_s, 3)}–${fmt(episode.end_s, 3)} s${episode.left_censored || episode.right_censored ? ' · 边界截断' : ''}`; segment.setAttribute('aria-label', segment.title);
        on(segment, 'click', () => { pause(); seek(episode.start_s); });
      });
      const marker = node('span', 'ke-band-cursor', track); contactRows.push(marker);
      const value = finite(continuity.observed_contact_duration_s) ? `${fmt(continuity.observed_contact_duration_s, 2)} s` : '未知'; node('span', 'ke-band-duration', row, value);
    });
    if (!channels.length) node('p', 'ke-empty', contacts, '没有可用的触觉通道');
    node('p', 'ke-note', contactPanel, '色段为达到最短时长的已确认片段；灰斜纹为原生缺测或采样缺口。其余浅色也可能含短片段或未初始化的迟滞状态，不能一律当作断触。右侧为所有观测接触时长。');

    const taskPanel = node('section', 'ke-panel', main); titleRow(taskPanel, '任务状态与接触效果', '同源时间证据');
    if (!taskSeries.length) node('p', 'ke-empty', taskPanel, '未提供可可靠同步的任务状态曲线。终态与原始判定仍保留在审计详情。');
    const taskGrid = node('div', 'ke-task-charts', taskPanel);
    taskSeries.forEach((series, i) => {
      const item = node('article', 'ke-task-chart-card', taskGrid), heading = node('div', 'ke-task-chart-heading', item);
      node('h3', '', heading, series.label || series.key || `任务信号 ${i + 1}`);
      const current = node('strong', '', heading, '—');
      const view = canvas(item, 'ke-task-chart', `${series.label || series.key || '任务信号'}原始时间曲线`);
      const ts = array(series.time_s), vs = array(series.values), step = cadence(ts);
      const references = goalReferences(series);
      if (references.length) {
        const legend = node('div', 'ke-goal-references', item); legend.setAttribute('role', 'note');
        references.forEach(ref => {
          const entry = node('div', 'ke-goal-reference', legend), line = node('p', 'ke-goal-reference-heading', entry);
          node('span', 'ke-goal-reference-swatch', line); node('strong', '', line, `${ref.label ? ref.label + ' · ' : ''}${referenceText(ref)}`);
          if (finite(ref.lower) && finite(ref.upper)) node('p', 'ke-goal-reference-band', entry, `淡色带：${referenceNumber(ref.lower)}–${referenceNumber(ref.upper)} ${displayUnit(ref.unit)}（${ref.relation === 'target' ? '到位几何容差' : '该项明确允许范围'}）`);
          if (ref.note) node('p', 'ke-goal-reference-note', entry, ref.note);
        });
        node('p', 'ke-goal-reference-disclaimer', legend, '虚线 / 淡色带只作单项阈值参考，不代表完整任务成功。');
      }
      addChart(view, [{time: ts, values: vs, gaps: series.gap_after, nominal: step, color: i % 2 ? '#d69230' : '#087e94', label: series.label || series.key}], {unit: displayUnit(series.unit) || '', current, references});
    });
    node('p', 'ke-note', taskPanel, '任务状态曲线仅在时钟来源可核实时联动。接触本身不证明有效作业；请结合终态和任务条件。');

    const metricsPanel = node('section', 'ke-panel', main); titleRow(metricsPanel, '逐通道力幅值对比', '完整轨迹 · 原生时长加权');
    const metricLegend = node('div', 'ke-metric-legend', metricsPanel);
    [['均值', '#087e94'], ['P95', '#d69230'], ['原生采样峰值', '#25394f']].forEach(([label, color]) => { const k = node('span', '', metricLegend); node('i', 'ke-dot', k).style.backgroundColor = color; node('span', '', k, label); });
    const metrics = channels.map((_, i) => { const intensity = object(overall.channels?.[channelName(i)]?.intensity); return {name: channelName(i), mean: intensity.all_observed?.mean_n, p95: intensity.all_observed?.p95_n, peak: intensity.sampled_peak_n}; });
    const metricMaximum = Math.max(0, ...metrics.flatMap(m => [m.mean, m.p95, m.peak]).filter(finite));
    metrics.forEach(metric => {
      const row = node('div', 'ke-metric-row', metricsPanel), label = node('div', 'ke-metric-label', row); node('strong', '', label, metric.name);
      node('span', '', label, `${fmt(metric.mean)} / ${fmt(metric.p95)} / ${fmt(metric.peak)} N`);
      const track = node('div', 'ke-metric-track', row); track.setAttribute('aria-label', `${metric.name}：均值 ${fmt(metric.mean)}，P95 ${fmt(metric.p95)}，原生采样峰值 ${fmt(metric.peak)} N`);
      [['mean', 'ke-mean'], ['p95', 'ke-p95'], ['peak', 'ke-peak']].forEach(([field, cls]) => { if (!finite(metric[field])) return; const mark = node('i', `ke-metric-mark ${cls}`, track); mark.style.left = `${clamp(metric[field] / (metricMaximum || 1) * 100, 0, 100)}%`; });
    });
    const metricRuler = node('div', 'ke-ruler', metricsPanel); node('span', '', metricRuler, '0 N'); node('span', '', metricRuler, `${fmt(metricMaximum)} N`);
    node('p', 'ke-note', metricsPanel, '均值与 P95 包含全部有效观测的零力阶段；这些是力幅值描述，不直接代表稳定性或任务质量。采样峰值不是物理时间内的真实峰值上界。');

    const spatialPanel = node('section', 'ke-panel ke-spatial-panel', side); titleRow(spatialPanel, '触觉阵列', '共享时间 · 两种视图');
    const spatialModes = node('div', 'ke-spatial-mode', spatialPanel); spatialModes.setAttribute('role', 'group'); spatialModes.setAttribute('aria-label', '触觉显示方式');
    const heatMode = button(spatialModes, '阵列热图', 'ke-mode-button'), circleMode = button(spatialModes, '接触力圆图', 'ke-mode-button');
    heatMode.setAttribute('aria-pressed', 'true'); circleMode.setAttribute('aria-pressed', 'false');
    const spatialClock = node('div', 'ke-spatial-clock', spatialPanel), spatialTimeLabel = node('strong', '', spatialClock, '—'), spatialLag = node('span', '', spatialClock);
    const heatLegend = node('div', 'ke-heat-legend', spatialPanel); node('span', '', heatLegend, '0'); node('i', 'ke-heat-gradient', heatLegend); node('span', '', heatLegend, `${fmt(taxelScale, 3)} N / 单元`);
    const heatNote = node('p', 'ke-note', spatialPanel, '所有通道、所有时刻采用同一固定色标。快照取游标之前已保存的样本，不插值；行列仅为阵列索引，不推断毫米坐标或真实皮肤压力。');
    const heatGrid = node('div', 'ke-heat-grid', spatialPanel);
    channels.forEach((channel, i) => {
      const card = node('article', 'ke-heat-card', heatGrid), head = node('div', 'ke-heat-heading', card); node('i', 'ke-dot', head).style.backgroundColor = COLORS[i % COLORS.length]; node('h3', '', head, channelName(i));
      const view = canvas(card, 'ke-heatmap', `${channelName(i)}法向力阵列，缺测单元显示斜纹`), value = node('p', 'ke-heat-reading', card, '无观测');
      gridViews.push({view, value, index: i});
    });
    const heatSelectionNote = node('p', 'ke-note ke-spatial-selection', spatialPanel);
    heatSelectionNote.textContent = spatial.status === 'insufficient_frame_budget' ? '快照预算不足以保留必需样本；阵列数据未导出，请查看原生曲线和审计说明。' : `${spatialTimes.length} 个实际快照 / ${times.length} 个原生样本；快照可能稀疏，时刻和距游标间隔始终列出。`;
    const glyphLegend = node('div', 'ke-glyph-legend', spatialPanel); glyphLegend.hidden = true;
    const normalLegend = node('div', 'ke-glyph-legend-row', glyphLegend); node('i', 'ke-glyph-circle-key', normalLegend); node('span', '', normalLegend, `圆面积 ∝ Fn · 虚线参考圆 ${fmt(normalScale, 3)} N`);
    node('p', 'ke-note', glyphLegend, '圆心为法向力加权中心；圆的尺寸只是固定力标尺，不代表接触面积或皮肤变形。所有通道、所有时刻共用面积与箭头尺度。');
    const glyphGeometryNote = node('p', 'ke-note ke-glyph-coordinate-note', glyphLegend);
    const coordinateKinds = array(glyph.channels).map(channel => channel.coordinate_status);
    glyphGeometryNote.textContent = coordinateKinds.length && coordinateKinds.every(kind => kind === 'physical') ? '坐标使用归档的局部 XY 位置投影（mm），横纵轴等比例。' : '含阵列索引坐标：无已验证物理几何时，仅显示 taxel 行列索引，不能解释为 mm。索引行 0 在上，物理 Y 正向朝上。';
    const glyphGrid = node('div', 'ke-glyph-grid', spatialPanel); glyphGrid.hidden = true;
    const glyphUnavailable = node('p', 'ke-empty ke-glyph-unavailable', glyphGrid, '此归档未提供接触力圆图数据；可继续查看原有阵列，不补造中心或切向力。');
    glyphUnavailable.hidden = array(glyph.time_s).length > 0;
    if (array(glyph.time_s).length) channels.forEach((channel, i) => {
      const card = node('article', 'ke-glyph-card', glyphGrid), head = node('div', 'ke-glyph-heading', card);
      node('i', 'ke-dot', head).style.backgroundColor = COLORS[i % COLORS.length]; node('h3', '', head, channelName(i));
      const coordinate = object(glyph.channels?.[i]);
      node('span', 'ke-glyph-coordinate-badge', card, coordinate.coordinate_status === 'physical' && coordinate.unit === 'mm' ? '局部 XY 投影 · mm' : array(coordinate.taxel_xy).length ? '阵列索引 · 非 mm' : '位置坐标缺失');
      const normalCanvas = canvas(card, 'ke-force-glyph', `${channelName(i)}法向力圆图`);
      const normalReading = node('p', 'ke-glyph-reading', card, 'Fn — N');
      const centerReading = node('p', 'ke-glyph-center', card, '中心未知');
      const changeBox = node('div', 'ke-glyph-change', card);
      node('p', 'ke-glyph-window', changeBox, `过去 ${fmt(glyph.window_s, 3)} s 的力变化`);
      const normalChange = node('p', 'ke-glyph-normal-change', changeBox, 'ΔFn — N');
      const tangentCanvas = canvas(changeBox, 'ke-tangent-glyph', `${channelName(i)}记录切向分量的变化向量，不是移动方向`);
      const tangentReading = node('p', 'ke-glyph-tangent-reading', changeBox, '|ΔFt| — N');
      const tangentNote = node('p', 'ke-glyph-tangent-note', changeBox);
      glyphViews.push({index: i, normalCanvas, tangentCanvas, normalReading, centerReading, normalChange, tangentReading, tangentNote});
    });
    const glyphFootnote = node('p', 'ke-note ke-glyph-footnote', spatialPanel, 'ΔFn 红色表示增加、蓝色表示减少（颜色只表符号）；ΔFt 箭头表示切向合力变化，既不是当前切向力，也不是运动位移。严格窗口数据缺失时显示未知。'); glyphFootnote.hidden = true;
    function setSpatialMode(mode) {
      spatialMode = mode;
      const showGlyph = mode === 'glyph';
      heatMode.setAttribute('aria-pressed', String(!showGlyph)); circleMode.setAttribute('aria-pressed', String(showGlyph));
      [heatLegend, heatNote, heatGrid, heatSelectionNote].forEach(item => { item.hidden = showGlyph; });
      [glyphLegend, glyphGrid, glyphFootnote].forEach(item => { item.hidden = !showGlyph; });
      renderSpatial();
    }
    on(heatMode, 'click', () => setSpatialMode('heatmap')); on(circleMode, 'click', () => setSpatialMode('glyph'));
    [heatMode, circleMode].forEach(modeButton => on(modeButton, 'keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault(); const showGlyph = event.key === 'End' || (event.key !== 'Home' && spatialMode === 'heatmap');
      setSpatialMode(showGlyph ? 'glyph' : 'heatmap'); (showGlyph ? circleMode : heatMode).focus();
    }));
    const eventPanel = node('section', 'ke-panel', side); titleRow(eventPanel, '任务事件', '带时间的原始证据');
    const eventList = node('div', 'ke-events', eventPanel);
    const eventButtons = [];
    array(viz.events).forEach(event => {
      if (!finite(event.time_s)) return;
      const b = button(eventList, '', 'ke-event'), clock = node('time', '', b, `${fmt(event.time_s, 3)} s`);
      node('span', '', b, event.label || event.name || event.type || '已记录事件');
      b.title = [event.evidence_kind, event.evidence, event.ground_truth === false ? '阶段声明不是任务真值' : ''].filter(Boolean).join(' · ');
      on(b, 'click', () => { pause(); seek(event.time_s); }); eventButtons.push({button: b, time: event.time_s});
    });
    if (!eventButtons.length) node('p', 'ke-empty', eventList, '没有可同步的事件记录');
    node('p', 'ke-note', eventPanel, '阶段声明仅是记录中的状态；不据此推断已完成任务或真实重试次数。');
    const warningsPanel = node('section', 'ke-panel ke-warnings', side); titleRow(warningsPanel, '证据与限制');
    const warnings = [...new Set([...array(report.warnings), ...array(viz.warnings)].map(item => typeof item === 'string' ? item : JSON.stringify(item)))];
    if (!warnings.length) node('p', 'ke-note', warningsPanel, '未记录额外输入警告。指标仍受保存采样、来源语义与任务监视器定义限制。');
    const warningList = node('ul', '', warningsPanel); warnings.forEach(warning => node('li', '', warningList, warning));
    const more = node('details', 'ke-more', warningsPanel); node('summary', '', more, '协议与来源语义');
    node('p', 'ke-note', more, `协议 SHA-256：${report.protocol_sha256 || '未提供'}`);
    const semantics = viz.semantics;
    const semanticItems = Array.isArray(semantics) ? semantics : semantics && typeof semantics === 'object' ? Object.entries(semantics).map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`) : semantics ? [String(semantics)] : [];
    semanticItems.forEach(line => node('p', 'ke-note', more, line));
    node('footer', 'ke-footer', root, '可视化用于检查已有记录；不重新计算任务成功、不生成统一评分。完整定义、阶段统计与输入摘要见审计详情。');

    function addChart(view, series, options) {
      const {low, high} = chartRange(series, options.references);
      const chart = {view, series, options, low, high, cache: null, geometry: null}; charts.push(chart);
      on(view, 'click', event => { const box = view.getBoundingClientRect(), g = chart.geometry; if (!g) return; pause(); seek((event.clientX - box.left - g.left) / g.plotWidth * span); });
      view.tabIndex = 0;
      on(view, 'keydown', event => { if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return; event.preventDefault(); pause(); if (event.key === 'Home') seek(0); else if (event.key === 'End') seek(end); else step(event.key === 'ArrowLeft' ? -1 : 1); });
    }
    function renderChart(chart) {
      const {view, series, options, low, high} = chart, {ctx, width, height, dpr} = fit(view);
      const precision = options.references?.length ? chartPrecision(high - low) : high - low < 1 ? 2 : 1;
      const g = {left: options.references?.length && precision >= 4 ? 66 : width < 390 ? 42 : 50, right: 16, top: 24, bottom: 28}; g.plotWidth = Math.max(1, width - g.left - g.right); g.plotHeight = height - g.top - g.bottom; chart.geometry = g;
      const x = t => g.left + t / span * g.plotWidth, y = v => height - g.bottom - (v - low) / (high - low) * g.plotHeight;
      if (!chart.cache || chart.cache.width !== view.width || chart.cache.height !== view.height) {
        const base = doc.createElement('canvas'); base.width = view.width; base.height = view.height;
        const b = base.getContext('2d'); b.setTransform(dpr, 0, 0, dpr, 0, 0);
        array(options.references).forEach(ref => {
          if (finite(ref.lower) && finite(ref.upper)) { b.fillStyle = '#8666ad18'; b.fillRect(g.left, y(ref.upper), g.plotWidth, Math.max(1, y(ref.lower) - y(ref.upper))); }
        });
        for (let i = 0; i <= 4; i++) { const v = low + (high - low) * i / 4, yy = y(v); b.strokeStyle = '#e8edf2'; b.lineWidth = .8; b.beginPath(); b.moveTo(g.left, yy); b.lineTo(width - g.right, yy); b.stroke(); text(b, fmt(v, precision), g.left - 8, yy, 'right'); }
        const ticks = width < 390 ? 3 : 5; for (let i = 0; i <= ticks; i++) text(b, fmt(end * i / ticks, end < 10 ? 1 : 0), x(end * i / ticks), height - 10, 'center');
        text(b, options.unit || '', g.left, 10); text(b, 's', width - 3, height - 10, 'right');
        series.forEach(s => {
          if (options.filter && !options.filter(s)) return;
          b.strokeStyle = s.color; b.fillStyle = s.color; b.lineWidth = 1.25; b.lineJoin = 'round';
          knownSegments(s.time, s.values, s.gaps, s.nominal).forEach(points => { b.beginPath(); points.forEach((p, i) => { if (i) b.lineTo(x(p[0]), y(p[1])); else b.moveTo(x(p[0]), y(p[1])); }); b.stroke(); if (points.length === 1) { b.beginPath(); b.arc(x(points[0][0]), y(points[0][1]), 1.6, 0, Math.PI * 2); b.fill(); } });
        });
        array(options.references).forEach(ref => {
          const yy = y(ref.target); b.strokeStyle = '#80609f'; b.lineWidth = 1.4; b.setLineDash([6, 4]); b.beginPath(); b.moveTo(g.left, yy); b.lineTo(width - g.right, yy); b.stroke(); b.setLineDash([]);
          const label = referenceText(ref), labelY = clamp(yy - 9, g.top + 7, height - g.bottom - 8);
          b.font = '10px system-ui,sans-serif'; const labelWidth = Math.min(g.plotWidth, b.measureText(label).width + 8);
          b.fillStyle = '#fffffff0'; b.fillRect(width - g.right - labelWidth, labelY - 7, labelWidth, 14);
          b.fillStyle = '#70558d'; b.textAlign = 'right'; b.textBaseline = 'middle'; b.fillText(label, width - g.right - 3, labelY, Math.max(1, g.plotWidth - 6));
        });
        if (!series.some(s => (!options.filter || options.filter(s)) && s.values.some(finite))) text(b, '无可显示的观测', width / 2, height / 2, 'center', '#788696', 13);
        chart.cache = base;
      }
      ctx.drawImage(chart.cache, 0, 0, width, height);
      ctx.strokeStyle = '#25394f'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x(currentTime), g.top); ctx.lineTo(x(currentTime), height - g.bottom); ctx.stroke(); ctx.setLineDash([]);
      const currentValues = [];
      series.forEach(s => {
        if (options.filter && !options.filter(s)) return;
        const i = observedIndex(s.time, currentTime, s.nominal, s.gaps), value = i >= 0 ? s.values[i] : null;
        const reading = options.references?.length && finite(value) && value !== 0 && Math.abs(value) < 1e-6
          ? referenceNumber(value) : options.references?.length ? fmt(value, Math.max(chartPrecision(high - low), options.unit === 'm' ? 6 : 4)) : fmt(value);
        currentValues.push(`${s.label || '观测'} ${reading} ${options.unit || ''}`);
        if (!finite(value)) { if (options.current) options.current.textContent = '—'; return; }
        ctx.fillStyle = s.color; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x(currentTime), y(value), 3.7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        if (options.current) options.current.textContent = `${reading} ${options.unit || ''}`;
      });
      view.setAttribute('aria-label', `${fmt(currentTime, 3)} 秒；${currentValues.join('；')}。${array(options.references).map(referenceText).join('；')}${options.references?.length ? '；参考线不代表完整任务成功。' : ''}点击或使用方向键选择时刻。`);
    }
    function rebuildCharts() { charts.forEach(chart => { chart.cache = null; }); }
    function heatColor(value) {
      const maximum = finite(taxelScale) && taxelScale > 0 ? taxelScale : 1;
      const q = clamp(value / maximum, 0, 1), stops = [[244, 248, 249], [165, 221, 227], [51, 155, 175], [6, 87, 115], [14, 47, 76]], scaled = q * 4, i = Math.min(3, Math.floor(scaled)), r = scaled - i;
      return `rgb(${stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * r)).join(',')})`;
    }
    function renderGlyphs() {
      const sample = glyphSample(glyph, currentTime, 0);
      spatialTimeLabel.textContent = sample.index >= 0 ? `原生采样 ${fmt(sample.sampleTime, 3)} s` : '无可用原生观测';
      spatialLag.textContent = sample.index >= 0 ? `距游标 ${fmt(sample.lag, 3)} s` : '不补零、不沿缺口延续';
      if (!glyphViews.length) return;
      // One pixel/N scale across every card; reserve a whole maximum-radius halo
      // around coordinate bounds so forces near edge taxels are never clipped.
      const commonWidth = Math.min(...glyphViews.map(view => view.normalCanvas.getBoundingClientRect().width || 140));
      const commonHeight = Math.min(...glyphViews.map(view => view.normalCanvas.getBoundingClientRect().height || 170));
      const fullSide = Math.max(20, Math.min(commonWidth - 36, commonHeight - 44));
      const forceRadius = Math.min(21, fullSide * .19), coordinateSide = Math.max(8, fullSide - 2 * forceRadius - 6);
      const tangentWidth = Math.min(...glyphViews.map(view => view.tangentCanvas.getBoundingClientRect().width || 140));
      const tangentHeight = Math.min(...glyphViews.map(view => view.tangentCanvas.getBoundingClientRect().height || 110));
      const vectorRadius = Math.max(8, Math.min(tangentWidth - 42, tangentHeight - 45) / 2);
      const maximum = normalScale, vectorMaximum = tangentScale;
      const componentBasis = glyph.tangent_coordinate_status === 'recorded_components';
      glyphViews.forEach(view => {
        const {index, normalCanvas, tangentCanvas, normalReading, centerReading, normalChange, tangentReading, tangentNote} = view;
        const data = glyphSample(glyph, currentTime, index), coordinate = object(glyph.channels?.[index]);
        const physical = coordinate.coordinate_status === 'physical' && coordinate.unit === 'mm';
        const bounds = coordinateBounds[physical ? 'mm' : 'index'], points = array(coordinate.taxel_xy).filter(vector2);
        const {ctx, width, height} = fit(normalCanvas), cx = width / 2 + 5, cy = height / 2 - 5;
        const half = coordinateSide / 2, left = cx - half, top = cy - half;
        const map = point => glyphPoint(point, bounds, coordinateSide, cx, cy, !physical);
        if (data.normal === null) hatch(ctx, cx - fullSide / 2, cy - fullSide / 2, fullSide, fullSide);
        if (points.length && bounds) {
          ctx.strokeStyle = '#bdc8d2'; ctx.lineWidth = .7;
          ctx.beginPath(); ctx.moveTo(left - forceRadius - 3, top - forceRadius); ctx.lineTo(left - forceRadius - 3, top + coordinateSide + forceRadius + 3); ctx.lineTo(left + coordinateSide + forceRadius, top + coordinateSide + forceRadius + 3); ctx.stroke();
          const xLow = bounds.centerX - bounds.range / 2, xHigh = bounds.centerX + bounds.range / 2;
          const yLow = bounds.centerY - bounds.range / 2, yHigh = bounds.centerY + bounds.range / 2;
          const tickDigits = physical || ![xLow, xHigh, yLow, yHigh].every(Number.isInteger) ? 1 : 0;
          text(ctx, fmt(xLow, tickDigits), left, top + coordinateSide + forceRadius + 11, 'center', '#8895a2', 8);
          text(ctx, fmt(xHigh, tickDigits), left + coordinateSide, top + coordinateSide + forceRadius + 11, 'center', '#8895a2', 8);
          text(ctx, fmt(physical ? yHigh : yLow, tickDigits), left - forceRadius - 6, top, 'right', '#8895a2', 8);
          text(ctx, fmt(physical ? yLow : yHigh, tickDigits), left - forceRadius - 6, top + coordinateSide, 'right', '#8895a2', 8);
          text(ctx, String(coordinate.axis_labels?.[1] || (physical ? 'Y' : 'row')), 4, 10, 'left', '#758594', 8);
          text(ctx, String(coordinate.axis_labels?.[0] || (physical ? 'X' : 'column')), width / 2, height - 7, 'center', '#758594', 8);
          ctx.fillStyle = '#a7b0bb';
          points.forEach(point => { const p = map(point); ctx.beginPath(); ctx.arc(p[0], p[1], 1.35, 0, Math.PI * 2); ctx.fill(); });
          if (data.normal !== null && data.normal > 0 && data.center) {
            const p = map(data.center), radius = glyphRadius(data.normal, maximum, forceRadius);
            // Invalid scales/centres remain unavailable instead of clipping or
            // silently shrinking a force that exceeds the declared global scale.
            const inside = p && p[0] - forceRadius >= 0 && p[0] + forceRadius <= width && p[1] - forceRadius >= 0 && p[1] + forceRadius <= height;
            if (inside && finite(radius) && radius <= forceRadius + 1e-7) {
              ctx.strokeStyle = '#c1a8a7'; ctx.lineWidth = .8; ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.arc(p[0], p[1], forceRadius, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
              ctx.fillStyle = 'rgba(220,89,81,.69)'; ctx.strokeStyle = '#c35b56'; ctx.lineWidth = .8;
              ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
              ctx.fillStyle = '#773f43'; ctx.beginPath(); ctx.arc(p[0], p[1], 1.5, 0, Math.PI * 2); ctx.fill();
            } else text(ctx, '中心或标尺不可用', cx, cy, 'center', '#986d56', 9);
          } else if (data.normal === 0) text(ctx, '零法向力', cx, cy, 'center', '#7c8b98', 10);
          else if (data.normal === null) text(ctx, '当前观测未知', cx, cy, 'center', '#697d8e', 9);
          else text(ctx, '中心未知', cx, cy, 'center', '#697d8e', 10);
        } else {
          if (data.normal !== null) hatch(ctx, cx - fullSide / 2, cy - fullSide / 2, fullSide, fullSide);
          text(ctx, '无位置坐标', cx, cy, 'center', '#7c8b98', 10);
        }
        normalReading.textContent = `Fn ${fmt(data.normal, 3)} N${data.normal === 0 ? ' · 零力' : ''}`;
        centerReading.textContent = data.normal === null ? '无观测；不是零力' : data.normal === 0 ? '无接触中心，不绘制力圆' : data.center && points.length ? `中心 (${fmt(data.center[0], 2)}, ${fmt(data.center[1], 2)}) ${physical ? 'mm' : '索引'}` : '法向力已记录；接触中心缺失';
        normalCanvas.setAttribute('aria-label', `${channelName(index)}，${normalReading.textContent}；${centerReading.textContent}；${sample.index >= 0 ? '实际采样 ' + fmt(data.sampleTime, 3) + ' 秒' : '无原生观测'}。圆面积代表法向力，不是接触面积。`);
        normalChange.classList.toggle('ke-positive', finite(data.normalChange) && data.normalChange > 0);
        normalChange.classList.toggle('ke-negative', finite(data.normalChange) && data.normalChange < 0);
        normalChange.textContent = `ΔFn ${finite(data.normalChange) && data.normalChange > 0 ? '+' : ''}${fmt(data.normalChange, 3)} N${data.normalChange === null ? ' · 未知' : ''}`;
        const tangent = fit(tangentCanvas), tc = tangent.ctx, tx = tangent.width / 2, ty = tangent.height / 2 - 3;
        const vector = glyphVector(data.tangentChange, vectorMaximum, vectorRadius);
        const magnitude = data.tangentChange ? Math.hypot(...data.tangentChange) : null;
        tangentReading.textContent = `|ΔFt| ${fmt(magnitude, 3)} N${magnitude === null ? ' · 未知' : ''}`;
        tangentNote.textContent = componentBasis ? '记录的切向分量；与上图坐标方向未核实' : '切向坐标基未声明；不绘制方向';
        if (!componentBasis || !vector) hatch(tc, tx - vectorRadius, ty - vectorRadius, vectorRadius * 2, vectorRadius * 2);
        if (componentBasis) {
          tc.strokeStyle = '#c0cbd5'; tc.lineWidth = .8; tc.beginPath(); tc.moveTo(tx - vectorRadius - 5, ty); tc.lineTo(tx + vectorRadius + 5, ty); tc.moveTo(tx, ty + vectorRadius + 5); tc.lineTo(tx, ty - vectorRadius - 5); tc.stroke();
          text(tc, String(glyph.tangent_axis_labels?.[0] || 't₁'), tx + vectorRadius + 7, ty - 7, 'center', '#7c8e9b', 9);
          text(tc, String(glyph.tangent_axis_labels?.[1] || 't₂'), tx + 9, ty - vectorRadius - 7, 'center', '#7c8e9b', 9);
          if (finite(vectorMaximum) && vectorMaximum > 0) { tc.strokeStyle = '#aebfc9'; tc.setLineDash([2, 3]); tc.beginPath(); tc.arc(tx, ty, vectorRadius, 0, Math.PI * 2); tc.stroke(); tc.setLineDash([]); }
          if (vector && vector.magnitude <= vectorMaximum + 1e-7) {
            tc.strokeStyle = '#16869a'; tc.fillStyle = '#16869a'; tc.lineWidth = 1.8;
            if (vector.magnitude === 0) { tc.beginPath(); tc.arc(tx, ty, 2, 0, Math.PI * 2); tc.fill(); }
            else {
              const ex = tx + vector.dx, ey = ty + vector.dy, angle = Math.atan2(vector.dy, vector.dx), arrow = Math.min(5, Math.hypot(vector.dx, vector.dy) * .6);
              tc.beginPath(); tc.moveTo(tx, ty); tc.lineTo(ex, ey); tc.stroke();
              tc.beginPath(); tc.moveTo(ex, ey); tc.lineTo(ex - arrow * Math.cos(angle - .45), ey - arrow * Math.sin(angle - .45)); tc.lineTo(ex - arrow * Math.cos(angle + .45), ey - arrow * Math.sin(angle + .45)); tc.closePath(); tc.fill();
            }
          } else text(tc, vector ? '标尺不一致' : '窗口数据未知', tx, ty + 12, 'center', '#7c8b98', 9);
          text(tc, finite(vectorMaximum) && vectorMaximum === 0 ? '已观测变化均为 0 N' : `参考圆 ${fmt(vectorMaximum, 3)} N`, tx, tangent.height - 7, 'center', '#82939f', 8);
        } else text(tc, '方向未知', tx, ty, 'center', '#7c8b98', 10);
        tangentCanvas.setAttribute('aria-label', `${channelName(index)}，${tangentReading.textContent}；${tangentNote.textContent}。箭头为切向合力变化，不是位移。`);
      });
    }
    function renderSpatial() {
      if (spatialMode === 'glyph') { renderGlyphs(); return; }
      let frame = leftIndex(spatialTimes, currentTime);
      const native = observedIndex(times, currentTime, nominal, source.gap_after);
      if (native < 0 || !spatialTimes.length || currentTime > (times[times.length - 1] ?? -1) + 1e-8) frame = -1;
      if (frame >= 0) {
        const original = spatial.source_sample_index?.[frame];
        if (Number.isInteger(original) && original <= native) {
          for (let i = original; i < native; i++) if (source.gap_after?.[i] === true) { frame = -1; break; }
        }
      }
      spatialTimeLabel.textContent = frame >= 0 ? `采样 ${fmt(spatialTimes[frame], 3)} s` : '无可用快照';
      spatialLag.textContent = frame >= 0 ? `距游标 ${fmt(Math.max(0, currentTime - spatialTimes[frame]), 3)} s` : '不补零、不沿缺口延续';
      gridViews.forEach(({view, value, index}) => {
        const channelFrame = native >= 0 && finite(forces[native]?.[index]) ? frame : -1;
        const {ctx, width, height} = fit(view), values = channelFrame >= 0 ? spatial.normal_taxel_force_n?.[channelFrame]?.[index] : null;
        const shape = channels[index]?.grid_shape, rows = Array.isArray(values) ? values.length : shape?.[0], columns = Array.isArray(values?.[0]) ? values[0].length : shape?.[1];
        if (!Number.isInteger(rows) || !Number.isInteger(columns) || rows <= 0 || columns <= 0) { hatch(ctx, 10, 8, width - 20, height - 20); text(ctx, '阵列数据缺失', width / 2, height / 2, 'center', '#697889', 12); value.textContent = '无观测'; return; }
        const cell = Math.min((width - 24) / columns, (height - 22) / rows), left = (width - columns * cell) / 2, top = (height - rows * cell - 12) / 2;
        let sum = 0, known = 0;
        for (let r = 0; r < rows; r++) for (let c = 0; c < columns; c++) {
          const force = values?.[r]?.[c], x = left + c * cell + .4, y = top + r * cell + .4;
          if (finite(force) && force >= 0) { ctx.fillStyle = heatColor(force); ctx.fillRect(x, y, cell - .8, cell - .8); sum += force; known++; }
          else hatch(ctx, x, y, cell - .8, cell - .8);
        }
        text(ctx, `${rows} × ${columns} · 行 0 在上`, width / 2, height - 6, 'center', '#86919d', 9);
        value.textContent = known === rows * columns ? `阵列合计 ${fmt(sum, 3)} N` : known ? `部分缺测 · ${known}/${rows * columns} 单元有效` : '无观测';
        view.setAttribute('aria-label', `${channelName(index)}，${channelFrame >= 0 ? '实际采样 ' + fmt(spatialTimes[channelFrame], 3) + ' 秒' : '当前通道没有可用观测'}，${value.textContent}。`);
      });
    }
    function render() {
      if (destroyed) return;
      timeText.textContent = fmt(currentTime, 3); slider.value = String(currentTime); slider.setAttribute('aria-valuetext', `${fmt(currentTime, 3)} 秒`); slider.style.setProperty('--ke-progress', `${currentTime / span * 100}%`);
      readingNodes.forEach(({reading}, i) => { reading.textContent = `${fmt(nativeValue(i, currentTime))} N`; });
      contactRows.forEach(marker => { marker.style.left = `${currentTime / span * 100}%`; });
      charts.forEach(renderChart); renderSpatial();
      eventButtons.forEach(({button, time}) => { button.classList.toggle('ke-event-active', Math.abs(time - currentTime) < Math.max(.025, (nominal || .01) / 2)); });
    }
    function seekVideo(time) {
      if (!video || !mappedMedia) return;
      const i = leftIndex(mediaTimes, time); if (i < 0) return;
      const target = videoTimes[i]; pendingVideoSeek = time;
      if (video.readyState < 1) return;
      if (Math.abs(video.currentTime - target) < 1e-5) { pendingVideoSeek = null; return; }
      try { programmaticSeek = true; video.currentTime = target; } catch (_) { programmaticSeek = false; }
    }
    function seek(time, syncVideo = true) { if (!finite(time)) return; currentTime = clamp(time, 0, end); render(); if (syncVideo) seekVideo(currentTime); }
    function step(direction) { const i = leftIndex(times, currentTime); seek(times[clamp(i + direction, 0, Math.max(0, times.length - 1))] ?? clamp(currentTime + direction * .1, 0, end)); }
    function updatePlay() { play.textContent = playing ? 'Ⅱ 暂停' : '▶ 播放'; play.setAttribute('aria-pressed', String(playing)); }
    function pause() { playing = false; previousTick = null; if (animation !== null) global.cancelAnimationFrame(animation); animation = null; if (mappedMedia && video) video.pause(); updatePlay(); }
    function tick(timestamp) {
      if (!playing || destroyed) return;
      if (previousTick !== null) seek(currentTime + Math.min(.25, (timestamp - previousTick) / 1000), false);
      previousTick = timestamp;
      if (currentTime >= end) { pause(); return; }
      animation = global.requestAnimationFrame(tick);
    }
    on(play, 'click', () => {
      if (playing) { pause(); return; }
      if (currentTime >= end) seek(0);
      if (video && mappedMedia) { seekVideo(currentTime); const promise = video.play(); if (promise?.catch) promise.catch(() => { playing = false; updatePlay(); }); }
      else { playing = true; updatePlay(); previousTick = null; animation = global.requestAnimationFrame(tick); }
    });
    on(previous, 'click', () => { pause(); step(-1); }); on(next, 'click', () => { pause(); step(1); });
    on(slider, 'input', () => { pause(); seek(Number(slider.value)); });
    on(slider, 'keydown', event => { if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return; event.preventDefault(); pause(); if (event.key === 'Home') seek(0); else if (event.key === 'End') seek(end); else step(event.key === 'ArrowLeft' ? -1 : 1); });
    let resizeFrame = null;
    const resize = () => { if (resizeFrame !== null) global.cancelAnimationFrame(resizeFrame); resizeFrame = global.requestAnimationFrame(() => { resizeFrame = null; rebuildCharts(); render(); }); };
    let observer = null;
    if (typeof global.ResizeObserver === 'function') { observer = new global.ResizeObserver(resize); observer.observe(root); }
    else on(global, 'resize', resize);
    updatePlay(); render();
    return {destroy() {
      if (destroyed) return;
      pause(); if (video) video.pause(); destroyed = true; cleanups.forEach(fn => fn()); if (observer) observer.disconnect();
      if (resizeFrame !== null) global.cancelAnimationFrame(resizeFrame);
      objectURLs.forEach(url => global.URL.revokeObjectURL(url)); objectURLs.clear(); root.remove();
    }};
  }
  global.KaiEvaluationDashboard = Object.freeze({mount, normalizeBulbAngleSeries});
  // Pure helpers are available in Node for source-data and gap regression tests.
  if (typeof module !== 'undefined' && module.exports) module.exports = {mount, leftIndex, observedIndex, knownSegments, relativeURL, glyphSample, glyphRadius, glyphBounds, glyphPoint, glyphVector, goalReferences, chartRange, referenceNumber, referenceText, normalizeBulbAngleSeries, displayUnit};
})(typeof window !== 'undefined' ? window : globalThis);
