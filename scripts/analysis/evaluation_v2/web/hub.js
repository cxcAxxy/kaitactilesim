/* Local, read-only hub. Trial clocks are deliberately independent. */
(function (global) {
  'use strict';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  function trialFolder(path) {
    return typeof path === 'string' && /^[a-zA-Z0-9_-]+\/report\.html$/.test(path) ? path.slice(0, -11) : null;
  }
  function sharedScales(visuals) {
    const maximum = values => { const known = values.filter(value => finite(value) && value >= 0); return known.length ? Math.max(...known) : null; };
    return {normal_force_max_n: maximum(visuals.map(v => v.contact_glyph?.force_scale_max_n)),
      taxel_force_max_n: maximum(visuals.map(v => v.spatial_frames?.scale_max_n)),
      tangent_change_max_n: maximum(visuals.map(v => v.contact_glyph?.tangent_change_scale_max_n))};
  }
  function mount(host, data) {
    const doc = host.ownerDocument, trials = Array.isArray(data.trials) ? data.trials : [];
    host.replaceChildren();
    const node = (tag, parent, text, cls) => { const el = doc.createElement(tag); if (text != null) el.textContent = text; if (cls) el.className = cls; parent?.appendChild(el); return el; };
    const status = value => ({pass:'通过', fail:'未通过', unknown:'无法判定', missing:'缺失', legacy_partial:'旧记录 · 部分', available:'可用', partial:'部分', unavailable:'不可比较', not_configured:'未配置', incompatible:'不兼容'})[value] || value || '未知';
    const badge = (parent, value) => node('span', parent, status(value), `hub-badge hub-${['pass','fail'].includes(value) ? value : 'unknown'}`);
    const bar = node('section', host, null, 'hub-stats');
    [[trials.length,'实际轨迹'], [data.validation?.task_count_with_trials || 0,'有轨迹的任务'], [(data.task_catalog || []).length,'任务契约'], [status(data.reference_status),'专家参照']].forEach(([value,label]) => {
      const card = node('article', bar); node('strong', card, String(value)); node('span', card, label);
    });
    node('p', host, '这是所选归档的描述性检查，不是统一性能排名。记录结果 ≠ 规则复核；缺少证据 ≠ 安全通过。单元测试与契约数量不计作仿真实验。', 'hub-notice');
    if (Number.isInteger(data.horizon_selection?.required_prediction_horizon)) {
      const cohort = node('p',host,null,'hub-notice hub-horizon-notice'), selection = data.horizon_selection;
      node('strong',cohort,`H${selection.required_prediction_horizon}-only · `);
      node('span',cohort,`输入 ${selection.input_count} 条，纳入 ${selection.included_count} 条，排除 ${selection.excluded_count} 条。依据明确的预测 horizon，非执行步数或采样率；排除项不参与任何本页统计。 `);
      const audit = node('a',cohort,'查看纳入 / 排除清单'); audit.href = 'horizon_selection.json';
    }
    const coverage = node('section', host, null, 'hub-section'); node('h2', coverage, '任务覆盖与证据');
    const cards = node('div', coverage, null, 'hub-coverage');
    (data.validation?.tasks || []).forEach(row => {
      const card = node('article', cards, null, 'hub-task-card'); node('h3', card, row.task);
      node('p', card, `${row.trial_count} 条轨迹 · ${row.contract_available ? '已定义契约' : '无专用契约'}`);
      const counts = row.recorded_outcome_counts || {}; node('p', card, `记录：通过 ${counts.pass || 0} / 未通过 ${counts.fail || 0} / 未知 ${counts.unknown || 0}`);
      const track = node('div', card, null, 'hub-coverage-track');
      [['pass',counts.pass],['fail',counts.fail],['unknown',counts.unknown]].forEach(([label,count]) => { const segment = node('span', track, null, `hub-segment-${label}`); segment.style.width = `${100 * (count || 0) / (row.trial_count || 1)}%`; });
      const detail = node('details', card); node('summary', detail, '证据缺口');
      const list = node('ul', detail); (row.coverage_gaps || []).forEach(gap => node('li', list, gap));
    });
    const library = node('section', host, null, 'hub-section'); node('h2', library, '轨迹库');
    const filters = node('div', library, null, 'hub-filters');
    const select = (label, values) => { const group = node('label', filters, label); const el = node('select', group); values.forEach(([value,text]) => { const option = node('option', el, text); option.value = value; }); return el; };
    const taskFilter = select('任务', [['','全部任务'], ...[...new Set(trials.map(t => t.task))].sort().map(t => [t,t])]);
    const resultFilter = select('记录结果', [['','全部结果'],['pass','通过'],['fail','未通过'],['unknown','无法判定']]);
    const searchGroup = node('label', filters, '编号、checkpoint 或路径'); const search = node('input', searchGroup); search.type = 'search'; search.placeholder = '搜索归档';
    const counter = node('p', library, '', 'hub-caption');
    const scroll = node('div', library, null, 'hub-table-scroll'); const table = node('table', scroll);
    const head = node('tr', node('thead', table)); ['并排','任务 / 试验','记录结果','规则复核','证据','报告'].forEach(t => node('th', head, t));
    const body = node('tbody', table), selected = new Set();
    const comparison = node('section', host, null, 'hub-section'); node('h2', comparison, '并排检查');
    node('p', comparison, '最多选择两条轨迹。两侧独立播放，不按时间对齐。统一色标和力圆量程只帮助肉眼比较，不代表任务、传感器或条件已满足统计可比性。', 'hub-caption');
    const actions = node('div', comparison, null, 'hub-actions');
    const compare = node('button', actions, '载入所选轨迹', 'hub-primary'); compare.type = 'button';
    const clear = node('button', actions, '清空'); clear.type = 'button';
    const feedback = node('p', comparison, '尚未选择轨迹', 'hub-caption'); feedback.setAttribute('role','status');
    const panes = node('div', comparison, null, 'hub-panes'); let controllers = [], generation = 0, abort = null;
    function destroyPanes() { controllers.forEach(c => c.destroy()); controllers = []; panes.replaceChildren(); }
    function selectionChanged() { compare.disabled = selected.size === 0; feedback.textContent = `已选 ${selected.size} / 2 条轨迹`; }
    function renderRows() {
      body.replaceChildren(); let count = 0;
      trials.forEach((trial,index) => {
        if (taskFilter.value && trial.task !== taskFilter.value || resultFilter.value && trial.outcome?.status !== resultFilter.value || search.value && !JSON.stringify([trial.task,trial.trial_id,trial.source,trial.cohort]).toLowerCase().includes(search.value.toLowerCase())) return;
        count++; const row = node('tr',body), folder = trialFolder(trial.report_path);
        const check = node('input',node('td',row)); check.type = 'checkbox'; check.checked = selected.has(index); check.disabled = !folder; check.setAttribute('aria-label',`选择 ${trial.task} ${trial.trial_id}`);
        check.addEventListener('change', () => { if (check.checked && selected.size >= 2) { check.checked = false; feedback.textContent = '最多选择两条；请先取消一条。'; return; } if (check.checked) selected.add(index); else selected.delete(index); selectionChanged(); });
        const identity = node('td',row); node('strong',identity,trial.task); node('small',identity,trial.trial_id); node('small',identity,`seed ${trial.seed ?? '未知'} · 预测 H${trial.prediction_horizon ?? '未知'}`); identity.title = trial.source || '';
        const recorded = node('td',row); badge(recorded, trial.outcome?.status);
        if (trial.outcome?.diagnostic_only === true || trial.outcome?.formal_metrics_valid === false) node('small',recorded,'诊断运行 · 非正式性能');
        else if (trial.outcome?.formal_metrics_valid !== true) node('small',recorded,'正式性未核实');
        badge(node('td',row), trial.task_review?.review?.status);
        node('td',row,status(trial.evidence_v2?.status)); const linkCell = node('td',row);
        if (folder) { const link = node('a',linkCell,'打开 ↗'); link.href = trial.report_path; }
      }); counter.textContent = `显示 ${count} / ${trials.length} 条。通过/未通过来自原记录；未知复核不覆盖原结果。`;
    }
    [taskFilter,resultFilter].forEach(el => el.addEventListener('change',renderRows)); search.addEventListener('input',renderRows);
    async function loadSelected() {
      const current = ++generation; abort?.abort(); abort = new AbortController(); destroyPanes();
      const picked = [...selected]; if (!picked.length) { selectionChanged(); return; }
      feedback.textContent = '正在载入原生曲线与报告…';
      try {
        const loaded = await Promise.all(picked.map(async index => {
          const folder = trialFolder(trials[index].report_path); if (!folder) throw Error('非法报告路径');
          const read = async name => { const response = await global.fetch(`${folder}${name}`, {signal:abort.signal}); if (!response.ok) throw Error(`读取失败 ${response.status}`); return response.json(); };
          const [report,visualization] = await Promise.all([read('report.json'),read('visualization.json')]);
          if (visualization.media?.url) {
            const url = visualization.media.url;
            if (!/^media\/[a-zA-Z0-9_.-]+$/.test(url)) throw Error('非法媒体路径');
            visualization.media = {...visualization.media,url:folder + url};
          }
          return {report,visualization,folder};
        }));
        if (current !== generation) return;
        const scales = sharedScales(loaded.map(item => item.visualization));
        loaded.forEach(item => { const panel = node('div',panes,null,'hub-pane'); controllers.push(global.KaiEvaluationDashboard.mount(panel,{...item,scales,auditURL:item.folder+'audit.html'})); });
        const scaleText = value => finite(value) ? value.toFixed(3) + ' N' : '未知';
        feedback.textContent = `已载入 ${loaded.length} 条 · 各自独立时钟 · 共用 Fn 参考 ${scaleText(scales.normal_force_max_n)}，单元色标 ${scaleText(scales.taxel_force_max_n)}。曲线 Y 轴仍各自标注，请按数值读取。`;
      } catch (error) { if (current !== generation || error.name === 'AbortError') return; destroyPanes(); feedback.textContent = `无法载入：${error.message}。请通过本地 HTTP 服务打开，而非 file://。`; }
    }
    compare.addEventListener('click',loadSelected);
    clear.addEventListener('click', () => { ++generation; abort?.abort(); selected.clear(); destroyPanes(); renderRows(); selectionChanged(); });
    node('p',host,'本地只读站点 · 不启动仿真、不更改数据 · 完整来源与限制见各报告 JSON', 'hub-footer');
    renderRows(); selectionChanged();
    return {destroy() { ++generation; abort?.abort(); destroyPanes(); host.replaceChildren(); }};
  }
  global.KaiEvaluationHub = Object.freeze({mount,trialFolder,sharedScales});
  if (typeof module !== 'undefined' && module.exports) module.exports = global.KaiEvaluationHub;
  if (global.document) { const host = global.document.getElementById('hub-content'), source = global.document.getElementById('hub-data'); if (host && source) { try { mount(host,JSON.parse(source.textContent)); } catch (error) { host.textContent = `索引载入失败：${error.message}`; } } }
})(typeof window !== 'undefined' ? window : globalThis);
