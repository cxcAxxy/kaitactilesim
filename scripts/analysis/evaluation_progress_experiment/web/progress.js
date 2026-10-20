/* Progress is an observed task quantity, never a normalized expert clock. */
(function () {
  'use strict';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = (value, digits=2) => finite(value) ? value.toLocaleString('zh-CN',{minimumFractionDigits:digits,maximumFractionDigits:digits}) : '—';
  const pct = value => finite(value) ? `${fmt(value * 100,1)}%` : '未知';
  const binFor = (progress, edges) => {
    if (!finite(progress) || progress < 0 || progress > 1) return -1;
    for(let i=0;i<edges.length-1;i++) if(progress < edges[i+1] || i===edges.length-2) return i;
    return -1;
  };
  const videoTime = (media,time) => {
    if(media?.sync_status!=='mapped' || !Array.isArray(media.time_s) || !Array.isArray(media.video_time_s)
      || media.time_s.length!==media.video_time_s.length || !media.time_s.length || !finite(time)) return null;
    if(time < media.time_s[0]-1e-8 || time > media.time_s.at(-1)+1e-8) return null;
    let result=null;
    media.time_s.forEach((t,i)=>{if(t<=time+1e-8 && finite(media.video_time_s[i])) result=media.video_time_s[i];});
    return result;
  };
  const safeMedia = value => typeof value==='string' && /^media\/[0-9]+\.mp4$/.test(value);
  const progressWidth = value => finite(value) ? `${value*100}%` : null;
  const unchangedDuration = row => row.status==='observed' && row.known_transition_duration_s>0 ? row.unchanged_progress_duration_s : null;
  const cadenceHz = points => {
    const dt=points.slice(1).map((p,i)=>p.time_s-points[i].time_s).filter(v=>finite(v)&&v>0).sort((a,b)=>a-b);
    if(!dt.length)return null;
    const middle=Math.floor(dt.length/2),median=dt.length%2?dt[middle]:(dt[middle-1]+dt[middle])/2;
    return 1/median;
  };
  const helpers = {binFor,videoTime,safeMedia,cadenceHz,progressWidth,unchangedDuration};
  if(typeof module!=='undefined' && module.exports) module.exports=helpers;
  if(typeof document==='undefined') return;
  const data=JSON.parse(document.getElementById('progress-data').textContent);
  const el=(tag,parent,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;parent.appendChild(n);return n;};
  const policies=data.episodes.filter(e=>e.role==='policy'), experts=data.episodes.filter(e=>e.role==='expert_reference');
  let bin=0,signal='board_normal',controllers=[];
  const positions=new Map();
  const policySelect=document.getElementById('policy-select'),expertSelect=document.getElementById('expert-select');
  const shortName=episode=>{
    const source=episode.source_report||'';
    const match=source.match(/([^/]+)\/report\.json$/);
    return match ? match[1] : episode.id;
  };
  for(const [select,episodes] of [[policySelect,policies],[expertSelect,experts]]){
    episodes.forEach((e,i)=>{const o=el('option',select,shortName(e));o.value=String(i);});
    if(!episodes.length){el('option',select,'没有可用参考');select.disabled=true;}
  }
  // Start with the originally requested checkpoint when it is represented.
  const requested=policies.findIndex(e=>e.id.includes('step24999'));
  if(requested>=0)policySelect.value=String(requested);
  const baseline=document.getElementById('baseline-link');
  if(/^\.\.\/[^/]+\/index\.html$/.test(data.baseline_url)) baseline.href=data.baseline_url;
  const overview=document.getElementById('overview');
  const card=(number,label,note)=>{const a=el('article',overview);el('strong',a,number);el('span',a,label);el('small',a,note);};
  card(`${policies.length} 条`,'模型白板轨迹 · H50',`原记录：成功 ${policies.filter(e=>e.outcome==='pass').length} / 失败 ${policies.filter(e=>e.outcome==='fail').length} / 未知 ${policies.filter(e=>e.outcome==='unknown').length}`);
  card(`${experts.length} 条`,'专家示范 · 案例参照','不估计专家总体成功率，不建立分布合格带');
  card('稳定版保留','独立入口，随时切回','旧代码和报告未覆盖；校验命令见 rollback.json');
  const panes=document.getElementById('comparison-panes'),buttons=document.getElementById('bin-buttons');
  const selected=()=>[policies[Number(policySelect.value)],experts[Number(expertSelect.value)]].filter(Boolean);
  function renderButtons(){
    buttons.replaceChildren();
    const episodes=selected(),edges=episodes[0]?.edges||[0,.2,.4,.6,.8,1];
    for(let i=0;i<edges.length-1;i++){
      const button=el('button',buttons);button.type='button';button.dataset.bin=String(i);button.setAttribute('aria-pressed',String(i===bin));
      el('strong',button,`${Math.round(edges[i]*100)}–${Math.round(edges[i+1]*100)}%`);
      for(const e of episodes)el('small',button,`${e.role==='policy'?'模型':'专家'} · ${e.bins[i].status==='observed'?'已观测':'未观测'}`);
      button.onclick=()=>{bin=i;render();};
    }
    document.getElementById('bin-caption').textContent=`当前 ${Math.round(edges[bin]*100)}–${Math.round(edges[bin+1]*100)}%`;
  }
  function mount(episode,scale){
    const expert=episode.role==='expert_reference',color=expert?'#6975b7':'#07838a';
    const pane=el('article',panes,undefined,`episode-pane ${expert?'expert':'policy'}`);pane.dataset.episodeId=episode.id;
    const head=el('div',pane,undefined,'pane-head');el('span',head,expert?'EXPERT / 专家示范参照':'POLICY / H50 模型','role');
    const outcome=episode.outcome==='pass'?'成功':episode.outcome==='fail'?'失败':'未知';el('span',head,`原记录 · ${outcome}`,`outcome ${episode.outcome}`);
    el('h3',pane,shortName(episode));el('p',pane,episode.id,'identity');
    const readout=el('div',pane,undefined,'progress-readout');el('span',readout,'终点清洁进度');el('strong',readout,pct(episode.final_progress));
    const track=el('div',pane,undefined,'progress-track');
    const width=progressWidth(episode.final_progress);
    if(width===null){track.classList.add('unknown');track.setAttribute('aria-label','终点进度未知，不是0%');}
    else el('span',track).style.width=width;
    el('p',pane,`自身记录时长 ${fmt(episode.duration_s)} s · 进度缺测 ${episode.missing_progress_samples} 点 · 间隔缺口 ${fmt(episode.gap_duration_s)} s`,'native-meta');
    el('p',pane,`记录间隔中位数对应约 ${fmt(cadenceHz(episode.points),1)} Hz（不等于传感器带宽）。`+(expert?'力与状态有积分前/后采样相位差。只作同记录关联，不认证瞬时同相位。':'手指力只保留时间点精确匹配的样本，不插值补齐。'),'evidence-warning');
    const descriptor=episode.signal_descriptors[signal];
    el('div',pane,`${descriptor?.label||signal} × 清洁进度`,'chart-title');
    el('p',pane,'横轴：记录清洁比例 0–100%　纵轴：N。点按原顺序连接，缺口断开。点击曲线可定位自身时刻。','chart-subtitle');
    const canvas=el('canvas',pane,undefined,'chart');canvas.setAttribute('aria-label',`${expert?'专家':'模型'} 接触力与清洁进度图`);
    const chartReadout=el('div',pane,'点击曲线查看原生样本。','chart-readout');chartReadout.setAttribute('aria-live','polite');
    const row=episode.bins[bin],stats=row.signals[signal]||{},empty=row.status!=='observed';
    const statsRoot=el('div',pane,undefined,'stats');
    for(const [label,value] of [['区间停留',`${fmt(empty?null:row.observed_duration_s)} s`],['均值 / P95',`${fmt(stats.mean_n)} / ${fmt(stats.p95_n)} N`],['采样峰值',`${fmt(stats.sampled_peak_n)} N`],['进度未变化时长',`${fmt(unchangedDuration(row))} s`],['载荷时间积分',`${fmt(stats.load_integral_ns)} N·s`],['力数据覆盖',pct(stats.coverage_fraction)]]){
      const cell=el('div',statsRoot,undefined,'stat');el('span',cell,label);el('strong',cell,value);
    }
    el('p',pane,empty?'这条轨迹没有该进度区间的观测。保留空白，不把失败终点拉伸为 100%，也不据此填成零载荷。':
      `该区间有 ${row.sample_count} 个记录样本，${row.visit_count} 段连续访问；相邻有效区间累计进度变化 ${finite(row.progress_delta)?`${fmt(row.progress_delta*100)} 个百分点`:'未知'}。这是同段记录的载荷与进展，不是力引起进展的证明。`,'bin-result');
    el('p',pane,`可评价进度变化的区间合计 ${fmt(empty?null:row.known_transition_duration_s)} s。进度差归入起点区间，跨区间不插值；载荷积分不是能耗。`,'fine');
    const seekRow=el('div',pane,undefined,'seek-row');const seek=el('button',seekRow,'跳至区间首个观测');seek.type='button';seek.disabled=!finite(row.first_time_s);
    el('span',seekRow,`首个观测：${fmt(row.first_time_s)} s（自身时钟）`);
    let video=null;
    if(episode.media?.status==='available' && safeMedia(episode.media.url)){
      video=el('video',pane);video.src=episode.media.url;video.controls=true;video.preload='metadata';video.playsInline=true;
    }else el('div',pane,'没有经验证的视频映射。保留原生记录查看，不猜测专家视频对应时间。','no-video');
    const native=el('div',pane,undefined,'native-control'),slider=el('input',native);slider.type='range';slider.min='0';slider.max=String(Math.max(0,episode.points.length-1));slider.step='1';slider.value='0';slider.disabled=!episode.points.length;slider.setAttribute('aria-label',`${expert?'专家':'模型'} 独立原生记录游标`);
    const clock=el('output',native,'0.00 s');
    const details=el('details',pane);el('summary',details,'查看采样、兼容性与来源限制');el('pre',details,JSON.stringify({metadata:episode.metadata,warnings:episode.warnings,semantics:episode.semantics},null,2),'metadata');
    let current=Math.min(positions.get(episode.id)||0,Math.max(0,episode.points.length-1));
    function draw(){
      const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
      canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);
      const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
      const w=rect.width,h=rect.height,left=48,right=14,top=20,bottom=34,pw=w-left-right,ph=h-top-bottom;
      const x=p=>left+p*pw,y=value=>top+(1-value/scale)*ph;
      ctx.fillStyle='#f0f6f5';ctx.fillRect(x(row.low),top,(row.high-row.low)*pw,ph);
      ctx.font='11px sans-serif';ctx.textAlign='right';ctx.fillStyle='#7d8f96';ctx.strokeStyle='#e4ecee';ctx.lineWidth=1;
      for(let i=0;i<=4;i++){const value=scale*i/4,yy=y(value);ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(w-right,yy);ctx.stroke();ctx.fillText(fmt(value,1),left-7,yy+4);}
      ctx.textAlign='center';for(let i=0;i<=5;i++)ctx.fillText(`${i*20}%`,x(i/5),h-10);
      ctx.fillText('N',left-24,12);ctx.strokeStyle=color;ctx.lineWidth=1.3;ctx.globalAlpha=.48;
      let previous=null;
      for(const point of episode.points){
        const value=point.signals[signal];
        if(!finite(point.progress)||!finite(value)){previous=null;continue;}
        if(previous&&!previous.gap_after){ctx.beginPath();ctx.moveTo(x(previous.progress),y(previous.signals[signal]));ctx.lineTo(x(point.progress),y(value));ctx.stroke();}
        ctx.fillStyle=color;ctx.fillRect(x(point.progress)-.8,y(value)-.8,1.6,1.6);previous=point;
      }
      ctx.globalAlpha=1;
      const point=episode.points[current];
      if(point&&finite(point.progress)&&finite(point.signals[signal])){ctx.beginPath();ctx.arc(x(point.progress),y(point.signals[signal]),4.5,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke();}
      canvas._geometry={x,y};
    }
    function setPoint(index,seekVideo=true){
      current=index;const point=episode.points[index];if(!point)return;positions.set(episode.id,index);
      slider.value=String(index);clock.textContent=`${fmt(point.time_s)} s`;
      chartReadout.textContent=`自身 ${fmt(point.time_s,3)} s · 清洁 ${pct(point.progress)} · ${fmt(point.signals[signal],3)} N`;
      if(video&&seekVideo){const target=videoTime(episode.media,point.time_s);if(target!==null){video.pause();video.currentTime=target;}else chartReadout.textContent+=' · 此样本无视频映射';}
      draw();
    }
    canvas.onclick=event=>{
      const rect=canvas.getBoundingClientRect(),mx=event.clientX-rect.left,my=event.clientY-rect.top;
      let distance=Infinity,index=-1;
      episode.points.forEach((point,i)=>{if(!finite(point.progress)||!finite(point.signals[signal]))return;const d=(canvas._geometry.x(point.progress)-mx)**2+(canvas._geometry.y(point.signals[signal])-my)**2;if(d<distance){distance=d;index=i;}});
      if(index>=0)setPoint(index);
    };
    slider.oninput=()=>setPoint(Number(slider.value));
    seek.onclick=()=>{const index=episode.points.findIndex(p=>p.time_s===row.first_time_s);if(index>=0)setPoint(index);};
    // Video playback only updates this episode's record cursor via its explicit map.
    if(video)video.ontimeupdate=()=>{
      const media=episode.media;if(media.sync_status!=='mapped'||!Array.isArray(media.video_time_s)||!Array.isArray(media.time_s))return;
      let t=null;media.video_time_s.forEach((value,i)=>{if(value<=video.currentTime+1e-8)t=media.time_s[i];});
      if(!finite(t))return;let index=-1;episode.points.forEach((p,i)=>{if(p.time_s<=t+1e-8)index=i;});
      if(index>=0&&index!==current&&!video.paused)setPoint(index,false);
    };
    if(video)video.addEventListener('loadedmetadata',()=>setPoint(current),{once:true});
    const observer=new ResizeObserver(()=>draw());observer.observe(canvas);draw();setPoint(current,false);
    return {destroy(){observer.disconnect();if(video){video.pause();video.removeAttribute('src');video.load();}},draw};
  }
  function render(){
    controllers.forEach(controller=>controller.destroy());controllers=[];panes.replaceChildren();renderButtons();
    const episodes=selected();let maximum=0;
    episodes.forEach(e=>e.points.forEach(p=>{if(finite(p.signals[signal]))maximum=Math.max(maximum,p.signals[signal]);}));
    const scale=maximum>0?maximum*1.08:1;
    episodes.forEach(e=>controllers.push(mount(e,scale)));
  }
  policySelect.onchange=render;expertSelect.onchange=render;
  document.getElementById('signal-select').onchange=event=>{signal=event.target.value;render();};
  render();
})();
