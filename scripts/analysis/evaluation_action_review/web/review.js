/* Native-recording evidence only. No cross-trajectory clock alignment or causal labels. */
(function () {
  'use strict';
  const finite=v=>typeof v==='number'&&Number.isFinite(v);
  const fmt=(v,n=2)=>finite(v)?v.toLocaleString('zh-CN',{minimumFractionDigits:n,maximumFractionDigits:n}):'未知';
  const pct=v=>finite(v)?`${fmt(v*100,3)}%`:'未知';
  function pastIndex(values,t) {
    if(!Array.isArray(values)||!values.length||!finite(t)||t<values[0]-1e-8)return -1;
    let lo=0,hi=values.length;
    while(lo<hi){const mid=(lo+hi)>>1;if(values[mid]<=t+1e-8)lo=mid+1;else hi=mid;}
    return lo-1;
  }
  function mappedTime(media,t,inverse=false){
    if(media?.sync_status!=='mapped')return null;
    const source=inverse?media.video_time_s:media.time_s,target=inverse?media.time_s:media.video_time_s;
    if(!Array.isArray(source)||!Array.isArray(target)||source.length!==target.length||!source.length||!finite(t)||t>source.at(-1)+1e-8)return null;
    const i=pastIndex(source,t);return i<0||!finite(target[i])?null:target[i];
  }
  const safeMedia=v=>typeof v==='string'&&/^media\/[0-9]+\.mp4$/.test(v);
  const segmentPosition=(t,segment)=>!segment||!finite(t)?'未知':t<segment.start_s?'前置上下文':t>segment.end_s?'后置上下文':'选段内';
  function sampleLocation(series,t){
    const index=pastIndex(series.map(p=>p.time_s),t);
    if(index<0||t>series.at(-1).time_s+1e-8)return {index:-1,gap:true};
    return {index,gap:Boolean(series[index].gap_after&&t>series[index].time_s+1e-8)};
  }
  const eventTitle=e=>!e?'全程记录 · 未选定片段':e.kind==='growth'?'这段动作中，清洁读数在增长':
    e.contact_stats?.known_sample_count===e.contact_stats?.sample_count&&e.contact_stats?.min_n>0?
    '清洁读数不变，但每个板面力样本仍为正':'这段动作中，清洁读数没有变化';
  function cropStyle(views,id){
    const c=views?.crops?.find(c=>c.id===id);
    if(!c||![c.x,c.y,c.width,c.height,views.source_width,views.source_height].every(finite)||c.width<=0||c.height<=0||c.x<0||c.y<0||c.x+c.width>views.source_width||c.y+c.height>views.source_height)return null;
    return {ratio:`${c.width} / ${c.height}`,width:`${views.source_width/c.width*100}%`,height:`${views.source_height/c.height*100}%`,left:`${-c.x/c.width*100}%`,top:`${-c.y/c.height*100}%`};
  }
  const helpers={finite,pastIndex,mappedTime,safeMedia,eventTitle,cropStyle,sampleLocation,segmentPosition};
  if(typeof module!=='undefined'&&module.exports)module.exports=helpers;
  if(typeof document==='undefined')return;
  const data=JSON.parse(document.getElementById('review-data').textContent);
  const $=id=>document.getElementById(id);
  const el=(tag,parent,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;parent.appendChild(n);return n;};
  const video=$('main-video'),contrastVideo=$('contrast-video'),selectionVideo=$('selection-video');
  let ep,current=0,event=null,contrast=null,selection=null,times=[],playing=false,currentGap=false,cursorTime=0;
  const policies=data.episodes;
  policies.forEach((p,i)=>{const o=el('option',$('episode-select'),p.display_name);o.value=String(i);});
  const original=policies.findIndex(p=>p.id.includes('step24999'));
  if(original>=0)$('episode-select').value=String(original);
  if(/^\.\.\/[^/]+\/index\.html$/.test(data.baseline_url))$('baseline-link').href=data.baseline_url;

  function applyView(target,viewport){
    const style=cropStyle(ep.media?.views,$('view-select').value);
    viewport.classList.toggle('crop',Boolean(style));
    viewport.style.aspectRatio=style?.ratio||'16 / 9';
    for(const key of ['width','height','left','top'])target.style[key]=style?.[key]||'';
    target.style.right=style?'auto':'';target.style.bottom=style?'auto':'';
    // Native controls would be clipped in a crop. Custom playback lives outside.
    target.controls=target===contrastVideo&&!style;
  }
  function row(parent,label,value){const r=el('div',parent,undefined,'fact-row');el('span',r,label);el('strong',r,value);}
  function facts(parent,selected){
    parent.replaceChildren();
    if(!selected){el('p',parent,'在全程时间条或候选列表选一段，查看起止读数与视频。','muted');return;}
    row(parent,'清洁读数',`${pct(selected.progress_start)} → ${pct(selected.progress_end)}`);
    row(parent,'持续 / 占全程记录',`${fmt(selected.duration_s)} s / ${fmt(selected.duration_s/ep.coverage.total_duration_s*100,1)}%`);
    row(parent,'板面法向力 · 样本范围',`${fmt(selected.contact_stats?.min_n,3)} – ${fmt(selected.contact_stats?.max_n,3)} N`);
    row(parent,'五指法向力合计 · 样本范围',`${fmt(selected.hand_stats?.min_n,3)} – ${fmt(selected.hand_stats?.max_n,3)} N`);
    const motion=selected.motion_stats;
    row(parent,'实测手腕采样路径（非板擦）',finite(motion?.wrist_path_m)?`${fmt(motion.wrist_path_m*100,2)} cm`:'缺少匹配记录');
    const detail=el('details',parent,undefined,'fact-detail');el('summary',detail,'展开读数增量、采样覆盖与手腕净位移');
    row(detail,'新增清洁比例',`${fmt(selected.progress_delta*100,3)} 个百分点`);
    if(finite(motion?.wrist_displacement_m))row(detail,'手腕端点净位移',`${fmt(motion.wrist_displacement_m*100,2)} cm`);
    el('p',detail,`板面力有效样本 ${selected.contact_stats?.known_sample_count??0}/${selected.contact_stats?.sample_count??0}；手指力 ${selected.hand_stats?.known_sample_count??0}/${selected.hand_stats?.sample_count??0}。采样范围不是连续时间的峰值保证。`,'fine');
    if(motion)el('p',detail,`手腕路径仅累加有记录的相邻点，覆盖 ${fmt(motion.known_duration_s)} s；存在缺测时是部分路径，不补齐。`,'fine');
    if(selected.progress_start===0&&selected.kind==='plateau')el('p',parent,'包含 0% 状态，可能覆盖接近或准备动作，不自动判失败。','fine');
    if(selected.progress_start===1&&selected.kind==='plateau')el('p',parent,'包含 100% 状态，可能覆盖结束动作，不自动判失败。','fine');
  }
  function showEvent(selected,seek=true){
    event=selected;
    $('event-title').textContent=eventTitle(selected);
    $('event-range').textContent=selected?`${fmt(selected.start_s,3)}–${fmt(selected.end_s,3)} s · 原始记录时钟`:'任意时刻均可回看；未标记不表示正常。';
    facts($('event-facts'),selected);
    for(const button of document.querySelectorAll('.event-card'))button.setAttribute('aria-pressed',String(button.dataset.id===selected?.id));
    if(seek&&selected)setTime(selected.start_s,true);
    positionNote();
    draw();
  }
  function positionNote(){
    $('position-note').textContent=!event?'当前为全程浏览，右侧没有选段统计。':
      cursorTime<event.start_s||cursorTime>event.end_s?
      '当前在选段外 / 上下文中；右侧统计仍是所选片段，不是当前画面的读数。':
      '当前位于选段内；右侧是整段统计，下方是当前原生记录读数。';
  }
  function listEvents(){
    const list=$('event-list');list.replaceChildren();
    const filter=$('event-filter').value,events=ep.events.filter(e=>filter==='all'||e.kind===filter);
    $('candidate-count').textContent=`全轨迹共 ${ep.events.length} 段候选，当前显示 ${events.length} 段。全部保留；短增长片段可能短于一帧，建议连同上下文查看。`;
    if(!events.length)el('p',list,'此筛选没有候选。仍可通过全程游标查看所有记录。','empty');
    events.forEach(e=>{
      const b=el('button',list,undefined,`event-card ${e.kind}`);b.type='button';b.dataset.id=e.id;b.setAttribute('aria-pressed',String(event?.id===e.id));
      el('strong',b,`${e.kind==='growth'?'读数增长':'读数不变'} · ${fmt(e.start_s,2)}–${fmt(e.end_s,2)} s`);
      el('span',b,`${fmt(e.duration_s)} s · ${pct(e.progress_start)} → ${pct(e.progress_end)}`);
      el('span',b,`板面力 ${fmt(e.contact_stats?.min_n)}–${fmt(e.contact_stats?.max_n)} N`);
      b.onclick=()=>{video.pause();showEvent(e);document.querySelector('.review-layout').scrollIntoView({behavior:'smooth',block:'start'});};
    });
  }
  function setTime(t,seek=false){
    const location=sampleLocation(ep.series,t),i=location.index;if(i<0)return;
    current=i;currentGap=location.gap;cursorTime=currentGap?t:times[i];$('scrubber').value=String(i);
    const p=ep.series[i];$('clock').textContent=currentGap?`${fmt(t,3)} s · 记录缺口`:`记录 ${fmt(p.time_s,3)} s`;
    $('instant').replaceChildren();
    for(const text of currentGap?['当前处于记录缺口，读数未知，不沿用缺口前数值']:[`清洁 ${pct(p.progress)}`,`板面 ${fmt(p.board_normal,3)} N`,`五指合计 ${fmt(p.hand_normal_sum,3)} N`])el('span',$('instant'),text);
    if(seek&&safeMedia(ep.media?.url)){
      const vt=mappedTime(ep.media,currentGap?t:p.time_s);
      if(vt!==null){video.pause();video.currentTime=vt;}
      else $('clock').textContent+=' · 无对应视频帧';
    }
    positionNote();
    draw();
  }
  function bounds(e){return e?[$('context').checked?e.context_start_s:e.start_s,$('context').checked?e.context_end_s:e.end_s]:[times[0],times.at(-1)];}
  async function playTarget(target,t){
    const vt=mappedTime(ep.media,t);
    if(vt===null)return;
    target.currentTime=vt;
    try{await target.play();}catch(error){$('frame-note').textContent=`浏览器未开始播放：${error.message}。可重试播放；数据曲线仍可查看。`;}
  }
  function setupContrast(){
    const select=$('contrast-select');select.replaceChildren();contrastVideo.pause();contrast=null;
    const growth=ep.events.filter(e=>e.kind==='growth');
    for(const e of growth){const option=el('option',select,`${fmt(e.start_s,3)}–${fmt(e.end_s,3)} s · +${fmt(e.progress_delta*100,3)} 个百分点 · ${fmt(e.duration_s)} s`);option.value=e.id;}
    select.disabled=!growth.length;$('open-contrast').disabled=!growth.length;
    $('contrast-replay').disabled=!growth.length;
    if(!growth.length){el('option',select,'这条轨迹没有记录到连续增长片段');facts($('contrast-facts'),null);return;}
    // Declared video-readability preference, not an action-quality threshold.
    select.value=(growth.find(e=>e.duration_s>=.3)||growth[0]).id;
    changeContrast();
  }
  function changeContrast(){
    contrast=ep.events.find(e=>e.id===$('contrast-select').value)||null;
    contrastVideo.pause();facts($('contrast-facts'),contrast);
    $('contrast-title').textContent=contrast?`对照段 · 清洁读数增长 · ${fmt(contrast.duration_s,3)} s`:'';
    const t=contrast?mappedTime(ep.media,contrast.context_start_s):null;
    if(t!==null)contrastVideo.currentTime=t;
    $('contrast-clock').textContent=contrast?`${fmt(contrast.start_s)}–${fmt(contrast.end_s)} s`:'';
  }
  function loadEpisode(){
    video.pause();contrastVideo.pause();selectionVideo.pause();$('contrast-panel').hidden=true;
    ep=policies[Number($('episode-select').value)];times=ep.series.map(p=>p.time_s);current=0;
    $('outcome').textContent=`原记录：${{pass:'成功',fail:'失败',unknown:'未知'}[ep.outcome]||'未知'} · 末端清洁 ${pct(ep.series.at(-1)?.progress)}`;
    $('scrubber').min='0';$('scrubber').max=String(Math.max(0,times.length-1));$('scrubber').value='0';
    const available=ep.media?.status==='available'&&ep.media?.sync_status==='mapped'&&safeMedia(ep.media?.url);
    $('video-missing').hidden=available;video.hidden=!available;
    $('play').disabled=!available;$('replay').disabled=!available;
    if(available){video.src=ep.media.url;contrastVideo.src=ep.media.url;selectionVideo.src=ep.media.url;}
    else{for(const target of [video,contrastVideo,selectionVideo]){target.removeAttribute('src');target.load();}}
    $('frame-note').textContent='视频约 10 帧/秒，依已存帧时钟定位；不是瞬时接触的证明。视角裁剪只放大原画面，不增加细节。';
    $('view-select').replaceChildren();
    for(const crop of ep.media?.views?.crops||[]){const o=el('option',$('view-select'),crop.label);o.value=crop.id;}
    const full=el('option',$('view-select'),'完整拼图');full.value='full';
    applyView(video,$('video-viewport'));applyView(contrastVideo,$('contrast-viewport'));applyView(selectionVideo,$('selection-viewport'));
    const c=ep.coverage;
    $('coverage-text').textContent=`全程记录 ${fmt(c.total_duration_s)} s = 候选覆盖 ${fmt(c.marked_duration_s)} s + 其余记录 ${fmt(c.unmarked_duration_s)} s + 记录缺口 ${fmt(c.gap_duration_s)} s；其余记录中进度缺测 ${fmt(c.unknown_progress_duration_s)} s。不同颜色是读数状态，不是动作好坏。`;
    $('source-link').href=ep.source_report_url;
    $('provenance').textContent=JSON.stringify({rules:ep.rules,motion_status:ep.motion_status,motion_warnings:ep.motion_warnings,motion_limitations:ep.motion_limitations,limitations:ep.limitations,metadata:ep.metadata,warnings:ep.warnings,source_report:ep.source_report},null,2);
    event=ep.events.find(e=>e.id===ep.preferred_event_id)||null;
    listEvents();showEvent(event,false);setupContrast();setTime(event?.start_s??times[0]);
    if(available)video.onloadedmetadata=()=>{video.playbackRate=Number($('speed').value);setTime(event?.start_s??times[0],true);};
  }
  const colors={plateau:'#c89a58',growth:'#58a69a',gap:'#90899d',other:'#e2e8e3'};
  function canvasContext(canvas){const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);return {ctx,w:rect.width,h:rect.height};}
  function geometry(w){const left=w<500?49:64,right=15;const start=times[0],span=Math.max(1e-9,times.at(-1)-start);return {left,right,x:t=>left+(t-start)/span*(w-left-right),t:x=>Math.max(start,Math.min(times.at(-1),start+(x-left)/(w-left-right)*span))};}
  function draw(){
    if(!ep||!times.length)return;
    drawCoverage();drawSignals();
  }
  function ticks(ctx,w,h,g){ctx.font='10px system-ui';ctx.textAlign='center';ctx.fillStyle='#698078';for(let i=0;i<=6;i++){const t=times[0]+(times.at(-1)-times[0])*i/6;ctx.fillText(`${fmt(t,0)} s`,g.x(t),h-4);}}
  function drawCoverage(){
    const canvas=$('coverage-chart'),{ctx,w,h}=canvasContext(canvas),g=geometry(w);canvas._geom=g;
    ctx.fillStyle=colors.other;ctx.fillRect(g.left,9,w-g.left-g.right,25);
    for(const e of ep.events){ctx.fillStyle=colors[e.kind];ctx.fillRect(g.x(e.start_s),9,Math.max(1,g.x(e.end_s)-g.x(e.start_s)),25);}
    ep.series.slice(0,-1).forEach((p,i)=>{if(p.gap_after||!finite(p.progress)||!finite(ep.series[i+1].progress)){ctx.fillStyle=colors.gap;ctx.fillRect(g.x(p.time_s),9,Math.max(1,g.x(times[i+1])-g.x(p.time_s)),25);}});
    if(event){ctx.strokeStyle='#254a45';ctx.lineWidth=2;ctx.strokeRect(g.x(event.start_s),6,Math.max(2,g.x(event.end_s)-g.x(event.start_s)),31);}
    ctx.strokeStyle='#183e40';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(g.x(cursorTime),1);ctx.lineTo(g.x(cursorTime),40);ctx.stroke();ticks(ctx,w,h,g);
  }
  function drawSignals(){
    const canvas=$('signal-chart'),{ctx,w,h}=canvasContext(canvas),g=geometry(w);canvas._geom=g;
    const fields=[{key:'progress',label:'清洁 %',scale:100,color:'#087a76',factor:100},
      {key:'board_normal',label:'板面 N',color:'#b87932',factor:1}];
    if($('hand-toggle').checked)fields.push({key:'hand_normal_sum',label:'五指 N',color:'#70678e',factor:1});
    const rowHeight=(h-32)/fields.length;
    fields.forEach((field,ri)=>{
      const top=ri*rowHeight+22,bottom=(ri+1)*rowHeight-12;
      const max=field.scale??ep.series.reduce((maximum,p)=>finite(p[field.key])?Math.max(maximum,p[field.key]):maximum,.01)*1.08;
      const y=v=>bottom-v*field.factor/max*(bottom-top);
      if(event){ctx.fillStyle='#f4eee2';ctx.fillRect(g.x(event.start_s),top-10,g.x(event.end_s)-g.x(event.start_s),bottom-top+10);}
      ctx.textAlign='right';ctx.font='10px system-ui';ctx.fillStyle='#718077';ctx.strokeStyle='#e7ece8';ctx.lineWidth=1;
      for(let j=0;j<=2;j++){const yy=bottom-j/2*(bottom-top);ctx.beginPath();ctx.moveTo(g.left,yy);ctx.lineTo(w-g.right,yy);ctx.stroke();ctx.fillText(fmt(max*j/2,field.key==='progress'?0:1),g.left-8,yy+3);}
      ctx.fillStyle=field.color;ctx.textAlign='left';ctx.font='bold 11px system-ui';ctx.fillText(field.label,g.left,top-9);
      let prev=null;ctx.strokeStyle=field.color;ctx.lineWidth=1.3;ctx.beginPath();
      for(const p of ep.series){
        if(!finite(p[field.key])){prev=null;continue;}
        if(prev&&!prev.gap_after){ctx.lineTo(g.x(p.time_s),y(prev[field.key]));ctx.lineTo(g.x(p.time_s),y(p[field.key]));}
        else ctx.moveTo(g.x(p.time_s),y(p[field.key]));prev=p;
      }
      ctx.stroke();
      const p=ep.series[current];if(!currentGap&&finite(p[field.key])){ctx.fillStyle=field.color;ctx.beginPath();ctx.arc(g.x(p.time_s),y(p[field.key]),3.5,0,2*Math.PI);ctx.fill();}
      ctx.strokeStyle='#55736b';ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(g.x(cursorTime),top-8);ctx.lineTo(g.x(cursorTime),bottom);ctx.stroke();
    });ticks(ctx,w,h,g);
  }
  $('coverage-chart').onclick=e=>{const canvas=e.currentTarget;const t=canvas._geom.t(e.clientX-canvas.getBoundingClientRect().left);video.pause();showEvent(ep.events.find(ev=>ev.start_s<=t&&ev.end_s>=t)||null,false);setTime(t,true);};
  $('signal-chart').onclick=e=>{const canvas=e.currentTarget;video.pause();setTime(canvas._geom.t(e.clientX-canvas.getBoundingClientRect().left),true);};
  $('scrubber').oninput=()=>{video.pause();setTime(times[Number($('scrubber').value)],true);};
  $('event-filter').onchange=listEvents;$('episode-select').onchange=loadEpisode;
  $('view-select').onchange=()=>{applyView(video,$('video-viewport'));applyView(contrastVideo,$('contrast-viewport'));applyView(selectionVideo,$('selection-viewport'));};
  $('hand-toggle').onchange=draw;
  $('whole').onclick=()=>{video.pause();$('loop').checked=false;showEvent(null,false);};
  $('replay').onclick=()=>playTarget(video,bounds(event)[0]);
  $('play').onclick=()=>{if(playing)video.pause();else{const t=mappedTime(ep.media,video.currentTime,true),[a,b]=bounds(event);playTarget(video,$('loop').checked&&(t===null||t<a||t>=b)?a:(t??times[current]));}};
  $('speed').onchange=()=>{for(const target of [video,contrastVideo,selectionVideo])target.playbackRate=Number($('speed').value);};
  video.onplay=()=>{playing=true;$('play').textContent='暂停';};video.onpause=()=>{playing=false;$('play').textContent='播放';};
  video.ontimeupdate=()=>{
    const t=mappedTime(ep.media,video.currentTime,true);
    if(t!==null&&!video.paused){setTime(t);const [a,b]=bounds(event);if($('loop').checked&&event&&t>=b-1e-8)playTarget(video,a);}
  };
  video.onended=()=>{if($('loop').checked&&event)playTarget(video,bounds(event)[0]);};
  $('open-contrast').onclick=()=>{
    video.pause();selection=event||ep.events.find(e=>e.id===ep.preferred_event_id)||null;
    $('contrast-panel').hidden=false;
    applyView(contrastVideo,$('contrast-viewport'));applyView(selectionVideo,$('selection-viewport'));
    $('selection-title').textContent=selection?`主选段 · ${selection.kind==='growth'?'读数增长':'读数不变'} · ${fmt(selection.start_s)}–${fmt(selection.end_s)} s`:'未选定片段';
    facts($('selection-facts'),selection);
    const t=selection?mappedTime(ep.media,selection.context_start_s):null;
    if(t!==null)selectionVideo.currentTime=t;
    $('selection-clock').textContent=selection?`${fmt(selection.start_s)}–${fmt(selection.end_s)} s`:'';
    $('selection-replay').disabled=!selection||!safeMedia(ep.media?.url);
    $('contrast-panel').scrollIntoView({behavior:'smooth',block:'start'});
  };
  $('close-contrast').onclick=()=>{contrastVideo.pause();selectionVideo.pause();$('contrast-panel').hidden=true;};
  $('contrast-select').onchange=changeContrast;
  $('contrast-replay').onclick=()=>{if(!contrastVideo.paused){contrastVideo.pause();return;}if(contrast)playTarget(contrastVideo,contrast.context_start_s);};
  contrastVideo.onplay=()=>$('contrast-replay').textContent='暂停对照片段';contrastVideo.onpause=()=>$('contrast-replay').textContent='播放这段（含上下文）';
  contrastVideo.ontimeupdate=()=>{const t=mappedTime(ep.media,contrastVideo.currentTime,true);if(t!==null)$('contrast-clock').textContent=`${fmt(t,3)} s · ${segmentPosition(t,contrast)}`;if(!contrastVideo.paused&&contrast&&t!==null&&t>=contrast.context_end_s)contrastVideo.pause();};
  $('selection-replay').onclick=()=>{if(!selectionVideo.paused){selectionVideo.pause();return;}if(selection)playTarget(selectionVideo,selection.context_start_s);};
  selectionVideo.onplay=()=>$('selection-replay').textContent='暂停主选段';selectionVideo.onpause=()=>$('selection-replay').textContent='播放主选段（含上下文）';
  selectionVideo.ontimeupdate=()=>{const t=mappedTime(ep.media,selectionVideo.currentTime,true);if(t!==null)$('selection-clock').textContent=`${fmt(t,3)} s · ${segmentPosition(t,selection)}`;if(!selectionVideo.paused&&selection&&t!==null&&t>=selection.context_end_s)selectionVideo.pause();};
  const observer=new ResizeObserver(draw);observer.observe($('signal-chart'));observer.observe($('coverage-chart'));
  loadEpisode();
})();
