/* Display-only extension. The archived dashboard and its shared cursor stay intact. */
(function(global){
  'use strict';
  const COLORS=['#087e94','#df982a','#6473b8','#b76186','#488763'];
  const finite=v=>typeof v==='number'&&Number.isFinite(v);
  const fmt=(v,n=3)=>finite(v)?v.toLocaleString('zh-CN',{minimumFractionDigits:n,maximumFractionDigits:n}):'未知';
  function indexAt(times,t,gaps,nominal){
    if(!times.length||!finite(t)||t<times[0]-1e-8||t>times.at(-1)+1e-8)return -1;
    let lo=0,hi=times.length;
    while(lo<hi){const mid=(lo+hi)>>>1;if(times[mid]<=t+1e-8)lo=mid+1;else hi=mid;}
    const i=lo-1,age=t-times[i];
    const declared=Array.isArray(gaps)&&gaps.length===times.length;
    return age>1e-8&&(gaps?.[i]===true||(!declared&&finite(nominal)&&age>nominal*1.5+1e-8))?-1:i;
  }
  function segments(times,values,gaps,nominal){
    const result=[];let run=[];
    const declared=Array.isArray(gaps)&&gaps.length===times.length;
    for(let i=0;i<times.length;i++){
      if(!finite(values[i])){if(run.length)result.push(run);run=[];continue;}
      if(run.length&&(gaps?.[i-1]===true||(!declared&&finite(nominal)&&times[i]-times[i-1]>1.5*nominal+1e-8))){result.push(run);run=[];}
      run.push([times[i],values[i]]);
    }
    if(run.length)result.push(run);return result;
  }
  const sumComplete=row=>{if(!Array.isArray(row)||row.length!==5||!row.every(v=>finite(v)&&v>=0))return null;const value=row.reduce((a,b)=>a+b,0);return finite(value)?value:null;};
  function stepTime(times,t,direction){
    if(direction>0){let i=0;while(i<times.length-1&&times[i]<=t+1e-8)i++;return times[i];}
    let i=times.length-1;while(i>0&&times[i]>=t-1e-8)i--;return times[i];
  }
  const helpers={indexAt,segments,sumComplete,stepTime};
  function mount(host,data,options={}){
    const anchor=host.querySelector('.ke-force-chart')?.closest('.ke-panel');
    const slider=host.querySelector('.ke-slider'),clock=host.querySelector('.ke-time-value');
    if(!anchor||!slider||!clock)throw new Error('Archived dashboard must be mounted first');
    const doc=host.ownerDocument,times=data.time_s,channels=data.channels;
    if(!Array.isArray(times)||!times.length||times.some((t,i)=>!finite(t)||(i>0&&t<=times[i-1]))||channels?.length!==5)throw new Error('Expected native five-finger load data');
    for(const key of ['normal_force_n','tangent_force_n'])if(!Array.isArray(data[key])||data[key].length!==times.length||data[key].some(row=>!Array.isArray(row)||row.length!==5))throw new Error(`Invalid ${key}`);
    for(const key of ['normal_total_n','tangent_total_n'])if(!Array.isArray(data[key])||data[key].length!==times.length)throw new Error(`Invalid ${key}`);
    const end=Number(slider.max)||times.at(-1)||1;
    const el=(tag,parent,text,cls)=>{const n=doc.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;parent.appendChild(n);return n;};
    const cleanups=[],panels=[],charts=[],ftReadings=[],selected=new Set([0,1,2,3,4]);
    let destroyed=false,frame=null;
    const on=(target,event,fn)=>{target.addEventListener(event,fn);cleanups.push(()=>target.removeEventListener(event,fn));};
    const banner=doc.createElement('section');banner.className='ke-load-banner';
    el('strong',banner,'H50 · 五指法向 / 切向载荷扩展版');
    el('p',banner,'保留原报告和评估结果，只补充触觉时间线。手指载荷与板擦—白板接触力是不同信号；总和为标量相加，不是矢量合力。');
    const nav=el('div',banner,undefined,'ke-load-nav');
    const jump=el('button',nav,'跳到五指载荷图 ↓','ke-button');jump.type='button';
    for(const [label,url] of [['原报告 ↗',data.source_page],['H50 稳定报告 ↗',data.stable_page],['载荷 JSON ↓','tactile_loads.json']]){
      if(typeof url==='string'&&!url.includes(':')&&!url.startsWith('//')){const a=el('a',nav,label);a.href=url;}
    }
    host.querySelector('.kai-eval').prepend(banner);
    const title=anchor.querySelector('h2'),oldTitle=title?.textContent;
    if(title)title.textContent='五指法向载荷时间线 · 逐指 Fn';
    const normalNote=el('p',anchor,'这里的五条线分别来自拇指、食指、中指、无名指、小指，不是板擦—白板法向力。下方总和始终包含这五指，隐藏曲线不会改变总和。','ke-note ke-load-clarification');
    let insertAfter=anchor;
    function panel(id,titleText,subtitle){
      const p=doc.createElement('section');p.className='ke-panel ke-load-panel';p.dataset.loadPanel=id;
      insertAfter.after(p);insertAfter=p;panels.push(p);
      const heading=el('div',p,undefined,'ke-section-title');el('h2',heading,titleText);el('span',heading,'N · 同轨迹原生记录','ke-subtle');
      el('p',p,subtitle,'ke-note');return p;
    }
    function chart(parent,id,series){
      const readout=el('div',parent,undefined,'ke-load-reading');readout.setAttribute('aria-live','off');
      const canvas=el('canvas',parent,undefined,'ke-load-chart');canvas.dataset.loadChart=id;canvas.setAttribute('role','img');canvas.setAttribute('aria-label',parent.querySelector('h2').textContent+'，点击与场景回放联动');canvas.tabIndex=0;
      const preview=el('div',parent,'悬浮查看数值；点击选择同一时刻。左右方向键逐原生样本移动。','ke-load-hover');preview.setAttribute('aria-live','polite');
      const models=series.map(s=>({...s,runs:segments(times,s.values,data.gap_after,data.nominal_interval_s)}));
      const maximum=models.reduce((m,s)=>s.values.reduce((n,v)=>finite(v)?Math.max(n,v):n,m),0);
      const view={canvas,readout,preview,models,scale:maximum>0?maximum*1.08:1};charts.push(view);
      const timeAt=event=>{const box=canvas.getBoundingClientRect();return Math.max(0,Math.min(end,(event.clientX-box.left-view.left)/(box.width-view.left-14)*end));};
      on(canvas,'click',event=>seek(timeAt(event)));
      on(canvas,'pointermove',event=>{
        const t=timeAt(event),i=indexAt(times,t,data.gap_after,data.nominal_interval_s);
        const values=models.filter(s=>s.finger===undefined||selected.has(s.finger)).map(s=>`${s.label} ${fmt(i<0?null:s.values[i])} N`);
        preview.textContent=i<0?`悬浮 ${fmt(t)} s · 此处记录未知，不沿用缺口前数值`:`悬浮 ${fmt(t)} s / 原生采样 ${fmt(times[i])} s · ${values.join('；')}`;
      });
      on(canvas,'pointerleave',()=>{preview.textContent='悬浮查看数值；点击选择同一时刻。左右方向键逐原生样本移动。';});
      on(canvas,'keydown',event=>{
        if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
        event.preventDefault();const t=Number(slider.value);
        seek(event.key==='Home'?times[0]:event.key==='End'?times.at(-1):stepTime(times,t,event.key==='ArrowLeft'?-1:1));
      });return view;
    }
    const normalPanel=panel('normal-total','五指法向载荷总和 · Σ Fnᵢ','将上图五个手指的 Fn 标量相加。不是净合力；任一手指缺测时，总和为未知。各图纵轴独立，请按 N 数值比较，不按曲线高度比较。');
    chart(normalPanel,'normal-total',[{label:'五指 Fn 总和',values:data.normal_total_n,color:'#087e94'}]);
    const tangentPanel=panel('tangent-fingers','五指切向载荷时间线 · 逐指 |Ft|','每个手指：按记录的两个切向分量，先对触觉单元求和，再取二维向量的模长。这里显示当前载荷，不是过去 0.1 秒的 ΔFt。');
    const checks=el('div',tangentPanel,undefined,'ke-load-fingers');
    const sourceNames=(options.sourceChannels||channels).map(c=>c.name);
    channels.forEach((c,i)=>{
      const label=el('label',checks),checkbox=el('input',label);checkbox.type='checkbox';checkbox.checked=true;checkbox.dataset.finger=c.name;
      const color=COLORS[Math.max(0,sourceNames.indexOf(c.name))%5];
      const dot=el('i',label,undefined,'ke-dot');dot.style.backgroundColor=color;el('span',label,c.label||c.name).title=c.name;
      const value=el('strong',label,'未知');ftReadings.push({value,i});
      on(checkbox,'change',()=>{if(checkbox.checked)selected.add(i);else selected.delete(i);render();});
    });
    chart(tangentPanel,'tangent-fingers',channels.map((c,i)=>({label:c.label||c.name,finger:i,values:data.tangent_force_n.map(row=>row[i]),color:COLORS[Math.max(0,sourceNames.indexOf(c.name))%5]})));
    const tangentTotalPanel=panel('tangent-total','五指切向载荷总和 · Σ |Ftᵢ|','将五指切向载荷的大小相加，不跨手指坐标系做矢量合成。它与法向总和分开显示，不作为滑移判定。');
    chart(tangentTotalPanel,'tangent-total',[{label:'五指 |Ft| 总和',values:data.tangent_total_n,color:'#995e86'}]);
    const details=el('details',tangentTotalPanel,undefined,'ke-more');el('summary',details,'查看定义、来源与缺测说明');
    el('pre',details,JSON.stringify({semantics:data.semantics,warnings:data.warnings,trial_id:data.trial_id,prediction_horizon:data.prediction_horizon,provenance:data.provenance},null,2),'ke-evidence-json');
    on(jump,'click',()=>anchor.scrollIntoView({behavior:'smooth',block:'start'}));
    function seek(time){slider.value=String(time);slider.dispatchEvent(new (doc.defaultView.Event)('input',{bubbles:true}));render();}
    function render(){
      if(destroyed)return;
      const t=Number(slider.value),index=indexAt(times,t,data.gap_after,data.nominal_interval_s);
      ftReadings.forEach(({value,i})=>{value.textContent=`${fmt(index<0?null:data.tangent_force_n[index][i])} N`;});
      charts.forEach(view=>{
        const {canvas,models}=view,rect=canvas.getBoundingClientRect(),width=Math.max(1,rect.width),height=Math.max(1,rect.height),dpr=Math.min(global.devicePixelRatio||1,2);
        canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
        const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
        const left=width<450?43:52,top=14,bottom=height-29,plotWidth=width-left-14;
        view.left=left;
        const x=v=>left+v/end*plotWidth,y=v=>bottom-v/view.scale*(bottom-top);
        ctx.font='10px system-ui';ctx.textAlign='right';ctx.fillStyle='#6b7c8e';ctx.strokeStyle='#e5ebf0';ctx.lineWidth=1;
        for(let i=0;i<=4;i++){const v=view.scale*i/4,yy=y(v);ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(width-14,yy);ctx.stroke();ctx.fillText(fmt(v,2),left-6,yy+3);}
        ctx.textAlign='center';for(let i=0;i<=6;i++)ctx.fillText(`${fmt(end*i/6,0)} s`,x(end*i/6),height-7);
        ctx.textAlign='left';ctx.fillText('N',8,12);
        const shown=models.filter(s=>s.finger===undefined||selected.has(s.finger));
        let hasData=false;
        shown.forEach(model=>{
          ctx.strokeStyle=model.color;ctx.fillStyle=model.color;ctx.lineWidth=1.35;
          for(const run of model.runs){
            hasData=true;ctx.beginPath();run.forEach(([tt,v],i)=>{if(i===0)ctx.moveTo(x(tt),y(v));else ctx.lineTo(x(tt),y(v));});ctx.stroke();
            if(run.length===1){ctx.beginPath();ctx.arc(x(run[0][0]),y(run[0][1]),2,0,Math.PI*2);ctx.fill();}
          }
          const v=index>=0?model.values[index]:null;
          if(finite(v)){ctx.beginPath();ctx.arc(x(times[index]),y(v),3,0,Math.PI*2);ctx.fill();}
        });
        ctx.strokeStyle='#2d465c';ctx.lineWidth=1;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(x(t),top);ctx.lineTo(x(t),bottom);ctx.stroke();ctx.setLineDash([]);
        if(!hasData){ctx.fillStyle='#697d8d';ctx.textAlign='center';ctx.fillText(shown.length?'没有有效观测，未知不是零':'曲线已隐藏；五指总和仍固定包含五指',left+plotWidth/2,(top+bottom)/2);}
        view.readout.replaceChildren();
        el('span',view.readout,`游标 ${fmt(t)} s · ${index>=0?'采样 '+fmt(times[index])+' s':'记录未知'}`);
        for(const model of shown)el('strong',view.readout,`${model.label}：${fmt(index<0?null:model.values[index])} N`).style.color=model.color;
      });
    }
    const schedule=()=>{if(frame===null)frame=global.requestAnimationFrame(()=>{frame=null;render();});};
    // The archived renderer updates text every seek/playback tick. Read the full
    // slider value, never the rounded clock label; no mutation of stable code.
    const observer=new global.MutationObserver(schedule);observer.observe(clock,{childList:true,characterData:true,subtree:true});
    const resize=new global.ResizeObserver(schedule);panels.forEach(p=>resize.observe(p));
    render();
    return {destroy(){if(destroyed)return;destroyed=true;observer.disconnect();resize.disconnect();if(frame!==null)global.cancelAnimationFrame(frame);cleanups.forEach(fn=>fn());panels.forEach(p=>p.remove());banner.remove();normalNote.remove();if(title)title.textContent=oldTitle;}};
  }
  global.KaiTactileLoadExtension=Object.freeze({mount});
  if(typeof module!=='undefined'&&module.exports)module.exports={...helpers,mount};
  if(typeof document!=='undefined'){
    const node=document.getElementById('kai-tactile-load-data'),host=document.getElementById('kai-evaluation');
    if(node&&host){
      try{
        const payload=JSON.parse(document.getElementById('kai-evaluation-data').textContent);
        global.kaiTactileLoadView=mount(host,JSON.parse(node.textContent),{sourceChannels:payload.visualization.channels});
      }catch(error){
        const p=document.createElement('p');p.className='ke-load-error';p.textContent='新增载荷图未能加载；原报告仍保留。请检查 tactile_loads.json 与来源。';host.prepend(p);global.console?.error('Tactile load extension:',error);
      }
    }
  }
})(typeof window!=='undefined'?window:globalThis);
