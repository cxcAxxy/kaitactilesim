const test = require('node:test');
const assert = require('node:assert/strict');
const h = require('../scripts/analysis/evaluation_action_review/web/review.js');
test('context is explicitly distinguished from the selected interval',()=>{
  const segment={start_s:10,end_s:15};
  assert.equal(h.segmentPosition(8,segment),'前置上下文');
  assert.equal(h.segmentPosition(12,segment),'选段内');
  assert.equal(h.segmentPosition(17,segment),'后置上下文');
});
test('native past-only lookup never rounds to a future sample',()=>{
  assert.equal(h.pastIndex([0,.1,.2],.19),1);
  assert.equal(h.pastIndex([0,.1],-.01),-1);
  assert.equal(h.pastIndex([],0),-1);
});
test('gaps never imply a held current measurement',()=>{
  const series=[{time_s:0},{time_s:1,gap_after:true},{time_s:10},{time_s:11}];
  assert.deepEqual(h.sampleLocation(series,5),{index:1,gap:true});
  assert.deepEqual(h.sampleLocation(series,1),{index:1,gap:false});
  assert.deepEqual(h.sampleLocation(series,10),{index:2,gap:false});
  assert.deepEqual(h.sampleLocation(series,12),{index:-1,gap:true});
});
test('video mapping is explicit, past-only and refuses extrapolation',()=>{
  const media={sync_status:'mapped',time_s:[10,10.1,10.2],video_time_s:[0,.1,.2]};
  assert.equal(h.mappedTime(media,10.15),.1);
  assert.equal(h.mappedTime(media,.15,true),10.1);
  assert.equal(h.mappedTime(media,9),null);
  assert.equal(h.mappedTime(media,10.3),null);
  assert.equal(h.mappedTime({...media,sync_status:'unknown'},10),null);
  assert.equal(h.mappedTime({...media,video_time_s:[0]},10),null);
});
test('only copied relative media can be loaded',()=>{
  assert.equal(h.safeMedia('media/001.mp4'),true);
  for(const path of ['../media/001.mp4','https://x/media/1.mp4','javascript:1','media/no.mp4'])assert.equal(h.safeMedia(path),false);
});
test('unknown force never becomes positive or failure diagnosis',()=>{
  assert.match(h.eventTitle(null),/全程/);
  assert.doesNotMatch(h.eventTitle({kind:'plateau',contact_stats:{sample_count:2,known_sample_count:1,min_n:1}}),/为正/);
  assert.match(h.eventTitle({kind:'plateau',contact_stats:{sample_count:2,known_sample_count:2,min_n:1}}),/为正/);
  assert.match(h.eventTitle({kind:'growth'}),/增长/);
});
test('display crops are bounded and not arbitrary inferred camera views',()=>{
  const views={source_width:1920,source_height:1080,crops:[{id:'head',x:214,y:78,width:434,height:324}]};
  assert.equal(h.cropStyle(views,'head').ratio,'434 / 324');
  assert.equal(h.cropStyle(views,'unknown'),null);
  assert.equal(h.cropStyle(null,'head'),null);
  assert.equal(h.cropStyle({...views,crops:[{id:'head',x:1900,y:0,width:434,height:324}]},'head'),null);
});
