'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const js=path.join(__dirname,'../scripts/analysis/evaluation_progress_experiment/web/progress.js');
const {binFor,videoTime,safeMedia,cadenceHz,progressWidth,unchangedDuration}=require(js);
test('progress bins do not stretch a failed endpoint to 100%',()=>{
  const edges=[0,.2,.4,.6,.8,1];
  assert.equal(binFor(.64,edges),3);assert.equal(binFor(1,edges),4);
  assert.equal(binFor(null,edges),-1);assert.equal(binFor(NaN,edges),-1);
});
test('video seek uses only the episode explicit past mapping, without extrapolation',()=>{
  const media={sync_status:'mapped',time_s:[0,1,2],video_time_s:[0,.1,.2]};
  assert.equal(videoTime(media,1.3),.1);assert.equal(videoTime(media,2),.2);
  assert.equal(videoTime(media,3),null);assert.equal(videoTime(media,-1),null);
  assert.equal(videoTime({...media,sync_status:'unverified'},1),null);
  assert.equal(videoTime({...media,video_time_s:[0]},1),null);
});
test('media URLs cannot escape exported local assets',()=>{
  assert(safeMedia('media/001.mp4'));
  for(const value of ['../source.mp4','https://host/video.mp4','javascript:x','media/../x.mp4','/media/001.mp4',null])assert(!safeMedia(value));
});
test('cadence description comes from actual recorded intervals',()=>{
  assert.equal(cadenceHz([{time_s:0},{time_s:.01},{time_s:.02}]),100);
  assert.equal(cadenceHz([]),null);
});
test('missing endpoint is not a 0% bar and unobserved transitions are not zero unchanged seconds',()=>{
  assert.equal(progressWidth(null),null);assert.equal(progressWidth(0),'0%');
  assert.equal(unchangedDuration({status:'observed',known_transition_duration_s:0,unchanged_progress_duration_s:0}),null);
  assert.equal(unchangedDuration({status:'observed',known_transition_duration_s:1,unchanged_progress_duration_s:0}),0);
});
test('page does not inject arbitrary source metadata as HTML',()=>{
  const source=fs.readFileSync(js,'utf8');
  assert(!source.includes('innerHTML'));assert(!source.includes('eval('));
});
