const test=require('node:test');
const assert=require('node:assert/strict');
const h=require('../scripts/analysis/evaluation_tactile_load_extension/web/tactile_loads.js');
test('keyboard stepping never skips the nearest subsequent native sample',()=>{
  assert.equal(h.stepTime([0,.1,.2],.05,1),.1);
  assert.equal(h.stepTime([0,.1,.2],.05,-1),0);
  assert.equal(h.stepTime([0,.1,.2],.1,1),.2);
  assert.equal(h.stepTime([0,.1,.2],.1,-1),0);
});
test('five-finger totals are complete-only scalar sums, not partial or vector sums',()=>{
  assert.equal(h.sumComplete([1,2,3,4,5]),15);
  assert.equal(h.sumComplete([0,0,0,0,0]),0);
  for(const row of [[1,2,null,4,5],[1,2,NaN,4,5],[1,2,Infinity,4,5],[1,2,-1,4,5],[1,2,3,4]])assert.equal(h.sumComplete(row),null);
});
test('cursor retains exact endpoints but never crosses explicit gaps or bounds',()=>{
  const times=[0,.1,1,1.1],gaps=[false,true,false,false];
  assert.equal(h.indexAt(times,.1,gaps,.1),1);
  assert.equal(h.indexAt(times,.5,gaps,.1),-1);
  assert.equal(h.indexAt(times,1,gaps,.1),2);
  assert.equal(h.indexAt(times,-.01,gaps,.1),-1);
  assert.equal(h.indexAt(times,1.2,gaps,.1),-1);
  assert.equal(h.indexAt(times,.5,null,.1),-1);
});
test('series break at gaps and unknown samples, preserving known zero',()=>{
  assert.deepEqual(h.segments([0,1,2,3,4],[0,2,null,0,4],[false,false,false,true,false],1),[[[0,0],[1,2]],[[3,0]],[[4,4]]]);
  assert.deepEqual(h.segments([0,1,10,11],[1,2,3,4],null,1),[[[0,1],[1,2]],[[10,3],[11,4]]]);
  assert.deepEqual(h.segments([0,1],[null,null],[false,false],1),[]);
});
