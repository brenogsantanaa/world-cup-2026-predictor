"""Generate web/index.html from web/predictions.json (round tabs + interactive bracket)."""
import json, pathlib

HERE = pathlib.Path(__file__).parent
data = json.load(open(HERE / "predictions.json"))

FLAG = {"Mexico":"🇲🇽","South Korea":"🇰🇷","South Africa":"🇿🇦","Czech Republic":"🇨🇿","Canada":"🇨🇦",
"Switzerland":"🇨🇭","Qatar":"🇶🇦","Bosnia and Herzegovina":"🇧🇦","Brazil":"🇧🇷","Morocco":"🇲🇦",
"Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Haiti":"🇭🇹","United States":"🇺🇸","Paraguay":"🇵🇾","Australia":"🇦🇺","Turkey":"🇹🇷",
"Spain":"🇪🇸","Egypt":"🇪🇬","Argentina":"🇦🇷","Algeria":"🇩🇿","France":"🇫🇷","Senegal":"🇸🇳",
"England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Germany":"🇩🇪","Curaçao":"🇨🇼","Norway":"🇳🇴","Iraq":"🇮🇶","Austria":"🇦🇹","Jordan":"🇯🇴",
"Portugal":"🇵🇹","DR Congo":"🇨🇩","Ecuador":"🇪🇨","Ivory Coast":"🇨🇮","Netherlands":"🇳🇱","Belgium":"🇧🇪",
"Croatia":"🇭🇷","Uruguay":"🇺🇾","Colombia":"🇨🇴","Japan":"🇯🇵","Ghana":"🇬🇭","Sweden":"🇸🇪",
"Saudi Arabia":"🇸🇦","New Zealand":"🇳🇿","Iran":"🇮🇷","Tunisia":"🇹🇳","Cabo Verde":"🇨🇻","Uzbekistan":"🇺🇿",
"Panama":"🇵🇦"}

for k in ["round1","round2","round3_played","round3_upcoming"]:
    for m in data[k]:
        m["hf"] = FLAG.get(m["home"], "⚽")
        m["af"] = FLAG.get(m["away"], "⚽")

payload = {"D": data, "FLAGS": FLAG}
DATA = json.dumps(payload)

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>World Cup 2026 — Model Predictions</title>
<style>
 :root{--berry:#8a1538;--red:#e8112d;--lime:#b6e300;--navy:#16235a;--green:#1bb55c;--yellow:#c9b400;--grey:#6b7280;}
 *{box-sizing:border-box;}body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1f2733;background:#f4f6f9;}
 .strip{display:flex;height:14px;}.strip>div{flex:1}.s1{background:var(--berry)}.s2{background:var(--red)}.s3{background:var(--lime)}.s4{background:var(--navy)}
 header{background:#fff;padding:22px 18px 14px;border-bottom:1px solid #e6e9ee;}
 h1{margin:0;font-size:22px;color:var(--navy);}.sub{color:var(--grey);font-size:13px;margin-top:4px;}
 .wrap{max-width:680px;margin:0 auto;padding:16px;}
 .tabs{display:flex;gap:6px;margin:6px 0 14px;flex-wrap:wrap;}
 .tab{flex:1;min-width:70px;text-align:center;padding:9px 6px;border-radius:9px;background:#e9edf2;color:#41506a;font-weight:700;cursor:pointer;font-size:13px;border:none;}
 .tab.on{background:var(--navy);color:#fff;}
 .banner{background:#fff;border:1px solid #e6e9ee;border-radius:12px;padding:14px 16px;margin-bottom:14px;}
 .banner b{font-size:30px;color:var(--navy);}.banner .cap{color:var(--grey);font-size:13px;}
 .sec{font-weight:800;color:var(--grey);text-transform:uppercase;letter-spacing:.5px;font-size:12px;margin:18px 2px 8px;}
 .card{background:#fff;border:1px solid #e6e9ee;border-radius:12px;padding:12px 14px;margin-bottom:10px;box-shadow:0 1px 2px rgba(0,0,0,.03);position:relative;}
 .row{display:flex;align-items:center;justify-content:space-between;gap:8px;}
 .team{display:flex;align-items:center;gap:8px;flex:1;font-weight:600;font-size:15px;}.team.away{justify-content:flex-end;text-align:right;}
 .flag{font-size:22px;line-height:1;}.mid{min-width:96px;text-align:center;}
 .score{font-size:25px;font-weight:800;color:var(--navy);}
 .pred{font-size:12px;color:var(--grey);font-weight:600;}.pred .ps{color:#41506a;font-weight:800;}
 .badge{position:absolute;top:8px;right:10px;font-size:12px;font-weight:800;}.ok{color:var(--green)}.no{color:var(--red)}
 .bar{display:flex;height:9px;border-radius:5px;overflow:hidden;margin-top:11px;}.bar>span{display:block;}
 .lab{display:flex;justify-content:space-between;font-size:11px;color:var(--grey);margin-top:5px;}
 /* bracket */
 .bkinfo{background:#fff;border:1px solid #e6e9ee;border-radius:12px;padding:12px 14px;margin-bottom:12px;font-size:13px;color:#41506a;}
 .bkinfo b{color:var(--navy);}
 .btns{display:flex;gap:8px;margin-bottom:10px;}
 .btn{padding:8px 12px;border-radius:8px;border:1px solid #cdd5e0;background:#fff;color:var(--navy);font-weight:700;font-size:13px;cursor:pointer;}
 .btn.pri{background:var(--navy);color:#fff;border-color:var(--navy);}
 .bkscroll{overflow-x:auto;-webkit-overflow-scrolling:touch;padding-bottom:10px;}
 .bk{display:flex;gap:10px;min-width:1300px;height:880px;}
 .col{display:flex;flex-direction:column;justify-content:space-around;flex:0 0 136px;}
 .col.center{flex:0 0 140px;justify-content:center;}
 .mt{background:#fff;border:1px solid #e6e9ee;border-radius:8px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.04);}
 .tm{display:flex;align-items:center;gap:5px;padding:5px 6px;cursor:pointer;font-size:12px;font-weight:600;border-left:3px solid transparent;}
 .tm:first-child{border-bottom:1px solid #eef1f5;}
 .tm:hover{background:#f3f6fb;}
 .tm.win{background:#eafaf1;border-left-color:var(--green);font-weight:800;color:#0f7a3d;}
 .tm.tbd{color:#aab2bf;cursor:default;font-style:italic;}
 .tm .fg{font-size:15px;}.tm .nm{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
 .tm .sc{font-size:14px;font-weight:800;color:var(--navy);min-width:12px;text-align:right;}
 .tm.win .sc{color:#0f7a3d;}
 .tm .pc{font-size:9px;color:var(--grey);font-weight:700;min-width:26px;text-align:right;}
 .champ{background:linear-gradient(135deg,var(--navy),#0d1838);color:#fff;border-radius:10px;padding:14px 10px;text-align:center;}
 .champ .ttl{font-size:11px;letter-spacing:1px;opacity:.75;}.champ .nm{font-size:16px;font-weight:800;margin-top:4px;}
 .champ .fg{font-size:26px;display:block;margin-bottom:4px;}
 footer{color:#9aa3b2;font-size:11px;text-align:center;padding:18px;}
</style></head><body>
<div class="strip"><div class="s1"></div><div class="s2"></div><div class="s3"></div><div class="s4"></div></div>
<header><div class="wrap" style="padding:0;"><h1>World Cup 2026 — My Model vs Reality</h1>
<div class="sub">Dixon-Coles goal model. Predicted scorelines checked against actual results, plus a live knockout bracket you can play with. For fun, not betting.</div></div></header>
<div class="wrap">
 <div class="tabs">
  <button class="tab" id="t1" onclick="show(1)">Round 1</button>
  <button class="tab" id="t2" onclick="show(2)">Round 2</button>
  <button class="tab" id="t3" onclick="show(3)">Round 3</button>
  <button class="tab on" id="t4" onclick="show(4)">Bracket</button></div>
 <div id="v1" style="display:none"></div><div id="v2" style="display:none"></div>
 <div id="v3" style="display:none"></div><div id="v4"></div>
</div>
<footer>Group stage updated through 26 Jun 2026 (66 matches). Bracket = Round of 32 per the official draw. Model trained on internationals since 1872 + the 2026 group games. No club/player data yet.</footer>
<script>
const P=__DATA__; const D=P.D; const FLAGS=P.FLAGS;

/* ---------- round tabs ---------- */
function bar(m){const s=(w,c)=>`<span style="width:${(w*100).toFixed(1)}%;background:${c}"></span>`;
 return `<div class="bar">${s(m.pH,'var(--green)')}${s(m.pD,'var(--yellow)')}${s(m.pA,'var(--red)')}</div>
 <div class="lab"><span>win ${(m.pH*100).toFixed(0)}%</span><span>draw ${(m.pD*100).toFixed(0)}%</span><span>win ${(m.pA*100).toFixed(0)}%</span></div>`;}
function predCard(m){return `<div class="card"><div class="row">
  <div class="team"><span class="flag">${m.hf}</span>${m.home}</div>
  <div class="mid"><div class="score">${m.hs} : ${m.as_}</div><div class="pred">predicted</div></div>
  <div class="team away">${m.away}<span class="flag">${m.af}</span></div></div>${bar(m)}</div>`;}
function cmpCard(m){return `<div class="card"><span class="badge ${m.correct?'ok':'no'}">${m.correct?(m.exact?'✓ exact':'✓ called it'):'✗ missed'}</span>
  <div class="row"><div class="team"><span class="flag">${m.hf}</span>${m.home}</div>
  <div class="mid"><div class="score">${m.actHS} : ${m.actAS}</div><div class="pred">predicted <span class="ps">${m.predHS}-${m.predAS}</span></div></div>
  <div class="team away">${m.away}<span class="flag">${m.af}</span></div></div>${bar(m)}</div>`;}
function bn(a){return `<div class="banner"><b>${a.outcomes}/${a.total}</b> winners called correctly <span class="cap">(${Math.round(a.outcomes/a.total*100)}%, ${a.exact} exact scorelines)</span></div>`;}
document.getElementById('v1').innerHTML=bn(D.acc1)+D.round1.map(cmpCard).join('');
document.getElementById('v2').innerHTML=bn(D.acc2)+D.round2.map(cmpCard).join('');
document.getElementById('v3').innerHTML=bn(D.acc3)+D.round3_played.map(cmpCard).join('')
  +(D.round3_upcoming.length?`<div class="sec">Still to play · predictions</div>`+D.round3_upcoming.map(predCard).join(''):'');

/* ---------- interactive bracket ---------- */
const B=D.bracket, PAIR=B.pair, SCORE=B.score, R32=B.r32;
const M={}, picks={};
R32.forEach((m,i)=>M['g'+i]={src:[{t:m.home},{t:m.away}],r32:m});
[[0,1],[2,3],[4,5],[6,7]].forEach((p,i)=>M['L16_'+i]={src:[{w:'g'+p[0]},{w:'g'+p[1]}]});
[[8,9],[10,11],[12,13],[14,15]].forEach((p,i)=>M['R16_'+i]={src:[{w:'g'+p[0]},{w:'g'+p[1]}]});
M['LQF_0']={src:[{w:'L16_0'},{w:'L16_1'}]};M['LQF_1']={src:[{w:'L16_2'},{w:'L16_3'}]};
M['RQF_0']={src:[{w:'R16_0'},{w:'R16_1'}]};M['RQF_1']={src:[{w:'R16_2'},{w:'R16_3'}]};
M['SF_L']={src:[{w:'LQF_0'},{w:'LQF_1'}]};M['SF_R']={src:[{w:'RQF_0'},{w:'RQF_1'}]};
M['F']={src:[{w:'SF_L'},{w:'SF_R'}]};
const ORDER=['L16_0','L16_1','L16_2','L16_3','R16_0','R16_1','R16_2','R16_3','LQF_0','LQF_1','RQF_0','RQF_1','SF_L','SF_R','F'];

function advA(a,b){ // model advancement prob of team a over b
 if(!a||!b||a===b) return 0.5;
 const s=[a,b].slice().sort(), k=s[0]+'|'+s[1], p=PAIR[k];
 if(!p) return 0.5;
 const advFirst=p[0]+0.5*p[1];
 return s[0]===a?advFirst:1-advFirst;
}
function team(ref){return ref.t?ref.t:winner(ref.w);}
function winner(id){const m=M[id];const a=team(m.src[0]),b=team(m.src[1]);
 if(!a||!b) return null;
 if(picks[id]===a||picks[id]===b) return picks[id];
 return advA(a,b)>=0.5?a:b;}
function pick(id,t){picks[id]=t;
 ORDER.forEach(mid=>{const m=M[mid];const a=team(m.src[0]),b=team(m.src[1]);
   if(picks[mid] && picks[mid]!==a && picks[mid]!==b) delete picks[mid];});
 render();}
function reset(){for(const k in picks) delete picks[k]; render();}

function scoreOf(a,b){ // predicted scoreline [aGoals,bGoals], consistent with who advances
 if(!a||!b||a===b) return [null,null];
 const s=[a,b].slice().sort(), p=SCORE[s[0]+'|'+s[1]];
 if(!p) return [null,null];
 return s[0]===a?[p[0],p[1]]:[p[1],p[0]];
}
function box(id){const m=M[id];const a=team(m.src[0]),b=team(m.src[1]),w=winner(id);
 const sc=scoreOf(a,b);
 function tmrow(t,goals){
  if(!t) return `<div class="tm tbd"><span class="fg">·</span><span class="nm">TBD</span><span class="sc"></span><span class="pc"></span></div>`;
  const pc = (a&&b)? Math.round((t===a?advA(a,b):advA(b,a))*100)+'%' : '';
  const cls = (w&&t===w)?'tm win':'tm';
  return `<div class="${cls}" onclick="pick('${id}','${t}')"><span class="fg">${FLAGS[t]||'⚽'}</span><span class="nm">${t}</span><span class="sc">${goals!=null?goals:''}</span><span class="pc">${pc}</span></div>`;
 }
 return `<div class="mt">${tmrow(a,sc[0])}${tmrow(b,sc[1])}</div>`;
}
function col(ids,cls){return `<div class="col ${cls||''}">${ids.map(box).join('')}</div>`;}

function renderBracket(){
 const champ=winner('F');
 const center=`<div class="col center"><div class="sec" style="text-align:center;margin:0 0 6px;">Final</div>${box('F')}
   <div class="champ" style="margin-top:10px;"><div class="ttl">CHAMPION</div><span class="fg">${champ?(FLAGS[champ]||'⚽'):'·'}</span><div class="nm">${champ||'—'}</div></div></div>`;
 const bk=`<div class="bk">
  ${col(['g0','g1','g2','g3','g4','g5','g6','g7'])}
  ${col(['L16_0','L16_1','L16_2','L16_3'])}
  ${col(['LQF_0','LQF_1'])}
  ${col(['SF_L'])}
  ${center}
  ${col(['SF_R'])}
  ${col(['RQF_0','RQF_1'])}
  ${col(['R16_0','R16_1','R16_2','R16_3'])}
  ${col(['g8','g9','g10','g11','g12','g13','g14','g15'])}
 </div>`;
 document.getElementById('v4').innerHTML=
  `<div class="banner"><b>${D.groupacc.outcomes}/${D.groupacc.total}</b> group-stage winners called <span class="cap">(${Math.round(D.groupacc.outcomes/D.groupacc.total*100)}%, ${D.groupacc.exact} exact scorelines)</span></div>
   <div class="bkinfo">This is the <b>Round of 32</b> from the official draw. Each tie shows my model's chance for each side, and it auto-fills the favorite all the way to the <b>Champion</b>. <b>Tap any team to mock your own result</b> — everything downstream re-resolves to the model's pick until you override it too.</div>
   <div class="btns"><button class="btn pri" onclick="reset()">Reset to model prediction</button></div>
   <div class="bkscroll">${bk}</div>`;
}

function show(n){for(const i of[1,2,3,4]){document.getElementById('v'+i).style.display=i===n?'block':'none';document.getElementById('t'+i).classList.toggle('on',i===n);}
 if(n===4) renderBracket();}
renderBracket();
</script></body></html>"""

out = HTML.replace("__DATA__", DATA)
(HERE / "index.html").write_text(out)
print("wrote index.html", len(out), "bytes")
