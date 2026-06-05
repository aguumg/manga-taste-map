// Headless smoke test for the v2 app: stub Plotly/DOM/localStorage, run the
// app script, exercise A (multi-axis), B (tag votes), C (artist), D (export/import),
// plus genre filter + synopsis expansion. Assert no errors + sane behavior.
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync(__dirname + "/../outputs/taste_map_app.html", "utf8");
const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
let app = scripts[scripts.length - 1];
if (!app.includes("const DATA")) throw new Error("app script not found");

// Expose internals for driving.
app += `
;globalThis.__t = { lib, votes, byId, titles, DATA,
  get anchorIds(){return anchorIds}, set anchorIds(v){anchorIds=v},
  get colorBy(){return colorBy}, set colorBy(v){colorBy=v},
  get activeGenres(){return activeGenres}, set activeGenres(v){activeGenres=v},
  setEnt, ent, isRead, overallOf, rebuildPref, recomputeSim, rankedList,
  tasteFit, tasteFit01, drawPlot, renderLib, renderTags, renderNearest,
  exportJson, exportCsv, importJson, saveVotes, artistTitlesHtml };`;

new vm.Script(app);
console.log("[parse] app script parses");

// ---- stubs ----
const store = {};
const downloads = [];
const elreg = {};
function makeEl(id){
  const el = { id, style:{}, dataset:{}, value:id==="topN"?"25":"", checked:false,
    _html:"", classList:{add(){},remove(){},contains(){return false},toggle(){return true}},
    get innerHTML(){return this._html}, set innerHTML(v){this._html=v},
    textContent:"", _kids:[], appendChild(c){this._kids.push(c);return c},
    addEventListener(){}, set onclick(f){}, set onchange(f){}, set oninput(f){},
    querySelector(){return makeEl("q")}, querySelectorAll(){return []},
    click(){}, contains(){return false} };
  return el;
}
const doc = { getElementById:id=>(elreg[id]||=makeEl(id)),
  querySelectorAll:()=>[], addEventListener:()=>{},
  createElement:tag=>{ const e=makeEl(tag);
    if(tag==="a"){ e._d=""; Object.defineProperty(e,"download",{set(v){e._d=v}}); }
    return e; } };
const Plotly = { calls:{newPlot:0,react:0},
  newPlot(d,tr,la){this.calls.newPlot++;this.last=tr;this.layout=la;},
  react(d,tr,la){this.calls.react++;this.last=tr;this.layout=la;} };
const localStorage = { getItem:k=>k in store?store[k]:null, setItem:(k,v)=>{store[k]=String(v)} };
let blobText="";
function Blob(p){blobText=p.join("");}
const URL={createObjectURL:()=>{downloads.push(blobText);return "blob:"},revokeObjectURL(){}};
function FileReader(){ this.readAsText=f=>{ this.result=f.__text; this.onload&&this.onload(); }; }

const sb = { document:doc, Plotly, localStorage, Blob, URL, FileReader,
  alert:m=>console.log("  [alert]",m), Date, Math, JSON, console, setTimeout:()=>{},
  Float64Array, Set, Map, parseInt };
vm.createContext(sb);
vm.runInContext(app, sb);
const T = sb.__t;
console.log("[boot] app ran, no throw. Plotly newPlot=" + Plotly.calls.newPlot);
if (Plotly.calls.newPlot < 1) throw new Error("plot never drawn");

// ---- seed check: the reader's REAL 43-entry export pre-loads ----
const seededLib = JSON.parse(store["taste_library_v2"]);
const nSeed = Object.keys(seededLib).length;
if (nSeed !== 43) throw new Error("expected 43 seed entries, got " + nSeed);
const berserk = seededLib["30002"];
if (!berserk || berserk.overall !== "loved") throw new Error("Berserk not seeded as loved overall");
if (!("art" in berserk) || berserk.art !== null) throw new Error("axes must be present & empty for user to fill");
// verdict distribution must match the export (13 loved/7 liked/4 meh/9 disliked/10 read-unrated)
const dist = {loved:0,liked:0,meh:0,disliked:0,unrated:0};
for(const k in seededLib){ const o=seededLib[k].overall; dist[o||"unrated"]++; }
console.log("[A.seed] 43 entries pre-loaded, verdicts:", JSON.stringify(dist));
if (dist.loved!==13||dist.liked!==7||dist.meh!==4||dist.disliked!==9||dist.unrated!==10)
  throw new Error("seed verdict distribution mismatch: " + JSON.stringify(dist));
console.log("[A.seed] verdicts match export (13 loved / 7 liked / 4 meh / 9 disliked / 10 read-unrated), axes empty ✓");

// ---- A: multi-axis rating preserves disliked-overall + art=5 ----
const someId = T.titles.find(t=>!T.overallOf(t.id)).id;
T.setEnt(someId, {overall:"disliked", art:5, story:2, read:true, note:"art carried it"});
const e = T.ent(someId);
if (e.overall!=="disliked" || e.art!==5) throw new Error("multi-axis not preserved");
console.log(`[A.axes] title ${someId}: overall=disliked but art=5 preserved (anti-poisoning) ✓`);

// ---- B: tag votes build pref vector; hating a power tag penalizes ----
const tagIdx = {}; T.DATA.tags.forEach((n,i)=>tagIdx[n]=i);
const sp = tagIdx["Super Power"];
T.votes[sp] = "hate"; T.saveVotes(); T.rebuildPref();
const solo = T.byId.get(105398);  // Solo Leveling carries Super Power
const tf = T.tasteFit(solo);
if (!(tf < 0)) throw new Error("hated tag did not produce negative taste-fit");
console.log(`[B.votes] hate 'Super Power' -> Solo Leveling taste-fit=${tf.toFixed(3)} (<0 = penalized) ✓`);
// switch colorBy to taste and confirm ranking puts non-SP above SP at top
T.colorBy = "taste"; T.recomputeSim();
const top = T.rankedList(15, true);
const topSP = top.filter(o=>T.byId.get(o.id).v.some(([j])=>j===sp)).length;
console.log(`[B.rank] colorBy=taste, top-15 carrying hated tag = ${topSP} (low is good)`);

// ---- C: artist index resolves; check the Vinland->Yukimura->Planetes case ----
const at = T.DATA.artistTitles;
const nArtists = Object.keys(at).length;
if (nArtists === 0) throw new Error("artistTitles empty — staff data missing");
const vin = T.byId.get(30642);   // Vinland Saga
if (!vin.ar.includes("Makoto Yukimura")) throw new Error("Vinland missing Yukimura");
const yuk = at["Makoto Yukimura"].map(id=>T.byId.get(id)).map(t=>t.e||t.r);
console.log(`[C.artist] ${nArtists} artists indexed; Makoto Yukimura -> [${yuk.join(", ")}]`);
if (!yuk.includes("Planetes")) throw new Error("Yukimura should map to Planetes too");
const aHtml = T.artistTitlesHtml(vin);
if (!/Planetes/.test(aHtml)) throw new Error("artist html missing Planetes");
console.log("[C.artist] 'more by artist' on Vinland Saga lists Planetes ✓");

// ---- synopsis present on a title (expand content) ----
const withSyn = T.titles.find(t=>t.syn && t.syn.length>40);
console.log(`[syn] sample synopsis (${withSyn.e||withSyn.r}): "${withSyn.syn.slice(0,60)}…" (${withSyn.syn.length} chars) ✓`);

// ---- genre filter ----
T.activeGenres = new Set(["Horror"]);
T.colorBy = "anchor"; T.recomputeSim();
const horrorList = T.rankedList(25, true);
const allHorror = horrorList.every(o=>T.byId.get(o.id).g.includes("Horror"));
console.log(`[genre] filter=Horror -> nearest list ${horrorList.length} items, all Horror: ${allHorror}`);
if (!allHorror) throw new Error("genre filter leaked non-Horror into nearest list");
T.activeGenres = new Set(T.DATA.genres); // reset

// ---- D: export round-trips axes + votes ----
downloads.length = 0; T.exportJson();
const ex = JSON.parse(downloads[0]);
if (ex.version !== "taste_library_v2") throw new Error("export version wrong");
if (ex.library[someId].art !== 5) throw new Error("export lost axis art=5");
if (!ex.tagVotes && !ex.tagVotesByName) throw new Error("export lost tag votes");
console.log(`[D.exportJSON] version=${ex.version}, lib entries=${Object.keys(ex.library).length}, axis art=${ex.library[someId].art}, votes=${Object.keys(ex.tagVotes).length}`);

downloads.length = 0; T.exportCsv();
const csvHead = downloads[0].split("\n")[0];
if (csvHead !== "id,title,country,read,overall,art,story,characters,pacing,note")
  throw new Error("csv header missing axes: " + csvHead);
console.log("[D.exportCSV] header includes all 4 axes + note ✓");

// import round-trip
T.importJson({ __text: JSON.stringify(ex) });
const reLib = JSON.parse(store["taste_library_v2"]);
const reVotes = JSON.parse(store["taste_tagvotes_v1"]);
if (reLib[someId].art !== 5) throw new Error("import lost axis");
if (reVotes[sp] !== "hate") throw new Error("import lost vote");
console.log(`[D.import] round-trip restored axes (art=${reLib[someId].art}) and votes (SuperPower=${reVotes[sp]}) ✓`);

console.log("\nALL HEADLESS A/B/C/D + GENRE CHECKS PASSED");
