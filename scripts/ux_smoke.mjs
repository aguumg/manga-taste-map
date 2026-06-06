// Headless smoke test: load the app, click around, assert no JS errors.
import { chromium } from 'playwright';
import { pathToFileURL } from 'url';

const APP = process.argv[2] || 'outputs/taste_map_app.html';
const url = pathToFileURL(APP).href;
const errors = [];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
page.on('pageerror', e => errors.push('pageerror: ' + e.message));

const log = [];
const ok = (c, m) => { const line = (c ? 'PASS' : 'FAIL') + '  ' + m; log.push(line); console.log(line); };

await page.goto(url, { waitUntil: 'load' });
await page.waitForTimeout(1500); // let Plotly draw

// 1. plot rendered
const plotKids = await page.locator('#plot .plotly').count();
ok(plotKids > 0, `plot rendered (#plot .plotly count=${plotKids})`);

// 2. tabs switch panes
for (const t of ['lib', 'tags', 'west', 'shelf', 'anchor']) {
  await page.click(`.tab[data-tab="${t}"]`);
  await page.waitForTimeout(150);
  const active = await page.locator(`#pane-${t}.active`).count();
  ok(active === 1, `tab "${t}" activates its pane`);
}

// 3. shelf content
await page.click('.tab[data-tab="shelf"]');
await page.waitForTimeout(200);
const rows = await page.locator('#shelfList .shelf-row').count();
ok(rows >= 40, `shelf rows render (${rows})`);
const mapBtns = await page.locator('#shelfList button[data-shmap]').count();
ok(mapBtns >= 40, `→map buttons present (${mapBtns})`);
const nomap = await page.locator('#shelfList .shelf-nomap').count();
ok(nomap === 0, `every shelf title links to a map dot (off-map=${nomap})`);
const altText = await page.locator('#shelfList .shelf-alt').first().innerText().catch(() => '');
ok(/maps to/.test(altText), `"(maps to: …)" shown for renamed titles ["${altText}"]`);

// 4. add-title feedback: add a known title, expect it to auto-link + flash
await page.fill('#shelfAdd', 'Chainsaw Man');
await page.click('#shelfAddBtn');
await page.waitForTimeout(300);
const added = await page.locator('#shelfList .shelf-row', { hasText: 'Chainsaw Man' }).count();
ok(added >= 1, `manually-added title appears (${added})`);
const addedRow = page.locator('#shelfList .shelf-row', { hasText: 'Chainsaw Man' }).first();
const addedMap = await addedRow.locator('button[data-shmap]').count();
ok(addedMap >= 1, `manually-added title auto-linked to map (data-shmap=${addedMap})`);

// 5. → map jump from shelf works (clicks first map button, expect tab change)
await page.locator('#shelfList button[data-shmap]').first().click();
await page.waitForTimeout(200);
const anchorOrLibActive = await page.locator('#pane-lib.active, #pane-west.active').count();
ok(anchorOrLibActive >= 1, `→map jumps to the linked title's tab`);

// 6. persistence: toggle highlight + reload, expect it remembered
await page.click('.tab[data-tab="anchor"]');
await page.waitForTimeout(150);
// Options panel is collapsed by default — expand it so the checkbox is visible
const optOpen = await page.locator('#highlightTop').isVisible();
if (!optOpen) { await page.click('#optHead'); await page.waitForTimeout(200); }
const hadHighlight = await page.locator('#highlightTop').isChecked();
await page.locator('#highlightTop').setChecked(!hadHighlight);
await page.waitForTimeout(300);
await page.reload({ waitUntil: 'load' });
await page.waitForTimeout(1200);
const afterReload = await page.locator('#highlightTop').isChecked();
ok(afterReload === !hadHighlight, `highlight-top toggle persists across reload (${hadHighlight}->${afterReload})`);

// 7. STALE localStorage regression: simulate an old shelf saved before the
// title-matcher existed (Demon Slayer with empty links) and confirm reload
// refreshes its map link from the rebuilt seed instead of keeping it dead.
await page.evaluate(() => {
  localStorage.setItem('taste_shelf_v1', JSON.stringify([
    { sid:'demon-slayer', title:'Demon Slayer', status:'done', source:'mine', note:'kept note', url:'', cid:null, wk:null }
  ]));
});
await page.reload({ waitUntil: 'load' });
await page.waitForTimeout(1200);
await page.click('.tab[data-tab="shelf"]');
await page.waitForTimeout(200);
const dsRow = page.locator('#shelfList .shelf-row[data-sid="demon-slayer"]');
const dsMap = await dsRow.locator('button[data-shmap]').count();
ok(dsMap >= 1, `stale shelf entry gets its map link refreshed on reload (${dsMap})`);
const dsNote = await dsRow.locator('input[data-shnote]').inputValue().catch(()=>'');
ok(dsNote === 'kept note', `user note preserved through the refresh ["${dsNote}"]`);

// 8. DELETE persistence: remove a seed row with ✕, reload, confirm it stays gone
//    (tombstone) and is NOT resurrected by the seed-merge.
await page.click('.tab[data-tab="shelf"]');
await page.waitForTimeout(200);
const before = await page.locator('#shelfList .shelf-row').count();
const victim = page.locator('#shelfList .shelf-row').first();
const victimSid = await victim.getAttribute('data-sid');
await victim.locator('.shelf-x').click();
await page.waitForTimeout(200);
await page.reload({ waitUntil: 'load' });
await page.waitForTimeout(1200);
await page.click('.tab[data-tab="shelf"]');
await page.waitForTimeout(200);
const stillGone = await page.locator(`#shelfList .shelf-row[data-sid="${victimSid}"]`).count();
ok(stillGone === 0, `deleted shelf row stays deleted after reload (sid=${victimSid})`);

// 9. "+ shelf" button on a My Library (Eastern) row adds + flips to ✓
await page.click('.tab[data-tab="lib"]');
await page.waitForTimeout(400);
const libBtn = page.locator('#libList .item .shelfbtn').first();
await libBtn.scrollIntoViewIfNeeded();
await libBtn.click();
await page.waitForTimeout(150);
const libBtnAfter = (await libBtn.innerText()).trim();
ok(libBtnAfter.includes('✓'), `My Library "+ shelf" button adds and flips to ✓ ["${libBtnAfter}"]`);

// 9b. same button exists on Western rows
await page.click('.tab[data-tab="west"]');
await page.waitForTimeout(300);
const westBtns = await page.locator('#westList .item .shelfbtn').count();
ok(westBtns > 0, `Western rows have a "+ shelf" button (${westBtns})`);

// 10. shelf search autocomplete: type a corpus title -> suggestion -> add linked
await page.click('.tab[data-tab="shelf"]');
await page.waitForTimeout(200);
await page.fill('#shelfAdd', 'Vinland Saga');
await page.waitForTimeout(350);
const suggCount = await page.locator('#shelfSugg .item').count();
ok(suggCount > 0, `shelf search shows DB suggestions (${suggCount})`);
await page.locator('#shelfSugg .item').first().click();
await page.waitForTimeout(300);
const vsRow = page.locator('#shelfList .shelf-row', { hasText: 'Vinland Saga' }).first();
const vsMap = await vsRow.locator('button[data-shmap]').count();
ok(vsMap >= 1, `picked suggestion added WITH a map link (data-shmap=${vsMap})`);

await browser.close();
console.log(log.join('\n'));
console.log('\nJS errors captured: ' + errors.length);
errors.slice(0, 15).forEach(e => console.log('  ' + e));
const failed = log.filter(l => l.startsWith('FAIL')).length;
console.log(`\n${failed === 0 && errors.length === 0 ? 'ALL GREEN' : failed + ' failures, ' + errors.length + ' errors'}`);
process.exit(failed === 0 && errors.length === 0 ? 0 : 1);
