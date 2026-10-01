// ponytail: self-check for pstay_compare.html — extract <script>, mock DOM, run it.
const fs = require('fs');
const html = fs.readFileSync('docs/pstay_compare.html', encoding='utf-8');
// pull the inline script (after the last <script>)
const m = html.match(/<script>([\s\S]*?)<\/script>/g);
const js = m[m.length - 1].replace(/<\/?script>/g, '');
// syntax check
try { new Function(js); console.log('syntax: OK'); }
catch (e) { console.error('syntax FAIL:', e.message); process.exit(1); }
// DOM mock
const NS = 'http://www.w3.org/2000/svg';
const store = {};
const mk = (tag) => {
  const o = { tagName: tag, children: [], style: {}, _attrs: {}, textContent: '' };
  o.setAttribute = (k, v) => { o._attrs[k] = v; };
  o.setAttributeNS = (k2, k, v) => { o._attrs[k] = v; };
  o.appendChild = (c) => { o.children.push(c); return c; };
  o.style = new Proxy({}, { get: () => '', set: () => true });
  return o;
};
const doc = {
  getElementById: (id) => {
    if (id === 'anal') return { innerHTML: '' };
    if (!store[id]) store[id] = mk('svg');
    return store[id];
  },
  createElementNS: (ns, tag) => mk(tag),
  documentElement: { getComputedStyle: () => ({ getPropertyValue: () => '' }) },
};
global.document = doc;
global.getComputedStyle = () => ({ getPropertyValue: () => '' });
try {
  (new Function('document', 'getComputedStyle', js))(doc, global.getComputedStyle);
} catch (e) { console.error('runtime FAIL:', e.message); process.exit(1); }
const counts = {};
for (const id of ['pathsA','pathsB','bands','cov']) {
  counts[id] = store[id] ? store[id].children.length : 0;
}
console.log('render counts:', JSON.stringify(counts));
console.log('anal set:', doc.getElementById('anal').innerHTML.length > 0);
console.log('all good');