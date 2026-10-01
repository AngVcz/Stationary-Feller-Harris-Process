// ponytail: self-check for gig_compare.html
const fs = require('fs');
const html = fs.readFileSync('docs/gig_compare.html', encoding='utf-8');
const m = html.match(/<script>([\s\S]*?)<\/script>/g);
const js = m[m.length - 1].replace(/<\/?script>/g, '');
try { new Function(js); console.log('syntax: OK'); }
catch (e) { console.error('syntax FAIL:', e.message); process.exit(1); }
// DOM mock
const NS = 'http://www.w3.org/2000/svg';
const store = {};
const mk = (tag) => {
  const o = { tagName: tag, children: [], _attrs: {}, textContent: '', style: {}, appendChild(c) { this.children.push(c); return c; }, setAttribute(k, v) { this._attrs[k] = v; }, setAttributeNS(n, k, v) { this._attrs[k] = v; } };
  o.style = new Proxy({}, { get: () => '', set: () => true });
  return o;
};
const byId = {};
const doc = {
  getElementById(id) {
    if (id === 'anal') return byId[id] || (byId[id] = { innerHTML: '' });
    if (id === 'heat' || id === 'dtiles') return byId[id] || (byId[id] = { children: [], appendChild(c) { this.children.push(c); return c; } });
    if (!store[id]) store[id] = mk('svg');
    return store[id];
  },
  createElement(t) { const o = mk(t); o.className = ''; o.style = new Proxy({}, { get: () => '', set: () => true }); return o; },
  createElementNS(ns, t) { return mk(t); },
  documentElement: {},
};
global.document = doc;
global.getComputedStyle = () => ({ getPropertyValue: (n) => n === '--bad' ? '#b94a3a' : n === '--good' ? '#3f8a4d' : n === '--ink' ? '#1a1d23' : n === '--a' ? '#1f6f8b' : n === '--b' ? '#c06a1f' : n === '--c' ? '#5a8f4d' : n === '--obs' ? '#1a1d23' : n === '--axis' ? '#9aa0a8' : n === '--grid' ? '#e6e8ec' : n === '--muted' ? '#6b7280' : '' });
try { (new Function('document', 'getComputedStyle', js))(doc, global.getComputedStyle); }
catch (e) { console.error('runtime FAIL:', e.message, e.stack); process.exit(1); }
console.log('svg children: cov=', store.cov ? store.cov.children.length : 0, 'bands=', store.bands ? store.bands.children.length : 0);
console.log('heat rows:', byId.heat ? byId.heat.children.length : 'n/a');
console.log('dtiles children:', byId.dtiles ? byId.dtiles.children.length : 'n/a');
console.log('anal set:', byId.anal && byId.anal.innerHTML.length > 0);
console.log('all good');