// ponytail: self-check for pit_predictive.html
const fs = require('fs');
const html = fs.readFileSync('docs/pit_predictive.html', encoding = 'utf-8');
const m = html.match(/<script>([\s\S]*?)<\/script>/g);
const js = m[m.length - 1].replace(/<\/?script>/g, '');
try { new Function(js); console.log('syntax: OK'); }
catch (e) { console.error('syntax FAIL:', e.message); process.exit(1); }
const mk = (tag) => {
  const o = { tagName: tag, children: [], _attrs: {}, textContent: '', style: {}, appendChild(c) { this.children.push(c); return c; }, setAttribute(k, v) { this._attrs[k] = v; }, setAttributeNS(n, k, v) { this._attrs[k] = v; } };
  o.style = new Proxy({}, { get: () => '', set: () => true });
  return o;
};
const store = {};
const byId = {};
const doc = {
  getElementById(id) {
    if (id === 'anal') return byId[id] || (byId[id] = { innerHTML: '', appendChild(c) { this.children = this.children || []; this.children.push(c); return c; }, appendElement() {} });
    if (!store[id]) store[id] = mk('div');
    return store[id];
  },
  createElement(t) { const o = mk(t); o.className = ''; o.style = new Proxy({}, { get: () => '', set: () => true }); return o; },
  createElementNS(ns, t) { return mk(t); },
  createTextNode(s) { return { textContent: String(s) }; },
  documentElement: {},
};
global.document = doc;
global.getComputedStyle = () => ({ getPropertyValue: (n) => '' });
try { (new Function('document', 'getComputedStyle', js))(doc, global.getComputedStyle); }
catch (e) { console.error('runtime FAIL:', e.message, e.stack); process.exit(1); }
const svgCount = (id) => { const c = doc.getElementById(id); let s = 0; const walk = (n) => { if (!n || !n.children) return; for (const ch of n.children) { if (ch.tagName === 'svg') s++; walk(ch); } }; walk(c); return s; };
['A', 'B', 'C', 'D', 'E'].forEach(id => console.log(id, 'svg=', svgCount(id)));
console.log('anal children:', byId.anal && byId.anal.children ? byId.anal.children.length : 0);
console.log('all good');