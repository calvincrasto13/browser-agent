// Injected via page.evaluate() — returns indexed interactive elements
// Mirrors browser-use's buildDomTree approach
() => {
  const INTERACTIVE = [
    'a', 'button', 'input', 'select', 'textarea',
    '[role="button"]', '[role="link"]', '[role="checkbox"]',
    '[role="menuitem"]', '[role="tab"]', '[role="option"]',
    '[role="radio"]', '[role="combobox"]', '[role="switch"]'
  ].join(',');

  const map = {};
  const lines = [];
  let idx = 0;

  const isVisible = (el) => {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    if (el.offsetParent === null && el.tagName !== 'BODY') return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
    return true;
  };

  document.querySelectorAll(INTERACTIVE).forEach(el => {
    if (!isVisible(el)) return;
    if (idx >= 150) return; // cap at 150 elements

    const tag  = el.tagName.toLowerCase();
    const type = el.type         ? ` type="${el.type}"`         : '';
    const name = el.name         ? ` name="${el.name}"`         : '';
    const ph   = el.placeholder  ? ` placeholder="${el.placeholder.slice(0,40)}"` : '';
    const href = el.href         ? ` href="${el.href.slice(0,80)}"` : '';
    const role = el.getAttribute('role') ? ` role="${el.getAttribute('role')}"` : '';
    const aria = el.getAttribute('aria-label') ? ` aria-label="${el.getAttribute('aria-label').slice(0,60)}"` : '';
    const sel  = el.id   ? `#${el.id}`
               : el.name ? `[name="${el.name}"]`
               : '';

    const text = (
      el.getAttribute('aria-label') ||
      el.getAttribute('placeholder') ||
      el.innerText?.trim() ||
      el.value ||
      el.getAttribute('title') ||
      ''
    ).slice(0, 80).replace(/\s+/g, ' ');

    map[idx] = { el, selector: sel, tag };
    lines.push(`[${idx}]<${tag}${type}${name}${ph}${href}${role}${aria}>${text}</${tag}>`);
    idx++;
  });

  // Store map globally in page context for actor to use
  window.__AGENT_ELEM_MAP = map;
  return { elements: lines, count: idx };
}
