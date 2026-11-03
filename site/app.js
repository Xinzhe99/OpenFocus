(() => {
  const $ = (sel, root = document) => root.querySelector(sel);

  // Theme: system -> light -> dark
  const themeBtn = $('#theme');
  if (themeBtn && window.__theme) {
    themeBtn.addEventListener('click', () => {
      const order = ['light', 'dark', 'system'];
      const next = order[(order.indexOf(window.__themeMode()) + 1) % order.length];
      window.__theme(next);
    });
  }

  // Every download button follows the visitor's OS, and /api/latest can
  // re-point them at the current release.
  const dlButtons = Array.from(document.querySelectorAll('[data-url-win]'));
  const applyDownload = () => {
    if (!dlButtons.length) return;
    const ua = navigator.userAgent;
    const os = /Mac/i.test(ua) ? 'mac' : /Win/i.test(ua) ? 'win' : 'linux';
    document.documentElement.dataset.os = os;
    const key = os[0].toUpperCase() + os.slice(1);
    dlButtons.forEach((el) => {
      const url = el.dataset['url' + key];
      if (url) el.href = url;
      const label = el.dataset['label' + key];
      const text = $('span', el);
      if (label && text) text.textContent = label;
    });
  };
  applyDownload();

  // Copy every install command (pip, curl, PowerShell)
  document.querySelectorAll('.copy').forEach((copyBtn) => {
    const source = copyBtn.closest('.install')?.querySelector('code');
    copyBtn.addEventListener('click', async () => {
      const text = (source ? source.textContent : '').replace(/^\$\s*/, '').trim();
      let ok = false;
      try {
        await navigator.clipboard.writeText(text);
        ok = true;
      } catch {
        const area = document.createElement('textarea');
        area.value = text;
        area.setAttribute('readonly', '');
        area.style.position = 'fixed';
        area.style.opacity = '0';
        document.body.appendChild(area);
        area.select();
        ok = document.execCommand('copy');
        area.remove();
      }
      if (!ok) return;
      const original = copyBtn.textContent;
      copyBtn.textContent = copyBtn.dataset.copied || original;
      copyBtn.classList.add('done');
      setTimeout(() => {
        copyBtn.textContent = original;
        copyBtn.classList.remove('done');
      }, 1600);
    });
  });

  // Mobile navigation
  const nav = $('#nav');
  const menuBtn = $('#menu');
  if (nav && menuBtn) {
    menuBtn.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      menuBtn.setAttribute('aria-expanded', String(open));
    });
    nav.querySelectorAll('.links a').forEach((a) => {
      a.addEventListener('click', () => {
        nav.classList.remove('open');
        menuBtn.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // Language menu
  const langBtn = $('#langbtn');
  const langMenu = $('#langmenu');
  if (langBtn && langMenu) {
    const close = () => {
      langMenu.hidden = true;
      langBtn.setAttribute('aria-expanded', 'false');
    };
    langBtn.addEventListener('click', () => {
      const show = langMenu.hidden;
      langMenu.hidden = !show;
      langBtn.setAttribute('aria-expanded', String(show));
    });
    langMenu.querySelectorAll('a[hreflang]').forEach((link) => {
      link.addEventListener('click', () => {
        const lang = link.getAttribute('hreflang');
        if (lang) document.cookie = `lang=${lang}; path=/; max-age=31536000; SameSite=Lax`;
      });
    });
    document.addEventListener('click', (e) => {
      if (!langMenu.hidden && !langMenu.contains(e.target) && !langBtn.contains(e.target)) close();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        close();
        if (nav) nav.classList.remove('open');
        if (menuBtn) menuBtn.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // Hairline under the nav once the page scrolls
  if (nav) {
    const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 4);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // Replace the version in text nodes only: .note also holds the release link
  // and <code> examples, which must survive the update.
  const applyReleaseTag = (tag) => {
    document.querySelectorAll('.meta, .note').forEach((root) => {
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      nodes.forEach((node) => {
        const text = node.nodeValue || '';
        if (!/v\d+\.\d+\.\d+/.test(text)) return;
        node.nodeValue = text.replace(/v\d+\.\d+\.\d+/g, () => tag);
      });
    });
  };

  // Keep version labels and download cards on the latest GitHub release.
  (async () => {
    try {
      const res = await fetch('/api/latest', { headers: { accept: 'application/json' } });
      if (!res.ok) return;
      const data = await res.json();
      if (!data || !data.tag) return;
      applyReleaseTag(data.tag);
      document.querySelectorAll('a[data-kind]').forEach((a) => {
        const asset = data.assets && data.assets[a.dataset.kind];
        if (!asset || !asset.url) return;
        a.href = '/download/latest/' + a.dataset.kind;
        const size = a.querySelector('span');
        if (size && asset.size) {
          const mb = Math.max(1, Math.round(asset.size / 1000000));
          size.textContent += ' · ' + mb + ' MB';
        }
      });
      dlButtons.forEach((el) => {
        if (data.assets && data.assets['windows-setup']) {
          el.dataset.urlWin = '/download/latest/windows-setup';
        }
        if (data.assets && data.assets['macos-dmg']) {
          el.dataset.urlMac = '/download/latest/macos-dmg';
        }
      });
      applyDownload();
    } catch {}
  })();
})();
