(() => {
  const $ = (sel, root = document) => root.querySelector(sel);

  // Theme: system -> light -> dark
  const themeBtn = $('#theme');
  if (themeBtn && window.__theme && window.__themeMode) {
    themeBtn.addEventListener('click', () => {
      const order = ['light', 'dark', 'system'];
      const next = order[(order.indexOf(window.__themeMode()) + 1) % order.length];
      window.__theme(next);
    });
    themeBtn.dataset.enhanced = '';
  }

  // The navigation download follows the visitor's OS; the hero shows both
  // desktop installers explicitly.
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
    nav.dataset.enhanced = '';
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
    const updateLanguageLinks = () => {
      langMenu.querySelectorAll('a[hreflang]').forEach((link) => {
        const target = new URL(link.href);
        target.hash = window.location.hash;
        link.href = target.href;
      });
    };
    updateLanguageLinks();
    window.addEventListener('hashchange', updateLanguageLinks);
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
    langMenu.hidden = true;
    langBtn.closest('.langs').dataset.enhanced = '';
  }

  // Hairline under the nav once the page scrolls
  if (nav) {
    const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 4);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // A build-time source version may differ from the current desktop download.
  // Show a version only after the live release metadata supplies it.
  const applyReleaseTag = (tag) => {
    document.querySelectorAll('[data-release-tag]').forEach((node) => {
      node.textContent = tag + ' · ';
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
