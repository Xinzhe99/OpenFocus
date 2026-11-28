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

  // Keep version labels and download cards on the current stable release.
  (async () => {
    try {
      const res = await fetch('/api/latest', { headers: { accept: 'application/json' } });
      if (!res.ok) return;
      const data = await res.json();
      if (!data || typeof data.tag !== 'string' || !/^v\d+(?:\.\d+)*$/.test(data.tag) ||
          data.tag.split('.').some((part) => /^0*999999999$/.test(part)) || data.actual_tag) return;
      applyReleaseTag(data.tag);
      const linuxStatus = $('#linux-status');
      const linuxDownload = $('#linux-download');
      if (linuxStatus && linuxDownload && data.assets && typeof data.assets === 'object') {
        const available = !!data.assets['linux-desktop'];
        linuxStatus.textContent = available ? linuxStatus.dataset.available : linuxStatus.dataset.unavailable;
        linuxDownload.hidden = !available;
        if (available) {
          const directory = `openfocus-${data.tag}`;
          const archive = `OpenFocus-${data.tag}-linux-x86_64.tar.gz`;
          $('#linux-command').textContent = `mkdir ${directory} &&\ntar -xzf ${archive} -C ${directory} &&\ncd ${directory} &&\n./OpenFocus/OpenFocus`;
          $('#linux-install').hidden = false;
        }
      }
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

  // Hydrate the Contributors Wall with the live synced dataset when present.
  (() => {
    const table = $('#contrib-table');
    const container = $('#contrib-rows');
    if (!table || !container) return;

    const lang = table.dataset.lang === 'zh' ? 'zh' : 'en';
    const repo = table.dataset.repo || 'https://github.com/Xinzhe99/OpenFocus';
    const nextLabel = table.dataset.next || 'Next (main)';
    const moreFmt = table.dataset.more || '{n} more';
    const statsFmt = table.dataset.statsFmt || '';
    const profileRe = /^https:\/\/github\.com\/[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$/;
    const tagRe = /^v[0-9]+(?:\.[0-9]+)*$/;

    const buildItemsList = (entries) => {
      const ul = document.createElement('ul');
      ul.className = 'contrib-items';
      for (const entry of entries) {
        const issueNum = Number(entry?.issue);
        const ver = typeof entry?.version === 'string' ? entry.version.trim() : '';
        const text = typeof entry?.[lang] === 'string' ? entry[lang].trim() : '';
        if (!Number.isSafeInteger(issueNum) || issueNum <= 0 || !text) continue;
        if (ver !== 'next' && !tagRe.test(ver)) continue;

        const li = document.createElement('li');
        li.className = 'contrib-item';

        const verLink = document.createElement('a');
        verLink.rel = 'noopener';
        if (ver === 'next') {
          verLink.className = 'contrib-ver is-next';
          verLink.href = `${repo}/commits/main`;
          verLink.textContent = nextLabel;
        } else {
          verLink.className = 'contrib-ver';
          verLink.href = `${repo}/releases/tag/${ver}`;
          verLink.textContent = ver;
        }

        const span = document.createElement('span');
        span.className = 'contrib-text';
        span.textContent = text;

        const issueLink = document.createElement('a');
        issueLink.className = 'contrib-issue';
        issueLink.rel = 'noopener';
        issueLink.href = `${repo}/issues/${issueNum}`;
        issueLink.textContent = `#${issueNum}`;

        li.append(verLink, span, issueLink);
        ul.appendChild(li);
      }
      return ul;
    };

    (async () => {
      try {
        const res = await fetch('/api/contributors', { headers: { accept: 'application/json' } });
        if (!res.ok) return;
        const data = await res.json();
        if (!data || !Array.isArray(data.contributors) || !data.contributors.length) return;

        const rows = [];
        let totalChanges = 0;
        for (const person of data.contributors) {
          const handle = typeof person?.handle === 'string' ? person.handle.trim() : '';
          const url = typeof person?.url === 'string' ? person.url.trim() : '';
          const items = Array.isArray(person?.contributions) ? person.contributions : [];
          if (!handle || !profileRe.test(url) || !items.length) continue;

          const row = document.createElement('div');
          row.className = 'contrib-row';

          const who = document.createElement('div');
          who.className = 'contrib-who';
          const handleLink = document.createElement('a');
          handleLink.className = 'contrib-handle';
          handleLink.href = url;
          handleLink.rel = 'noopener';
          handleLink.textContent = `@${handle}`;
          const tag = document.createElement('span');
          tag.className = 'contrib-tag';
          tag.textContent = typeof person.platform === 'string' && person.platform.trim() ? person.platform.trim() : 'GitHub';
          who.append(handleLink, tag);

          const changes = document.createElement('div');
          changes.className = 'contrib-changes';
          const headList = buildItemsList(items.slice(0, 3));
          changes.appendChild(headList);
          const tail = items.slice(3);
          if (tail.length) {
            const details = document.createElement('details');
            details.className = 'contrib-more';
            const summary = document.createElement('summary');
            summary.textContent = moreFmt.replace('{n}', String(tail.length));
            details.append(summary, buildItemsList(tail));
            changes.appendChild(details);
          }
          totalChanges += headList.childElementCount + (tail.length ? detailsCount(changes) : 0);
          row.append(who, changes);
          rows.push(row);
        }
        if (!rows.length) return;
        container.replaceChildren(...rows);
        const statsEl = $('#contrib-stats');
        if (statsEl && statsFmt) {
          statsEl.textContent = statsFmt
            .replace('{people}', String(rows.length))
            .replace('{changes}', String(totalChanges));
        }
      } catch {}
    })();

    function detailsCount(containerEl) {
      return containerEl.querySelectorAll('.contrib-more .contrib-item').length;
    }
  })();
})();
