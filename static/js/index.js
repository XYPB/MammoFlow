const navToggle = document.querySelector('[data-nav-toggle]');
const nav = document.querySelector('[data-nav]');

if (navToggle && nav) {
  navToggle.addEventListener('click', () => {
    const isOpen = nav.classList.toggle('is-open');
    navToggle.setAttribute('aria-expanded', String(isOpen));
  });

  nav.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      nav.classList.remove('is-open');
      navToggle.setAttribute('aria-expanded', 'false');
    });
  });
}

const copyButton = document.querySelector('[data-copy-citation]');
const citation = document.querySelector('#citation code');

if (copyButton && citation) {
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(citation.textContent.trim());
      copyButton.textContent = 'Copied';
      window.setTimeout(() => {
        copyButton.textContent = 'Copy';
      }, 1800);
    } catch (error) {
      copyButton.textContent = 'Select';
    }
  });
}
