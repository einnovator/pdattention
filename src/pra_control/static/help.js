(() => {
  'use strict';

  const pages = [
    { slug:'index', title:'Control Plane help', description:'Start here: workspace layout, navigation, themes, and access.' },
    { slug:'fleet', title:'Fleet and engines', description:'Inspect engine health, drift, capabilities, sessions, storage, and observability.' },
    { slug:'registry', title:'Registry and governance', description:'Models, bundles, profiles, qualifications, compatibility, policies, and approvals.' },
    { slug:'routers', title:'Routers and routes', description:'Inspect external routing data planes, eligibility, drift, and reconciliation.' },
    { slug:'agent', title:'PRA Agent', description:'Use governed chat for fleet questions and operator assistance.' },
    { slug:'activity', title:'Audit and alerts', description:'Review alert state and the immutable operator action trail.' },
  ];
  const cache = new Map();
  const history = [];
  let current = null;

  const page = slug => pages.find(item => item.slug === slug) || pages[0];
  const fetchPage = async slug => {
    if (cache.has(slug)) return cache.get(slug);
    const response = await fetch(`/static/help/${encodeURIComponent(slug)}.md`, { credentials:'same-origin' });
    if (!response.ok) throw new Error(`Help page could not be loaded (${response.status})`);
    const text = await response.text();
    cache.set(slug, text);
    return text;
  };
  const safeMarkdown = text => {
    if (!globalThis.marked || !globalThis.DOMPurify) return $('<pre>').text(text)[0].outerHTML;
    return globalThis.DOMPurify.sanitize(globalThis.marked.parse(text));
  };
  const updateBack = () => $('#help-back').prop('disabled', history.length === 0);
  const loadPage = async (slug, remember=true) => {
    const selected = page(slug);
    if (remember && current && current !== selected.slug) history.push(current);
    current = selected.slug;
    $('#help-title').text(selected.title);
    $('#help-content').html('<div class="loading-state" role="status" aria-live="polite"><span class="loading-spinner" aria-hidden="true"></span><span>Loading help</span></div>');
    $('#help-search-results').addClass('d-none').empty();
    $('#help-search').val('');
    updateBack();
    try {
      $('#help-content').html(safeMarkdown(await fetchPage(selected.slug))).scrollTop(0);
    } catch (error) {
      $('#help-content').html($('<div class="alert alert-danger">').text(error.message));
    }
  };
  const open = async slug => {
    $('#help-overlay,#help-drawer').addClass('open');
    $('#help-drawer').attr('aria-hidden','false');
    $('body').addClass('help-open');
    await loadPage(slug || current || 'index', false);
    window.setTimeout(() => $('#help-search').trigger('focus'), 210);
  };
  const close = () => {
    $('#help-overlay,#help-drawer').removeClass('open');
    $('#help-drawer').attr('aria-hidden','true');
    $('body').removeClass('help-open');
    $('#help-toggle').trigger('focus');
  };
  const showSearch = async term => {
    const query = term.trim().toLowerCase();
    if (!query) { $('#help-search-results').addClass('d-none').empty(); return; }
    const documents = await Promise.all(pages.map(async item => ({ item, text:await fetchPage(item.slug) })));
    const matches = documents.filter(({item,text}) => `${item.title} ${item.description} ${text}`.toLowerCase().includes(query));
    const host = $('#help-search-results').removeClass('d-none').empty();
    if (!matches.length) { host.append($('<div class="empty-state py-4">').text('No help pages match this search.')); return; }
    matches.forEach(({item,text}) => {
      const plain=text.replace(/[#>*_`\[\]()|-]/g,' ').replace(/\s+/g,' ').trim();
      const at=plain.toLowerCase().indexOf(query);
      const snippet=plain.slice(Math.max(0,at-45),Math.min(plain.length,at+query.length+90));
      const button=$('<button type="button" class="help-result">').attr('data-help-page',item.slug);
      button.append($('<strong>').text(item.title),$('<span>').text(snippet));
      host.append(button);
    });
  };

  $(document).on('click','#help-toggle,#open-help-menu',()=>open('index'))
    .on('click','#help-close,#help-overlay',close)
    .on('click','#help-home',()=>loadPage('index'))
    .on('click','#help-back',()=>{const previous=history.pop();if(previous)loadPage(previous,false);})
    .on('click','[data-help-page]',function(){loadPage($(this).data('help-page'));})
    .on('click','#help-content a',function(event){const href=$(this).attr('href')||'';const match=href.match(/(?:^|\/)([a-z0-9-]+)\.md(?:#.*)?$/i);if(match){event.preventDefault();loadPage(match[1].toLowerCase());}})
    .on('input','#help-search',function(){showSearch($(this).val()).catch(error=>$('#help-search-results').removeClass('d-none').text(error.message));})
    .on('keydown',event=>{if(event.key==='Escape'&&$('#help-drawer').hasClass('open'))close();});

  window.PRAHelp = { open, close, loadPage };
})();
