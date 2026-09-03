(() => {
  'use strict';

  const state = {
    me: null,
    csrf: '',
    fleet: { items: [], summary: {} },
    fleetLoaded: false,
    panels: new Map(),
    registryRecords: {},
    fleetFilters: { text: '', engine: '', model: '', status: '' },
    fleetSort: { key: 'name', direction: 'asc' },
    socket: null,
    reconnectTimer: null,
    retry: 500,
    agentBuffer: null,
    agentModels: [],
    activeAgentTarget: '',
  };

  const FIELD_HELP = {
    engine: 'Runtime family serving this instance, such as MLX, vLLM, SGLang, or OpenVINO.',
    engine_state: 'Immediate reachability reported by the engine management endpoint. This is independent of Registry drift.',
    desired_state: 'Comparison between the engine observation and the Registry desired deployment.',
    status: 'Current normalized state for this resource.',
    model: 'Model identity currently loaded by the runtime.',
    loaded_models: 'Number of runtime model aliases currently loaded on this engine.',
    profile: 'Named PRA quality and resource policy applied to the model.',
    mode: 'Selected-context transport mode currently used by the runtime.',
    ttft_p95: 'The 95th percentile time to first generated token for the measured interval.',
    alerts: 'Current conditions that may need operator attention.',
    total: 'Number of engine instances visible to this Control Plane.',
    in_sync: 'Engines whose observed model deployment matches Registry intent.',
    drift: 'Engines whose observed model, profile, mode, or revision differs from Registry intent.',
    offline: 'Engines whose management endpoint could not be reached.',
    unknown: 'Reachable engines for which no comparable Registry desired deployment is available.',
    revision: 'Immutable or monotonic version used to compare desired and observed state.',
    fingerprint: 'Content-derived model identity reported by the engine.',
    bundle: 'PRA bundle associated with the loaded model.',
    capabilities: 'Features advertised by this runtime or router.',
    storage: 'PRA native-context residency and lifecycle information.',
    sessions: 'Long-running runtime or agent sessions visible to the current identity.',
    route: 'Stable public model alias and its eligible backend pools.',
    policy: 'Deterministic eligibility, preference, weighting, and fallback rules.',
    health: 'Service-reported liveness or readiness, separate from configuration drift.',
    default: 'A Control Plane field reported by the selected service.',
  };

  const esc = value => $('<div>').text(value == null ? '' : String(value)).html();
  const api = (url, options = {}) => $.ajax({
    url,
    contentType: 'application/json',
    ...options,
    headers: { ...(options.headers || {}), ...(state.csrf ? { 'X-CSRF-Token': state.csrf } : {}) },
  });
  const panelQuery = (root, selector) => $(root).find(selector);
  const slug = value => String(value).replace(/[^a-zA-Z0-9_.:-]+/g, '-');
  const titleCase = value => String(value || '').replaceAll('_', ' ').replace(/\b\w/g, char => char.toUpperCase());
  const helpKey = value => String(value || 'default').toLowerCase().replaceAll(' ', '_').replace(/[^a-z0-9_]/g, '');
  const infoButton = (key, label = titleCase(key)) => `<button class="field-info" data-info-key="${esc(helpKey(key))}" data-info-label="${esc(label)}" aria-label="Explain ${esc(label)}" aria-haspopup="dialog"><i data-lucide="info"></i></button>`;
  const infoLabel = (key, label = titleCase(key)) => `${esc(label)}${infoButton(key, label)}`;
  const status = value => {
    const normalized = String(value || 'UNKNOWN').toUpperCase().replaceAll(' ', '_');
    return `<span class="status-pill status-${esc(normalized)}">${esc(normalized.replaceAll('_', ' '))}</span>`;
  };
  const engineHealth = row => {
    if (row.status === 'OFFLINE') return 'OFFLINE';
    const health = String(row.health || '').toUpperCase();
    return ['HEALTHY', 'READY', 'OK', 'ONLINE'].includes(health) ? 'READY' : (health || 'REACHABLE');
  };
  const notify = (text, level = 'success') => {
    const id = `toast-${Date.now()}`;
    $('#toast-stack').append(`<div id="${id}" class="toast text-bg-${level}" role="status"><div class="d-flex"><div class="toast-body">${esc(text)}</div><button class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div></div>`);
    bootstrap.Toast.getOrCreateInstance(document.getElementById(id), { delay: 4000 }).show();
  };
  const renderError = (error, root) => {
    const message = error.responseJSON?.error?.message || error.responseJSON?.detail || error.statusText || error.message || error;
    $(root).attr('aria-busy', 'false');
    panelQuery(root, '.workspace-content').html(`<div class="alert alert-danger">${esc(message)}</div>`);
  };
  const loadingMarkup = label => `<div class="loading-state" role="status" aria-live="polite"><span class="loading-spinner" aria-hidden="true"></span><span>${esc(label)}</span></div>`;
  const setPanelLoading = (root, label) => {
    $(root).attr('aria-busy', 'true');
    panelQuery(root, '.summary-strip').addClass('d-none');
    panelQuery(root, '.workspace-content').html(loadingMarkup(label));
  };
  const closeInfo = () => globalThis.tippy?.hideAll();

  const fieldHelpContent = reference => {
    const key = reference.dataset.infoKey;
    const label = reference.dataset.infoLabel || titleCase(key);
    const content = document.createElement('div');
    content.className = 'field-help';
    const eyebrow = document.createElement('div');
    eyebrow.className = 'field-help-eyebrow';
    eyebrow.textContent = 'Field guide';
    const title = document.createElement('strong');
    title.className = 'field-help-title';
    title.textContent = label;
    const description = document.createElement('p');
    description.textContent = FIELD_HELP[key] || `This field reports the ${label.toLowerCase()} value supplied by the active PRA service.`;
    content.append(eyebrow, title, description);
    return content;
  };

  if (globalThis.tippy) {
    globalThis.tippy.delegate(document.body, {
      target: '.field-info',
      trigger: 'click',
      placement: 'right-start',
      interactive: true,
      hideOnClick: true,
      maxWidth: 320,
      theme: 'pra',
      appendTo: () => document.body,
      content: fieldHelpContent,
      onShow(instance) { globalThis.tippy.hideAll({ exclude: instance }); },
    });
  }

  const renderValue = (value, depth = 0) => {
    if (value == null || value === '') return '<span class="empty-value">not reported</span>';
    if (typeof value === 'boolean') return status(value ? 'ENABLED' : 'DISABLED');
    if (Array.isArray(value)) {
      if (!value.length) return '<span class="empty-value">none</span>';
      if (value.every(item => typeof item !== 'object')) return `<div class="value-chips">${value.map(item => `<span>${esc(item)}</span>`).join('')}</div>`;
      return `<div class="nested-list">${value.map(item => `<div>${renderValue(item, depth + 1)}</div>`).join('')}</div>`;
    }
    if (typeof value === 'object') {
      const entries = Object.entries(value);
      if (!entries.length) return '<span class="empty-value">none</span>';
      return `<dl class="detail-grid ${depth ? 'nested-detail-grid' : ''}">${entries.map(([key, item]) => `<div class="detail-field"><dt>${infoLabel(key)}</dt><dd>${renderValue(item, depth + 1)}</dd></div>`).join('')}</dl>`;
    }
    return `<span class="field-value">${esc(value)}</span>`;
  };
  const compactValue = value => {
    if (value == null || value === '') return '<span class="empty-value">not reported</span>';
    if (Array.isArray(value)) return esc(value.map(item => typeof item === 'object' ? (item.id || item.name || 'record') : item).join(', ') || 'none');
    if (typeof value === 'object') return renderValue(value, 1);
    return esc(value);
  };

  const component = () => {
    const host = document.createElement('div');
    host.className = 'workspace-host';
    host.append(document.querySelector('#workspace-template').content.cloneNode(true));
    return {
      element: host,
      init(params) {
        const spec = params.params || { type: 'fleet' };
        state.panels.set(params.api.id, { root: host, spec });
        renderPanel(host, spec).catch(error => renderError(error, host));
      },
      dispose() {
        for (const [id, panel] of state.panels.entries()) if (panel.root === host) state.panels.delete(id);
      },
    };
  };

  const dockHost = document.getElementById('dockview');
  const dv = globalThis['dockview-core'].createDockview(dockHost, { createComponent: component });
  const layoutKey = 'pra-control-central-tabs-v3';
  const savedLayout = localStorage.getItem(layoutKey);
  if (savedLayout) {
    try { dv.fromJSON(JSON.parse(savedLayout)); } catch (_) { localStorage.removeItem(layoutKey); }
  }
  dv.onDidLayoutChange(() => localStorage.setItem(layoutKey, JSON.stringify(dv.toJSON())));
  dv.onDidActivePanelChange(() => closeInfo());
  dv.onDidRemovePanel(() => { if (!dv.panels.length) openView('fleet'); });

  const openCentral = (id, title, spec, preserveExisting = false) => {
    closeInfo();
    const existing = dv.getPanel(id);
    if (existing) {
      existing.api.setActive();
      const tracked = state.panels.get(id);
      if (tracked && !preserveExisting) {
        tracked.spec = spec;
        renderPanel(tracked.root, spec).catch(error => renderError(error, tracked.root));
      }
      return;
    }
    const reference = dv.activePanel || dv.panels[0];
    const options = { id, component: 'workspace', title, params: spec };
    if (reference) options.position = { referencePanel: reference, direction: 'within' };
    dv.addPanel(options).api.setActive();
  };
  const openView = view => {
    const titles = { fleet: 'Fleet', recommendations: 'Recommendations', audit: 'Audit log', alerts: 'Alerts', routers: 'Routers', routes: 'Routes' };
    openCentral(`view:${view}`, titles[view] || titleCase(view), { type: view });
    markNav(view);
  };
  const openRegistry = resource => {
    openCentral(`registry:${resource}`, titleCase(resource), { type: 'registry', resource });
    markNav(null, resource);
  };
  const openEngine = (name, section = null) => {
    openCentral(
      `engine:${slug(name)}`, name,
      { type: 'engine', name, section: section || 'summary' },
      section == null,
    );
    markNav();
  };
  const openRouter = routerId => openCentral(`router:${slug(routerId)}`, routerId, { type: 'router', routerId });

  const panelHeading = (root, title, subtitle, actions = '') => {
    panelQuery(root, '.view-title').text(title);
    panelQuery(root, '.view-subtitle').text(subtitle);
    panelQuery(root, '.view-actions').html(actions);
    lucide.createIcons();
  };
  const markNav = (view, registry) => {
    $('.section-nav button').removeClass('active');
    if (view) $(`.section-nav button[data-open-view="${view}"]`).addClass('active');
    if (registry) $(`.section-nav button[data-registry="${registry}"]`).addClass('active');
  };

  async function renderPanel(root, spec) {
    const renderers = { fleet: renderFleet, recommendations: renderRecommendations, audit: renderActivity, alerts: renderActivity, routers: renderRouters, routes: renderRoutes, registry: renderRegistry, engine: renderEngine, router: renderRouter };
    const renderer = renderers[spec.type];
    if (!renderer) throw new Error(`Unknown panel type: ${spec.type}`);
    closeInfo();
    setPanelLoading(root, `Loading ${titleCase(spec.type)}`);
    try {
      await renderer(root, spec);
      $(root).attr('aria-busy', 'false');
      lucide.createIcons();
    } catch (error) {
      renderError(error, root);
      throw error;
    }
  }

  const loadFleet = async () => {
    const affected = [...state.panels.values()].filter(panel => ['fleet', 'engine', 'alerts', 'recommendations'].includes(panel.spec.type));
    affected.forEach(panel => setPanelLoading(panel.root, 'Loading remote fleet state'));
    try {
      state.fleet = await api('/api/fleet');
      state.fleetLoaded = true;
    } catch (error) {
      affected.forEach(panel => renderError(error, panel.root));
      throw error;
    }
    $('#connection-summary').addClass('live').attr('title', 'Control Plane connected');
    for (const panel of affected) renderPanel(panel.root, panel.spec).catch(error => renderError(error, panel.root));
    await loadAgentModels();
  };

  const renderSummary = (root, summary) => {
    const values = [['total', summary.total], ['in_sync', summary.healthy], ['drift', summary.drift], ['offline', summary.offline], ['unknown', summary.unknown]];
    panelQuery(root, '.summary-strip').removeClass('d-none').html(values.map(([key, value]) => `<div class="summary-item"><div class="summary-value">${value ?? 0}</div><div class="summary-label">${infoLabel(key)}</div></div>`).join(''));
  };
  const fleetRows = () => {
    const filters = state.fleetFilters;
    const rows = state.fleet.items.filter(row => {
      const text = `${row.name} ${row.engine || ''} ${row.model || ''} ${(row.models || []).map(model => `${model.runtime_model_id || ''} ${model.model_id || ''}`).join(' ')}`.toLowerCase();
      return (!filters.text || text.includes(filters.text.toLowerCase()))
        && (!filters.engine || row.engine === filters.engine)
        && (!filters.status || row.status === filters.status)
        && (!filters.model || (row.models || []).some(model => (model.model_id || model.runtime_model_id) === filters.model));
    });
    const getter = {
      name: row => row.name,
      engine_state: row => engineHealth(row),
      status: row => row.status,
      engine: row => row.engine || '',
      loaded_models: row => row.model_count || 0,
      profile: row => row.profile || '',
      ttft_p95: row => Number(row.metrics?.ttft_p95_ms ?? Number.POSITIVE_INFINITY),
      alerts: row => (row.alerts || []).length,
    }[state.fleetSort.key];
    return rows.sort((a, b) => {
      const left = getter(a); const right = getter(b);
      const result = typeof left === 'number' ? left - right : String(left).localeCompare(String(right));
      return state.fleetSort.direction === 'asc' ? result : -result;
    });
  };
  const sortHeader = (key, label) => {
    const active = state.fleetSort.key === key;
    const icon = active ? (state.fleetSort.direction === 'asc' ? 'arrow-up' : 'arrow-down') : 'chevrons-up-down';
    return `<th><span class="sort-header"><button class="sort-button ${active ? 'active' : ''}" data-sort="${key}">${esc(label)}<i class="sort-icon" data-lucide="${icon}"></i></button>${infoButton(key, label)}</span></th>`;
  };
  async function renderFleet(root) {
    panelHeading(root, 'Fleet', 'Reachable engine state and Registry desired-state comparison.');
    if (!state.fleetLoaded) {
      setPanelLoading(root, 'Loading remote fleet state');
      return;
    }
    renderSummary(root, state.fleet.summary);
    const engines = [...new Set(state.fleet.items.map(row => row.engine).filter(Boolean))].sort();
    const models = [...new Set(state.fleet.items.flatMap(row => (row.models || []).map(model => model.model_id || model.runtime_model_id)).filter(Boolean))].sort();
    const statuses = [...new Set(state.fleet.items.map(row => row.status).filter(Boolean))].sort();
    const rows = fleetRows();
    panelQuery(root, '.workspace-content').html(`
      <div class="fleet-filters" aria-label="Fleet filters">
        <div class="filter-search"><i data-lucide="search"></i><input class="form-control form-control-sm" data-fleet-filter="text" value="${esc(state.fleetFilters.text)}" placeholder="Search fleet"></div>
        <select class="form-select form-select-sm" data-fleet-filter="engine"><option value="">All engines</option>${engines.map(value => `<option ${value === state.fleetFilters.engine ? 'selected' : ''}>${esc(value)}</option>`).join('')}</select>
        <select class="form-select form-select-sm" data-fleet-filter="model"><option value="">All models</option>${models.map(value => `<option ${value === state.fleetFilters.model ? 'selected' : ''}>${esc(value)}</option>`).join('')}</select>
        <select class="form-select form-select-sm" data-fleet-filter="status"><option value="">All desired states</option>${statuses.map(value => `<option ${value === state.fleetFilters.status ? 'selected' : ''}>${esc(value)}</option>`).join('')}</select>
        <span class="filter-count">${rows.length} of ${state.fleet.items.length}</span>
      </div>
      <div class="table-responsive"><table class="table table-hover data-grid"><thead><tr>
        ${sortHeader('name', 'Engine')}${sortHeader('engine_state', 'Engine state')}${sortHeader('status', 'Desired state')}${sortHeader('engine', 'Runtime')}${sortHeader('loaded_models', 'Loaded models')}${sortHeader('profile', 'Profile / mode')}${sortHeader('ttft_p95', 'TTFT p95')}${sortHeader('alerts', 'Alerts')}
      </tr></thead><tbody>${rows.map(row => `<tr>
        <td><button class="table-link" data-open-engine="${esc(row.name)}">${esc(row.name)}</button><div class="text-secondary">${esc(row.cluster || '')} / ${esc(row.environment || '')}</div></td>
        <td>${status(engineHealth(row))}</td><td>${status(row.status)}</td>
        <td>${esc(row.engine || 'not reported')}<div class="text-secondary">${esc(row.engine_version || '')}</div></td>
        <td><strong>${esc(row.model_count ?? 0)}</strong><div class="text-secondary text-truncate-cell">${esc((row.models || []).map(model => model.runtime_model_id || 'default').join(', ') || 'none')}</div></td>
        <td>${esc(row.profile || 'not reported')}<div class="text-secondary">${esc(row.mode || '')}</div></td>
        <td>${esc(row.metrics?.ttft_p95_ms ?? 'not measured')}</td><td>${esc((row.alerts || []).join(', ') || 'none')}</td>
      </tr>`).join('') || '<tr><td colspan="8"><div class="empty-state">No engines match these filters</div></td></tr>'}</tbody></table></div>`);
  }

  const engineTabs = (name, active) => `<nav class="engine-tabs">${['summary', 'capabilities', 'config', 'models', 'sessions', 'resources', 'storage', 'observability', 'audit'].map(tab => `<button data-engine-section="${tab}" data-engine="${esc(name)}" class="${tab === active ? 'active' : ''}">${titleCase(tab)}</button>`).join('')}</nav>`;
  const engineActions = name => {
    const row = state.fleet.items.find(item => item.name === name) || {};
    const actions = ['prefetch', 'promote', 'demote', 'evict', 'maintenance', ...(row.capabilities?.dynamic_model_load ? ['load-model'] : [])];
    return `<button class="icon-button" data-observe="${esc(name)}" title="Open Grafana"><i data-lucide="chart-no-axes-combined"></i></button><div class="dropdown"><button class="icon-button" data-bs-toggle="dropdown" title="Engine actions"><i data-lucide="ellipsis-vertical"></i></button><ul class="dropdown-menu dropdown-menu-end">${actions.map(action => `<li><button class="dropdown-item" data-action="${action}" data-engine="${esc(name)}">${titleCase(action)}</button></li>`).join('')}</ul></div>`;
  };
  async function renderEngine(root, spec) {
    const row = state.fleet.items.find(item => item.name === spec.name) || {};
    panelHeading(root, spec.name, `${row.engine || 'Engine'} / ${row.cluster || 'unknown cluster'} / ${row.environment || 'unknown environment'}`, engineActions(spec.name));
    panelQuery(root, '.summary-strip').addClass('d-none');
    panelQuery(root, '.workspace-content').html(`${engineTabs(spec.name, spec.section)}<div class="engine-section">${loadingMarkup(`Loading ${spec.section}`)}</div>`);
    try {
      const value = await api(`/api/engines/${encodeURIComponent(spec.name)}/${spec.section}`);
      const host = panelQuery(root, '.engine-section');
      if (Array.isArray(value?.items) && value.items.every(item => item.runtime_model_id)) {
        host.html(`<div class="model-runtime-list">${value.items.map(model => {
          const drift = row.drift?.models?.[model.runtime_model_id];
          return `<section class="model-runtime"><header><div><strong>${esc(model.runtime_model_id)}</strong><span>${esc(model.model_id)}</span></div>${status(drift?.status || 'UNKNOWN')}</header>${renderValue({ revision: model.revision, fingerprint: model.model_fingerprint, bundle: model.pra_bundle_id, profile: model.profile, mode: model.execution_mode, state: model.runtime_state })}${row.capabilities?.dynamic_model_unload ? `<button class="btn btn-sm btn-outline-danger" data-action="unload-model" data-engine="${esc(spec.name)}" data-runtime-model-id="${esc(model.runtime_model_id)}">Unload</button>` : ''}</section>`;
        }).join('')}</div>`);
      } else host.html(renderValue(value));
    } catch (error) {
      panelQuery(root, '.engine-section').html(`<div class="alert alert-danger">${esc(error.responseJSON?.detail || error.statusText || error)}</div>`);
    }
  }

  async function renderRegistry(root, spec) {
    const canWrite = ['Approver', 'Administrator'].includes(state.me?.role) && !['audit', 'instances'].includes(spec.resource);
    panelHeading(root, titleCase(spec.resource), 'Authoritative Registry records and qualification provenance.', canWrite ? `<button class="icon-button primary" data-registry-create="${esc(spec.resource)}" title="Create record"><i data-lucide="plus"></i></button>` : '');
    panelQuery(root, '.summary-strip').addClass('d-none');
    panelQuery(root, '.workspace-content').html(loadingMarkup('Loading Registry'));
    try {
      const result = await api(`/api/registry/${spec.resource}`);
      const records = result.items || result || [];
      state.registryRecords[spec.resource] = records;
      if (!records.length) return panelQuery(root, '.workspace-content').html('<div class="empty-state">No records found</div>');
      const keys = [...new Set(records.flatMap(row => Object.keys(row)))].slice(0, 8);
      const canApprove = canWrite && ['bundles', 'profiles', 'policies', 'qualifications', 'deployments'].includes(spec.resource);
      panelQuery(root, '.workspace-content').html(`<div class="table-responsive"><table class="table data-grid"><thead><tr>${keys.map(key => `<th>${infoLabel(key)}</th>`).join('')}${canWrite ? `<th>${infoLabel('actions')}</th>` : ''}</tr></thead><tbody>${records.map((row, index) => `<tr>${keys.map(key => `<td>${compactValue(row[key])}</td>`).join('')}${canWrite ? `<td class="text-nowrap"><button class="icon-button d-inline-grid" data-registry-edit="${esc(spec.resource)}" data-record-index="${index}" title="Edit record"><i data-lucide="pencil"></i></button>${canApprove ? `<button class="btn btn-sm btn-outline-success" data-registry-approve="${esc(spec.resource)}" data-record-id="${esc(row.id)}">Approve</button>` : ''}</td>` : ''}</tr>`).join('')}</tbody></table></div>`);
    } catch (error) { renderError(error, root); }
  }

  async function renderRouters(root) {
    panelHeading(root, 'Routers', 'External routing data planes reconciled from Registry intent.');
    panelQuery(root, '.summary-strip').addClass('d-none');
    try {
      const rows = (await api('/api/routers')).items || [];
      panelQuery(root, '.workspace-content').html(rows.length ? `<div class="table-responsive"><table class="table table-hover data-grid"><thead><tr><th>${infoLabel('route', 'Router')}</th><th>${infoLabel('engine', 'Kind')}</th><th>${infoLabel('health')}</th><th>${infoLabel('location', 'Region / cluster')}</th><th>${infoLabel('revision', 'Desired')}</th><th>${infoLabel('revision', 'Observed')}</th><th>${infoLabel('drift')}</th></tr></thead><tbody>${rows.map(row => `<tr><td><button class="table-link" data-open-router="${esc(row.id)}">${esc(row.id)}</button></td><td>${esc(row.kind)}</td><td>${status(row.health)}</td><td>${esc(row.region)} / ${esc(row.cluster)}</td><td>${esc(row.desired_revision)}</td><td>${esc(row.observed_revision)}</td><td>${row.desired_revision === row.observed_revision && !row.last_error ? status('IN_SYNC') : status('DRIFT')}</td></tr>`).join('')}</tbody></table></div>` : '<div class="empty-state">No routers registered</div>');
    } catch (error) { renderError(error, root); }
  }

  async function renderRoutes(root) {
    panelHeading(root, 'Routes', 'Stable public aliases, route kinds, policies, and model-pool membership.');
    panelQuery(root, '.summary-strip').addClass('d-none');
    try {
      const rows = (await api('/api/routes')).items || [];
      panelQuery(root, '.workspace-content').html(rows.length ? `<div class="table-responsive"><table class="table data-grid"><thead><tr><th>${infoLabel('route')}</th><th>${infoLabel('model', 'Public model')}</th><th>${infoLabel('route_kind', 'Kind')}</th><th>${infoLabel('policy')}</th><th>${infoLabel('pools')}</th><th>${infoLabel('status', 'Enabled')}</th></tr></thead><tbody>${rows.map(row => `<tr><td>${esc(row.id)}</td><td><strong>${esc(row.public_model)}</strong></td><td>${esc(row.route_kind)}</td><td>${esc(row.policy_id)}</td><td>${esc([...(row.pool_ids || []), ...(row.fallback_pool_ids || []).map(id => `${id} (fallback)`)].join(', '))}</td><td>${status(row.enabled ? 'ENABLED' : 'DISABLED')}</td></tr>`).join('')}</tbody></table></div>` : '<div class="empty-state">No routes registered</div>');
    } catch (error) { renderError(error, root); }
  }

  async function renderRouter(root, spec) {
    panelHeading(root, spec.routerId, 'Observed router state, desired configuration, and deterministic reconciliation drift.');
    panelQuery(root, '.summary-strip').addClass('d-none');
    try {
      const value = await api(`/api/routers/${encodeURIComponent(spec.routerId)}`);
      const drift = value.drift || {}; const router = value.router || {}; const operations = drift.operations || [];
      const apply = ['Approver', 'Administrator'].includes(state.me?.role) && operations.length ? `<button class="btn btn-sm btn-danger" data-router-apply="${esc(spec.routerId)}"><i data-lucide="upload-cloud"></i> Apply ${operations.length} changes</button>` : '';
      panelHeading(root, spec.routerId, `${router.kind || 'router'} / ${router.region || 'unknown region'} / ${router.cluster || 'unknown cluster'}`, apply);
      const summary = { desired_revision: drift.desired_revision, observed_revision: drift.observed_revision, changes: operations.length, capabilities: (value.capabilities || []).length };
      panelQuery(root, '.workspace-content').html(`${renderValue(summary)}<h2 class="section-heading">Reconciliation plan</h2>${operations.length ? `<div class="operation-list">${operations.map(item => `<section class="operation-item"><header>${status(item.action)} <strong>${esc(item.resource_id)}</strong></header><div class="operation-sides"><div><h3>Before</h3>${renderValue(item.before)}</div><div><h3>After</h3>${renderValue(item.after)}</div></div></section>`).join('')}</div>` : '<div class="alert alert-success">Router is in sync.</div>'}<h2 class="section-heading">Capabilities</h2>${renderValue(value.capabilities || [])}`);
    } catch (error) { renderError(error, root); }
  }

  async function renderRecommendations(root) {
    panelHeading(root, 'Recommendations', 'Read-only optimization suggestions. Every change requires human approval.');
    panelQuery(root, '.summary-strip').addClass('d-none');
    try {
      const rows = (await api('/api/recommendations')).items || [];
      panelQuery(root, '.workspace-content').html(rows.length ? `<div class="table-responsive"><table class="table data-grid"><thead><tr><th>${infoLabel('engine')}</th><th>${infoLabel('recommendation')}</th><th>${infoLabel('reason')}</th><th>${infoLabel('policy')}</th></tr></thead><tbody>${rows.map(row => `<tr><td><button class="table-link" data-open-engine="${esc(row.engine)}">${esc(row.engine)}</button></td><td>${esc(row.kind)}</td><td>${esc(row.reason)}</td><td>Approval required</td></tr>`).join('')}</tbody></table></div>` : '<div class="empty-state">No active recommendations</div>');
    } catch (error) { renderError(error, root); }
  }

  async function renderActivity(root, spec) {
    const alertsView = spec.type === 'alerts';
    panelHeading(root, alertsView ? 'Alerts' : 'Audit log', alertsView ? 'Current fleet conditions requiring operator attention.' : 'Governed engine actions and Registry mutations.', '<button class="icon-button" data-refresh-panel title="Refresh"><i data-lucide="refresh-cw"></i></button>');
    panelQuery(root, '.summary-strip').addClass('d-none');
    const rows = alertsView ? state.fleet.items.flatMap(row => (row.alerts || []).map(alert => ({ timestamp: new Date().toISOString(), action: 'alert', target: row.name, reason: alert, result: row.status }))) : ((await api('/api/audit?limit=100')).items || []);
    panelQuery(root, '.workspace-content').html(`<div class="event-list central-event-list">${rows.map(row => `<div class="event-row"><time>${esc(new Date(row.timestamp).toLocaleString())}</time><strong>${esc(row.action)}</strong><span>${esc(row.target)} <span class="muted">${esc(row.reason || '')}</span></span><span>${esc(row.result)}</span></div>`).join('') || '<div class="empty-state">No events</div>'}</div>`);
  }

  const userInitials = name => String(name || '?').trim().split(/\s+/).slice(0, 2).map(part => part[0]).join('').toUpperCase();
  const syncThemeControl = () => {
    const dark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
    $('#theme-toggle').html(`<i data-lucide="${dark ? 'sun' : 'moon'}"></i><span>${dark ? 'Light' : 'Dark'} theme</span>`);
    dockHost.classList.toggle('dockview-theme-dark', dark); dockHost.classList.toggle('dockview-theme-light', !dark);
    lucide.createIcons();
  };
  const toggleTheme = () => {
    const next = document.documentElement.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-bs-theme', next); localStorage.setItem('pra-control-theme', next); syncThemeControl();
  };
  const loadIdentity = async () => {
    const value = await api('/api/auth/me'); state.me = value; state.csrf = value.csrf_token;
    const name = value.display_name || value.subject;
    $('.user-avatar-initials').text(userInitials(name)); $('#user-menu-toggle').attr('title', `${name} / ${value.role}`);
    $('#user-menu-name').text(name); $('#user-menu-role').text(value.role); $('#user-menu-subject').text(value.subject); $('#user-menu-provider').text(value.provider);
    syncThemeControl(); $('#login-screen').addClass('d-none');
  };
  const showLogin = async () => {
    $('#login-screen').removeClass('d-none'); const result = await $.getJSON('/api/auth/providers'); const external = result.items.filter(item => item.kind !== 'local');
    $('#local-login > .form-label,#login-user,#login-password,#local-login > button').toggle(result.items.some(item => item.kind === 'local'));
    $('#provider-logins').html(external.map(item => `<a class="btn btn-outline-secondary" href="/api/auth/login/${encodeURIComponent(item.name)}">Continue with ${esc(item.name)}</a>`).join(''));
  };

  const setLeftCollapsed = collapsed => {
    $('#control-layout').toggleClass('left-collapsed', collapsed);
    $('#left-toggle').attr('title', collapsed ? 'Expand navigation' : 'Collapse navigation').html(`<i data-lucide="${collapsed ? 'panel-left-open' : 'panel-left-close'}"></i>`);
    localStorage.setItem('pra-control-left-collapsed', collapsed ? '1' : '0'); window.setTimeout(() => dv.layout(), 0); lucide.createIcons();
  };
  const setChatCollapsed = collapsed => {
    $('#control-layout').toggleClass('chat-collapsed', collapsed); $('#chat-expand-edge').toggleClass('d-none', !collapsed);
    localStorage.setItem('pra-control-chat-collapsed', collapsed ? '1' : '0'); window.setTimeout(() => dv.layout(), 0); lucide.createIcons();
  };
  const wireChatResize = () => {
    const layout = document.getElementById('control-layout'); const handle = document.getElementById('chat-resize'); const key = 'pra-control-chat-ratio-v3';
    let ratio = Math.min(0.45, Math.max(0.22, Number(localStorage.getItem(key)) || 0.30)); let dragging = false;
    const apply = value => { ratio = Math.min(0.45, Math.max(0.22, value)); layout.style.setProperty('--chat-width', `${ratio * 100}%`); localStorage.setItem(key, ratio); dv.layout(); };
    apply(ratio);
    handle.addEventListener('pointerdown', event => { dragging = true; handle.classList.add('dragging'); handle.setPointerCapture(event.pointerId); });
    handle.addEventListener('pointermove', event => { if (dragging) apply((layout.getBoundingClientRect().right - event.clientX) / layout.clientWidth); });
    handle.addEventListener('pointerup', event => { dragging = false; handle.classList.remove('dragging'); handle.releasePointerCapture(event.pointerId); });
  };

  const loadAgentModels = async () => {
    const current = state.activeAgentTarget || $('#agent-model').val() || '';
    $('#agent-model').prop('disabled', true).html('<option>Loading models...</option>');
    try {
      state.agentModels = (await api('/api/agent/models')).items || [];
      $('#agent-model').html('<option value="">Manager-only fallback</option>' + state.agentModels.map(row => `<option value="${esc(row.target_id)}" ${row.target_id === current ? 'selected' : ''} ${row.reachable ? '' : 'disabled'}>${row.reachable ? '[online]' : '[offline]'} ${esc(row.target_id)} / ${esc(row.model_id)}</option>`).join(''));
    } catch (_) {
      $('#agent-model').html('<option value="">Manager-only fallback</option>');
    } finally {
      $('#agent-model').prop('disabled', false);
    }
  };
  const disconnectAgent = () => {
    clearTimeout(state.reconnectTimer); state.reconnectTimer = null;
    if (state.socket) { state.socket.onclose = null; state.socket.close(); state.socket = null; }
  };
  const connectAgent = (token = localStorage.getItem('pra-control-agent-token') || '', after = localStorage.getItem('pra-control-agent-sequence') || '0') => {
    if (state.socket && [WebSocket.CONNECTING, WebSocket.OPEN].includes(state.socket.readyState)) return;
    clearTimeout(state.reconnectTimer); $('#agent-status-label').text('Connecting');
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${scheme}://${location.host}/ws/agent?resume_token=${encodeURIComponent(token)}&after=${encodeURIComponent(after)}`); state.socket = socket;
    socket.onopen = () => { state.retry = 500; clearTimeout(state.reconnectTimer); $('#agent-dot').addClass('live'); $('#agent-status-label').text('Connected'); };
    socket.onerror = () => $('#agent-status-label').text('Connection interrupted');
    socket.onclose = event => {
      if (state.socket !== socket) return;
      state.socket = null; $('#agent-dot').removeClass('live');
      if (event.code === 4401) { $('#agent-status-label').text('Sign in required'); return; }
      const delay = state.retry; $('#agent-status-label').text(`Disconnected / retry in ${Math.ceil(delay / 1000)}s`);
      state.reconnectTimer = window.setTimeout(() => connectAgent(), delay); state.retry = Math.min(state.retry * 2, 30000);
    };
    socket.onmessage = event => handleAgent(JSON.parse(event.data));
  };
  const handleAgent = event => {
    if (event.sequence) localStorage.setItem('pra-control-agent-sequence', event.sequence);
    if (event.type === 'session') {
      localStorage.setItem('pra-control-agent-token', event.resume_token); state.activeAgentTarget = event.settings?.target_id || ''; $('#agent-model').val(state.activeAgentTarget); return;
    }
    if (event.type === 'session.updated') {
      state.activeAgentTarget = event.settings?.target_id || ''; $('#agent-model').val(state.activeAgentTarget); notify(`Agent model: ${state.activeAgentTarget || 'manager-only fallback'}`); return;
    }
    if (event.type === 'ping') { state.socket?.send(JSON.stringify({ type: 'pong' })); return; }
    if (event.type === 'tool.started' || event.type === 'tool.completed') $('#agent-messages').append(`<div class="tool-event">${esc(event.type)} / ${esc(event.tool)}</div>`);
    else if (event.type === 'message.delta') { if (!state.agentBuffer) state.agentBuffer = $('<div class="message assistant">').appendTo('#agent-messages'); state.agentBuffer.text(state.agentBuffer.text() + event.text); }
    else if (event.type === 'message.completed') state.agentBuffer = null;
    else if (event.type === 'error') $('#agent-messages').append(`<div class="message assistant text-danger">${esc(event.detail)}</div>`);
    $('#agent-messages').scrollTop($('#agent-messages')[0].scrollHeight);
  };
  const newAgentSession = async () => {
    const selected = $('#agent-model').val() || null;
    const value = await api('/api/agent/sessions', { method: 'POST', data: JSON.stringify({ target_id: selected }) });
    disconnectAgent(); localStorage.setItem('pra-control-agent-token', value.resume_token); localStorage.setItem('pra-control-agent-sequence', '0'); state.activeAgentTarget = value.settings?.target_id || '';
    $('#agent-messages').html('<div class="agent-empty">New session ready.</div>'); connectAgent(value.resume_token, '0');
  };
  const showSessions = async () => {
    $('#sessions-list').html(loadingMarkup('Loading sessions'));
    bootstrap.Modal.getOrCreateInstance(document.getElementById('sessions-modal')).show(); const rows = (await api('/api/agent/sessions')).items || [];
    $('#sessions-list').html(rows.length ? `<div class="session-list">${rows.map(row => `<button class="session-item" data-resume-session="${esc(row.resume_token)}"><span><strong>${esc(row.settings?.target_id || 'Manager-only fallback')}</strong><small>${esc(new Date(row.updated_at).toLocaleString())} / ${row.event_count} events</small></span><i data-lucide="arrow-right"></i></button>`).join('')}</div>` : '<div class="empty-state">No previous sessions</div>'); lucide.createIcons();
  };

  const openAction = (engine, action, values = {}) => {
    const highImpact = ['evict', 'demote', 'unload-model'].includes(action);
    $('#action-form').data({ engine, action, values }); $('#action-description').text(`${titleCase(action)} on ${engine}`); $('#action-values-group').toggleClass('d-none', action !== 'load-model');
    $('#action-values').val(action === 'load-model' ? JSON.stringify({ runtime_model_id: 'model-alias', model_id: 'organization/model', profile: 'BALANCED', execution_mode: 'selected-context' }, null, 2) : '{}');
    $('#action-reason').val(''); $('#action-confirmed').prop('checked', !highImpact).closest('.form-check').toggle(highImpact); bootstrap.Modal.getOrCreateInstance(document.getElementById('action-modal')).show();
  };
  const openRegistryEditor = (resource, id = null, transition = null, values = {}) => {
    $('#registry-form').data({ resource, id, transition }); $('#registry-values').val(JSON.stringify(values, null, 2)); $('#registry-reason').val('');
    $('#registry-modal .modal-title').text(transition ? `${titleCase(transition)} ${resource} record` : `Create ${resource} record`); bootstrap.Modal.getOrCreateInstance(document.getElementById('registry-modal')).show();
  };

  $(document)
    .on('click', '[data-open-view]', function () { openView($(this).data('open-view')); })
    .on('click', '[data-registry]', function () { openRegistry($(this).data('registry')); })
    .on('click', '[data-open-engine]', function () { openEngine($(this).data('open-engine')); })
    .on('click', '[data-open-router]', function () { openRouter($(this).data('open-router')); })
    .on('click', '[data-engine-section]', function () { openEngine($(this).data('engine'), $(this).data('engine-section')); })
    .on('input change', '[data-fleet-filter]', function () { state.fleetFilters[$(this).data('fleet-filter')] = $(this).val(); const panel = [...state.panels.values()].find(item => item.root.contains(this)); if (panel) renderFleet(panel.root); })
    .on('click', '[data-sort]', function () { const key = $(this).data('sort'); state.fleetSort.direction = state.fleetSort.key === key && state.fleetSort.direction === 'asc' ? 'desc' : 'asc'; state.fleetSort.key = key; const panel = [...state.panels.values()].find(item => item.root.contains(this)); if (panel) renderFleet(panel.root); })
    .on('click', '[data-action]', function (event) { event.stopPropagation(); openAction($(this).data('engine'), $(this).data('action'), $(this).data('runtime-model-id') ? { runtime_model_id: $(this).data('runtime-model-id') } : {}); })
    .on('click', '[data-observe]', async function (event) { event.stopPropagation(); const links = await api(`/api/observability/links?engine=${encodeURIComponent($(this).data('observe'))}`); if (links.grafana) window.open(links.grafana, '_blank', 'noopener'); else notify('Grafana is not configured', 'warning'); })
    .on('click', '[data-registry-create]', function () { openRegistryEditor($(this).data('registry-create')); })
    .on('click', '[data-registry-edit]', function () { const resource = $(this).data('registry-edit'); const row = state.registryRecords[resource][Number($(this).data('record-index'))]; openRegistryEditor(resource, row.id, 'patch', row); })
    .on('click', '[data-registry-approve]', function () { openRegistryEditor($(this).data('registry-approve'), $(this).data('record-id'), 'approve'); })
    .on('click', '[data-router-apply]', function () { const id = $(this).data('router-apply'); $('#router-apply-form').data('router', id); $('#router-apply-description').text(`Apply Registry desired state to ${id}. Last-good routing remains active if verification fails.`); $('#router-apply-reason').val(''); $('#router-apply-confirmed').prop('checked', false); bootstrap.Modal.getOrCreateInstance(document.getElementById('router-apply-modal')).show(); })
    .on('click', '[data-refresh-panel]', function () { const panel = [...state.panels.values()].find(item => item.root.contains(this)); if (panel) renderPanel(panel.root, panel.spec); })
    .on('click', '[data-resume-session]', function () { disconnectAgent(); const token = $(this).data('resume-session'); localStorage.setItem('pra-control-agent-token', token); localStorage.setItem('pra-control-agent-sequence', '0'); $('#agent-messages').empty(); bootstrap.Modal.getInstance(document.getElementById('sessions-modal'))?.hide(); connectAgent(token, '0'); });

  $(document).on('submit', '#action-form', async event => {
    event.preventDefault(); const form = $(event.currentTarget); const { engine, action } = form.data(); let values = form.data('values') || {};
    if (action === 'load-model') { try { values = JSON.parse($('#action-values').val()); } catch (_) { return notify('Model load JSON is invalid', 'danger'); } }
    try { await api(`/api/engines/${encodeURIComponent(engine)}/actions/${action}`, { method: 'POST', data: JSON.stringify({ values, reason: $('#action-reason').val(), confirmed: $('#action-confirmed').prop('checked') }) }); bootstrap.Modal.getInstance(document.getElementById('action-modal'))?.hide(); notify(`${titleCase(action)} accepted for ${engine}`); await loadFleet(); } catch (error) { notify(error.responseJSON?.detail || 'Action failed', 'danger'); }
  });
  $(document).on('submit', '#registry-form', async event => {
    event.preventDefault(); const { resource, id, transition } = $(event.currentTarget).data(); let values;
    try { values = JSON.parse($('#registry-values').val()); } catch (_) { return notify('Record JSON is invalid', 'danger'); }
    const editing = transition === 'patch'; const path = editing ? `/api/registry/${resource}/${encodeURIComponent(id)}` : transition ? `/api/registry/${resource}/${encodeURIComponent(id)}/${transition}` : `/api/registry/${resource}`;
    try { await api(path, { method: editing ? 'PATCH' : 'POST', data: JSON.stringify({ values, reason: $('#registry-reason').val() }) }); bootstrap.Modal.getInstance(document.getElementById('registry-modal'))?.hide(); notify('Registry mutation accepted'); openRegistry(resource); } catch (error) { notify(error.responseJSON?.detail || 'Registry mutation failed', 'danger'); }
  });
  $(document).on('submit', '#router-apply-form', async event => {
    event.preventDefault(); const id = $(event.currentTarget).data('router');
    if (!$('#router-apply-confirmed').prop('checked')) return notify('Explicit confirmation is required', 'warning');
    try { await api(`/api/routers/${encodeURIComponent(id)}/apply`, { method: 'POST', data: JSON.stringify({ reason: $('#router-apply-reason').val(), confirmed: true, values: {} }) }); bootstrap.Modal.getInstance(document.getElementById('router-apply-modal'))?.hide(); notify(`Router ${id} reconciled`); openRouter(id); } catch (error) { notify(error.responseJSON?.error?.message || error.responseJSON?.detail || 'Router reconciliation failed', 'danger'); }
  });
  $(document).on('submit', '#local-login', async event => { event.preventDefault(); try { await $.ajax({ url: '/api/auth/login/local', method: 'POST', contentType: 'application/json', data: JSON.stringify({ username: $('#login-user').val(), password: $('#login-password').val() }) }); location.reload(); } catch (error) { $('#login-error').text(error.responseJSON?.detail || 'Sign-in failed'); } });
  $(document).on('submit', '#agent-form', event => {
    event.preventDefault(); const text = $('#agent-input').val().trim();
    if (!text || state.socket?.readyState !== WebSocket.OPEN) return;
    if (text === '/clear') { $('#agent-messages').empty(); $('#agent-input').val(''); return; }
    if (text === '/new') { $('#agent-input').val(''); newAgentSession().catch(error => notify(error.responseJSON?.detail || error, 'danger')); return; }
    $('.agent-empty').remove(); $('<div class="message user">').text(text).appendTo('#agent-messages'); state.socket.send(JSON.stringify({ type: 'message', message_id: crypto.randomUUID(), text })); $('#agent-input').val('');
  });

  $('#left-toggle').on('click', () => setLeftCollapsed(!$('#control-layout').hasClass('left-collapsed')));
  $('#chat-collapse').on('click', () => setChatCollapsed(true)); $('#chat-expand-edge').on('click', () => setChatCollapsed(false));
  $('[data-panel-toggle="left"]').on('click', () => setLeftCollapsed(!$('#control-layout').hasClass('left-collapsed')));
  $('[data-panel-toggle="chat"]').on('click', () => setChatCollapsed(!$('#control-layout').hasClass('chat-collapsed')));
  $('[data-close-active-tab]').on('click', () => dv.activePanel?.api.close());
  $('[data-open-help]').on('click', function () { $('#help-toggle').trigger('click'); });
  $('#agent-new').on('click', () => newAgentSession().catch(error => notify(error.responseJSON?.detail || error, 'danger')));
  $('#agent-sessions').on('click', () => showSessions().catch(error => notify(error.responseJSON?.detail || error, 'danger')));
  $('#agent-tips').on('click', () => { $('#agent-input').val('/tips'); $('#agent-form').trigger('submit'); });
  $('#agent-model').on('change', function () { if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify({ type: 'model.select', target_id: this.value || null })); });
  $('#refresh-all').on('click', async () => { try { await loadFleet(); notify('Control Plane refreshed'); } catch (error) { $('#connection-summary').removeClass('live'); notify(error.responseJSON?.detail || 'Refresh failed', 'danger'); } });
  $('#theme-toggle').on('click', toggleTheme);
  $('#logout').on('click', async () => { disconnectAgent(); await api('/api/auth/logout', { method: 'POST', data: '{}' }); location.reload(); });
  window.addEventListener('resize', () => dv.layout());

  setLeftCollapsed(localStorage.getItem('pra-control-left-collapsed') === '1'); setChatCollapsed(localStorage.getItem('pra-control-chat-collapsed') === '1'); wireChatResize();
  if (!dv.panels.length) openView('fleet');
  (async () => {
    try { await loadIdentity(); await loadFleet(); connectAgent(); lucide.createIcons(); }
    catch (error) { $('#connection-summary').removeClass('live'); if (error.status === 401) await showLogin(); else notify(error.responseJSON?.detail || error, 'danger'); }
  })();
})();
