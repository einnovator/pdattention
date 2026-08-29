(() => {
  const state = { session: null, sessions: [], socket: null, detailTab: 'runtime' };
  const component = (name) => {
    const node = document.querySelector(`#${name}-template`).content.cloneNode(true);
    const host = document.createElement('div'); host.className = 'h-100'; host.append(node); return host;
  };
  const dv = dockview.createDockview(document.getElementById('dockview'), { createComponent: e => component(e.name) });
  const left = dv.addPanel({ id:'conversations', component:'conversations', title:'Conversations' });
  const chat = dv.addPanel({ id:'chat', component:'chat', title:'Chat', position:{ referencePanel:left, direction:'right' } });
  dv.addPanel({ id:'details', component:'details', title:'Inspect', position:{ referencePanel:chat, direction:'right' } });
  const refresh = async () => { state.sessions = await $.getJSON('/api/sessions'); renderSessions(); };
  const renderSessions = () => {
    $('#conversations').empty(); state.sessions.forEach(s => $('<button class="conversation">').toggleClass('active', s.session_id===state.session?.session_id).text(s.session_id).on('click',()=>select(s)).appendTo('#conversations'));
  };
  const select = async (session) => { state.session = await $.getJSON(`/api/sessions/${session.session_id}`); $('#profile-label').text(`${state.session.profile} · ${state.session.session_id}`); renderSessions(); renderDetails(); connect(); };
  const renderDetails = () => { if (!state.session) return; const value = state.detailTab==='tasks' ? state.session.tasks : state.detailTab==='context' ? state.session.records : {agent_profile:state.session.profile, session_id:state.session.session_id, version:state.session.version}; $('#details').text(JSON.stringify(value,null,2)); };
  const connect = () => { if (state.socket) state.socket.close(); const scheme=location.protocol==='https:'?'wss':'ws'; state.socket=new WebSocket(`${scheme}://${location.host}/ws/sessions/${state.session.session_id}`); state.socket.onopen=()=>$('#connection').addClass('live'); state.socket.onclose=()=>{ $('#connection').removeClass('live'); setTimeout(()=>state.session&&connect(),1500); }; state.socket.onmessage=e=>{ const event=JSON.parse(e.data); if(event.type==='message.user'||event.type==='message.assistant') $('<div class="message">').addClass(event.type.endsWith('user')?'user':'assistant').text(event.text).appendTo('#messages'); }; };
  $(document).on('submit','#composer',async e=>{ e.preventDefault(); const text=$('#prompt').val().trim(); if(!text||!state.session)return; $('#prompt').val(''); await $.ajax({url:`/api/sessions/${state.session.session_id}/messages`,method:'POST',contentType:'application/json',data:JSON.stringify({text})}); state.session=await $.getJSON(`/api/sessions/${state.session.session_id}`); renderDetails(); });
  $(document).on('click','.detail-tabs button',function(){ state.detailTab=$(this).data('tab'); $('.detail-tabs button').removeClass('active'); $(this).addClass('active'); renderDetails(); });
  $('#new-session').on('click',async()=>{ const profiles=await $.getJSON('/api/profiles'); const session=await $.ajax({url:'/api/sessions',method:'POST',contentType:'application/json',data:JSON.stringify({profile:profiles.default_profile})}); await refresh(); await select(session); });
  window.addEventListener('resize',()=>dv.layout()); lucide.createIcons(); refresh();
})();
