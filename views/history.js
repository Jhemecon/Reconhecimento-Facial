window.AppViews = window.AppViews || {};

window.AppViews.historico = () => `
  <section class="page active" id="page-historico" aria-labelledby="historyTitle">
    <div class="page-intro"><p class="eyebrow">AUDITORIA</p><h1 id="historyTitle">Histórico de tentativas</h1><p class="muted">Registro local e temporário das simulações feitas nesta sessão.</p></div>
    <div class="content-card history-card">
      <div class="history-toolbar"><div><h2>Registros</h2><p id="historyDescription">Nenhuma tentativa simulada nesta sessão.</p></div><label class="search-field"><span aria-hidden="true">⌕</span><input id="historySearch" type="search" placeholder="Buscar por nome ou matrícula" aria-label="Buscar por nome ou matrícula"></label></div>
      <div class="table-wrap"><table><thead><tr><th>Data e horário</th><th>Aluno / matrícula</th><th>Resultado</th><th>Registro de presença</th></tr></thead><tbody id="historyTable"></tbody></table><div class="empty-state" id="historyEmpty"><span aria-hidden="true">▤</span><strong>Nenhum registro encontrado</strong><p>As verificações simuladas aparecerão aqui.</p></div></div>
      <div class="table-footnote"><span aria-hidden="true">ⓘ</span>Este histórico existe somente nesta sessão; recarregar a página apaga os dados.</div>
    </div>
  </section>`;
