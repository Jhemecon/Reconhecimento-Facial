window.AppViews = window.AppViews || {};

window.AppViews.inicio = () => `
  <section class="page active" id="page-inicio" aria-labelledby="welcomeTitle">
    <div class="welcome-row">
      <div>
        <p class="eyebrow">PROJETO ACADÊMICO · ÁLGEBRA LINEAR II</p>
        <h1 id="welcomeTitle">Olá, seja bem-vindo<span class="accent">.</span></h1>
        <p class="muted welcome-copy">Uma demonstração didática de como conceitos de álgebra linear podem apoiar um fluxo de verificação de presença.</p>
      </div>
      <a class="button button-primary" href="#reconhecimento" data-page="reconhecimento"><span aria-hidden="true">◎</span> Iniciar demonstração</a>
    </div>

    <div class="notice-banner">
      <span class="notice-icon" aria-hidden="true">✦</span>
      <div><strong>Protótipo educacional — não é um sistema institucional.</strong><p>As verificações e os registros exibidos são simulados. Não há reconhecimento facial real nem decisão de acesso.</p></div>
    </div>

    <div class="section-heading"><div><h2>Visão geral</h2><p>Acompanhe os dados fictícios desta demonstração.</p></div><span class="demo-tag">DADOS DE EXEMPLO</span></div>
    <div class="stats-grid">
      <article class="stat-card"><div class="stat-top"><span>Alunos cadastrados</span><span class="stat-icon violet" aria-hidden="true">♙</span></div><strong id="studentCount">0</strong><small>neste protótipo, nesta sessão</small></article>
      <article class="stat-card"><div class="stat-top"><span>Tentativas simuladas</span><span class="stat-icon blue" aria-hidden="true">◎</span></div><strong id="attemptCount">0</strong><small>registros desta sessão</small></article>
      <article class="stat-card"><div class="stat-top"><span>Verificações demonstrativas</span><span class="stat-icon green" aria-hidden="true">✓</span></div><strong id="recognizedCount">0</strong><small>resultados simulados</small></article>
    </div>

    <div class="section-heading section-heading-spaced"><div><h2>Acesso rápido</h2><p>Escolha uma etapa para explorar o fluxo.</p></div></div>
    <div class="quick-grid">
      <a class="quick-card" href="#cadastro" data-page="cadastro"><span class="quick-icon violet" aria-hidden="true">＋</span><span><strong>Cadastrar aluno</strong><small>Preencha os dados de exemplo</small></span><span class="arrow" aria-hidden="true">↗</span></a>
      <a class="quick-card" href="#reconhecimento" data-page="reconhecimento"><span class="quick-icon blue" aria-hidden="true">◎</span><span><strong>Testar reconhecimento</strong><small>Visualize uma verificação simulada</small></span><span class="arrow" aria-hidden="true">↗</span></a>
      <a class="quick-card" href="#historico" data-page="historico"><span class="quick-icon amber" aria-hidden="true">▤</span><span><strong>Consultar histórico</strong><small>Veja as tentativas desta sessão</small></span><span class="arrow" aria-hidden="true">↗</span></a>
    </div>

    <div class="math-card">
      <div class="math-symbol" aria-hidden="true">⟨u, v⟩</div>
      <div><span class="eyebrow">CONEXÃO COM A DISCIPLINA</span><h2>Da imagem à representação vetorial</h2><p>Em um sistema real, características poderiam ser representadas por vetores e comparadas por medidas de similaridade. O cálculo e a decisão pertencem ao backend — esta interface apenas ilustra o fluxo.</p></div>
      <span class="math-decoration" aria-hidden="true">v ∈ ℝⁿ</span>
    </div>
  </section>`;
