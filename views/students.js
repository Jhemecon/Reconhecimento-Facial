window.AppViews = window.AppViews || {};

window.AppViews.cadastro = () => `
  <section class="page active" id="page-cadastro" aria-labelledby="registerTitle">
    <div class="page-intro"><p class="eyebrow">GESTÃO DE ALUNOS</p><h1 id="registerTitle">Cadastro de alunos</h1><p class="muted">Adicione dados fictícios para explorar a interface. O cadastro não é enviado nem salvo.</p></div>
    <div class="content-card form-card">
      <div class="card-heading"><div><h2>Informações do aluno</h2><p>Os campos são usados somente nesta sessão do navegador.</p></div><span class="demo-tag">MODO DEMONSTRAÇÃO</span></div>
      <form id="studentForm">
        <div class="form-grid">
          <label class="field field-wide">Nome completo<input type="text" name="name" autocomplete="name" placeholder="Ex.: Ana Souza" required maxlength="100"></label>
          <label class="field">Matrícula<input type="text" name="registration" placeholder="Ex.: 2024001" required maxlength="30"></label>
          <div class="field image-field">
            <span class="field-label">Imagem de referência <span class="optional">(opcional)</span></span>
            <label class="upload-box" for="studentImage">
              <span class="upload-symbol" aria-hidden="true">↑</span><span><strong>Escolher uma imagem</strong><small>JPG ou PNG · pré-visualização local, não armazenada</small></span>
              <input id="studentImage" name="image" type="file" accept="image/png,image/jpeg">
            </label>
            <div class="image-preview" id="studentPreview" hidden></div>
          </div>
        </div>
        <div class="form-disclaimer"><span aria-hidden="true">ⓘ</span>Imagem usada apenas para pré-visualização no navegador. Nenhum processamento facial será realizado.</div>
        <div class="form-actions"><button class="button button-primary" type="submit">Adicionar aluno de demonstração</button></div>
      </form>
    </div>
    <div class="content-card list-card"><div class="card-heading"><div><h2>Alunos desta sessão</h2><p id="studentListDescription">Nenhum aluno adicionado.</p></div><span class="count-pill" id="studentListCount">0</span></div><div class="student-list" id="studentList"><div class="empty-state compact-empty"><span aria-hidden="true">♙</span><p>Os alunos adicionados aparecerão aqui.</p></div></div></div>
  </section>`;
