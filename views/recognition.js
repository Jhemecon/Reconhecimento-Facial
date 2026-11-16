window.AppViews = window.AppViews || {};

window.AppViews.reconhecimento = () => `
  <section class="page active" id="page-reconhecimento" aria-labelledby="recognitionTitle">
    <div class="page-intro"><p class="eyebrow">VERIFICAÇÃO DE PRESENÇA</p><h1 id="recognitionTitle">Reconhecimento facial</h1><p class="muted">Explore visualmente as etapas. Esta demonstração não realiza identificação biométrica.</p></div>
    <div class="recognition-layout">
      <div class="content-card camera-card">
        <div class="card-heading"><div><h2>Captura de imagem</h2><p>Use a câmera ou selecione uma imagem para pré-visualizar.</p></div><span class="camera-status"><span class="status-dot status-dot-muted"></span><span id="cameraStatus">Câmera desligada</span></span></div>
        <div class="camera-stage" id="cameraStage">
          <video id="cameraVideo" autoplay playsinline muted hidden></video>
          <div class="camera-placeholder" id="cameraPlaceholder"><span class="camera-frame" aria-hidden="true"><span></span></span><strong>A imagem aparecerá aqui</strong><small>A câmera só será solicitada quando você clicar no botão.</small></div>
          <img id="recognitionImagePreview" alt="Pré-visualização da imagem selecionada" hidden>
          <div class="camera-overlay" id="cameraOverlay" hidden><span class="live-dot"></span>PRÉ-VISUALIZAÇÃO LOCAL</div>
        </div>
        <div class="capture-actions"><button class="button button-secondary" id="cameraButton" type="button"><span aria-hidden="true">◉</span> Ativar câmera</button><label class="button button-secondary upload-button" for="recognitionImage">↑ Selecionar imagem<input id="recognitionImage" type="file" accept="image/png,image/jpeg"></label></div>
        <div class="simulation-box"><div><span class="simulation-icon" aria-hidden="true">✳</span><span><strong>Simulação visual</strong><small>Não compara rostos nem registra presença real.</small></span></div><button class="button button-primary" id="simulateButton" type="button">Simular verificação</button></div>
        <div class="result-panel" id="recognitionResult" aria-live="polite" hidden></div>
      </div>
      <aside class="content-card steps-card"><div class="card-heading"><div><h2>Etapas do fluxo</h2><p>Como seria uma integração futura.</p></div></div><ol class="steps-list"><li class="step-done"><span>1</span><div><strong>Identificação inicial</strong><small>Matrícula ou credencial</small></div></li><li class="step-active"><span>2</span><div><strong>Apresentação voluntária</strong><small>Captura ou imagem fornecida</small></div></li><li><span>3</span><div><strong>Processamento no backend</strong><small>Detecção e representação vetorial</small></div></li><li><span>4</span><div><strong>Resultado e auditoria</strong><small>Retorno da API e histórico</small></div></li></ol><div class="integration-note"><strong>Ponto de integração</strong><p>Conectar a captura e os dados a uma API quando os endpoints do grupo forem definidos. A lógica de reconhecimento permanece no backend.</p></div></aside>
    </div>
  </section>`;
