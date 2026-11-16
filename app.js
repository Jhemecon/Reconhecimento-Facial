const state = {
  students: [],
  attempts: [],
  cameraStream: null,
  studentImageUrl: null,
  recognitionImageUrl: null,
  toastTimer: null
};

const pageNames = {
  inicio: "Visão geral",
  cadastro: "Alunos",
  reconhecimento: "Reconhecimento",
  historico: "Histórico"
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 3000);
}

function navigate(page) {
  const selectedPage = pageNames[page] ? page : "inicio";
  $("#pageContent").innerHTML = window.AppViews[selectedPage]();
  $$(".nav-link").forEach((link) => {
    const active = link.dataset.page === selectedPage;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  $("#pageTitle").textContent = pageNames[selectedPage];
  $("#sidebar").classList.remove("open");
  $("#menuToggle").setAttribute("aria-expanded", "false");
  if (selectedPage !== "reconhecimento") {
    stopCamera();
    clearRecognitionImage();
  }
  if (selectedPage !== "cadastro") clearStudentImage();
  if (window.location.hash !== `#${selectedPage}`) window.history.replaceState(null, "", `#${selectedPage}`);
  updateDashboard();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function handleNavigation(event) {
  const link = event.target.closest("[data-page]");
  if (!link) return;
  event.preventDefault();
  navigate(link.dataset.page);
}

function updateDashboard() {
  const studentCount = $("#studentCount");
  if (studentCount) studentCount.textContent = state.students.length;
  const attemptCount = $("#attemptCount");
  if (attemptCount) attemptCount.textContent = state.attempts.length;
  const recognizedCount = $("#recognizedCount");
  if (recognizedCount) recognizedCount.textContent = state.attempts.filter((attempt) => attempt.recognized).length;
  const studentListCount = $("#studentListCount");
  if (studentListCount) studentListCount.textContent = state.students.length;
  const studentListDescription = $("#studentListDescription");
  if (studentListDescription) {
    studentListDescription.textContent = state.students.length
      ? `${state.students.length} aluno${state.students.length === 1 ? "" : "s"} nesta sessão.`
      : "Nenhum aluno adicionado.";
  }
  const historyDescription = $("#historyDescription");
  if (historyDescription) {
    historyDescription.textContent = state.attempts.length
      ? `${state.attempts.length} tentativa${state.attempts.length === 1 ? "" : "s"} simulada${state.attempts.length === 1 ? "" : "s"} nesta sessão.`
      : "Nenhuma tentativa simulada nesta sessão.";
  }
  renderStudents();
  renderHistory();
}

function renderStudents() {
  const list = $("#studentList");
  if (!list) return;
  if (!state.students.length) {
    list.innerHTML = '<div class="empty-state compact-empty"><span aria-hidden="true">♙</span><p>Os alunos adicionados aparecerão aqui.</p></div>';
    return;
  }
  list.innerHTML = state.students.map((student) => `
    <div class="student-row">
      <span class="avatar" aria-hidden="true">${escapeHtml(student.name.trim().charAt(0).toUpperCase())}</span>
      <span><strong>${escapeHtml(student.name)}</strong><small>Matrícula ${escapeHtml(student.registration)}</small></span>
    </div>`).join("");
}

function renderHistory() {
  const search = $("#historySearch");
  const table = $("#historyTable");
  const emptyState = $("#historyEmpty");
  if (!table || !emptyState) return;
  const query = search.value.trim().toLocaleLowerCase("pt-BR");
  const matches = state.attempts.filter((attempt) =>
    `${attempt.name} ${attempt.registration}`.toLocaleLowerCase("pt-BR").includes(query));
  table.innerHTML = matches.map((attempt) => `
    <tr>
      <td>${escapeHtml(attempt.timestamp)}</td>
      <td><strong>${escapeHtml(attempt.name)}</strong><small>${escapeHtml(attempt.registration)}</small></td>
      <td><span class="result-pill ${attempt.recognized ? "" : "fail"}">${attempt.recognized ? "Reconhecido · simulado" : "Não reconhecido · simulado"}</span></td>
      <td>${attempt.recognized ? "Presença simulada" : "Sem registro"}</td>
    </tr>`).join("");
  emptyState.classList.toggle("visible", matches.length === 0);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}

function setImagePreview(file, preview, kind) {
  if (!file) return;
  if (!["image/jpeg", "image/png"].includes(file.type)) {
    showToast("Escolha uma imagem JPG ou PNG.");
    return;
  }
  if (file.size > 5 * 1024 * 1024) {
    showToast("A imagem deve ter no máximo 5 MB.");
    return;
  }
  const stateKey = kind === "student" ? "studentImageUrl" : "recognitionImageUrl";
  if (state[stateKey]) URL.revokeObjectURL(state[stateKey]);
  state[stateKey] = URL.createObjectURL(file);

  if (kind === "student") {
    preview.hidden = false;
    preview.innerHTML = `<img src="${state[stateKey]}" alt="Pré-visualização da imagem selecionada"><span>${escapeHtml(file.name)}</span><button type="button" id="removeStudentImage">Remover</button>`;
    $("#removeStudentImage").addEventListener("click", () => {
      URL.revokeObjectURL(state.studentImageUrl);
      state.studentImageUrl = null;
      $("#studentImage").value = "";
      preview.hidden = true;
      preview.replaceChildren();
    });
  } else {
    stopCamera();
    $("#recognitionImagePreview").src = state[stateKey];
    $("#recognitionImagePreview").hidden = false;
    $("#cameraPlaceholder").hidden = true;
    $("#cameraOverlay").hidden = false;
  }
}

function clearStudentImage() {
  if (state.studentImageUrl) URL.revokeObjectURL(state.studentImageUrl);
  state.studentImageUrl = null;
  const input = $("#studentImage");
  if (input) input.value = "";
  const preview = $("#studentPreview");
  if (preview) {
    preview.hidden = true;
    preview.replaceChildren();
  }
}

function clearRecognitionImage() {
  if (state.recognitionImageUrl) URL.revokeObjectURL(state.recognitionImageUrl);
  state.recognitionImageUrl = null;
  const input = $("#recognitionImage");
  if (input) input.value = "";
  const preview = $("#recognitionImagePreview");
  if (preview) {
    preview.removeAttribute("src");
    preview.hidden = true;
  }
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }
  const video = $("#cameraVideo");
  if (video) {
    video.srcObject = null;
    video.hidden = true;
  }
  const status = $("#cameraStatus");
  if (status) status.textContent = "Câmera desligada";
  const button = $("#cameraButton");
  if (button) button.innerHTML = '<span aria-hidden="true">◉</span> Ativar câmera';
  const overlay = $("#cameraOverlay");
  if (overlay) overlay.hidden = true;
  if ($("#recognitionImagePreview")?.hidden !== false && $("#cameraPlaceholder")) {
    $("#cameraPlaceholder").hidden = false;
  }
}

async function toggleCamera() {
  if (state.cameraStream) {
    stopCamera();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("Câmera indisponível. Use localhost/HTTPS ou selecione uma imagem.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    state.cameraStream = stream;
    $("#recognitionImagePreview").hidden = true;
    $("#cameraPlaceholder").hidden = true;
    $("#cameraVideo").srcObject = stream;
    $("#cameraVideo").hidden = false;
    $("#cameraOverlay").hidden = false;
    $("#cameraStatus").textContent = "Câmera ativa";
    $("#cameraButton").innerHTML = '<span aria-hidden="true">■</span> Desativar câmera';
  } catch (error) {
    if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
      showToast("Permissão para a câmera negada. Você ainda pode selecionar uma imagem.");
    } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      showToast("Nenhuma câmera encontrada. Você ainda pode selecionar uma imagem.");
    } else {
      showToast("Não foi possível iniciar a câmera. Tente selecionar uma imagem.");
    }
  }
}

function simulateRecognition() {
  const student = state.students[0] ?? null;
  const recognized = Boolean(student);
  const attempt = {
    name: student ? student.name : "Pessoa não identificada",
    registration: student ? student.registration : "—",
    recognized,
    timestamp: new Intl.DateTimeFormat("pt-BR", { dateStyle: "short", timeStyle: "short" }).format(new Date())
  };
  state.attempts.unshift(attempt);

  const result = $("#recognitionResult");
  result.hidden = false;
  result.classList.toggle("result-fail", !recognized);
  result.innerHTML = recognized
    ? `<strong>Resultado demonstrativo: reconhecido</strong><p>${escapeHtml(student.name)} · matrícula ${escapeHtml(student.registration)}. Presença simulada. Nenhuma imagem foi comparada.</p>`
    : "<strong>Resultado demonstrativo: não reconhecido</strong><p>Nenhum aluno cadastrado nesta sessão. Nenhuma imagem foi comparada e nenhuma presença foi registrada.</p>";
  updateDashboard();
  showToast("Verificação simulada adicionada ao histórico desta sessão.");
}

document.addEventListener("click", handleNavigation);
document.addEventListener("click", (event) => {
  const menuToggle = event.target.closest("#menuToggle");
  if (!menuToggle) return;
  const open = $("#sidebar").classList.toggle("open");
  menuToggle.setAttribute("aria-expanded", String(open));
});
document.addEventListener("change", (event) => {
  if (event.target.matches("#studentImage")) {
    setImagePreview(event.target.files[0], $("#studentPreview"), "student");
  }
  if (event.target.matches("#recognitionImage")) {
    setImagePreview(event.target.files[0], $("#recognitionImagePreview"), "recognition");
  }
});
document.addEventListener("input", (event) => {
  if (event.target.matches("#historySearch")) renderHistory();
});
document.addEventListener("click", (event) => {
  if (event.target.closest("#cameraButton")) toggleCamera();
  if (event.target.closest("#simulateButton")) simulateRecognition();
});
document.addEventListener("submit", (event) => {
  if (!event.target.matches("#studentForm")) return;
  event.preventDefault();
  const form = event.target;
  const formData = new FormData(form);
  const name = String(formData.get("name")).trim();
  const registration = String(formData.get("registration")).trim();
  if (!name || !registration) {
    showToast("Preencha o nome e a matrícula.");
    return;
  }
  if (state.students.some((student) => student.registration.toLocaleLowerCase("pt-BR") === registration.toLocaleLowerCase("pt-BR"))) {
    showToast("Já existe um aluno com essa matrícula nesta sessão.");
    return;
  }
  state.students.push({ name, registration });
  form.reset();
  clearStudentImage();
  updateDashboard();
  showToast("Aluno adicionado apenas para esta demonstração.");
});

window.addEventListener("hashchange", () => navigate(window.location.hash.slice(1)));
window.addEventListener("beforeunload", () => {
  stopCamera();
  if (state.studentImageUrl) URL.revokeObjectURL(state.studentImageUrl);
  if (state.recognitionImageUrl) URL.revokeObjectURL(state.recognitionImageUrl);
});

navigate(window.location.hash.slice(1) || "inicio");
updateDashboard();
