const authScreen = document.querySelector("#auth-screen");
const workspace = document.querySelector("#workspace");
const authForm = document.querySelector("#auth-form");
const authError = document.querySelector("#auth-error");
const authToggle = document.querySelector("#auth-toggle");
const authSubmit = document.querySelector("#auth-submit");
const messageForm = document.querySelector("#message-form");
const messageInput = document.querySelector("#message-input");
const messagesNode = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const conversationList = document.querySelector("#conversation-list");
const conversationTitle = document.querySelector("#conversation-title");
const documentList = document.querySelector("#document-list");
const documentInput = document.querySelector("#document-input");
const sendButton = document.querySelector("#send-button");
const toast = document.querySelector("#toast");

let token = localStorage.getItem("assistant-token");
let activeConversation = null;
let registering = false;
let toastTimer;

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail = "The request could not be completed.";
    try { detail = (await response.json()).detail || detail; } catch {}
    if (response.status === 401 && token) signOut();
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 3600);
}

function signOut() {
  localStorage.removeItem("assistant-token");
  token = null;
  activeConversation = null;
  clearContextFields();
  workspace.classList.add("hidden");
  authScreen.classList.remove("hidden");
}

function clearContextFields() {
  document.querySelectorAll(".context-fields input, .context-fields textarea")
    .forEach((field) => { field.value = ""; });
}

function readMessageContext() {
  const customer_identifier = document.querySelector("#customer-identifier").value.trim();
  const readFields = (fields) => Object.fromEntries(
    fields.map(([key, selector]) => [key, document.querySelector(selector).value.trim()])
      .filter(([, value]) => value)
  );
  const business_context = readFields([
    ["business_id", "#business-context-id"],
    ["name", "#business-context-name"],
    ["details", "#business-context-details"]
  ]);
  const task_context = readFields([
    ["task_id", "#task-context-id"],
    ["current_task", "#task-context-current"],
    ["details", "#task-context-details"],
    ["status", "#task-context-status"],
    ["notes", "#task-context-notes"]
  ]);
  return {
    ...(customer_identifier ? { customer_identifier } : {}),
    ...(Object.keys(business_context).length ? { business_context } : {}),
    ...(Object.keys(task_context).length ? { task_context } : {})
  };
}

function addMessage(role, content, citations = []) {
  welcome.classList.add("hidden");
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = content;
  article.append(bubble);
  if (citations.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "citations";
    for (const citation of citations) {
      const source = document.createElement("span");
      source.className = "citation";
      source.textContent = `[${citation.index}] ${citation.filename}` +
        (citation.page_number ? ` · page ${citation.page_number}` : "");
      sourceList.append(source);
    }
    article.append(sourceList);
  }
  messagesNode.append(article);
  messagesNode.scrollTop = messagesNode.scrollHeight;
}

async function loadConversations() {
  const { conversations } = await api("/api/conversations");
  conversationList.replaceChildren();
  for (const conversation of conversations) {
    const button = document.createElement("button");
    button.className = "conversation-item" +
      (conversation.id === activeConversation ? " active" : "");
    button.textContent = conversation.title;
    button.title = conversation.title;
    button.addEventListener("click", () => openConversation(conversation.id));
    conversationList.append(button);
  }
  return conversations;
}

async function loadDocuments() {
  const { documents } = await api("/api/documents");
  documentList.replaceChildren();
  for (const document of documents) {
    const entry = document.createElement("div");
    entry.className = "document-entry";
    entry.textContent = document.filename;
    entry.title = document.filename;
    documentList.append(entry);
  }
}

async function openConversation(id) {
  clearContextFields();
  activeConversation = id;
  const { messages } = await api(`/api/conversations/${encodeURIComponent(id)}/messages`);
  const conversations = await loadConversations();
  conversationTitle.textContent =
    conversations.find((conversation) => conversation.id === id)?.title || "Conversation";
  document.querySelector("#sidebar").classList.remove("mobile-open");
  messagesNode.replaceChildren(welcome);
  welcome.classList.toggle("hidden", messages.length > 0);
  for (const message of messages) {
    addMessage(message.role, message.content, message.citations);
  }
}

async function showWorkspace() {
  authScreen.classList.add("hidden");
  workspace.classList.remove("hidden");
  try {
    await Promise.all([loadConversations(), loadDocuments()]);
  } catch (error) {
    showToast(error.message);
  }
}

authToggle.addEventListener("click", () => {
  registering = !registering;
  authSubmit.textContent = registering ? "Create account" : "Sign in";
  authToggle.textContent = registering ? "I already have an account" : "Create an account";
  document.querySelector("#password").autocomplete =
    registering ? "new-password" : "current-password";
  authError.textContent = "";
});

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  authError.textContent = "";
  authSubmit.disabled = true;
  try {
    const result = await api(
      registering ? "/api/auth/register" : "/api/auth/login",
      {
        method: "POST",
        body: JSON.stringify({
          email: document.querySelector("#email").value,
          password: document.querySelector("#password").value
        })
      }
    );
    token = result.access_token;
    localStorage.setItem("assistant-token", token);
    await showWorkspace();
  } catch (error) {
    authError.textContent = error.message;
  } finally {
    authSubmit.disabled = false;
  }
});

document.querySelector("#new-chat").addEventListener("click", async () => {
  try {
    clearContextFields();
    const conversation = await api("/api/conversations", {
      method: "POST",
      body: JSON.stringify({ title: "New conversation" })
    });
    activeConversation = conversation.id;
    conversationTitle.textContent = conversation.title;
    document.querySelector("#sidebar").classList.remove("mobile-open");
    messagesNode.replaceChildren(welcome);
    welcome.classList.remove("hidden");
    await loadConversations();
    messageInput.focus();
  } catch (error) {
    showToast(error.message);
  }
});

messageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;
  sendButton.disabled = true;
  try {
    if (!activeConversation) {
      const conversation = await api("/api/conversations", {
        method: "POST",
        body: JSON.stringify({ title: text.slice(0, 70) || "New conversation" })
      });
      activeConversation = conversation.id;
    }
    addMessage("user", text);
    messageInput.value = "";
    const pending = document.createElement("article");
    pending.className = "message assistant";
    pending.textContent = "Thinking…";
    pending.setAttribute("aria-live", "polite");
    messagesNode.append(pending);
    messagesNode.scrollTop = messagesNode.scrollHeight;
    const result = await api(
      `/api/conversations/${encodeURIComponent(activeConversation)}/messages`,
      {
        method: "POST",
        body: JSON.stringify({ message: text, ...readMessageContext() })
      }
    );
    pending.remove();
    addMessage("assistant", result.response, result.citations);
    conversationTitle.textContent = text.slice(0, 70);
    await loadConversations();
  } catch (error) {
    showToast(error.message);
    await loadConversations();
  } finally {
    sendButton.disabled = false;
    messageInput.focus();
  }
});

async function uploadSelectedFile() {
  const file = documentInput.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  documentInput.value = "";
  showToast(`Processing ${file.name}…`);
  try {
    const result = await api("/api/documents", { method: "POST", body });
    await loadDocuments();
    showToast(`${result.filename} is ready (${result.chunk_count} chunks).`);
  } catch (error) {
    showToast(error.message);
  }
}

document.querySelector("#upload-trigger").addEventListener("click", () => documentInput.click());
document.querySelector("#mobile-upload").addEventListener("click", () => documentInput.click());
document.querySelector("#mobile-menu").addEventListener("click", () => {
  document.querySelector("#sidebar").classList.toggle("mobile-open");
});
document.querySelector("#close-sidebar").addEventListener("click", () => {
  document.querySelector("#sidebar").classList.remove("mobile-open");
});
documentInput.addEventListener("change", uploadSelectedFile);
document.querySelector("#sign-out").addEventListener("click", signOut);

if (token) showWorkspace();
