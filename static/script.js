function showTab(tabId) {
  document.querySelectorAll(".tab-content").forEach((tab) => {
    tab.classList.remove("active");
  });
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.remove("active");
  });

  document.getElementById(tabId).classList.add("active");
  event.target.classList.add("active");
}


const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    fileName.textContent = `Selected: ${e.target.files[0].name}`;
  }
});


async function analyzeText() {
  const text = document.getElementById("text-input").value.trim();

  if (!text) {
    showError("Please enter some text to analyze");
    return;
  }

  const formData = new FormData();
  formData.append("text", text);

  await sendRequest(formData);
}


async function analyzeFile() {
  const file = fileInput.files[0];

  if (!file) {
    showError("Please select a file to analyze");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  await sendRequest(formData);
}


async function sendRequest(formData) {
  //  loading
  document.getElementById("loading").style.display = "block";
  document.getElementById("results").style.display = "none";
  document.getElementById("error").style.display = "none";

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      displayResults(data);
    } else {
      showError(data.error || "Analysis failed");
    }
  } catch (error) {
    showError(`Error: ${error.message}`);
  } finally {
    document.getElementById("loading").style.display = "none";
  }
}


function displayResults(data) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.style.display = "block";

  const card = document.getElementById("result-card");
  const classification = document.getElementById("classification");

  classification.textContent = data.classification;

  if (data.classification === "HUMAN") {
    card.className = "result-card human";
  } else {
    card.className = "result-card ai";
  }

  document.getElementById("confidence").textContent = data.confidence;

  document.getElementById("metric-rank").textContent =
    data.metrics.effective_rank.toFixed(4);
  document.getElementById("metric-var").textContent =
    data.metrics.total_variance.toFixed(6);
  document.getElementById("threshold-rank").textContent =
    data.thresholds.effective_rank.toFixed(4);
  document.getElementById("threshold-var").textContent =
    data.thresholds.total_variance.toFixed(6);

  document.getElementById("combined-score").textContent =
    data.metrics.combined_score.toFixed(4);
  document.getElementById("n-sentences").textContent =
    data.analysis.n_sentences;
  document.getElementById("margin").textContent =
    data.analysis.margin.toFixed(4);

  if (data.votes) {
    document.getElementById("vote-rank").textContent =
      data.votes.by_rank || "N/A";
    document.getElementById("vote-var").textContent =
      data.votes.by_variance || "N/A";
  }

  const explanation = document.getElementById("explanation");
  if (data.classification === "AI") {
    explanation.innerHTML = `
            <strong>⚠️ Likely AI-Generated</strong>
            Low geometric complexity detected:<br>
            • Effective Rank below threshold<br>
            • Limited semantic diversity<br>
            • Constrained generation pattern
        `;
  } else {
    explanation.innerHTML = `
            <strong>✓ Likely Human-Written</strong>
            High geometric complexity detected:<br>
            • Effective Rank above threshold<br>
            • Rich semantic diversity<br>
            • Natural writing pattern
        `;
  }

  resultsDiv.scrollIntoView({ behavior: "smooth", block: "start" });
}


function showError(message) {
  const errorDiv = document.getElementById("error");
  errorDiv.textContent = message;
  errorDiv.style.display = "block";

  document.getElementById("results").style.display = "none";
  document.getElementById("loading").style.display = "none";

  // Scroll to error
  errorDiv.scrollIntoView({ behavior: "smooth" });
}
