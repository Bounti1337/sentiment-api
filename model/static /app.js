async function predict() {
    const text = document.getElementById("text").value;
    const resultDiv = document.getElementById("result");

    resultDiv.innerHTML = "⏳ Analyzing...";

    const response = await fetch("/v1/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    resultDiv.innerHTML = `
        <b>Label:</b> ${data.label}<br>
        <b>Confidence:</b> ${data.confidence.toFixed(2)}
    `;
}
