let chartInstance = null;
const historyData = { labels: [], observed: [], predicted: [] };

async function fetchJSON(url) {
    const res = await fetch(url);
    return await res.json();
}

function initChart() {
    const ctx = document.getElementById("forecastChart").getContext("2d");
    chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: historyData.labels,
            datasets: [
                { label: "Quan trắc thực tế", data: historyData.observed, borderColor: "#2d3748", backgroundColor: "#2d3748", tension: 0.2 },
                { label: "Mô hình dự báo (+1h)", data: historyData.predicted, borderColor: "#3182ce", borderDash: [5, 5], backgroundColor: "#3182ce", tension: 0.2 }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

async function updateDashboard() {
    try {
        const [health, payload] = await Promise.all([
            fetchJSON("/health"),
            fetchJSON("/api/v1/forecasts/latest")
        ]);

        const badge = document.getElementById("statusBadge");
        if (health.is_warming_up) {
            badge.className = "badge warmup";
            badge.innerText = `Đang nạp buffer (${health.buffer_ready_stations}/12 trạm)`;
        } else if (health.consumer_connected) {
            badge.className = "badge connected";
            badge.innerText = "Trực tuyến (Live)";
        } else {
            badge.className = "badge";
            badge.innerText = "Chờ kết nối Dự án 1...";
        }

        const select = document.getElementById("stationSelect");
        if (select.children.length === 0 && payload.forecasts.length > 0) {
            payload.forecasts.forEach(f => {
                const opt = document.createElement("option");
                opt.value = f.station;
                opt.innerText = f.station.toUpperCase();
                select.appendChild(opt);
            });
            select.addEventListener("change", updateDashboard);
        }

        const currentStation = select.value;
        const currentData = payload.forecasts.find(f => f.station === currentStation);

        if (currentData) {
            document.getElementById("observedPm25").innerText = currentData.observed_pm25 ?? "--";
            document.getElementById("predictedPm25").innerText = currentData.predicted_pm25;

            const timeLabel = new Date(currentData.observation_time).toLocaleTimeString();
            if (!historyData.labels.includes(timeLabel)) {
                if (historyData.labels.length >= 15) {
                    historyData.labels.shift();
                    historyData.observed.shift();
                    historyData.predicted.shift();
                }
                historyData.labels.push(timeLabel);
                historyData.observed.push(currentData.observed_pm25);
                historyData.predicted.push(currentData.predicted_pm25);
                chartInstance.update();
            }
        }
    } catch (e) {
        console.error("Lỗi cập nhật dashboard:", e);
    }
}

window.addEventListener("DOMContentLoaded", () => {
    initChart();
    updateDashboard();
    setInterval(updateDashboard, 5000);
});