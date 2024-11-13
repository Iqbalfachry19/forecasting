import { backend } from "../../declarations/backend";

let predictionChart; // Variable to hold the chart instance

document.getElementById("predict").onclick = predict;
document.getElementById("fetchHistorical").onclick = fetchHistoricalPrices;

async function predict(event) {
  event.preventDefault();

  const message = document.getElementById("message");
  const loader = document.getElementById("loader");
  const resultsDiv = document.getElementById("results");
  const futureStepsInput = document.getElementById("future_steps");

  message.innerText = "";
  resultsDiv.innerHTML = ""; // Clear previous results
  loader.className = "loader";

  const future_steps = parseInt(futureStepsInput.value);

  try {
    // Call the backend prediction service
    const result = await backend.predict(BigInt(future_steps));

    if (result.Ok) {
      renderResults(resultsDiv, result.Ok.values);
    } else {
      throw result.Err;
    }
  } catch (err) {
    message.innerText = "Failed to predict prices: " + JSON.stringify(err);
  } finally {
    loader.className = "loader invisible"; // Ensure loader is hidden
  }

  return false;
}

async function fetchHistoricalPrices(event) {
  event.preventDefault();

  const message = document.getElementById("message");
  const loader = document.getElementById("loader");

  message.innerText = "";
  loader.className = "loader";

  try {
    // Call the backend historical price service
    const result = await backend.fetch_historical_price();

    if (result.Ok) {
      renderHistoricalData(result.Ok.values);
    } else {
      throw result.Err;
    }
  } catch (err) {
    message.innerText = "Failed to fetch historical prices: " + JSON.stringify(err);
  } finally {
    loader.className = "loader invisible"; // Hide loader after completion
  }
}

function renderResults(element, values) {
  element.innerHTML = "<h2>Predicted Prices:</h2>";
  const ul = document.createElement("ul");

  // Prepare data for the chart
  const labels = values.map((_, index) => `Day ${index + 1}`);
  const data = values.map(value => value.toFixed(2)); // Formatting values for display

  // Destroy existing chart instance if it exists
  if (predictionChart) {
    predictionChart.destroy();
  }

  // Create the chart
  const ctx = document.getElementById("predictionChart").getContext("2d");
  predictionChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Predicted Prices',
        data: data,
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderWidth: 1,
        fill: true,
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Price ($)'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Days'
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
        },
        title: {
          display: true,
          text: 'Future Price Predictions'
        }
      }
    }
  });

  element.appendChild(ul);
  values.forEach((value, index) => {
    const li = document.createElement("li");
    li.innerText = `Day ${index + 1}: $${value.toFixed(2)}`; // Format the predicted value
    ul.appendChild(li);
  });

  document.getElementById("predictionChart").style.display = "block"; // Show the chart
}

// Function to render historical data on the chart
function renderHistoricalData(values) {
  const labels = values.map((_, index) => `Day -${values.length - index}`);
  const data = values.map(value => value.toFixed(2)); // Formatting values for display

  if (predictionChart) {
    predictionChart.data.labels = [...labels, ...predictionChart.data.labels];
    predictionChart.data.datasets.unshift({
      label: 'Historical Prices',
      data: data,
      borderColor: 'rgba(255, 99, 132, 1)',
      backgroundColor: 'rgba(255, 99, 132, 0.2)',
      borderWidth: 1,
      fill: true,
    });
    predictionChart.update();
  }
}
