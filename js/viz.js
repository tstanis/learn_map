const statusElement = document.getElementById('status');
const messageElement = document.getElementById('message');
const imagesElement = document.getElementById('images');

function logStatus(message) {
  statusElement.innerText = message;
  console.log(message);
}

function trainingLog(message) {
  messageElement.innerText = `${message}\n`;
  console.log(message);
}

function showTestResults(batch, predictions, labels) {
  const testExamples = batch.xs.shape[0];
  imagesElement.innerHTML = '';
  for (let i = 0; i < testExamples; i++) {
    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

    const div = document.createElement('div');
    div.className = 'pred-container';

    const canvas = document.createElement('canvas');
    canvas.className = 'prediction-canvas';
    draw(image, canvas);

    const pred = document.createElement('div');

    const prediction = predictions[i];
    const label = labels[i];
    const correct = prediction === label;

    pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
    pred.innerText = `pred: ${prediction} `+ `label: ${label} `;

    div.appendChild(pred);
    div.appendChild(canvas);

    imagesElement.appendChild(div);
  }
}

function setTrainButtonCallback(callback) {
    const trainButton = document.getElementById('train');
    trainButton.addEventListener('click', () => {
      trainButton.setAttribute('disabled', true);
      callback();
    });
}

const lossLabelElement = document.getElementById('loss-label');
const accuracyLabelElement = document.getElementById('accuracy-label');
const lossValues = [[], []];
function plotLoss(batch, loss, set) {
    const series = set === 'train' ? 0 : 1;
    lossValues[series].push({x: batch, y: loss});
    const lossContainer = document.getElementById('loss-canvas');
    tfvis.render.linechart(
        {values: lossValues, series: ['train', 'validation']}, lossContainer, {
            xLabel: 'Batch #',
            yLabel: 'Loss',
            width: 400,
            height: 300,
        });
     
    if (loss) {
        lossLabelElement.innerText = `last loss: ${loss.toFixed(3)}`;
    }
}

const accuracyValues = [[], []];
function plotAccuracy(batch, accuracy, set) {
  const accuracyContainer = document.getElementById('accuracy-canvas');
  const series = set === 'train' ? 0 : 1;
  accuracyValues[series].push({x: batch, y: accuracy});
  tfvis.render.linechart(
      {values: accuracyValues, series: ['train', 'validation']},
      accuracyContainer, {
        xLabel: 'Batch #',
        yLabel: 'Accuracy',
        width: 400,
        height: 300,
      });
  accuracyLabelElement.innerText =
      `last accuracy: ${(accuracy * 100).toFixed(1)}%`;
}

function drawLegacy(image, canvas) {
    
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      imageData.data[j + 0] = data[i] * 255;
      imageData.data[j + 1] = data[i] * 255;
      imageData.data[j + 2] = data[i] * 255;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

function draw(image, canvas) {
    COLORS = {'1000': 'white', '0100' : 'black', '0010' : 'green', '0001': 'red'};
    const ctx = canvas.getContext('2d');
    canvas.width = 100;
    canvas.height = 100;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    var img_width = image.shape[1];
    var img_height = image.shape[2];
    var x_run = canvas.width / img_width;
    var y_run = canvas.height / img_height;
    
    for (var x = 0; x < img_width; x++) {
        for (var y = 0; y < img_height; y++) {
            var pixel_value = [image.get(0, x, y, 0), image.get(0, x, y, 1), image.get(0, x, y, 2), image.get(0, x, y, 3)].join("");
            ctx.fillStyle = COLORS[pixel_value];
            ctx.fillRect(x * x_run + 1, y * y_run + 1, x_run - 2, y_run - 2);
        }
    }
}