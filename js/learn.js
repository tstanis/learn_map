
var canvas = document.getElementById('map_canvas')
var ctx = canvas.getContext('2d');
const AGENT = 3;
const GOAL = 2;
const WALL = 1;
const EMPTY = 0;
const FIRST = 4;

MAP_COLORS = { [AGENT]: 'red', [GOAL]: 'green', [WALL]: 'black', [EMPTY]: 'white', [FIRST] : 'blue'}
var randSeed = 'foo';
var rand = new Random(randSeed);
const MAP_WIDTH = 10, MAP_HEIGHT = 10;


class Map {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.map = Array.from(Array(this.width), () => Array.from(Array(this.height), () => 0));
    }

    randomize(percent_walls) {
        this.goal = [rand.randrange(0, this.width), rand.randrange(0, this.height)];
        this.agent = [rand.randrange(0, this.width), rand.randrange(0, this.height)];
        this.map[this.agent[0]][this.agent[1]] = AGENT;
        this.map[this.goal[0]][this.goal[1]] = GOAL;
        var num_walls = Math.floor((this.width * this.height) * percent_walls);
        while (num_walls > 0) {
            var x = rand.randrange(0, this.width);
            var y = rand.randrange(0, this.height);
            if (this.map[x][y] != WALL) {
                this.map[x][y] = WALL;
                num_walls -= 1;
            }
        }
    }


    compute_one_hot_naive_first_step() {
        var diff = [this.goal[0] - this.agent[0], this.goal[1] - this.agent[1]];
        var output = [0, 0, 0, 0];
        if (Math.abs(diff[0]) >= Math.abs(diff[1])) {
            if (diff[0] >= 0) {
                output[0] = 1;
            } else if (diff[0] <= 0) {
                output[1] = 1;
            }
        } else if (Math.abs(diff[0]) <= Math.abs(diff[1])) {
            if (diff[1] >= 0) {
                output[2] = 1;
            } else if (diff[1] <= 0) {
                output[3] = 1;
            }
        }
        return output;
    }
    compute_naive_first_step_dir() {
        var diff = [this.goal[0] - this.agent[0], this.goal[1] - this.agent[1]];
        if (Math.abs(diff[0]) > Math.abs(diff[1])) {
            return [(diff[0] > 0 ? 1 : -1),0];
        } else {
            return [0, (diff[1] > 0 ? 1 : -1)];
        }
    }

    compute_naive_first_step() {
        var dir = this.compute_naive_first_step_dir();
        return [this.agent[0] + dir[0], this.agent[1] + dir[1]];
    }

    compute_best_path() {
       path = []
    }

    draw(ctx, canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        var x_run = canvas.width / this.width;
        var y_run = canvas.height / this.height;
        var first_step = this.compute_naive_first_step();
        console.log("Naive: " + first_step);
        for (var x = 0; x < this.width; x++) {
            for (var y = 0; y < this.height; y++) {
                //console.log(x + "," + y + " = " + this.map.get(x,y) + " = " + MAP_COLORS[Math.floor(this.map.get(x, y))])
                if (x == first_step[0] && y == first_step[1]) {
                    ctx.fillStyle = MAP_COLORS[FIRST];
                } else {
                    ctx.fillStyle = MAP_COLORS[Math.floor(this.map[x][y])];
                }
                ctx.fillRect(x * x_run + 1, y * y_run + 1, x_run - 2, y_run - 2);
            }
        }
        ctx.fillStyle = 'blue';
    }
}

// the_map = new Map(50, 50)
// the_map.randomize(0.05);
// the_map.draw(ctx, canvas);

// var frameNum = 0;

// function drawFrame() {
//     frameNum += 1;
//     if (frameNum % 60 == 0) {
//         the_map = new Map(50, 50)
//         the_map.randomize(0.05);
//         the_map.draw(ctx, canvas);
//     }
//     window.requestAnimationFrame(drawFrame);
// };

//window.requestAnimationFrame(drawFrame);

const NUM_CLASSES = 4;
function generateBatch(batchSize) {
    var maps = []
    for (var i = 0; i < batchSize; ++i) {
        var map = new Map(MAP_WIDTH, MAP_HEIGHT)
        map.randomize(0.05);
        maps.push(map)
    }
    var map_tensors = []
    var map_labels = []
    maps.forEach(function(map) {
        map_tensors.push(map.map);
        map_labels.push(map.compute_one_hot_naive_first_step())
    });

    const xs = tf.oneHot(tf.tensor(map_tensors, null, 'int32').reshape([batchSize * MAP_WIDTH * MAP_HEIGHT]), 4).reshape([batchSize, MAP_WIDTH, MAP_HEIGHT, 4]);
    const labels = tf.tensor2d(
        map_labels, [map_labels.length, NUM_CLASSES], 'int32');
    return {xs, labels};
}

function createConvModel() {
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [MAP_WIDTH, MAP_HEIGHT, 4],
        kernelSize: 3,
        filters: 4,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 10,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
    model.add(tf.layers.dense({
        units: 4,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax'
    }))

    const LEARNING_RATE = 0.15;
    const optimizer = tf.train.sgd(LEARNING_RATE);

    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    return model;
}

function createDenseModel() {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape: [MAP_WIDTH, MAP_HEIGHT, 4]}));
    model.add(tf.layers.dense({units: 42, activation: 'relu'}));
    model.add(tf.layers.dense({units: 4, activation: 'softmax'}));

    const LEARNING_RATE = 0.15;
    const optimizer = 'rmsprop';//tf.train.sgd(LEARNING_RATE);

    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    return model;
  }

async function train(model, onIteration) {
    // How many examples the model should "see" before making a parameter update.
    const BATCH_SIZE = 320;
    // How many batches to train the model for.
    const TRAIN_BATCHES = 10000;

    // Every TEST_ITERATION_FREQUENCY batches, test accuracy over TEST_BATCH_SIZE examples.
    // Ideally, we'd compute accuracy over the whole test set, but for performance
    // reasons we'll use a subset.
    const TEST_BATCH_SIZE = 1000;
    const TEST_ITERATION_FREQUENCY = 5;
    var trainBatchCount = 0;
    for (let i = 0; i < TRAIN_BATCHES; i++) {
        const trainBatch = generateBatch(BATCH_SIZE);
    
        let testBatch;
        let validationData;
        // Every few batches test the accuracy of the mode.
        if (i % TEST_ITERATION_FREQUENCY === 0) {
            testBatch = generateBatch(TEST_BATCH_SIZE);
            validationData = [
                testBatch.xs, testBatch.labels
            ];
        }
    
        // The entire dataset doesn't fit into memory so we call fit repeatedly
        // with batches.
        const history = await model.fit(
            trainBatch.xs,
            trainBatch.labels,
            {
            batchSize: BATCH_SIZE,
            validationData,
            epochs: 1,
            callbacks: {
                    onBatchEnd: async (batch, logs) => {
                    trainBatchCount++;
                    console.log(
                        `Training... (` +
                        `${(trainBatchCount / TRAIN_BATCHES * 100).toFixed(1)}%` +
                        ` complete). To stop training, refresh or close page.`);
                    plotLoss(trainBatchCount, logs.loss, 'train');
                    plotAccuracy(trainBatchCount, logs.acc, 'train');
                    if (onIteration && batch % 10 === 0) {
                        onIteration('onBatchEnd', batch, logs);
                    }
                    await tf.nextFrame();
                    },
                    onEpochEnd: async (epoch, logs) => {
                    valAcc = logs.val_acc;
                    plotLoss(trainBatchCount, logs.val_loss, 'validation');
                    plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
                    if (onIteration) {
                        onIteration('onEpochEnd', epoch, logs);
                    }
                    await tf.nextFrame();
                    }
                }
            });
    
        const loss = history.history.loss[0];
        const accuracy = history.history.acc[0];
    }
}

function labelToString(label) {
    var result = "";
    if (label.get(0) == 1) {
        result += "R"
    }
    if (label.get(1) == 1) {
        result += "L"
    }
    if (label.get(2) == 1) {
        result += "D"
    }
    if (label.get(4) == 1) {
        result += "U"
    }
}

PREDICTION = { 0: 'R', 1: 'L', 2: 'D', 3: 'U'};

const testExamples = 100;
const examples = generateBatch(testExamples);

async function showPredictions(model) {
  
    // Code wrapped in a tf.tidy() function callback will have their tensors freed
    // from GPU memory after execution without having to call dispose().
    // The tf.tidy callback runs synchronously.
    tf.tidy(() => {
      const output = model.predict(examples.xs);
  
      // tf.argMax() returns the indices of the maximum values in the tensor along
      // a specific axis. Categorical classification tasks like this one often
      // represent classes as one-hot vectors. One-hot vectors are 1D vectors with
      // one element for each output class. All values in the vector are 0
      // except for one, which has a value of 1 (e.g. [0, 0, 0, 1, 0]). The
      // output from model.predict() will be a probability distribution, so we use
      // argMax to get the index of the vector element that has the highest
      // probability. This is our prediction.
      // (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
      // dataSync() synchronously downloads the tf.tensor values from the GPU so
      // that we can use them in our normal CPU JavaScript code
      // (for a non-blocking version of this function, use data()).
      const axis = 1;
      const labels = Array.from(examples.labels.dataSync()).map(pred => labelToString(pred));
      const predictions = Array.from(output.argMax(axis).dataSync()).map(pred => PREDICTION[pred]);
  
      showTestResults(examples, predictions, labels);
    });
  }

  setTrainButtonCallback(async () => {
    console.log('Creating model...');
    const model = createConvModel();
    model.summary();
  
    logStatus('Starting model training...');
    await train(model, () => showPredictions(model));
  });


