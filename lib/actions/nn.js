import * as tf from '@tensorflow/tfjs';
import { loadTitanicSplits } from './dataset.js';
import { saveModelToPublic, savePreprocessingStats } from './modelStorage.js';

const { train, val, test, stats } = await loadTitanicSplits();
const numFeatures = train.xs.shape[1];

const nnModel = tf.sequential();

// Capa 1
nnModel.add(tf.layers.dense({
  units: 32,
  activation: 'relu',
  inputShape: [numFeatures],
}));

// Capa 2
nnModel.add(tf.layers.dense({
  units: 16,
  activation: 'relu',
}));

// Capa de salida
nnModel.add(tf.layers.dense({
  units: 1,
  activation: 'sigmoid',
}));

nnModel.compile({
  optimizer: tf.train.adam(0.001),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

await nnModel.fit(train.xs, train.ys, {
  epochs: 50,
  validationData: [val.xs, val.ys],
  batchSize: 32,
  shuffle: true,
});

const evalOutput = await nnModel.evaluate(test.xs, test.ys);
const toNumber = (tensor) => tensor.dataSync()[0];
let lossValue;
let evalAccuracy;
if (Array.isArray(evalOutput)) {
  const [lossTensor, accTensor] = evalOutput;
  lossValue = toNumber(lossTensor);
  evalAccuracy = toNumber(accTensor);
} else {
  lossValue = toNumber(evalOutput);
}

// Prediccion sobre test
const probs = nnModel.predict(test.xs);
const preds = probs.greater(tf.scalar(0.5)).cast('int32');

const labelsTensor = test.ys.squeeze().toInt();
const predsTensor = preds.squeeze().toInt();
const confusionTensor = tf.math.confusionMatrix(labelsTensor, predsTensor, 2);
const [[tn, fp], [fn, tp]] = confusionTensor.arraySync();

const safeDivide = (num, denom) => (denom === 0 ? 0 : num / denom);
const total = tp + tn + fp + fn;
const accuracy = safeDivide(tp + tn, total);
const precision = safeDivide(tp, tp + fp);
const recall = safeDivide(tp, tp + fn);
const f1 = safeDivide(2 * precision * recall, precision + recall);

console.log('Matriz de confusion:');
confusionTensor.print();
console.log('Error:', lossValue.toFixed(4));
if (evalAccuracy !== undefined) {
  console.log('Exactitud (tf.evaluate):', evalAccuracy.toFixed(4));
}
console.log('Precision:', precision.toFixed(4));
console.log('Exhaustividad:', recall.toFixed(4));
console.log('F1-Score:', f1.toFixed(4));

await saveModelToPublic(nnModel, 'nn');
await savePreprocessingStats(stats);

export { nnModel };
