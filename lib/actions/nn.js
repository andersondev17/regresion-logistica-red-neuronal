import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import { loadTitanicSplits } from './dataset.js';

const { train, val, test } = await loadTitanicSplits();
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
if (Array.isArray(evalOutput)) {
  const [lossTensor, accTensor] = evalOutput;
  console.log('Loss:', toNumber(lossTensor).toFixed(4));
  console.log('Accuracy:', toNumber(accTensor).toFixed(4));
} else {
  console.log('Loss:', toNumber(evalOutput).toFixed(4));
}
