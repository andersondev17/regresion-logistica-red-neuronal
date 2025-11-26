import * as tf from '@tensorflow/tfjs';
import { loadTitanicSplits } from './dataset.js';
import { saveModelToPublic, savePreprocessingStats } from './modelStorage.js';

const { train, val, test, stats } = await loadTitanicSplits();

// Modelo de regresion logistica
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [2] }));

model.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

// Entrenar
await model.fit(train.xs, train.ys, {
  epochs: 40,
  validationData: [val.xs, val.ys],
  batchSize: 32,
});

const evalOutput = await model.evaluate(test.xs, test.ys);
const extractScalar = (tensor) => tensor.dataSync()[0];
let lossValue;
let evalAccuracy;
if (Array.isArray(evalOutput)) {
  const [lossTensor, accTensor] = evalOutput;
  lossValue = extractScalar(lossTensor);
  evalAccuracy = extractScalar(accTensor);
} else {
  lossValue = extractScalar(evalOutput);
}

// Evaluacion en test
const probs = model.predict(test.xs);
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

await saveModelToPublic(model, 'logistic');
await savePreprocessingStats(stats);

// Exportar modelo en memoria
export { model };
