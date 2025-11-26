import * as tf from '@tensorflow/tfjs';
import { loadTitanicSplits } from './dataset.js';

const { train, val, test } = await loadTitanicSplits();

// Modelo de regresión logística
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [5] }));

model.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

// Entrenar
await model.fit(train.xs, train.ys, {
  epochs: 40,
  validationData: [val.xs, val.ys],
  batchSize: 32,
});

// Evaluación
const probs = model.predict(test.xs);
const preds = probs.greater(tf.scalar(0.5)).cast('int32');
const yPred = Array.from(preds.dataSync());
const yTrue = Array.from(test.ys.dataSync());

let TP = 0, TN = 0, FP = 0, FN = 0;
for (let i = 0; i < yTrue.length; i++) {
  if (yTrue[i] === 1 && yPred[i] === 1) TP++;
  else if (yTrue[i] === 0 && yPred[i] === 0) TN++;
  else if (yTrue[i] === 0 && yPred[i] === 1) FP++;
  else if (yTrue[i] === 1 && yPred[i] === 0) FN++;
}

const safeDivide = (num, denom) => denom === 0 ? 0 : num / denom;

console.log('Matriz de confusión:', { TP, TN, FP, FN });
const total = TP + TN + FP + FN;
console.log('Exactitud:', safeDivide(TP + TN, total).toFixed(4));
console.log('Precision:', safeDivide(TP, TP + FP).toFixed(4));
console.log('Exhaustividad:', safeDivide(TP, TP + FN).toFixed(4));
console.log('F1-Score:', safeDivide(2 * safeDivide(TP, TP + FP) * safeDivide(TP, TP + FN),
                                      safeDivide(TP, TP + FP) + safeDivide(TP, TP + FN)).toFixed(4));

// Exportar modelo para usar en web
export { model };

