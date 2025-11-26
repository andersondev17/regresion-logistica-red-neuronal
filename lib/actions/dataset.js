import * as tf from '@tensorflow/tfjs';
import fetch from 'node-fetch';

const CSV_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv';

async function fetchCSV(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`No se pudo obtener el CSV (${res.status} ${res.statusText})`);
  const text = await res.text();
  const lines = text.split('\n').filter(line => line.trim() !== '');
  const headers = lines[0].split(',');

  return lines.slice(1).map(line => {
    const values = line.split(',');
    const row = {};
    headers.forEach((h, idx) => { row[h] = values[idx]; });
    return row;
  });
}

function preprocessRows(rows) {
  const processed = rows.map(row => {
    let Age = parseFloat(row.Age);
    if (Number.isNaN(Age)) Age = null;

    return {
      Age,
      Sex: row.Sex.toLowerCase() === 'male' ? 1 : 0,
      Pclass: parseInt(row.Pclass, 10),
      SibSp: parseInt(row.SibSp, 10),
      Parch: parseInt(row.Parch, 10),
      Survived: parseInt(row.Survived, 10),
    };
  });

  // Rellenar Age con la media
  const ages = processed.map(r => r.Age).filter(a => a !== null);
  const meanAge = ages.reduce((a, b) => a + b, 0) / ages.length;
  processed.forEach(r => { if (r.Age === null) r.Age = meanAge; });

  // Normalizar numéricas entre 0 y 1
  const features = ['Age', 'Pclass', 'SibSp', 'Parch'];
  features.forEach(f => {
    const vals = processed.map(r => r[f]);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    processed.forEach(r => { r[f] = (r[f] - min) / (max - min); });
  });

  return { processed, meanAge };
}

function toTensorPairs(samples) {
  const xs = samples.map(s => [s.Sex, s.Age, s.Pclass, s.SibSp, s.Parch]);
  const ys = samples.map(s => [s.Survived]);
  return {
    xs: tf.tensor2d(xs),
    ys: tf.tensor2d(ys),
  };
}

export async function loadTitanicSplits({ trainSplit = 0.7, valSplit = 0.15 } = {}) {
  const rows = await fetchCSV(CSV_URL);
  const { processed } = preprocessRows(rows);

  const shuffled = processed.slice();
  tf.util.shuffle(shuffled);

  const total = shuffled.length;
  const trainEnd = Math.floor(total * trainSplit);
  const valEnd = trainEnd + Math.floor(total * valSplit);

  const trainSamples = shuffled.slice(0, trainEnd);
  const valSamples = shuffled.slice(trainEnd, valEnd);
  const testSamples = shuffled.slice(valEnd);

  return {
    train: toTensorPairs(trainSamples),
    val: toTensorPairs(valSamples),
    test: toTensorPairs(testSamples),
  };
}

export { CSV_URL };



