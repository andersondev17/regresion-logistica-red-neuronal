import * as tf from '@tensorflow/tfjs';
import fetch from 'node-fetch';

const CSV_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv';

function splitCSVLine(line) {
  const cells = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      cells.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }

  cells.push(current.trim());
  return cells;
}

function parseCSV(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim() !== '');
  const headers = splitCSVLine(lines[0]);

  return lines.slice(1).map((line) => {
    const values = splitCSVLine(line);
    return headers.reduce((row, header, idx) => {
      row[header] = values[idx] ?? '';
      return row;
    }, {});
  });
}

async function fetchCSV(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`No se pudo obtener el CSV (${res.status})`);
  const text = await res.text();
  return parseCSV(text);
}

function preprocessRows(rows) {
  const processed = rows.map((row) => {
    let Age = row.Age ? parseFloat(row.Age.trim()) : null;
    if (Number.isNaN(Age)) Age = null;

    const Sex = row.Sex.toLowerCase() === 'male' ? 1 : 0;
    const Survived = parseInt(row.Survived, 10);

    return { Age, Sex, Survived };
  });

  const ages = processed.map((r) => r.Age).filter((age) => age !== null);
  const meanAge = ages.reduce((a, b) => a + b, 0) / ages.length;
  processed.forEach((r) => { if (r.Age === null) r.Age = meanAge; });

  const ageValues = processed.map((r) => r.Age);
  const ageMin = Math.min(...ageValues);
  const ageMax = Math.max(...ageValues);
  const ageRange = ageMax - ageMin || 1;
  processed.forEach((r) => {
    r.Age = (r.Age - ageMin) / ageRange;
  });

  return {
    samples: processed,
    stats: {
      ageMean: meanAge,
      ageMin,
      ageMax,
      ageRange,
    },
  };
}

function toTensorPairs(samples) {
  const xs = samples.map((sample) => [sample.Sex, sample.Age]);
  const ys = samples.map((sample) => [sample.Survived]);
  return {
    xs: tf.tensor2d(xs),
    ys: tf.tensor2d(ys),
  };
}

export async function loadTitanicSplits({ trainSplit = 0.7, valSplit = 0.15 } = {}) {
  const rows = await fetchCSV(CSV_URL);
  const { samples, stats } = preprocessRows(rows);

  const shuffled = samples.slice().sort(() => Math.random() - 0.5);

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
    stats,
  };
}

export { CSV_URL };
