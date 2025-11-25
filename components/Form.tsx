"use client";

import { useState } from "react";
import { Button } from "./ui/Button";

export default function PredictionForm({
  action
}: {
  action: (formData: FormData) => Promise<any>
}) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  async function handleAction(formData: FormData) {
    setLoading(true);
    setResult(null);

    const response = await action(formData);
    setResult(response);
    setLoading(false);
  }

  return (
    <div className="space-y-10">
      <form className="space-y-4" action={handleAction}>
        {/* Sexo */}
        <div className="space-y-1">
          <label htmlFor="sex" className="text-zinc-900 dark:text-zinc-100 leading-8">
            Sexo *
          </label>
          <select
            id="sex"
            name="sex"
            className="w-full rounded-xl border border-zinc-300 bg-white px-4 py-3 text-zinc-900 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            <option value="">Seleccione el sexo</option>
            <option value="male">Masculino</option>
            <option value="female">Femenino</option>
          </select>
        </div>

        {/* Edad */}
        <div className="space-y-1">
          <label htmlFor="age" className="text-zinc-900 dark:text-zinc-100 leading-8">
            Edad *
          </label>
          <input
            id="age"
            name="age"
            type="number"
            placeholder="Ingresa tu edad"
            className="w-full rounded-xl border border-zinc-300 bg-white px-4 py-3 text-zinc-900 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            min="0"
            max="120"
            required
          />
        </div>

        <Button
          type="submit"
          fullWidth
          disabled={loading}
          variant="primary"
        >
          {loading ? "Calculando..." : "Calcular"}
        </Button>
      </form>

      {/* Resultados */}
      {result && (
        <div className={`p-4 rounded-xl border ${result.success
          ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
          : 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
          }`}>
          {result.success ? (
            <div className="text-center">
              <h3 className="text-lg font-semibold text-green-900 dark:text-green-100">
                {result.prediction === "Sobrevive" ? "✅ SOBREVIVIRÍA" : "❌ NO SOBREVIVIRÍA"}
              </h3>
              <p className="text-green-700 dark:text-green-300 mt-2">
                Probabilidad: <strong>{(result.survivalProbability * 100).toFixed(1)}%</strong>
              </p>
            </div>
          ) : (
            <div className="text-center">
              <h3 className="text-lg font-semibold text-red-900 dark:text-red-100">
                Error
              </h3>
              <p className="text-red-700 dark:text-red-300">
                {result.error}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}