import Form from "@/components/Form";

export default function Home() {

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-zinc-900">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
        <div className="flex flex-col items-center gap-6 text-center sm:items-start sm:text-left">
          <h1 className="max-w-xs text-3xl font-semibold leading-10 tracking-tight text-black dark:text-zinc-50">
            Predicción Supervivencia Titanic  
          </h1>
          <p className="max-w-md text-lg leading-8 text-zinc-600 dark:text-zinc-400">
            Llene los campos{" "}
            <a
              className="font-medium text-zinc-950 dark:text-zinc-50"
            >
              campos
            </a>{" "}
            para empezar a {" "}
            <a
              className="font-medium text-zinc-950 dark:text-zinc-50"
            >
              Predecir.
            </a>
            <Form />
          </p>
        </div>
      </main>
    </div>
  );
}
