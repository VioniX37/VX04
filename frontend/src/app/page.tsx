import { NewRunForm } from "@/components/NewRunForm";

export default function Home() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">New AutoML run</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Upload a dataset and describe what you want in plain language. The agents parse your request, retrieve
          ML knowledge, plan several pipelines, evaluate them in parallel, then write, run and verify the code.
        </p>
      </div>
      <NewRunForm />
    </div>
  );
}
