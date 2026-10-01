import { NewRunForm } from "@/components/NewRunForm";

const STEPS = [
  { k: "01", t: "Understand", d: "Prompt Agent turns your request into a verified task spec" },
  { k: "02", t: "Plan", d: "Manager drafts several pipelines from KB, web and memory" },
  { k: "03", t: "Ground", d: "Plans compete on real data samples — weakest dropped" },
  { k: "04", t: "Train", d: "Operation Agent writes, runs and debugs the winner" },
  { k: "05", t: "Verify", d: "Scored on a held-out test split against your goals" },
];

export default function Home() {
  return (
    <div className="space-y-8">
      <section className="fade-up">
        <p className="eyebrow mb-3">new automl run</p>
        <h1 className="text-gradient max-w-3xl text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
          From a sentence to a verified model.
        </h1>
        <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-muted">
          Point at a dataset and describe the goal in plain language. A team of Gemini agents plans competing
          pipelines, tests them on real data, then trains, debugs and verifies the winner, all of it live.
        </p>
        <ol className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-border bg-border sm:grid-cols-5">
          {STEPS.map((s) => (
            <li key={s.k} className="bg-surface/80 p-3.5 backdrop-blur">
              <p className="font-mono text-[10px] text-accent">{s.k}</p>
              <p className="mt-1 text-sm font-semibold">{s.t}</p>
              <p className="mt-0.5 text-xs leading-snug text-muted">{s.d}</p>
            </li>
          ))}
        </ol>
      </section>
      <NewRunForm />
    </div>
  );
}
