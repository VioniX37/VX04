import { NewRunForm } from "@/components/NewRunForm";

export default function Home() {
  return (
    <div className="space-y-8">
      <section className="fade-up">
        <h1 className="text-gradient text-3xl font-semibold tracking-tight">New run</h1>
      </section>
      <NewRunForm />
    </div>
  );
}
