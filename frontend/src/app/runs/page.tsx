import { RunList } from "@/components/RunList";

export default function RunsPage() {
  return (
    <div className="space-y-6">
      <div className="fade-up">
        <h1 className="text-gradient text-3xl font-semibold tracking-tight">Runs</h1>
      </div>
      <RunList />
    </div>
  );
}
