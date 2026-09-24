import { RunList } from "@/components/RunList";

export default function RunsPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Runs</h1>
      <RunList />
    </div>
  );
}
