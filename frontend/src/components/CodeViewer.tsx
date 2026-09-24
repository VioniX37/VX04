"use client";

import { useState } from "react";

import { Button } from "./ui";

export function CodeViewer({ code, filename = "train.py" }: { code: string; filename?: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <div className="flex items-center justify-between bg-surface-muted px-3 py-1.5">
        <span className="font-mono text-xs text-muted">{filename}</span>
        <Button variant="ghost" className="px-2 py-1 text-xs" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      <pre className="max-h-[28rem] overflow-auto bg-code-bg p-4 font-mono text-xs leading-relaxed text-code-fg">
        {code}
      </pre>
    </div>
  );
}
