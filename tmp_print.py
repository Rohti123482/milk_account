import sys

if len(sys.argv) < 2:
    print("Usage: python tmp_print.py <needle> [context]")
    sys.exit(1)

needle = sys.argv[1]
context = int(sys.argv[2]) if len(sys.argv) > 2 else 60

with open("app.py", "r", encoding="utf-8", errors="replace") as f:
    lines = f.read().splitlines()

hits = [i for i, l in enumerate(lines, 1) if needle in l]
if not hits:
    print(f"No matches for: {needle}")
    sys.exit(0)

out = []
for idx in hits:
    start = max(1, idx - 3)
    end = min(len(lines), idx + context)
    out.append(f"--- match at line {idx} ---")
    for j in range(start, end + 1):
        out.append(f"{j}: {lines[j-1]}")

sys.stdout.buffer.write("\n".join(out).encode("utf-8", errors="replace"))
