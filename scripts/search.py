import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.connection import close_pool, open_pool
from app.services.retrieval import retrieve

if __name__ == "__main__":
    print("=== Knowledge Base Retrieval CLI ===")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    open_pool()
    try:
        while True:
            try:
                q = input("Query> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            if not q or q.lower() in ("exit", "quit", "q"):
                break

            results = retrieve(q)
            if not results:
                print("No matching chunks found.\n")
                continue

            print(f"\nFound {len(results)} matching chunk(s):")
            print("=" * 70)
            for i, r in enumerate(results, 1):
                print(f"[{i}] Similarity: {r.similarity:.4f} (Distance: {r.distance:.4f})")
                print(f"    Document: {r.filename} | Page: {r.page_number or 'N/A'} | Chunk #{r.chunk_index}")
                if r.question:
                    print(f"    Q-Anchor: {r.question}")
                print("    Content :")
                indented_content = "\n".join(f"    {line}" for line in r.content.strip().splitlines())
                print(indented_content)
                print("-" * 70)
    finally:
        close_pool()