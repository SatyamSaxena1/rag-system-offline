import os
import sys
import json
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.rag_pipeline import RAGPipeline


def main():
    ap = argparse.ArgumentParser(description='Ask questions via RAGPipeline and optionally export answers with citations.')
    ap.add_argument('--config', default='configs/default.yaml', help='Path to config YAML')
    ap.add_argument('--mode', choices=['strict','balanced','loose'], help='Behavior mode override for this run')
    ap.add_argument('--show-citations', action='store_true', help='Print citations after each answer')
    ap.add_argument('--export', help='Path to JSONL file to export results {question, answer, sources}')
    ap.add_argument('--queries', nargs='*', help='One or more queries; if omitted, reads from stdin (one per line)')
    args = ap.parse_args()

    if args.mode:
        os.environ['RAG_BEHAVIOR_MODE'] = args.mode

    pipe = RAGPipeline(args.config)

    out_f = None
    if args.export:
        os.makedirs(os.path.dirname(args.export), exist_ok=True)
        out_f = open(args.export, 'a', encoding='utf-8')

    def handle(q: str):
        ans = pipe.run(q)
        print(f"\nQ: {q}")
        print(f"A: {ans}")
        sources = getattr(pipe, '_last_sources', []) or []
        if args.show_citations and sources:
            print('Sources:')
            for s in sources:
                if s:
                    print(f" - {s}")
        if out_f:
            rec = {
                'question': q,
                'answer': ans,
                'sources': [s for s in sources if s],
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    if args.queries:
        for q in args.queries:
            handle(q)
    else:
        # Read from stdin
        try:
            for line in sys.stdin:
                q = line.strip()
                if not q:
                    continue
                handle(q)
        except KeyboardInterrupt:
            pass
    if out_f:
        out_f.close()


if __name__ == '__main__':
    main()
