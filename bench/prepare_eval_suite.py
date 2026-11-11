"""Export existing local benchmark files to dependency-free evaluation inputs."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(path):
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == '.parquet':
            return pq.read_table(path).to_pylist()
        with path.open('rb') as f:
            return ipc.open_stream(f).read_all().to_pylist()

    def write(name, rows):
        path = args.out / (name + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=True) + '\n' for r in rows))
        return dict(rows=len(rows), path=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    root = args.dataset_root
    manifest = {}
    hs = read(root / 'hellaswag/validation/data-00000-of-00001.arrow')
    assert len(hs) == 10042 and all(0 <= int(r['label']) < 4 for r in hs)
    manifest['hellaswag'] = write('hellaswag', hs)
    lo = read(root / 'lambada_openai/test/data-00000-of-00001.arrow')
    assert len(lo) == 5153 and all(' ' in r['text'] for r in lo)
    manifest['lambada_openai'] = write('lambada_openai', lo)
    wt = read(root / 'wikitext-2-raw-v1/test-00000-of-00001.parquet')
    manifest['wikitext2'] = write('wikitext2', wt)
    tests, devs = [], {}
    for folder in sorted((root / 'mmlu').iterdir()):
        if not folder.is_dir() or folder.name == 'all':
            continue
        testfile = folder / 'test-00000-of-00001.parquet'
        if not testfile.exists():
            continue
        subject = folder.name
        dev = read(folder / 'dev-00000-of-00001.parquet')
        assert len(dev) == 5
        devs[subject] = dev
        for row in read(testfile):
            assert 0 <= int(row['answer']) < 4 and len(row['choices']) == 4
            tests.append(dict(row, subject=subject))
    assert len(devs) == 57 and len(tests) == 14042
    manifest['mmlu'] = write('mmlu', tests)
    dev_path = args.out / 'mmlu_dev.json'
    dev_path.write_text(json.dumps(devs, ensure_ascii=False))
    manifest['mmlu_dev'] = dict(path=dev_path.name,
        sha256=hashlib.sha256(dev_path.read_bytes()).hexdigest())
    manifest['sources'] = sources
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'sources'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
