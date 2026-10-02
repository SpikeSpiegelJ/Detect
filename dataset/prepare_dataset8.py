"""Audit an external LabelMe/YOLO dataset, then merge and repartition it with an existing YOLO dataset."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml

from split_data import Sample, SPLITS, audit_sources, audit_pair_directories, class_counts, stratified_split
from LabelMeToYOLO import convert_labelme_to_yolo

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BASE = ROOT / "repartition_v6_dataset7_9c"
SOURCE = PROJECT / "dataset_8"
OUTPUT = ROOT / "repartition_v7_dataset8_9c"
REPORT = ROOT / "dataset8_premerge_audit.json"
SEED = 20260909


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def inspect(base=BASE, source=SOURCE, report_path=REPORT, seed=SEED):
    """Audit source annotations and group visually similar samples to prevent split leakage."""
    names = yaml.safe_load((ROOT / "data_repartition_v6_dataset7_9c.yaml").read_text())["names"]
    mapping = {v: k for k, v in names.items()}
    existing, external, issues, missing_images, missing_labels = audit_sources(base, source)
    if missing_images or missing_labels:
        issues.append(f"Unpaired files: {missing_images}, {missing_labels}")
    samples = existing + external
    snapshots = {}
    pixels, thumbnails, hashes, aspects = [], [], [], []
    for sample in samples:
        snapshots[str(sample.image)] = digest(sample.image)
        snapshots[str(sample.label)] = digest(sample.label)
        im = cv2.imdecode(np.fromfile(sample.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        pixels.append(sha256(str(im.shape).encode() + im.tobytes()).hexdigest())
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        thumbnails.append(thumb)
        low = cv2.dct(thumb.astype(np.float32))[:8, :8].flatten()[1:]
        hashes.append(sum(int(v > np.median(low)) << i for i, v in enumerate(low)))
        aspects.append(w / h)
        if sample.origin != source.name:
            continue
        jp = sample.image.with_suffix('.json')
        snapshots[str(jp)] = digest(jp)
        data = json.loads(jp.read_text(encoding='utf-8-sig'))
        if (data['imageWidth'], data['imageHeight']) != (w, h):
            issues.append(f"JSON dimensions mismatch: {jp}")
        _, expected = convert_labelme_to_yolo(jp, mapping)
        actual = [list(map(float, line.split())) for line in sample.label.read_text().splitlines() if line.strip()]
        if len(actual) != len(expected) or not np.allclose(sorted(actual), sorted(expected), atol=1e-6, rtol=0):
            issues.append(f"JSON/TXT annotation mismatch: {jp}")
    parent = list(range(len(samples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    # Conservative similarity grouping, not deletion; all annotations are retained.
    for i in range(len(samples)):
        for j in range(i):
            exact = pixels[i] == pixels[j]
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if not exact and (distance > 4 or abs(aspects[i]/aspects[j]-1) > 0.05):
                continue
            mae = float(np.abs(thumbnails[i].astype(float)-thumbnails[j]).mean())
            if exact or mae <= 12:
                parent[find(i)] = find(j)
                pairs.append({'a': str(samples[j].image), 'b': str(samples[i].image), 'exact_pixels': exact,
                              'phash_distance': distance, 'thumbnail_mae': mae})
    groups = {}
    for i, sample in enumerate(samples):
        groups.setdefault(find(i), []).append(sample)
    report = {'seed': seed, 'names': names, 'base_images': len(existing), 'added_images': len(external),
              'added_instances': dict(class_counts(external)), 'total_instances': dict(class_counts(samples)),
              'issues': issues, 'similar_pairs': pairs, 'groups': len(groups), 'source_sha256': snapshots,
              'note': 'Similarity is a conservative heuristic, not proof of absence of all near duplicates.'}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','similar_pairs')}, ensure_ascii=False))
    print(f'Similarity pairs: {len(pairs)}; report: {report_path}', flush=True)
    if issues:
        raise ValueError('Audit failed; originals unchanged. See premerge report.')
    return samples, list(groups.values()), report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE, help='Existing split dataset to merge.')
    parser.add_argument('--source', type=Path, default=SOURCE, help='External images, LabelMe JSON, and YOLO labels.')
    parser.add_argument('--output', type=Path, default=OUTPUT, help='New independently copied split dataset.')
    parser.add_argument('--report', type=Path, default=REPORT, help='Premerge audit report path.')
    parser.add_argument(
        '--config', type=Path, default=ROOT / 'data_repartition_v7_dataset8_9c.yaml', help='Output dataset YAML.'
    )
    parser.add_argument('--seed', type=int, default=SEED, help='Deterministic stratified split seed.')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    base, source, output = args.base.resolve(), args.source.resolve(), args.output.resolve()
    report_path, config_path = args.report.resolve(), args.config.resolve()
    if args.apply and output.exists():
        raise FileExistsError(output)
    samples, groups, report = inspect(base, source, report_path, args.seed)
    if not args.apply:
        return
    # Reuse the existing multilabel splitter at group level to prevent similarity leakage.
    representatives = [
        Sample(
            group[0].image,
            group[0].label,
            group[0].origin,
            tuple(cls for sample in group for cls in sample.classes),
        )
        for group in groups
    ]
    by_image = {g[0].image: g for g in groups}
    split_groups = stratified_split(representatives, dict(train=.7, val=.15, test=.15), args.seed)
    manifest = {'seed': args.seed, 'ratios': [0.7,0.15,0.15], 'splits': {}}
    assigned = {}
    for split in SPLITS:
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
        manifest['splits'][split] = []
        for rep in split_groups[split]:
            for s in by_image[rep.image]:
                name = f"{s.origin}_{report['source_sha256'][str(s.image)][:20]}"
                im = output/'images'/split/(name+s.image.suffix.lower())
                lab = output/'labels'/split/(name+'.txt')
                if im.exists() or lab.exists():
                    raise ValueError(f'Output name collision: {im}')
                shutil.copy2(s.image, im)
                shutil.copy2(s.label, lab)
                if (
                    digest(im) != report['source_sha256'][str(s.image)]
                    or digest(lab) != report['source_sha256'][str(s.label)]
                ):
                    raise ValueError('Copy verification failed')
                assigned[str(s.image)] = split
                manifest['splits'][split].append(
                    {'image': str(s.image), 'label': str(s.label), 'name': im.name, 'origin': s.origin}
                )
    post = {'splits': {}}
    total = Counter()
    for split in SPLITS:
        verified, errors, only_images, only_labels = audit_pair_directories(
            output/'images'/split, output/'labels'/split, split
        )
        if errors or only_images or only_labels:
            raise ValueError(f'Post-audit failed: {errors}, {only_images}, {only_labels}')
        total.update(class_counts(verified))
        post['splits'][split] = {'images':len(verified), 'instances':dict(class_counts(verified)),
                                'backgrounds':sum(not s.classes for s in verified)}
    assert len(assigned) == len(samples)
    assert total == class_counts(samples)
    assert all(assigned[p['a']] == assigned[p['b']] for p in report['similar_pairs'])
    assert all(digest(Path(p)) == h for p,h in report['source_sha256'].items())
    post.update(source_files_unchanged=True, copies_verified=True, similarity_cross_split_pairs=0,
                total_images=len(samples), total_instances=dict(total))
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'audit.json').write_text(json.dumps(post,ensure_ascii=False,indent=2),encoding='utf-8')
    config = {
        'path': output.as_posix(),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 9,
        'names': report['names'],
    }
    config_path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    print(json.dumps(post, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
