"""Render 11 coverage x 9 relative snow-thickness settings on one fixed scene."""
import argparse
import json
from pathlib import Path

from snow_control import control_design, run_control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Optional clean mountain image; otherwise generate one with FuzzyDiff.')
    parser.add_argument('--mask', type=Path, help='Optional same-size terrain mask (white = mountain). Otherwise use CPU CLIPSeg.')
    parser.add_argument('--seed', type=int, default=142)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='Print controls and cost; no model loading or writes.')
    mode.add_argument('--artifacts', action='store_true', help='Rebuild the figure from saved images, on CPU.')
    default = Path('/kaggle/working/outputs') if Path('/kaggle/working').exists() else Path('./outputs')
    parser.add_argument('--output', type=Path, default=default / 'snow_control')
    args = parser.parse_args()
    manifest_path = args.output / 'control_manifest.json'
    if args.artifacts:
        if args.source or args.mask:
            parser.error('--artifacts uses saved source/mask; omit --source and --mask.')
        from snow_control_figure import export_control_grid
        status = export_control_grid(args.output)
        return 0 if status['complete'] == status['expected'] else 1
    design = control_design(args.seed, args.source, args.mask)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))['design']
        # Cached inputs can resume without retaining their external original paths.
        for name, argument in (('input_source_sha256', args.source), ('input_mask_sha256', args.mask)):
            if argument is None:
                design[name] = previous[name]
    if args.plan:
        print(json.dumps({'rendered_images': 99, 'base_generations': 0 if design['input_source_sha256'] else 1,
                          'segmentation': 'provided mask' if design['input_mask_sha256'] else 'CLIPSeg on CPU',
                          'output': str(args.output), 'design': design}, indent=2))
        return 0
    print('Fixed-scene control: one reference, 99 CPU renderings. D is an appearance proxy, not snow depth.', flush=True)
    run_control(args.output, design, source=args.source, mask=args.mask)
    from snow_control_figure import export_control_grid
    status = export_control_grid(args.output)
    print('Inspect source.png and mask_overlay.png. Outputs:', args.output.resolve())
    return 0 if status['complete'] == status['expected'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
