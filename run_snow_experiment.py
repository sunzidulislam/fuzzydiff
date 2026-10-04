"""Run the confirmed pilot (default), or resume the complete 225-image snow study."""
import argparse
import json
from pathlib import Path

from snow_experiment import make_design, jobs_for, run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Expand to all scenes and seeds (225 images).')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='Print design/job counts; no model loading or writes.')
    mode.add_argument('--artifacts', action='store_true', help='Rebuild grids/reviewer pack from saved outputs on CPU.')
    default = Path('/kaggle/working/outputs') if Path('/kaggle/working').exists() else Path('./outputs')
    parser.add_argument('--output', type=Path, default=default / 'snow_experiment')
    options = parser.parse_args()
    design = make_design()
    jobs = jobs_for(design, options.full)
    if options.plan:
        print(json.dumps({'scope': 'full' if options.full else 'pilot', 'images': len(jobs),
                          'base_generations': sum(j['method'] != 'refined' for j in jobs),
                          'refinements': sum(j['method'] == 'refined' for j in jobs),
                          'output': str(options.output), 'design': design}, indent=2))
        return 0
    if not options.artifacts:
        from snow_gpu import SnowBackend
        manifest = run_study(options.output, design, SnowBackend(design), options.full)
    else:
        manifest = json.loads((options.output / 'manifest.json').read_text(encoding='utf-8'))
        # Artifact regeneration uses the saved design and scope, even after code changes.
        jobs = jobs_for(manifest['design'], manifest.get('scope') == 'full' or options.full)
    from snow_artifacts import export_artifacts
    status = export_artifacts(options.output, full=options.full or manifest.get('scope') == 'full')
    complete = status['complete']
    print(f'{complete}/{len(jobs)} images complete. Outputs: {options.output.resolve()}', flush=True)
    if complete != len(jobs):
        print('Failures/missing outputs remain visible in the manifest and grids. Rerun to retry.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
