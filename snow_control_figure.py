"""Reference-style 9 x 11 snow parameter grid, with a source image beneath."""
import json
from pathlib import Path

from PIL import Image

from snow_control import file_hash, _verified_asset
from snow_experiment import atomic_json


def export_control_grid(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    root = Path(root)
    manifest = json.loads((root / 'control_manifest.json').read_text(encoding='utf-8'))
    design = manifest['design']
    source = _verified_asset(root, manifest['source'], 'source')
    _verified_asset(root, manifest['mask'], 'mask')
    rows, columns = len(design['coverage']), len(design['thickness'])
    figure = plt.figure(figsize=(10.8, 16.2))
    left, bottom, width, height = 0.12, 0.255, 0.82, 0.67
    gap = 0.002
    missing = []
    complete = 0
    for row, coverage in enumerate(design['coverage']):
        for column, thickness in enumerate(design['thickness']):
            identifier = f'coverage_{row:02d}_thickness_{column:02d}'
            record = manifest['runs'].get(identifier, {})
            axis = figure.add_axes([left + column * width / columns + gap / 2,
                                    bottom + row * height / rows + gap / 2,
                                    width / columns - gap, height / rows - gap])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            path = root / record.get('image', f'images/{identifier}.png')
            valid = (record.get('status') == 'complete' and path.is_file() and
                     file_hash(path) == record.get('sha256'))
            if valid:
                with Image.open(path) as image:
                    axis.imshow(image.convert('RGB'), aspect='equal')
                complete += 1
            else:
                missing.append(identifier)
                axis.set_facecolor('#eee')
                axis.text(0.5, 0.5, record.get('status', 'missing').upper(), transform=axis.transAxes,
                          fontsize=6, ha='center', va='center', color='#a2372a')
        figure.text(left - 0.018, bottom + (row + 0.5) * height / rows, f'{coverage:.0%}',
                    va='center', ha='right', fontsize=9)
    for column, thickness in enumerate(design['thickness']):
        figure.text(left + (column + 0.5) * width / columns, bottom - 0.015,
                    f'{thickness:.2f}', ha='center', va='top', fontsize=9)
    figure.patches.extend([
        FancyArrowPatch((left - 0.011, bottom - 0.002), (left - 0.011, bottom + height + 0.011),
                        transform=figure.transFigure, arrowstyle='->', mutation_scale=22,
                        linewidth=2.5, color='black'),
        FancyArrowPatch((left - 0.013, bottom - 0.005), (left + width + 0.025, bottom - 0.005),
                        transform=figure.transFigure, arrowstyle='->', mutation_scale=22,
                        linewidth=2.5, color='#79d894')])
    figure.text(0.5, 0.963, 'Fine-Grained Snow Control', fontsize=19, fontweight='bold', ha='center')
    figure.text(0.022, bottom + height / 2, 'Terrain-mask coverage  C', rotation=90,
                fontsize=11, va='center', ha='center')
    figure.text(left + width, bottom - 0.048, 'Relative snow thickness  D', fontsize=11, ha='right')
    # Match the source aspect ratio, instead of stretching non-square user images.
    with Image.open(source) as image:
        thumbnail_height = 0.105
        thumbnail_width = min(0.26, thumbnail_height * 16.2 / 10.8 * image.width / image.height)
        axis = figure.add_axes([0.5 - thumbnail_width / 2, 0.069, thumbnail_width, thumbnail_height])
        axis.imshow(image.convert('RGB'))
        axis.axis('off')
    figure.text(0.5, 0.182, 'Clean reference (C = 0)', fontsize=10, ha='center')
    figure.text(0.5, 0.048, 'One fixed scene and terrain mask; zero-coverage cells repeat the reference.',
                fontsize=8.5, ha='center')
    figure.text(0.5, 0.028, 'Image-space snow rendering. C: selected mask area. D: opacity/texture proxy, not physical depth.',
                fontsize=8.5, ha='center')
    for suffix in ('png', 'pdf'):
        figure.savefig(root / f'snow_control_grid.{suffix}', dpi=300, facecolor='white')
    plt.close(figure)
    status = {'rows': rows, 'columns': columns, 'expected': rows * columns,
              'complete': complete, 'missing': missing, 'source_sha256': manifest['source']['sha256'],
              'mask_sha256': manifest['mask']['sha256'], 'interpretation': design['interpretation']}
    atomic_json(root / 'control_artifact_status.json', status)
    print(f'Exported {columns} x {rows} grid ({complete}/{rows * columns} valid cells).', flush=True)
    return status
