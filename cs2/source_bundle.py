"""Small source inventory for handoff and readable return diffs (no ML imports)."""
from pathlib import Path


def source_files(root):
    root=Path(root)
    excluded={'config_optimized.json','config_module_controls.json','config_expanded.json','config_reference_baselines.json',
              'PACKAGE_MANIFEST.json','PACKAGE_BUILD.json','PACKAGE_VERIFICATION.json'}
    def is_source(path):
        if not path.is_file() or path.name in excluded:return False
        if path.suffix=='.txt':
            return path.name.lower().startswith('requirements') or path.name.upper() in {'LICENSE.TXT','NOTICE.TXT','README.TXT','THIRD_PARTY.TXT'}
        return path.suffix in {'.py','.json','.md','.ps1'}
    result=[p for p in root.iterdir() if is_source(p)]
    for folder in ('cs2','tools','figures','tests'):
        result.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix=='.py' and '__pycache__' not in p.parts)
    return sorted(result)
