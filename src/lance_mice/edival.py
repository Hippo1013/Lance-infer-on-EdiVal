"""EdiVal full-history entry point, using 512 output pixels and prefix caching."""
import sys
from .runner import main as run


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    return run(['--profile', 'dp2', '--gpus', '0,1', '--cache-mode', 'prefix', '--resolution', '512',
                *args, '--benchmark', 'edival'])


if __name__ == '__main__':
    raise SystemExit(main())
