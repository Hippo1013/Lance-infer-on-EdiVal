"""ImgEdit entry point using the accepted DP2 + prefix profile by default."""
import sys
from .runner import main as run


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    return run(['--profile', 'dp2', '--gpus', '0,1', '--cache-mode', 'prefix',
                *args, '--benchmark', 'imgedit'])


if __name__ == '__main__':
    raise SystemExit(main())
