#!/usr/bin/env python3
"""check_stamps.py — verify a STAMPS course environment.

Run from anywhere after `pip install -e .`:

    python check_stamps.py

Prints one line per package and ends with READY when everything needed for
the course is present.  Anything else: post the whole output.
"""
import importlib
import os
import sys

REQUIRED_BY_STAMPS = [('numpy', 'numpy'), ('scipy', 'scipy'), ('pandas', 'pandas')]
REQUIRED_BY_COURSE = [('matplotlib', 'matplotlib'), ('nlopt', 'nlopt')]
OPTIONAL_LATER = [('sklearn', 'scikit-learn'), ('xgboost', 'xgboost'), ('jupyterlab', 'jupyterlab')]


def version_of(modname):
    mod = importlib.import_module(modname)
    return getattr(mod, '__version__', None) or 'installed'


def check(items, hard):
    ok = True
    for modname, pipname in items:
        try:
            print(f'  ok      {modname:<14} {version_of(modname)}')
        except Exception as e:  # noqa: BLE001
            ok = ok and not hard
            tag = 'MISSING' if hard else 'missing'
            print(f'  {tag:<7} {modname:<14} pip install {pipname}    ({type(e).__name__})')
    return ok


def main():
    print(f'python {sys.version.split()[0]}  ({sys.executable})')
    print()
    print('required by stamps and this course:')
    ok = check(REQUIRED_BY_STAMPS, hard=True)
    ok = check(REQUIRED_BY_COURSE, hard=True) and ok
    print()
    print('optional (needed in later weeks):')
    check(OPTIONAL_LATER, hard=False)
    print()
    print('stamps itself:')
    try:
        import stamps  # noqa: WPS433
        from stamps.bme.softconverter import probaUniform  # noqa: F401
        from stamps.bme.BMEprobaEstimations import BMEPosteriorMoments  # noqa: F401
        from stamps.estimation.kriging import kriging  # noqa: F401
        where = os.path.dirname(os.path.abspath(stamps.__file__))
        home = os.path.expanduser('~')
        if where.startswith(home):
            where = '~' + where[len(home):]
        ver = getattr(stamps, '__version__', None)
        if ver is None:
            try:
                from importlib.metadata import version
                ver = version('stamps')
            except Exception:  # noqa: BLE001
                ver = '?'
        print(f'  ok      stamps {ver:<8} imported from {where}')
        if 'site-packages' in where and 'stamps' not in os.path.basename(os.path.dirname(where)):
            print('  WARNING stamps is not running from your clone — did you `pip install stamps` from PyPI?')
            print('          Fix: pip uninstall -y stamps, then pip install -e . inside the clone.')
            ok = False
    except Exception as e:  # noqa: BLE001
        print(f'  MISSING stamps: {type(e).__name__}: {e}')
        if 'stamps.stamps' in str(e):
            print('          You are using an old notebook or path: write `from stamps.<module> import ...`')
        ok = False
    print()
    if ok:
        print('READY \u2014 you can start L1.')
        return 0
    print('NOT READY \u2014 fix the MISSING lines above, then run this again.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
