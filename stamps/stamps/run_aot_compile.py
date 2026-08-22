import os
import platform
import struct
import shutil
import subprocess
from numba.pycc import CC

FILE_FOLDER = os.path.dirname(__file__)
if platform.platform().startswith('Darwin'):
    OS = 'Mac'
elif platform.platform().startswith('Windows'):
    OS = 'Windows'

ARCHITECTURE = 8 * struct.calcsize("P") # 32 or 64 bit 
if ARCHITECTURE == 64:
    ARCHITECTURE = 'x64'
elif ARCHITECTURE == 32:
    ARCHITECTURE = 'x86'
else:
    raise ValueError('No Architecture Supported.')

ROOT_FOLDER = os.path.join(FILE_FOLDER, OS, ARCHITECTURE)


if __name__ == "__main__":
    package_path = os.path.abspath(os.path.dirname(__file__))
    compile_files_py = []
    for dd, sub_dd, ff in os.walk( package_path ):
        compile_files_py += [(dd,i) for i in ff if i.endswith('__aot.py')]
    for dd, ff in compile_files_py:
        subprocess.call(['python', os.path.join(dd,ff)])
        module_path = dd
        module_name = ff[:-8]

        compiled_files = []
        for f in os.listdir(module_path):
            if f.startswith(module_name) and f.endswith(('.so', '.pyd')):
                compiled_files.append(f)
    
        path_p_to_m = os.path.relpath(module_path, package_path)

        for f in compiled_files:
            shutil.move(
                os.path.join(module_path,f),
                os.path.join(package_path, 'aot', path_p_to_m, f)
                )

    # add __init__.py if needed
    for d, sub_d, fs in os.walk(os.path.join(package_path,'aot')):
        if '__init__.py' not in fs: # need
            finit = open(os.path.join(d,'__init__.py'),'w')
            finit.close()
