import os
import glob

print("Checking scratch directory files:")
for f in sorted(glob.glob('scratch/*.py') + sorted(glob.glob('src/*.py'))):
    print(f, os.path.getmtime(f))
