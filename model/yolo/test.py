import re
import sys
from ultralytics.cfg import entrypoint
if __name__ == 'main':
    sys.argv[0] = re.sub(r'(-script.pyw|.exe)?$', '', sys.argv[0])

    sys.exit(entrypoint())