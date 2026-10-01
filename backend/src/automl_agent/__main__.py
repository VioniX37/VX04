"""Allow ``python -m automl_agent ...`` as an alias for the ``automl-agent`` CLI."""

import sys

from automl_agent.cli import main

sys.exit(main())
