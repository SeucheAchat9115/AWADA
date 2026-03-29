#!/bin/bash
# Shared helper for reading values from a benchmark YAML config file.
#
# Usage:
#   source scripts/lib/config_helper.sh
#   CONFIG_FILE="configs/benchmarks/<benchmark>.yaml"
#   SOURCE_DATASET=$(_cfg source_dataset)
#
# _cfg <key> reads the value of <key> from $CONFIG_FILE.
# Lists are printed as space-separated words; null/missing keys print empty string.

_cfg() {
    python3 -c "
import yaml, sys
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
val = cfg.get('$1')
if val is None:
    print('')
elif isinstance(val, list):
    print(' '.join(str(x) for x in val))
else:
    print(val)
"
}
