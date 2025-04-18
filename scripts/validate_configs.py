#!/usr/bin/env python3

import os
import sys
import logging

# Add the project root to PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config_validator import validate_all_configs

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if validate_all_configs():
        logging.info("All configuration files are valid")
    else:
        logging.error("Some configuration files are invalid")
        exit(1) 