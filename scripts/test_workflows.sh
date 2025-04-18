#!/bin/bash

# Check if act is installed
if ! command -v act &> /dev/null; then
    echo "act is not installed. Installing..."
    # For Windows (using scoop)
    if [[ "$OSTYPE" == "msys" ]]; then
        scoop install act
    # For macOS
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install act
    # For Linux
    else
        curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
    fi
fi

# Run CI workflow
echo "Testing CI workflow..."
act -W .github/workflows/ci.yml

# Run CD workflow
echo "Testing CD workflow..."
act -W .github/workflows/cd.yml 