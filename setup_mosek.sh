#!/bin/bash

REPO_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ -e "${REPO_ROOT}/mosek.lic" ]; then
    echo "Found MOSEK license and set environment variable successfully!"
    export MOSEKLM_LICENSE_FILE=${REPO_ROOT}/mosek.lic
else
    echo "No MOSEK license found in directory ${REPO_ROOT}! Make sure it is named mosek.lic and is located in the repository root."
fi