#!/bin/bash

nvcc -o radicals -x cu -I CLI11/include -lnvToolsExt solverRadicals.cu
