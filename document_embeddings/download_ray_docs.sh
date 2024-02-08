#!/usr/bin/env bash

wget -e robots=off --recursive --no-clobber --page-requisites --html-extension --convert-links --restrict-file-names=windows --domains docs.ray.io --no-parent https://docs.ray.io/en/master/
