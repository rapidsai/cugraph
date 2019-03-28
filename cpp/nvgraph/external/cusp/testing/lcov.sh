#!/bin/sh

# Generate filtered LCOV docs
lcov --capture --directory . --output-file coverage.info &&\
lcov --extract coverage.info "*/cusp/*" -o filtered_coverage.info &&\
coveralls-lcov --repo-token $REPO_TOKEN filtered_coverage.info
# genhtml filtered_coverage.info --output-directory coverage_html
