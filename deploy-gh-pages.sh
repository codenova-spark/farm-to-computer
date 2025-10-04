#!/bin/bash
# Script to push the gh-pages branch
# Run this script after merging the PR to deploy GitHub Pages

echo "Pushing gh-pages branch to remote..."
git push origin gh-pages

echo ""
echo "GitHub Pages deployment initiated!"
echo "Once pushed, configure GitHub Pages in repository settings:"
echo "  Settings → Pages → Source: gh-pages branch"
echo ""
echo "Your site will be available at:"
echo "  https://codenova-spark.github.io/farm-to-computer/"
