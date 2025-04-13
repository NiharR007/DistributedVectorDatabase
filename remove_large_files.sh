#!/bin/bash

# Create a fresh clone of the repository
cd ..
mkdir temp_repo
cd temp_repo
git clone --no-hardlinks file:///Users/niharpatel/Desktop/Boston_University_Courses/AdvanceDBMS/DistributedVectorDatabase repo
cd repo

# Remove the large files from history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch data/coco/val2017.zip data/coco_embeddings.json venv/lib/python3.9/site-packages/torch/lib/libtorch_cpu.dylib' \
  --prune-empty --tag-name-filter cat -- --all

# Force garbage collection and remove old references
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now

# Push to GitHub with force
echo "Now you can push to GitHub with:"
echo "git remote add origin YOUR_GITHUB_REPO_URL"
echo "git push -u --force origin main"
