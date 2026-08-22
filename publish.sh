#!/bin/bash
DEV=~/Documents/MyDoc_iCloud/packages/stamps/stamps_v3
REL=~/Documents/MyDoc_iCloud/packages/stamps/stamps_repo

rsync -av --delete \
  --exclude='__pycache__' --exclude='.DS_Store' --exclude='*.egg-info' \
  "$DEV/stamps/" "$REL/stamps/"

rsync -av --delete \
  --exclude='.ipynb_checkpoints' --exclude='.DS_Store' \
  "$DEV/tutorials/" "$REL/tutorials/"

echo ""
echo "同步完成。到 GitHub Desktop (stamps_repo) 檢視變更，commit 後 Push origin。"
