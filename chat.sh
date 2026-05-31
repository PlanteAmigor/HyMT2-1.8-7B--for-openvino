#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"
if [ ! -d "$VENV" ]; then
    if [[ "$LANG" =~ zh_CN|zh-|zh_ ]]; then
        echo "虚拟环境不存在，请先运行: bash \"$DIR/setup.sh\""
    else
        echo "Virtual environment not found. Run first: bash \"$DIR/setup.sh\""
    fi
    exit 1
fi
exec "$VENV/bin/python" "$DIR/chat.py" "$@"
