#!/bin/bash

HF_TOKEN="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"
BASE_URL="https://hf-mirror.com/datasets/suddhu/Feelsight/resolve/main"

FILES=(
  "feelsight.tar.gz"
  "feelsight_real.tar.gz"
  "feelsight_occlusion.tar.gz"
  "assets.tar.gz"
)

mkdir -p Feelsight
cd Feelsight || exit 1

for FILE in "${FILES[@]}"; do
  echo "🔽 Downloading $FILE..."
  wget --header="Authorization: Bearer $HF_TOKEN" \
       --content-disposition \
       "$BASE_URL/$FILE" || { echo "❌ Failed to download $FILE"; exit 1; }
done

echo "📦 Extracting all .tar.gz files..."
for f in *.tar.gz; do
  if file "$f" | grep -q 'gzip compressed data'; then
    echo "📂 Extracting $f..."
    tar -xzf "$f" && rm "$f"
  else
    echo "⚠️ Skipping $f: not a valid gzip archive"
  fi
done

echo "✅ Done! All Feelsight dataset files are downloaded and extracted."