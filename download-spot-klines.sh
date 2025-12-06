# Bash script who permit to download the perpetuals futures klines simultaneously.
# That's mean that the script create few sub-processes for download the data asynchronously

YEARS=("2017" "2018" "2019" "2020" "2021" "2022" "2023" "2024" "2025")
MONTHS=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")

# Use 1-minute klines (much smaller than 1-second)
BASE_URL="https://data.binance.vision/data/spot/monthly/klines/BTCUSDC/1m/BTCUSDC-1m-"

# Directory containing zip files
SOURCE_DIR="downloads/klines"

mkdir -p $SOURCE_DIR

# Function who download the URL, this function is called asynchronously by several child processes
download_url() {
  url=$1

  response=$(wget --server-response -q --directory-prefix=$SOURCE_DIR ${url} 2>&1 | awk 'NR==1{print $2}')
  if [ ${response} == '404' ]; then
    echo "File not exist: ${url}"
  else
    echo "downloaded: ${url}"
  fi
}

# Main loop who iterate over all the arrays and launch child processes
for year in ${YEARS[@]}; do
  for month in ${MONTHS[@]}; do
    url="${BASE_URL}${year}-${month}.zip"
    download_url "${url}" &
  done
done
# Wait for all background downloads to finish before unzipping
wait

# Directory to extract files to
DEST_DIR="data/klines"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through all zip files in the source directory
shopt -s nullglob
zip_files=("$SOURCE_DIR"/*.zip)
if [ ${#zip_files[@]} -eq 0 ]; then
  echo "No zip files downloaded; nothing to unzip."
else
  for zip_file in "${zip_files[@]}"; do
      echo "Unzipping $zip_file into $DEST_DIR"
      unzip -o "$zip_file" -d "$DEST_DIR" &
  done
  wait
  echo "All files have been unzipped successfully!"
fi
