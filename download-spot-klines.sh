# Bash script who permit to download the perpetuals futures klines simultaneously.
# That's mean that the script create few sub-processes for download the data asynchronously

YEARS=("2017" "2018" "2019" "2020" "2021" "2022" "2023" "2024")
MONTHS=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")


BASE_URL="https://data.binance.vision/data/spot/monthly/klines/BTCUSDC/1s/BTCUSDC-1s-"

mkdir -p downloads/klines

# Function who download the URL, this function is called asynchronously by several child processes
download_url() {
  url=$1

  response=$(wget --server-response -q --directory-prefix='downloads/klines' ${url} 2>&1 | awk 'NR==1{print $2}')
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
