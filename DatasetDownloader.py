#!/bin/python
from bing_image_downloader import downloader

query = input("Enter Search Term: ")
amount = int(input("Amount to download: "))

folderName = query.replace(' ', '_');

downloader.download( 
  query, #Query String
  limit=100,
  output_dir=folderName,
  adult_filter_off=False,
  force_replace=False,
  timeout=60,
  verbose=True
)

print("DONE")