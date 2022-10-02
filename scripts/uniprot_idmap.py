#!/usr/bin/env python
"""
The purpose of this script is to extract a tsv file
containing the relevant annotations (transmembrane and transit)
from https://www.uniprot.org/id-mapping
by programmatically acccessing the REST API. 

The code herein is based on the suggestions published in:
https://www.uniprot.org/help/id_mapping
"""

import sys
import re
import time
import json
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import csv


def parse_accession_list(list_fh):
	"""
	This function is meant to access the "false_negative.txt, etc."
	files created by the module misclassified. 
	
	list_fh = Filehandler for Uniprot accession number list file 
	"""
	with open(list_fh, 'r') as fh:
		accessions = []
		for line in fh:
			accessions.append(line.strip())
	return accessions
	
	
def check_response(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.json())
        raise

def decode_results(response, file_format, compressed):
	if compressed:
		decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
		if file_format == "json":
			j = json.loads(decompressed.decode("utf-8"))
			return j
		elif file_format == "tsv":
			return [line for line in decompressed.decode("utf-8").split("\n") if line]
		elif file_format == "xlsx":
			return [decompressed]
		elif file_format == "xml":
			return [decompressed.decode("utf-8")]
		else:
			return decompressed.decode("utf-8")
	elif file_format == "json":
		return response.json()
	elif file_format == "tsv":
		return [line for line in response.text.split("\n") if line]
	elif file_format == "xlsx":
		return [response.content]
	elif file_format == "xml":
		return [response.text]
	return response.text

def get_id_mapping_results_stream(url):
	if "/stream/" not in url:
		url = url.replace("/results/", "/results/stream/")
	request = session.get(url)
	check_response(request)
	parsed = urlparse(url)
	query = parse_qs(parsed.query)
	file_format = query["format"][0] if "format" in query else "json"
	compressed = (
		query["compressed"][0].lower() == "true" if "compressed" in query else False
	)
	return decode_results(request, file_format, compressed)

def get_data_frame_from_tsv_results(tsv_results):
	reader = csv.DictReader(tsv_results, delimiter="\t", quotechar='"')
	return pd.DataFrame(list(reader))


	
def tsv_extractor(accessions):
	"""
	This function gets the TSV file needed for downstream analysis.
	It will use the Uniprot ID mapping REST API.
	It will submit the job and retreive the results
	
	accessions= list produced by parse_accession_list()
	"""
	#Submit the job
	files = {'from': (None, 'UniProtKB_AC-ID'),	'to': (None, 'UniProtKB'), 'ids': (None, ",".join(accessions)),}
	response = requests.post('https://rest.uniprot.org/idmapping/run', files=files)
	job_code = response.json()['jobId']
	url = 'https://rest.uniprot.org/idmapping/uniprotkb/results/stream/'+ job_code \
	+'?compressed=true&fields=accession%2Creviewed%2Cid%2Cft_transmem%2Cft_transit&format=tsv'
	
	#Get TSV
	tsv_results = get_id_mapping_results_stream(url)
	df = get_data_frame_from_tsv_results(tsv_results)
	return df
	


#REST Uniprot variables
POLLING_INTERVAL = 3
API_URL = "https://rest.uniprot.org"
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))	
	

if __name__ == '__main__':
	#Opening files
	try:
		accession_list_fh = sys.argv[1]			
	except IndexError:
		accession_list_fh = input("insert the accesion_list text file path   ")		
	
    
	
	#Workflow
	accession_list = parse_accession_list(accession_list_fh) #list of relevant accession numbers
	df = tsv_extractor(accession_list)
	#df.to_csv('debug.csv')
	
	

