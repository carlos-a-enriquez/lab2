#!/usr/bin/env python
"""
The purpose of this script is to extract a tsv file
containing the relevant annotations (transmembrane and transit)
from https://www.uniprot.org/id-mapping
by programmatically acccessing the REST API. 

The code herein is based on the suggestions published in:
https://www.uniprot.org/help/id_mapping

Note: The get_id_mapping_results_link() function was excluded
to ensure that the tsv file is always downloaded with the specific
file configuration that we need for the analysis (
in terms of the columns that we need (transmembrane and transit), etc.)
Therefore, the url template is found hardcoded into the tsv_extractor() function. 

In addition, the search API was chosen in place of the stream API of Uniprot 
ID mapping. 
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
	
	
def check_id_mapping_results_ready(job_id):
	while True:
		request = session.get(f"{API_URL}/idmapping/status/{job_id}")
		check_response(request)
		j = request.json()
		if "jobStatus" in j:
			if j["jobStatus"] == "RUNNING":
				print(f"Retrying in {POLLING_INTERVAL}s")
				time.sleep(POLLING_INTERVAL)
			else:
				raise Exception(j["jobStatus"])
		else:
			return bool(j["results"] or j["failedIds"])
            

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
	
def print_progress_batches(batch_index, size, total):
	n_fetched = min((batch_index + 1) * size, total)
	print(f"Fetched: {n_fetched} / {total}", file=sys.stderr)
	
	
def get_next_link(headers):
	re_next_link = re.compile(r'<(.+)>; rel="next"')
	if "Link" in headers:
		match = re_next_link.match(headers["Link"])
		if match:
			return match.group(1)
			
			
def get_xml_namespace(element):
    m = re.match(r"\{(.*)\}", element.tag)
    return m.groups()[0] if m else ""

			
def merge_xml_results(xml_results):
	merged_root = ElementTree.fromstring(xml_results[0])
	for result in xml_results[1:]:
		root = ElementTree.fromstring(result)
		for child in root.findall("{http://uniprot.org/uniprot}entry"):
			merged_root.insert(-1, child)
	ElementTree.register_namespace("", get_xml_namespace(merged_root[0]))
	return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)



def combine_batches(all_results, batch_results, file_format):
    if file_format == "json":
        for key in ("results", "failedIds"):
            if key in batch_results and batch_results[key]:
                all_results[key] += batch_results[key]
    elif file_format == "tsv":
        return all_results + batch_results[1:]
    else:
        return all_results + batch_results
    return all_results

	
def get_batch(batch_response, file_format, compressed):
	batch_url = get_next_link(batch_response.headers)
	while batch_url:
		batch_response = session.get(batch_url)
		batch_response.raise_for_status()
		yield decode_results(batch_response, file_format, compressed)
		batch_url = get_next_link(batch_response.headers)



	
def get_id_mapping_results_search(url):
	parsed = urlparse(url)
	query = parse_qs(parsed.query)
	file_format = query["format"][0] if "format" in query else "json"
	if "size" in query:
		size = int(query["size"][0])
	else:
		size = 500
		query["size"] = size
	compressed = (
		query["compressed"][0].lower() == "true" if "compressed" in query else False
	)
	parsed = parsed._replace(query=urlencode(query, doseq=True))
	url = parsed.geturl()
	request = session.get(url)
	check_response(request)
	results = decode_results(request, file_format, compressed)
	total = int(request.headers["x-total-results"])
	print_progress_batches(0, size, total)
	for i, batch in enumerate(get_batch(request, file_format, compressed), 1):
		results = combine_batches(results, batch, file_format)
		print_progress_batches(i, size, total)
	if file_format == "xml":
		return merge_xml_results(results)
	return results

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
	job_id = response.json()['jobId']
	
	#url = 'https://rest.uniprot.org/idmapping/uniprotkb/results/stream/'+ job_code \
	#+'?compressed=true&fields=accession%2Creviewed%2Cid%2Cft_transmem%2Cft_transit&format=tsv'
	url = 'https://rest.uniprot.org/idmapping/uniprotkb/results/'+job_id \
	+'?compressed=true&fields=accession%2Creviewed%2Cid%2Cft_transmem%2Cft_transit&format=tsv&size=500'
	
	#Get TSV
	if check_id_mapping_results_ready(job_id):
		#link = get_id_mapping_results_link(job_id)
		tsv_results = get_id_mapping_results_search(url)
    # Equivalently using the stream endpoint which is more demanding
    # on the API and so is less stable:
    # results = get_id_mapping_results_stream(link)
		df = get_data_frame_from_tsv_results(tsv_results)
	return df
	


#REST Uniprot variables
POLLING_INTERVAL = 3
API_URL = "https://rest.uniprot.org"
retries = Retry(total=10, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
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
	
	

