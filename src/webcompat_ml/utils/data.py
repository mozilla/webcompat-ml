import pandas

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from tqdm import tqdm


class DatasetBuilder(object):
    """Webcompat-ml dataset builder"""

    def __init__(self, es_url, index, doc_type):
        self.es_url = es_url
        self.index = index
        self.doc_type = doc_type
        self.scroll_time = "60m"

    def get_es(self):
        """Get ES client"""
        es = Elasticsearch(self.es_url)
        return es

    def q_closed(self):
        """Create filter for all closed webcompat issues"""
        q = {"query": {"match": {"state": "closed"}}}
        return q

    def scan(self, query):
        """Scan ES for query"""
        es = self.get_es()
        count = es.count(body=query, index=self.index, doc_type=self.doc_type)["count"]
        res = scan(es, query=query, index=self.index, doc_type=self.doc_type)
        return (res, count)

    def get_dataset(self, query):
        """Retrieve all data from ES"""
        results, count = self.scan(query)
        dataset = []

        for item in tqdm(results, total=count):
            dataset.append(item["_source"])
        return dataset

    def fetch(self):
        """Build webcompat-ml dataset"""
        query = self.q_closed()
        dataset = self.get_dataset(query)
        df = pandas.DataFrame(dataset)
        self.df = df

    def extract_needsdiagnosis(self, events):
        """Extract needsdiagnosis"""
        flag = False

        if events:
            for event in events:
                if (
                    event["event"] == "milestoned"
                    and event["milestone"]["title"] == "needsdiagnosis"
                ):
                    flag = True
        return flag

    def extract_title(self, row):
        """Extract renamed title"""
        for event in row["events"]:
            if event["event"] == "renamed":
                row["title"] = event["rename"]["from"]
        return row
