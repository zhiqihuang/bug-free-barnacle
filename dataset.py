from collections import defaultdict
import os, re, json
from typing import List, Tuple
import pickle
import random
from torch.utils.data import Dataset, DataLoader
import mmap
from tqdm import tqdm
from datasets import load_dataset
HF_TOKEN = "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"

# reads the number of lines in a file
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def read_queries(path):
    queries = {}
    file_type = path.split('.')[-1]
    assert file_type in ['tsv', 'jsonl'], "unsupported query file formt"

    with open(path, 'r') as f:
        for line in tqdm(f, total=get_num_lines(path), desc='read queries'):
            if file_type == 'tsv':
                data = line.rstrip('\n').split('\t')
                assert len(data) == 2
                qid, qtxt = data
                queries[qid] = qtxt
            else:
                data = json.loads(line.rstrip('\n'))
                qid, qtxt = data["id"], data["question"]
                queries[qid] = qtxt
    return queries

def read_qidpidtriples(file_path):
    qidpidtriples = []
    with open(file_path, 'r') as file:
        for line in tqdm(file, total=get_num_lines(file_path), desc='loading qidpidtriples'):
            line = line.strip()
            qid, pos_pid, neg_pid = line.split('\t')
            qidpidtriples.append((int(qid), int(pos_pid), int(neg_pid)))
    return qidpidtriples

def load_hf_mmarco(subset):
    data_dict = {}
    dataset = load_dataset("unicamp-dl/mmarco", subset)
    for key in dataset.keys():
        for data in tqdm(dataset[key], desc=f"{subset}-{key}"):
            data_dict[data['id']] = data['text']
    return data_dict

def load_pickle(path):
    data_dict = {}
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

class TriplesDataset(Dataset):
    def __init__(self, qidpidtriples, query_dataset, doc_dataset):
        self.query = query_dataset
        self.doc = doc_dataset
        self.qidpidtriples = qidpidtriples
        assert list(self.doc.keys()) == list(self.query.keys())
        self.langs = sorted(list(self.doc.keys()))
        
    def __len__(self):
        return len(self.qidpidtriples)

    def __getitem__(self, idx):
        qtxt, pos_dtxt, neg_dtxt = [], [], []
        qid, pos_pid, neg_pid = self.qidpidtriples[idx]
        for lang in self.langs:
            qtxt.append(self.query[lang][qid])
            pos_dtxt.append(self.doc[lang][pos_pid])
            neg_dtxt.append(self.doc[lang][neg_pid])
        return qtxt, pos_dtxt, neg_dtxt
    
class TevatronoTriplesDataset(Dataset):
    def __init__(self, qidpidtriples, query_dataset, doc_dataset, train_n_passages=2):
        self.query = query_dataset
        self.doc = doc_dataset
        self.qidpidtriples = qidpidtriples
        self.train_n_passages = train_n_passages
        assert list(self.doc.keys()) == list(self.query.keys())
        self.langs = sorted(list(self.doc.keys()))
        
    def __len__(self):
        return len(self.qidpidtriples)

    def __getitem__(self, idx):
        qtxt, pos_dtxt, neg_dtxt = [], [], []
        qid, pos_pids, neg_pids = self.qidpidtriples[idx]
        pos_pid = random.choice(pos_pids)
        neg_pid = random.sample(neg_pids, self.train_n_passages-1)
        for lang in self.langs:
            qtxt.append(self.query[lang][qid])
            pos_dtxt.append(self.doc[lang][pos_pid])
        for pid in neg_pid:
            neg_dtxt += [self.doc[lang][pid] for lang in self.langs]
        return qtxt, pos_dtxt, neg_dtxt
    
class NeuCLIRTevatronoTriplesDataset(Dataset):
    def __init__(self, qidpidtriples, query_dataset, doc_dataset, train_n_passages=2):
        self.query = query_dataset
        self.doc = doc_dataset
        self.qidpidtriples = qidpidtriples
        self.train_n_passages = train_n_passages
        self.langs = sorted(list(self.doc.keys()))
        
    def __len__(self):
        return len(self.qidpidtriples)

    def __getitem__(self, idx):
        qtxt, pos_dtxt, neg_dtxt = [], [], []
        qid, pos_pids, neg_pids = self.qidpidtriples[idx]
        pos_pid = random.choice(pos_pids)
        neg_pid = random.sample(neg_pids, self.train_n_passages-1)
        for lang in self.langs:
            qtxt.append(self.query["english"][qid])
            pos_dtxt.append(self.doc[lang][pos_pid])
        for pid in neg_pid:
            neg_dtxt += [self.doc[lang][pid] for lang in self.langs]
        return qtxt, pos_dtxt, neg_dtxt

def get_train_multi_parallel_dataset(args):
    # load query and document datasets
    qidpidtriples = read_qidpidtriples(args.qidpidtriples)
    query_dataset, doc_dataset = {}, {}
    for lang in tqdm(args.langs, desc='loading data by language'):
        doc_dataset[lang] = load_pickle(args.data_dir + f"/collection-{lang}.pickle")
        query_dataset[lang] = load_pickle(args.data_dir + f"/queries-{lang}.pickle")
    qidpidtriples = qidpidtriples[:args.num_train]
    # create datasets
    train_dataset = TriplesDataset(qidpidtriples, query_dataset, doc_dataset)
    return train_dataset

def get_train_neuclir_tevatrono(args):
    # load query and document datasets
    tevatron_msmarco = load_dataset("Tevatron/msmarco-passage")
    qidpidtriples = []
    invalid = 0
    for data in tqdm(tevatron_msmarco['train'], desc='loading qidpidtriples'):
        if len([int(psg['docid']) for psg in data['negative_passages']]) >= args.train_n_passages-1:
            qidpidtriples.append([int(data['query_id']), [int(psg['docid']) for psg in data['positive_passages']], [int(psg['docid']) for psg in data['negative_passages']]])
        else:
            invalid += 1
    print(f"number of invalid: {invalid}, due to the number of negative passages.")
    query_dataset, doc_dataset = {}, {}
    for lang in tqdm(args.langs, desc='loading data by language'):
        doc_dataset[lang] = load_pickle(args.data_dir + f"/collection-{lang}.pickle")
    query_dataset["english"] = load_pickle(args.data_dir + f"/queries-english.pickle")
    # create datasets
    train_dataset = NeuCLIRTevatronoTriplesDataset(qidpidtriples, query_dataset, doc_dataset, args.train_n_passages)
    return train_dataset

def get_train_multi_parallel_tevatrono(args):
    # load query and document datasets
    tevatron_msmarco = load_dataset("Tevatron/msmarco-passage")
    qidpidtriples = []
    invalid = 0
    for data in tqdm(tevatron_msmarco['train'], desc='loading qidpidtriples'):
        if len([int(psg['docid']) for psg in data['negative_passages']]) >= args.train_n_passages-1:
            qidpidtriples.append([int(data['query_id']), [int(psg['docid']) for psg in data['positive_passages']], [int(psg['docid']) for psg in data['negative_passages']]])
        else:
            invalid += 1
    print(f"number of invalid: {invalid}, due to the number of negative passages.")
    query_dataset, doc_dataset = {}, {}
    for lang in tqdm(args.langs, desc='loading data by language'):
        doc_dataset[lang] = load_pickle(args.data_dir + f"/collection-{lang}.pickle")
        query_dataset[lang] = load_pickle(args.data_dir + f"/queries-{lang}.pickle")
    # create datasets
    train_dataset = TevatronoTriplesDataset(qidpidtriples, query_dataset, doc_dataset, args.train_n_passages)
    return train_dataset

class QueryDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        content = self.dataset[idx]
        assert len(content) == 2
        qid, qtxt = content
        return [qid, qtxt]

def read_multi_parallel(path, languages, repeat, num_train):
    multi_parallel = defaultdict(list)
    train_multi_parallel = []
    if 'english' in languages:
        lang_label = {lang: i for i, lang in enumerate(languages)}
    else:
        lang_label = {lang: i+1 for i, lang in enumerate(languages)}
        lang_label["english"] = 0
    with open(path, 'r') as f:
        for line in tqdm(f, total=get_num_lines(path), desc='read multi parallel'):
            data = line.rstrip('\n').split('\t')
            if len(data) == 3:
                en, fn, lang = data
                if lang in languages:
                    multi_parallel[lang].append([en, lang_label["english"], fn, lang_label[lang]])
            else:
                en, fn, lang, _ = data
                if lang in languages:
                    multi_parallel[lang].append([en, lang_label["english"], fn, lang_label[lang]])

    num_langs = len(multi_parallel)
    train_per_lang = num_train // num_langs

    for lang in multi_parallel:
        if len(multi_parallel[lang]) < train_per_lang:
            diff = train_per_lang - len(multi_parallel[lang])
            multi_parallel[lang] += multi_parallel[lang][:diff]
        random.shuffle(multi_parallel[lang])

    for i in range(0, train_per_lang, repeat):
        train_multi_parallel += sum([multi_parallel[lang][i:i+repeat] for lang in multi_parallel], [])
    return train_multi_parallel, multi_parallel

def read_multi_collections(path, languages):
    multilingual = defaultdict(list)
    with open(path, 'r') as f:
        for line in tqdm(f, total=get_num_lines(path), desc='read multi parallel'):
            data = line.rstrip('\n').split('\t')
            if len(data) == 3:
                en, fn, lang = data
                if lang in languages:
                    multilingual["english"].append(en)
                    multilingual[lang].append(fn)
            else:
                en, fn, lang, _ = data
                if lang in languages:
                    multilingual["english"].append(en)
                    multilingual[lang].append(fn)
    
    for lang in multilingual:
        random.shuffle(multilingual[lang])
    
    multi_pairs = defaultdict(list)
    prev = 0
    for lang in languages:
        if lang != "english":
            size = len(multilingual[lang])
            en_sample = multilingual["english"][prev:prev+size]
            multi_pairs[lang] = [[en, "english", fn, lang] for en, fn in zip(en_sample, multilingual[lang])]
            prev += size
    return multi_pairs

class MultilingualParallelDataset(Dataset):
    def __init__(self, dataset, binary=True):
        self.dataset = dataset
        self.binary = binary

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        en, en_label, fn, fn_label = self.dataset[idx]
        if self.binary:
            return [(en, fn), (0, 1)]
        return [(en, fn), (en_label, fn_label)]

def get_multi_parallel_dataset(args):
    languages = sorted(args.langs)
    train_multi_parallel, _  = read_multi_parallel(args.parallel_collection, languages, args.nproc_per_node, args.num_train)
    parallel_train = MultilingualParallelDataset(train_multi_parallel)
    return parallel_train

def format_query(query: str, prefix: str = '') -> str:
    return f'{prefix} {query.strip()}'.strip()

def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'{prefix} {title.strip()} {text.strip()}'.strip()

class TevatronTrainDataset(Dataset):
    def __init__(self, train_n_passages=2, query_prefix='', passage_prefix=''):
        self.train_data = load_dataset("Tevatron/msmarco-passage", split='train')
        self.train_n_passages = train_n_passages
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = [format_query(query, self.query_prefix)]

        pos_psg = random.choice(group_positives)
        
        formated_pos_passages = [format_passage(pos_psg['text'], pos_psg['title'], self.passage_prefix)]

        negative_size = self.train_n_passages-1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        else:
            _offset = random.randint(0, negative_size)
            negs = [x for x in group_negatives]
            random.shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        formated_neg_passages = []
        for neg_psg in negs:
            formated_neg_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.passage_prefix))

        return formated_query, formated_pos_passages, formated_neg_passages

def get_train_tevatron(args):
    if args.add_prefix:
        train_dataset = TevatronTrainDataset(args.train_n_passages, query_prefix='query:', passage_prefix='passage:')
    else:
        train_dataset = TevatronTrainDataset(args.train_n_passages)
    # # load query and document datasets
    # tevatron_msmarco = load_dataset("Tevatron/msmarco-passage")
    # qidpidtriples = []
    # invalid = 0
    # for data in tqdm(tevatron_msmarco['train'], desc='loading qidpidtriples'):
    #     if len([int(psg['docid']) for psg in data['negative_passages']]) >= args.train_n_passages-1:
    #         qidpidtriples.append([int(data['query_id']), [int(psg['docid']) for psg in data['positive_passages']], [int(psg['docid']) for psg in data['negative_passages']]])
    #     else:
    #         invalid += 1
    # print(f"number of invalid: {invalid}, due to the number of negative passages.")
    # query_dataset, doc_dataset = {}, {}
    # lang = "english"
    # doc_dataset[lang] = load_pickle(args.data_dir + f"/collection-{lang}.pickle")
    # query_dataset[lang] = load_pickle(args.data_dir + f"/queries-{lang}.pickle")
    # # create datasets
    # train_dataset = TevatronoTriplesDataset(qidpidtriples, query_dataset, doc_dataset, args.train_n_passages)
    return train_dataset
