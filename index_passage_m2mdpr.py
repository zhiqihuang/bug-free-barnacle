import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import datasets
import logging
import os
import faiss
from arguments import get_index_parser
from tqdm import tqdm
import pickle
from models.dpr import mDPRBase
from models.leace import mDPRScrubber
from models.dpr_adapter import mDPRAdapter, mDPRZC3
from concept_erasure import LeaceFitter
from util.dataset import get_num_lines
from datasets import load_dataset
from util.util import set_seed

from datasets import disable_caching
disable_caching()

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()

def read_collection(args):
    # docs = {"pid":[], "text":[]}
    # with open(args.collection, "r") as f:
    #     for line in tqdm(f, total=get_num_lines(args.collection), desc="read docs"):
    #         data = line.rstrip("\n").split("\t")
    #         assert len(data) >= 2, data
    #         docid, doctxt = data[:2]
    #         docs["pid"].append(docid)
    #         docs["text"].append(doctxt)
    # ds = datasets.Dataset.from_dict(docs)
    ds = load_dataset("csv", delimiter="\t", header=None, names=['pid', 'text'], usecols=[0, 1], data_files=args.collection)
    return ds['train']

def save_vectors(ds, args, buffer_size=500000):
    n = len(ds)
    for i in range(0, n, buffer_size):
        docids = ds[i:i+buffer_size]["pid"]
        vectors = ds[i:i+buffer_size][args.index_name]
        with open(os.path.join(args.output_dir, f"{args.index_name}-{i//buffer_size}.id"), "wb") as f:
            pickle.dump(docids, f)
        with open(os.path.join(args.output_dir, f"{args.index_name}-{i//buffer_size}.vec"), "wb") as f:
            pickle.dump(vectors, f)

def indexing(model, tokenizer, args, ds, fitter=None):
    model.eval()
    def encode(examples):
        toks = tokenizer(examples["text"],
            padding="longest", 
            return_tensors="pt", 
            max_length=args.doc_maxlen,
            truncation=True)
        d_ids = toks["input_ids"].to(args.device)
        d_mask = toks["attention_mask"].to(args.device)
        if fitter is not None:
            npys = model.doc(d_ids, d_mask, fitter=fitter).cpu().numpy()
        else:
            npys = model.doc(d_ids, d_mask).cpu().numpy()
        return {args.index_name: npys}
    
    with torch.no_grad():
        ds_with_embeddings = ds.map(encode, batched=True, batch_size=args.batch_size, remove_columns=["text"])
    
    if args.save_vectors:
        logger.info("save vectors ...")
        save_vectors(ds_with_embeddings, args)
    else:
        logger.info("build index ...")
        ds_with_embeddings.add_faiss_index(column=args.index_name, metric_type=faiss.METRIC_INNER_PRODUCT) # !important metric_type=faiss.METRIC_INNER_PRODUCT
        
        logger.info("save index ...")
        ds_with_embeddings.save_faiss_index(args.index_name, os.path.join(args.output_dir, f"{args.index_name}.faiss"))

        logger.info("save passage ids to huggingface dataset ...")
        hf_ds = os.path.join(args.output_dir, "hf_ds")
        os.makedirs(hf_ds, exist_ok=True)
        ds_with_embeddings.drop_index(args.index_name)
        removed_columns = [col for col in ds_with_embeddings.column_names if col != "pid"]
        ds_ids = ds_with_embeddings.remove_columns(removed_columns) # only keep "pid" column
        ds_ids.save_to_disk(hf_ds)

def main(args):
    set_seed(args.seed)
    args.rank = 0 # single gpu, set rank to 0
    args.device = torch.cuda.current_device()
    os.makedirs(args.output_dir, exist_ok=True)

    args.num_langs = len(args.langs)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, from_slow=True)

    if args.use_pooler:
        base_encoder = AutoModel.from_pretrained(args.base_model_name, add_pooling_layer=True)
    else:
        base_encoder = AutoModel.from_pretrained(args.base_model_name, add_pooling_layer=False)
    if args.checkpoint and 'adapter' in args.checkpoint:
        model = mDPRAdapter(base_encoder, args)
    elif args.checkpoint and 'zc3' in args.checkpoint:
        model = mDPRZC3(base_encoder, args)
    elif args.fitter_checkpoint is not None or (args.checkpoint and 'leace' in args.checkpoint and 'self' not in args.checkpoint):
        model = mDPRScrubber(base_encoder, args)
    else:
        model = mDPRBase(base_encoder, args)
    model.to(args.device)

    # load checkpoint
    if args.checkpoint is not None:
        model.load(args.checkpoint)
    fitter = None
    if args.fitter_checkpoint is not None:
        fitter = LeaceFitter(model.base_encoder.config.hidden_size, args.num_langs, dtype=torch.float64, device=args.device)
        params = torch.load(args.fitter_checkpoint)
        fitter.__dict__ = params
    logger.info("model loaded")
    
    # read collection
    ds = read_collection(args)
    logger.info("dataset loaded")

    # indexing
    logger.info("begin indexing ...")
    indexing(model, tokenizer, args, ds, fitter=fitter)

    logger.info("done!")

if __name__ == "__main__":
    parser = get_index_parser()
    args = parser.parse_args()
    main(args)