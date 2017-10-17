import os
import sys
import logging
from tqdm import tqdm
from illustration2vec import i2v
from PIL import Image
from pymongo import MongoClient
from bson.objectid import ObjectId

from config import config

logging.getLogger('illus2vec')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

IMAGES_PATH = config['images_path']


logger.info('Loading chainer models...')
illust2vec = i2v.make_i2v_with_chainer(
    "./illustration2vec/illust2vec_tag_ver200.caffemodel", "./illustration2vec/tag_list.json")
logger.info('Load completed.')


logger.info('Connect to database.')
client = MongoClient("localhost", 27017)
db = client.animeloop_tags


def to_tags(db, filename, loopid):
    image = Image.open(filename)
    result = illust2vec.estimate_plausible_tags([image], threshold=0.5)[0]

    tag_shcema = {
        'loopid': ObjectId(loopid),
        'source': 'illustration2vec',
        'lang': 'en'
    }

    for key in result.keys():
        for item in result[key]:
            tag = tag_shcema.copy()
            if key is 'rating':
                tag['type'] = 'safe'
            else:
                tag['type'] = key
            tag['value'] = item[0]
            tag['confidence'] = item[1]
            check = db.tags.find_one({'loopid': ObjectId(loopid), 'type': tag['type'], 'value': tag['value']})
            if check is None:
                db.tags.insert_one(tag)

    db.tagscheck.insert_one({'loopid': ObjectId(loopid)})


logger.info('Loading files list...')
files = os.listdir(IMAGES_PATH)

progress_bar = tqdm(files, 'Estimating tags', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
for file in progress_bar:
    if not os.path.isdir(file):
        filename = IMAGES_PATH + '/' + file
        loopid = os.path.splitext(file)[0]

        check = db.tagscheck.find_one({'loopid': ObjectId(loopid)})
        if check is None:
            to_tags(db, filename, loopid)
