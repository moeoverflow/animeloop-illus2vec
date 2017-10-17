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


def to_tags(filename, loopid):
    image = Image.open(filename)
    result = illust2vec.estimate_plausible_tags([image], threshold=0.5)[0]

    tag_shcema = {
        'loopid': ObjectId(loopid),
        'source': 'illustration2vec',
        'lang': 'en'
    }

    # Extract tags from database to memory.
    saved_tags = list(db.tags.find({'loopid': ObjectId(loopid)}))

    def exist_in_tagslist(loopid, type, value):
        for t in saved_tags:
            if str(t['loopid']) == loopid and t['type'] == type and t['value'] == value:
                return True
        return False

    for key in result.keys():
        for item in result[key]:
            tag = tag_shcema.copy()
            if key is 'rating':
                tag['type'] = 'safe'
            else:
                tag['type'] = key
            tag['value'] = item[0]
            tag['confidence'] = item[1]

            if not exist_in_tagslist(loopid, tag['type'], tag['value']):
                db.tags.insert_one(tag)

    db.tagscheck.insert_one({'loopid': ObjectId(loopid)})


saved_tagscheck = map(lambda tc: str(tc['loopid']), list(db.tagscheck.find({})))

logger.info('Loading files list...')
files = os.listdir(IMAGES_PATH)

progress_bar = tqdm(files, 'Estimating tags', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
for file in progress_bar:
    if not os.path.isdir(file):
        filename = IMAGES_PATH + '/' + file
        loopid = os.path.splitext(file)[0]

        if loopid not in saved_tagscheck:
            to_tags(filename, loopid)
