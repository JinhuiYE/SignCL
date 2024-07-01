
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig

from hftrim.ModelTrimmers import MBartTrimmer
from PIL import Image
# import utils

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
from hftrim.TokenizerTrimmer import TokenizerTrimmer
import pickle
import os
import gzip
def build_data_file_format():

    import json

    # Assuming the labels files are in JSON format
    def read_labels_file(filename):
        split = filename
        filename = f"./data/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.{filename}.corpus.csv"


        import pandas as pd
        df = pd.read_csv(filename, delimiter="|")
        data_format = {}
        name = df["name"]
        gloss = df["orth"]
        text = df["translation"]
        for index in range(len(name)):
            file = os.path.join(split, name[index] )
            item = {
                "name": file,
                "gloss": gloss[index],
                "text": text[index],
                "length": -1,
                "imgs_path": None
            }

            data_format[file] = item

            pass



        return data_format

    def save_labels_file(filename, data):

        filename = f"./data/BBC/labels.{filename}"

        # # 保存gloss到prototype的映射
        # with open(filename, 'wb') as f:
        #     pickle.dump(data, f)

        with gzip.open(filename, "wb") as f:
            pickle.dump(data, f)

    def update_imgs_path(label_data, base_dir):
        updated_folders = []
        missing_folders = []
        for sample in label_data:
            sample = label_data[sample]
            folder_path = os.path.join(base_dir, sample['name'])
            if os.path.isdir(folder_path):
                images = sorted(os.listdir(folder_path))  # 确保顺序
                valid_images = []  # 用于存储可以成功读取的图片路径
                for img in images:
                    img_path = os.path.join(folder_path, img)
                    try:
                        with Image.open(img_path) as img_file:  # 尝试打开图片
                            valid_images.append(os.path.join(sample['name'], img))  # 如果成功，添加路径
                    except (IOError, OSError):
                        print(f"无法读取图片: {img_path}")  # 打印无法读取的图片路径，或进行其他处理
                if sample['length'] != len(valid_images):
                    sample['imgs_path'] = valid_images
                    sample['length'] = len(valid_images)
                    updated_folders.append(sample['name'])
            else:
                missing_folders.append(sample['name'])
        return updated_folders, missing_folders

    def process_labels(base_dir, labels_paths):
        for labels_path in labels_paths:
            label_data = read_labels_file(labels_path)
            updated_folders, missing_folders = update_imgs_path(label_data, base_dir)

            # Filter out samples with missing folders
            for mis_case in missing_folders:
                
                del label_data[mis_case]

            save_labels_file(f"{labels_path}", label_data)

            # print(f"Updated folders for {labels_path}: {updated_folders}")
            if missing_folders:
                print(f"Missing folders for {labels_path}: {missing_folders}")
            else:
                print(f"No missing folders for {labels_path}.")

    # Base directory where the train, dev, test folders are located
    base_dir = './data/PHOENIX-2014-T/features/fullFrame-210x260px'

    # Paths to the label files
    labels_paths = ['dev', 'test', 'train'] # train 只有 5K

    # Assuming the script is run from the directory containing the labels files
    process_labels(base_dir, labels_paths)

# build_data_file_format()


raw_data = load_dataset_file('data/Phonexi-2014T/labels.train')

data = []

for key,value in raw_data.items():
    sentence = value['text']
    # gloss = value['gloss']
    data.append(sentence)
    # data.append(gloss.lower())

tokenizer = MBartTokenizer.from_pretrained("./pretrain_models/MBart_proun", src_lang="de_DE", tgt_lang="de_DE")

model = MBartForConditionalGeneration.from_pretrained("./pretrain_models/MBart_proun")
configuration = model.config

# trim tokenizer
tt = TokenizerTrimmer(tokenizer)
tt.make_vocab(data)
tt.make_tokenizer()

# trim model
mt = MBartTrimmer(model, configuration, tt.trimmed_tokenizer)
mt.make_weights(tt.trimmed_vocab_ids)
mt.make_model()

new_tokenizer = tt.trimmed_tokenizer
new_model = mt.trimmed_model

new_tokenizer.save_pretrained('pretrain_models/MBart_trimmed')
new_model.save_pretrained('pretrain_models/MBart_trimmed')

## mytran_model
configuration = MBartConfig.from_pretrained('pretrain_models/mytran/config.json')
configuration.vocab_size = new_model.config.vocab_size
mytran_model = MBartForConditionalGeneration._from_config(config=configuration)
mytran_model.model.shared = new_model.model.shared

mytran_model.save_pretrained('pretrain_models/mytran/')











