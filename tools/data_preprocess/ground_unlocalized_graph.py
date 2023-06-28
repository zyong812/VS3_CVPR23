import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from tqdm import tqdm
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import json
from maskrcnn_benchmark.data.datasets import VG150Dataset
import argparse

parser = argparse.ArgumentParser(description="Grounding")
parser.add_argument("--gpu_size", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

def load(url):
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.show()

# # Use this command for evaluate the GLPT-T model
# # ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth
# config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
# weight_file = "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"

# Use this command to evaluate the GLPT-L model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
config_file = "configs/pretrain/glip_Swin_L.yaml"
weight_file = "MODEL/glip_large_model.pth"

# update the config options with the config file
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.LANGUAGE_BACKBONE.TOKENIZER_LOCAL_FILES_ONLY", True])
cfg.merge_from_list(["MODEL.DYHEAD.NUM_CLASSES", 200])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.,
    show_mask_heatmaps=False
)

# grounding unlocalized scene graphs
# while True:
#     pil_image = Image.open("./DATASET/examples/000000022935.jpg").convert("RGB")
#     image = np.array(pil_image)[:, :, [2, 1, 0]]
#     caption = 'a women wearing blue t-shirt in front, and another women in yellow stand in behind'
#
#     result, box_predictions = glip_demo.run_on_web_image(image, caption, 0.)
#     imshow(result, caption)


########################### grounding VG unlocalized scene graphs ###########################
vg_dataset = VG150Dataset(split='train',
                         img_dir="./DATASET/VG150/VG_100K",
                         roidb_file="./DATASET/VG150/VG-SGG-with-attri.h5",
                         dict_file="./DATASET/VG150/VG-SGG-dicts-with-attri.json",
                         image_file="./DATASET/VG150/image_data.json",
                         num_val_im=0, filter_empty_rels=False, filter_non_overlap=False,
                         filter_duplicate_rels=False)
print(len(vg_dataset))

all_image_groundings = {}
for index in tqdm(range(len(vg_dataset))):
    # if index < 770: continue
    if index % args.gpu_size != args.local_rank: continue
    pil_image = Image.open(vg_dataset.filenames[index]).convert("RGB")
    gt = vg_dataset.get_groundtruth(index, evaluation=True)

    # caption from unlocalized triplets & objects
    obj_labels = gt.get_field("labels").tolist()
    relation_triplets = gt.get_field("relation_tuple").unique(dim=0)
    pseudo_caption, caption_range2inst = "", np.zeros(0)
    for (s, o, p) in relation_triplets:
        subj_name = vg_dataset.ind_to_classes[obj_labels[s]]
        obj_name = vg_dataset.ind_to_classes[obj_labels[o]]
        triplet_text = f"{subj_name} {vg_dataset.ind_to_predicates[p]} {obj_name}. "
        range2instance = np.zeros(len(triplet_text))-1
        range2instance[:len(subj_name)] = s
        range2instance[-len(obj_name)-2:-2] = o

        pseudo_caption += triplet_text
        caption_range2inst = np.concatenate([caption_range2inst, range2instance])

    if len(relation_triplets) > 0:
        rel_ins_ids = relation_triplets[:, :2].unique()
    else:
        rel_ins_ids = []
    for ins_id in range(len(obj_labels)):
        if ins_id not in rel_ins_ids:
            obj_name = vg_dataset.ind_to_classes[obj_labels[ins_id]]
            obj_text = f"{obj_name}. "

            range2instance = np.zeros(len(obj_text))-1
            range2instance[:len(obj_name)] = ins_id
            pseudo_caption += obj_text
            caption_range2inst = np.concatenate([caption_range2inst, range2instance])

    # GLIP grounding
    input_image = np.array(pil_image)[:, :, [2, 1, 0]]
    _, box_predictions = glip_demo.run_on_web_image(input_image, pseudo_caption, 0.)
    # imshow(_, pseudo_caption)

    entities_2_boxes, instanceid_boxes = {}, {}
    all_box_ent_ids = box_predictions.get_field('labels')
    all_boxes = box_predictions.bbox
    all_box_scores = box_predictions.get_field('scores')
    all_ent_ids = all_box_ent_ids.unique()
    for ent_id in all_ent_ids:
        s, ind = all_box_scores[all_box_ent_ids == ent_id].topk(1)
        ent_box = all_boxes[all_box_ent_ids == ent_id][ind[0]]

        ent_span = glip_demo.entities_caption_span[ent_id.item()-1][0]
        ent_inst_id = np.unique(caption_range2inst[ent_span[0]:ent_span[1]])
        # assert len(ent_inst_id) == 1
        matched_instance_id = int(ent_inst_id.max())

        entities_2_boxes[ent_id.item()] = {
            'instance_label': glip_demo.entities[ent_id.item()-1],
            'instance_id': matched_instance_id,
            'sore': s[0].item(),
            'bbox': ent_box.tolist(),
        }
        if matched_instance_id in instanceid_boxes:
            if s[0].item() > instanceid_boxes[matched_instance_id]['score']:
                instanceid_boxes[matched_instance_id] = {
                    'caption_instance_label': glip_demo.entities[ent_id.item()-1],
                    'score': s[0].item(),
                    'bbox': ent_box.tolist(),
                }
        else:
            instanceid_boxes[matched_instance_id] = {
                'caption_instance_label': glip_demo.entities[ent_id.item()-1],
                'score': s[0].item(),
                'bbox': ent_box.tolist(),
            }

    all_image_groundings[vg_dataset.img_info[index]['image_id']] = instanceid_boxes

# save to json files
save_file_name = f'vg150_unlocalized_graph_groundings_with_glipL_{args.gpu_size}-{args.local_rank}.json'
with open(save_file_name, 'w', encoding='utf-8') as f:
    json.dump(all_image_groundings, f, ensure_ascii=False, indent=4)
