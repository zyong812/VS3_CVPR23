import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from tqdm import tqdm
import re
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import json
import os
from maskrcnn_benchmark.data.datasets import VG150Dataset
import argparse
import sng_parser
from maskrcnn_benchmark.data.preprocess_utils import map_caption_concepts_to_vg150_categories

parser = argparse.ArgumentParser(description="Grounding")
parser.add_argument("--sg_parser", type=str, default='java')
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

vg_test_dataset = VG150Dataset(split='test',
                         img_dir="./DATASET/VG150/VG_100K",
                         roidb_file="./DATASET/VG150/VG-SGG-with-attri.h5",
                         dict_file="./DATASET/VG150/VG-SGG-dicts-with-attri.json",
                         image_file="./DATASET/VG150/image_data.json",
                         num_val_im=0, filter_empty_rels=False, filter_non_overlap=False,
                         filter_duplicate_rels=False)
print(len(vg_test_dataset))
coco_ids_in_vg_test = [x['coco_id'] for x in vg_test_dataset.img_info if x['coco_id'] is not None]

########################### caption data process ############################################
# parse text scene graphs from captions
if args.sg_parser == 'java':
    processed_coco_caption_file = 'DATASET/VG150/weak_supervisions/coco_caption_with_java_parsed_graph_grounding=glipL.json'
    with open('DATASET/coco/annotations/captions_train2017_with_java_parsed_sg.json', 'r') as fin:
        coco_captions = json.load(fin) # with java parsed SG
else:
    processed_coco_caption_file = 'DATASET/VG150/weak_supervisions/coco_caption_with_parsed_graph_grounding=glipL.json'
    with open('DATASET/coco/annotations/captions_train2017.json', 'r') as fin:
        coco_captions = json.load(fin) # original caption annotation

if os.path.exists(processed_coco_caption_file):
    with open(processed_coco_caption_file, 'r') as fin:
        coco_caption_scene_graph_info = json.load(fin)
else:
    # parse scene graphs
    coco_imgid2captions = {}
    caption_object_vocabs, caption_relation_vocabs = set(), set()
    if args.sg_parser == 'java':
        # for cap in tqdm(coco_captions['annotations'][:100]):
        for cap in tqdm(coco_captions['annotations']):
            image_id = cap['image_id']
            if image_id in coco_ids_in_vg_test: continue

            py_like_sg = {'entities': [], 'relations': []}
            cap_graph = json.loads(cap['text_scene_graph'])
            for e in cap_graph['objects']:
                char_span = e['char_span'][0]
                new_ent = {
                    'span': cap_graph['sentence'][char_span[0]:char_span[1]],
                    'char_span': char_span,
                    'lemma_head': e['names'][0],
                    'org_parsed_info': e
                }
                py_like_sg['entities'].append(new_ent)
                caption_object_vocabs.add(new_ent['lemma_head'])
            for r in cap_graph['relationships']:
                new_rel = {
                    'subject': r['subject'],
                    'object': r['object'],
                    'relation': r['predicate'], # todo: get original word, get char span
                    'lemma_relation': r['predicate'],
                    'org_parsed_info': r
                }
                py_like_sg['relations'].append(new_rel)
                caption_relation_vocabs.add(new_rel['lemma_relation'])
            cap['text_scene_graph'] = py_like_sg

            if image_id in coco_imgid2captions:
                coco_imgid2captions[image_id].append(cap)
            else:
                coco_imgid2captions[image_id] = [cap]
    elif args.sg_parser == 'python':
        # for cap in tqdm(coco_captions['annotations'][:100]):
        for cap in tqdm(coco_captions['annotations']):
            image_id = cap['image_id']
            if image_id in coco_ids_in_vg_test: continue

            cap_graph = sng_parser.parse(cap['caption']) # text parsing
            for e in cap_graph['entities']:
                caption_object_vocabs.add(e['lemma_head'])
            for r in cap_graph['relations']:
                caption_relation_vocabs.add(r['lemma_relation'])
            cap['text_scene_graph'] = cap_graph

            if image_id in coco_imgid2captions:
                coco_imgid2captions[image_id].append(cap)
            else:
                coco_imgid2captions[image_id] = [cap]

    coco_caption_scene_graph_info = {
        'img_captions_with_parsed_graph': coco_imgid2captions,
        'caption_object_vocabs': list(caption_object_vocabs),
        'caption_relation_vocabs': list(caption_relation_vocabs)
    }
    with open(processed_coco_caption_file, 'w', encoding='utf-8') as fout:
        json.dump(coco_caption_scene_graph_info, fout, ensure_ascii=False, indent=4)

print(f"#image = {len(coco_caption_scene_graph_info['img_captions_with_parsed_graph'])}")
print(f'#caption_object_vocabs={len(coco_caption_scene_graph_info["caption_object_vocabs"])}')
print(f'#caption_relation_vocabs={len(coco_caption_scene_graph_info["caption_relation_vocabs"])}')

# map caption objects/predicates to VG150 categories
if 'caption_object_vocabs_to_vg_objs' not in coco_caption_scene_graph_info:
    caption_object_vocabs_to_vg_objs, caption_relation_vocabs_to_vg_rels = map_caption_concepts_to_vg150_categories(
        coco_caption_scene_graph_info['caption_object_vocabs'], coco_caption_scene_graph_info['caption_relation_vocabs'], vg_test_dataset
    )
    coco_caption_scene_graph_info.update({
        'caption_object_vocabs_to_vg_objs': caption_object_vocabs_to_vg_objs,
        'caption_relation_vocabs_to_vg_rels': caption_relation_vocabs_to_vg_rels
    })
    with open(processed_coco_caption_file, 'w', encoding='utf-8') as fout:
        json.dump(coco_caption_scene_graph_info, fout, ensure_ascii=False, indent=4)

# mark valid images, captions and triplets for VG150 training
if 'vg150_valid_infos' not in coco_caption_scene_graph_info:
    vg150_valid_image_ids, vg150_valid_captions_count, vg150_valid_relation_triplet_count = [], 0, 0
    for img_id, caption_infos in tqdm(coco_caption_scene_graph_info['img_captions_with_parsed_graph'].items()):
        for cap in caption_infos:
            caption_text = cap['caption']

            # map to vg150 categories & find char spans & mark valid captions and relations for VG150 training
            matched_spans = []
            for ent in cap['text_scene_graph']['entities']:
                ent['vg150_obj_category'] = coco_caption_scene_graph_info['caption_object_vocabs_to_vg_objs'][ent['lemma_head']]
                for m in re.finditer(re.escape(ent['span']), caption_text):
                    ent_char_span = [m.start(), m.end()]
                    if ent_char_span not in matched_spans:
                        matched_spans.append(ent_char_span)
                        ent['char_span'] = ent_char_span
                        break

            matched_spans = []
            for rel in cap['text_scene_graph']['relations']:
                rel['vg150_rel_category'] = coco_caption_scene_graph_info['caption_relation_vocabs_to_vg_rels'][rel['lemma_relation']]
                for m in re.finditer(re.escape(rel['relation']), caption_text):
                    rel_char_span = [m.start(), m.end()]
                    if rel_char_span not in matched_spans:
                        matched_spans.append(rel_char_span)
                        rel['char_span'] = rel_char_span
                        break

                sub_ent = cap['text_scene_graph']['entities'][rel['subject']]
                obj_ent = cap['text_scene_graph']['entities'][rel['object']]
                if len(sub_ent['vg150_obj_category']) > 0 and len(obj_ent['vg150_obj_category']) > 0 and len(rel['vg150_rel_category']) > 0:
                    rel['vg150_valid'] = True
                    vg150_valid_relation_triplet_count += 1
                else:
                    rel['vg150_valid'] = False

            # caption valid for language-supervised SGG for VG150 when at least one valid relation exist
            valid_rels = [r for r in cap['text_scene_graph']['relations'] if r['vg150_valid']]
            if len(valid_rels) > 0:
                cap['vg150_valid'] = True
                vg150_valid_captions_count += 1
            else:
                cap['vg150_valid'] = False

        # image valid for language-supervised SGG for VG150 when at least one valid caption exist
        valid_caps = [c for c in caption_infos if c['vg150_valid']]
        if len(valid_caps) > 0:
            vg150_valid_image_ids.append(img_id)

    coco_caption_scene_graph_info.update(
        {
            'vg150_valid_infos': {
                'vg150_valid_image_ids': vg150_valid_image_ids,
                'vg150_valid_image_count': len(vg150_valid_image_ids),
                "vg150_valid_captions_count": vg150_valid_captions_count,
                "vg150_valid_relation_triplet_count": vg150_valid_relation_triplet_count
            }
        }
    )
    with open(processed_coco_caption_file, 'w', encoding='utf-8') as fout:
        json.dump(coco_caption_scene_graph_info, fout, ensure_ascii=False, indent=4)

vg150_valid_info = coco_caption_scene_graph_info['vg150_valid_infos']
print(f"\nvg150_valid_image_count={vg150_valid_info['vg150_valid_image_count']}")
print(f"vg150_valid_captions_count={vg150_valid_info['vg150_valid_captions_count']}")
print(f"vg150_valid_relation_triplet_count={vg150_valid_info['vg150_valid_relation_triplet_count']}")

########################### obtain grounded boxes of entities ###########################
if 'sg_grounding_infos' not in coco_caption_scene_graph_info:
    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.,
        show_mask_heatmaps=False
    )

    grounded_img_ids = []
    all_sg_box_count, grounded_sg_box_count = 0, 0
    coco_imgid2filename = {x['id']: x['file_name'] for x in coco_captions['images']}
    for index, (img_id, caption_infos) in enumerate(tqdm(coco_caption_scene_graph_info['img_captions_with_parsed_graph'].items())):
        if index % args.gpu_size != args.local_rank: continue
        grounded_img_ids.append(img_id)
        img_filename = f"DATASET/coco/train2017/{coco_imgid2filename[int(img_id)]}"
        pil_image = Image.open(img_filename).convert("RGB")
        input_image = np.array(pil_image)[:, :, [2, 1, 0]]

        for cap in caption_infos:
            if not cap['vg150_valid']: continue # only ground for vg150_valid captions
            caption = cap['caption']
            caption_char2inst = np.zeros(len(caption)) - 1
            for sg_ent_id, sg_ent in enumerate(cap['text_scene_graph']['entities']):
                caption_char2inst[sg_ent['char_span'][0]:sg_ent['char_span'][1]] = sg_ent_id
                # print(caption[sg_ent['char_span'][0]:sg_ent['char_span'][1]])
            all_sg_box_count += len(cap['text_scene_graph']['entities'])

            # grounding
            _, box_predictions = glip_demo.run_on_web_image(input_image, caption, 0., show_result=False)
            # imshow(_, caption)

            visual_boxes = box_predictions.bbox
            visual_boxes_ent_ids = box_predictions.get_field('labels')
            visual_boxes_scores = box_predictions.get_field('scores')
            for v_ent_id in visual_boxes_ent_ids.unique():
                s, ind = visual_boxes_scores[visual_boxes_ent_ids == v_ent_id].topk(1)
                ent_box = visual_boxes[visual_boxes_ent_ids == v_ent_id][ind[0]]

                # match with parsed SG entities
                ent_span = glip_demo.entities_caption_span[v_ent_id.item()-1][0]
                ent_inst_id = np.unique(caption_char2inst[ent_span[0]:ent_span[1]])
                matched_instance_id = int(ent_inst_id.max()) # check!!
                # print(f"{glip_demo.entities[v_ent_id.item()-1]} -> {cap['text_scene_graph']['entities'][matched_instance_id]}")

                if 'ground_box' in cap['text_scene_graph']['entities'][matched_instance_id]:
                    if s.item() > cap['text_scene_graph']['entities'][matched_instance_id]['ground_score']:
                        cap['text_scene_graph']['entities'][matched_instance_id].update({
                            'ground_box': ent_box.tolist(),
                            'ground_score': s.item(),
                            'ground_char_span': ent_span,
                        })
                else:
                    grounded_sg_box_count += 1
                    cap['text_scene_graph']['entities'][matched_instance_id].update({
                        'ground_box': ent_box.tolist(),
                        'ground_score': s.item(),
                        'ground_char_span': ent_span,
                    })

    coco_caption_scene_graph_info['sg_grounding_infos'] = {
        'all_sg_box_count': all_sg_box_count,
        'grounded_sg_box_count': grounded_sg_box_count,
        'grounded_img_ids': grounded_img_ids
    }
    with open(f"{processed_coco_caption_file.replace('.json', '')}_{args.gpu_size}-{args.local_rank}.json", 'w', encoding='utf-8') as fout:
        json.dump(coco_caption_scene_graph_info, fout, ensure_ascii=False, indent=4)

sg_grounding_infos = coco_caption_scene_graph_info['sg_grounding_infos']
print(f"\nall_sg_box_count={sg_grounding_infos['all_sg_box_count']}")
print(f"grounded_sg_box_count={sg_grounding_infos['grounded_sg_box_count']}")

