import json, pickle
if __name__=='__main__':
    ann_path = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/cocoa/COCO_amodal_val2014_with_classes_amodal.json'

    with open(ann_path, "r") as f:
        ann_data = json.load(f)

    cat_list = []
    for ann_cat in ann_data['categories']:
        cat_list.append(ann_cat['name'])

    print(cat_list)
    with open('/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/cocoa/cocoa_cat_list', 'wb') as fp:
        pickle.dump(cat_list, fp)

