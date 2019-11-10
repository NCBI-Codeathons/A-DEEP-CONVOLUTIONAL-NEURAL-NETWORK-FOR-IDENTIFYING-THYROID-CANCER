import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import tqdm
from torchvision import datasets, models, transforms

def main(args: argparse.Namespace) -> None:
    BATCH_SIZE = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.arch == 'resnet':
        model_ft = models.resnet50(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 1)
        model_ft = model_ft.to(device)
        feature_model = nn.Sequential(*list(model_ft.children())[:-1])
    else:
        model_ft = models.mobilenet_v2(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.classifier = nn.Linear(num_ftrs,1)
        model_ft = model_ft.to(device)
        feature_model = lambda x: model_ft.features(x).mean([2, 3])
    
    model_file = os.path.abspath(args.model_file)
    model_ft.load_state_dict(torch.load(model_file))

    data_dir = os.path.abspath(args.data_dir)

    xform = transforms.Compose([
            transforms.CenterCrop((288,288)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    image_dataset = datasets.ImageFolder(data_dir, xform)

    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, num_workers=4)

    dataset_size = len(image_dataset)

    model_ft.eval()

    running_corrects = 0
    num_pred = 0
    out_file = os.path.abspath(args.out_file)
    with open(out_file, 'w') as out_fp:
        column_names = ['idx','prob0','target'] + ["val{}".format(x) for x in range(num_ftrs)]
        out_fp.write(','.join(column_names)+'\n')
        for i, (inputs, labels) in tqdm.tqdm(enumerate(dataloader)):
            inputs = inputs.to(device)
            probs = F.sigmoid(model_ft(inputs)).cpu()
            # features = model_ft.features(inputs).mean([2, 3]).cpu()
            features = feature_model(inputs).cpu()
            for offset in range(probs.shape[0]):
                out_arr = [i + offset, probs[offset].tolist()[0], labels[offset].tolist()]
                out_arr.extend(features[offset].squeeze().tolist())
                out_arr = [str(x) for x in out_arr]
                out_fp.write(",".join(out_arr)+'\n')


if __name__ == "__main__":
    '''
    /project/hackathon/hackers01/shared/Organized/Pathology/
    '''
    parser = argparse.ArgumentParser('Make metafile for pedraza Pedraza dataset')
    parser.add_argument('--data-dir', default=os.environ['DATA_DIR'])
    parser.add_argument('--arch',default='mobilenet',help='Model architecture')
    parser.add_argument('model_file', help='Pretrained model weights')
    parser.add_argument('out_file')
    args = parser.parse_args()

    main(args)