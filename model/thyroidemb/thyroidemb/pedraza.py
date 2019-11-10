import os
import glob
import argparse
import tqdm
import xml.etree.ElementTree as ET
import random
from typing import List, Dict
import re
from shutil import copyfile

random.seed(1337)

SPLIT_MAP = {
    '4b': 'non_benign',
    '4a': 'non_benign',
    '5': 'non_benign',
    '4c': 'non_benign',
    '2': 'benign',
    '3': 'benign',
}

def split_folders(args: argparse.Namespace) -> None:
    meta_file = os.path.abspath(args.meta_file)
    print('Looking for Meta file in {}'.format(meta_file))

    partitions: Dict[str, List[str]] = {}
    patient_files: Dict[str, List[str]] = {}
    skipped = True

    print("Accumulating class clounts")
    with open(meta_file, 'r') as meta_fp:
        for line in tqdm.tqdm(meta_fp):
            if skipped:
                skipped = False
            else:
                parts = re.split(r'\t',line)  # line.strip().split('\t')
                idx, filepath = parts[1], parts[0], 
                tirads = parts[6] if parts[6] != '' else 'UNK'

                patient_images = patient_files.get(idx, [])
                patient_images.append(filepath)
                patient_files[idx] = patient_images

                partition_tirad = partitions.get(tirads,[])
                partition_tirad.append(idx)
                partitions[tirads] = partition_tirad

    train: Dict[str, List[str]] = {}
    test: Dict[str, List[str]] = {}

    print(partitions)
    print("Found class counts of: ")
    for tirad, idxs in partitions.items():
        print("\t{}: {}".format(tirad, len(idxs)))
        random.shuffle(idxs)
        split_n = int(0.9 * len(idxs))
        train[tirad] = idxs[:split_n]
        test[tirad] = idxs[split_n:]

    def create_folder(partition: str, tirades: Dict[str, List[str]]):
        data_dir = os.path.abspath(args.data_dir)
        print('Looking for image files in {}'.format(data_dir))
        
        os.makedirs(os.path.join('pedraza',partition), exist_ok=True)
        partition_folder = os.path.abspath(os.path.join('pedraza',partition))

        print('Moving partition files to {}'.format(partition_folder))
        for tirad, idxs in tqdm.tqdm(tirades.items()):
            label_dir = SPLIT_MAP.get(tirad, None)
            if label_dir is None:
                continue
            os.makedirs(os.path.join('pedraza',partition,label_dir), exist_ok=True)
            for idx in idxs:
                for patient_file in patient_files[idx]:
                    copyfile(os.path.join(data_dir, patient_file),
                    os.path.join(partition_folder,label_dir, patient_file))

    create_folder('train', train)
    create_folder('test',test)


def build_meta(args: argparse.Namespace) -> None:
    data_dir = os.path.abspath(args.data_dir)
    print('Looking for XML files in {}'.format(data_dir))

    xml_files = sorted(glob.glob(os.path.join(data_dir,'*.xml')))
    print('Found {} files'.format(len(xml_files)))
    
    out_file = os.path.abspath(args.meta_file)
    with open(out_file, 'w') as out_fp:
        out_fp.write('filepath\tcomposition\tmargins\tcalcifications\ttirads\treportbacaf\treporteco\n')
        for xml_file in tqdm.tqdm(xml_files):
            idx = os.path.splitext(os.path.basename(xml_file))[0]
            with open(out_file, 'r') as xml_fp:
                root = ET.parse(xml_file).getroot()
                composition = root.find('composition').text or ''
                echogenicity = root.find('echogenicity').text or ''
                margins = root.find('margins').text or ''
                calcifications = root.find('calcifications').text or ''
                tirads = root.find('tirads').text or ''
                reportbacaf = root.find('reportbacaf').text or ''
                reporteco = root.find('reporteco').text or ''
                img_files = glob.glob(os.path.join(data_dir,"{}_*.jpg".format(idx)))
                for img_file in img_files:
                    out_fp.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        os.path.basename(img_file),
                        idx,
                        composition.strip().replace('\n',''),
                        echogenicity.strip().replace('\n',''),
                        margins.strip().replace('\n',''),
                        calcifications.strip().replace('\n',''),
                        tirads.strip().replace('\n',''),
                        reportbacaf.strip().replace('\n',''),
                        reporteco.strip().replace('\n','')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Make metafile for pedraza Pedraza dataset')
    parser.add_argument('--data-dir',default=os.environ['DATA_DIR'])
    parser.add_argument('--meta-file', default='pedraza.tsv', help='')
    parser.add_argument('cmd')
    args = parser.parse_args()

    if (args.cmd) == 'split':
        pass
        split_folders(args)
    else:
        build_meta(args)